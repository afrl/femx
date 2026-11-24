import argparse
from datetime import datetime
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from fe_jax.helper import *
from fe_jax.profiling import SegmentProfiler


CONTACT_MODELS = {
    "tanh": elastic_contact_truss_tanh,
    "quadratic": elastic_contact_truss_piecewise_quadratic,
    "piecewise_quadratic": elastic_contact_truss_piecewise_quadratic,
    "piecewise_linear": elastic_contact_truss_piecewise_linear,
    "exponential": elastic_contact_truss_exponential,
    "constant": elastic_contact_truss_constant,
    "linear": elastic_contact_truss_linear,
}

DEFAULT_CYLINDER_POINTS = np.array([
    [5.000000000, 0.860000000,  2.000000000],
    [5.000000000, 0.860000000,  1.200000006],
    [5.000000000, 0.860000000,  0.400000003],
    [5.000000000, 0.860000000, -0.400000003],
    [5.000000000, 0.860000000, -1.200000006],
    [5.000000000, 0.860000000, -2.000000000],
])


def _resolve_contact_model(contact_input: jax.tree_util.Partial | str):
    if isinstance(contact_input, str):
        key = contact_input.lower().strip()
        if key not in CONTACT_MODELS:
            raise ValueError(
                f"Unknown contact model '{contact_input}'. Available choices: {list(CONTACT_MODELS.keys())}"
            )
        return CONTACT_MODELS[key], key

    # Extract name from Partial
    model = contact_input
    name = "custom"
    if hasattr(model, "args") and len(model.args) > 0:
        arg0 = model.args[0]
        if hasattr(arg0, "func"):
            name = arg0.func.__name__.lstrip("_").replace("contact_stiffness_", "")
        elif hasattr(arg0, "__name__"):
            name = arg0.__name__.lstrip("_").replace("contact_stiffness_", "")
    elif hasattr(model, "__name__"):
        name = model.__name__.lstrip("_")
    return model, name


def _resolve_strain_measure(strain_input: StrainMeasure | str):
    strain_measure = StrainMeasure.from_value(strain_input)
    strain_name = strain_measure.name.lower()
    return strain_measure, strain_name


def make_boundary_conditions(
    fabric,
    old_fibers_n: int,
    old_points_n: int,
    fabric_n: int,
    pin_y: float,
):
    bcs = []

    # Fix both ends of original bundle fibers.
    for i in fabric.fiber_offsets[:old_fibers_n]:
        for c in range(3):
            bcs.append(DirichletBC(index=i, component=c, value=0.0, bc_type=BCType.NODE))

    for i in fabric.fiber_offsets[1:old_fibers_n + 1]:
        for c in range(3):
            bcs.append(DirichletBC(index=i - 1, component=c, value=0.0, bc_type=BCType.NODE))

    # Prescribe pin/cylinder nodes.
    for i in range(old_points_n, fabric_n):
        bcs.append(DirichletBC(index=i, component=0, value=0.0, bc_type=BCType.NODE))
        bcs.append(DirichletBC(index=i, component=1, value=pin_y, bc_type=BCType.NODE))
        bcs.append(DirichletBC(index=i, component=2, value=0.0, bc_type=BCType.NODE))

    return bcs


def make_bc_schedule(
    fabric,
    old_fibers_n: int,
    old_points_n: int,
    fabric_n: int,
    n_load_steps: int,
    dir_step: float,
    schedule: list[str] | tuple[str, ...] = ("LOAD",),
):
    dyn_bcs = []
    labels = []

    for k in range(1, n_load_steps + 1):
        pin_y = dir_step

        for stage in schedule:
            stage = stage.upper()
            if stage not in ("LOAD", "RELAX"):
                raise ValueError(f"Unknown schedule stage: {stage}")

            dyn_bcs.append(make_boundary_conditions(
                fabric=fabric,
                old_fibers_n=old_fibers_n,
                old_points_n=old_points_n,
                fabric_n=fabric_n,
                pin_y=pin_y,
            ))
            labels.append((stage, k))

    return dyn_bcs, labels


def run_mold(
    fabric: VTMSBundle | VTMSFabric | None = None,
    fabric_path: str = "experiments/sandboxes/rigidmold/pin_and_bundle.bdb",
    contact_model: jax.tree_util.Partial | str = elastic_contact_truss_piecewise_quadratic,
    contact_params: ContactParams | None = None,
    strain_measure: StrainMeasure | str = StrainMeasure.GREEN_LAGRANGE,
    nonlinear_max_iter: int = 75,
    linear_max_iter: int = 300,
    linear_solve_type: LinearSolverType = LinearSolverType.CG_JAX_SCIPY_W_INFO,
    damp_Newton_diag: float = 1.0,
    pseudoT: int = 30,
    dir_step: float = -0.02,
    pre_strain: float | list[float] | None = 0.141373887,
    cylinder_points: np.ndarray | None = None,
    cylinder_diameter: float = 1.0,
    date_str: str | None = None,
    filename_base: str | None = None,
    debug_info: DebugInfo | NullDebugInfo = NULL_DEBUG_INFO,
    use_profiler: bool = True,
    prof_file: str | None = None,
):
    """
    Executes the 19Tow rigid pin mold simulation.
    """
    if fabric is None:
        fabric = read_fib(fabric_path)

    contact_fn, contact_name = _resolve_contact_model(contact_model)
    strain_enum, strain_name = _resolve_strain_measure(strain_measure)

    if cylinder_points is None:
        cylinder_points = DEFAULT_CYLINDER_POINTS

    cylinder_points = np.asarray(cylinder_points, dtype=np.float64)
    if cylinder_points.ndim != 2 or cylinder_points.shape[1] != fabric.points.shape[1]:
        raise ValueError("cylinder_points must have shape (N, D), matching fabric.points")

    old_points_n = fabric.points.shape[0]
    old_fibers_n = fabric.fiber_offsets.shape[0] - 1
    old_bundles_n = fabric.get_n_bundles()
    cylinder_material_id = old_bundles_n

    if isinstance(fabric, VTMSBundle):
        fabric = VTMSFabric(
            name=fabric.name,
            material_ids=np.array([fabric.material_id, cylinder_material_id]),
            diameters=np.array([fabric.diameter, cylinder_diameter]),
            points=np.vstack([fabric.points, cylinder_points]),
            fiber_offsets=np.concatenate([
                fabric.fiber_offsets,
                [old_points_n + cylinder_points.shape[0]],
            ]),
            bundle_offsets=np.array([0, old_fibers_n, old_fibers_n + 1]),
        )
    else:
        fabric = VTMSFabric(
            name=fabric.name,
            material_ids=np.concatenate([fabric.material_ids, [cylinder_material_id]]),
            diameters=np.concatenate([fabric.diameters, [cylinder_diameter]]),
            points=np.vstack([fabric.points, cylinder_points]),
            fiber_offsets=np.concatenate([
                fabric.fiber_offsets,
                [old_points_n + cylinder_points.shape[0]],
            ]),
            bundle_offsets=np.concatenate([fabric.bundle_offsets, [old_fibers_n + 1]]),
        )

    fabric_n = fabric.points.shape[0]
    rigid_mold = None

    if contact_params is None:
        contact_params = ContactParams(
            self_adjacency_block=30000,
            contact_constitutive_model=contact_fn,
            D_stiffness_to_E_ratio=6.0,
            M_stiffness_to_E_ratio=0.001 if contact_name == "tanh" else 0.0,
            M_to_D_ratio=1.00,
            C_to_D_ratio=0.745,
            contact_search_alpha=2.0,
            contact_backend=ContactBackend.AUTO,
        )

    if not isinstance(debug_info, NullDebugInfo):
        debug_info.file.attrs['contact_stiffness_model'] = contact_name
        debug_info.file.attrs['contact_D_stiffness_to_E_ratio'] = contact_params.D_stiffness_to_E_ratio
        debug_info.file.attrs['contact_M_to_D_ratio'] = contact_params.M_to_D_ratio
        debug_info.file.attrs['contact_M_stiffness_to_E_ratio'] = contact_params.M_stiffness_to_E_ratio
        debug_info.file.attrs['contact_self_adjacency_block'] = contact_params.self_adjacency_block

    dyn_bcs, stage_labels = make_bc_schedule(
        fabric=fabric,
        old_fibers_n=old_fibers_n,
        old_points_n=old_points_n,
        fabric_n=fabric_n,
        n_load_steps=pseudoT,
        dir_step=dir_step,
        schedule=("LOAD","RELAX"),
    )

    d = np.linalg.norm(fabric.points[None, :, :] - fabric.points[:, None, :], axis=-1)
    min_dist = d[d.nonzero()].min()
    print('dynamic boundary conditions generated!')

    E = 0.5
    tow_A = (fabric.get_diameter(0) / 2) ** 2 * np.pi
    cylinder_A = (cylinder_diameter / 2) ** 2 * np.pi
    max_disp = min(min_dist, fabric.diameters[0]) / 2
    print(f"Max linear displacement limit: {max_disp}")

    solver_short = linear_solve_type.name.split("_")[0]
    if date_str is None:
        date_str = f"{datetime.now():%b}{datetime.now().day}"

    if filename_base is None:
        filename_base = f"rigid_mold/{date_str}/{strain_name}/{contact_name}/NL{nonlinear_max_iter:03d}_{pseudoT:02d}steps_{solver_short}"

    if prof_file is None:
        prof_file = f"prof/rigid_mold/{date_str}/{strain_name}/{contact_name}/NL{nonlinear_max_iter:03d}_{pseudoT:02d}steps_{solver_short}.json"

    profiler_name = f"19Tow ({contact_name}, {strain_name}, NL={nonlinear_max_iter}, steps={pseudoT})"

    def _execute():
        return solve_fiber_mechanics_bvp(
            fabric=fabric,
            rigid_mold=rigid_mold,
            materials=[
                VTMSFiberMaterial(id=0, E=E, A=tow_A),
                VTMSFiberMaterial(id=1, E=E, A=cylinder_A),
            ],
            boundary_conditions=dyn_bcs,
            contact_options=contact_params,
            solver_options=SolverOptions(
                linear_solve_type=linear_solve_type,
                nonlinear_max_iter=nonlinear_max_iter,
                linear_max_iter=linear_max_iter,
                damp_Newton_diag=damp_Newton_diag,
                max_backtracks=8,
                max_linear_displacement=max_disp,
            ),
            plot_convergence=False,
            filename_base=filename_base,
            pseudotime_iters=len(dyn_bcs),
            blow_up_threshold=10,
            pre_strain=pre_strain,
            debug_info=debug_info,
            strain_measure=strain_enum,
        )

    if use_profiler:
        print(f"Running 19Tow with SegmentProfiler ('{profiler_name}')...")
        with SegmentProfiler(name=profiler_name, sync_jax=True) as prof:
            u, _, _ = _execute()
        print("\n")
        prof.print_summary()
        os.makedirs(os.path.dirname(prof_file), exist_ok=True)
        prof.to_json(prof_file)
        print(f"\nSaved profile JSON to {prof_file}")
    else:
        u, _, _ = _execute()

    u = u.reshape((-1, 3))

    if not isinstance(debug_info, NullDebugInfo):
        print('close debug HDF5 file')
        debug_info.file.close()

    return u, fabric, dyn_bcs


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Run 19Tow rigid mold simulation.")
    parser.add_argument(
        "--strain-measure",
        type=str,
        default="green_lagrange",
        choices=["green_lagrange", "linear", "engineering", "logarithmic"],
        help="Strain measure formulation (default: green_lagrange)",
    )
    parser.add_argument(
        "--contact-model",
        type=str,
        default="piecewise_quadratic",
        choices=list(CONTACT_MODELS.keys()),
        help="Contact constitutive model (default: piecewise_quadratic)",
    )
    parser.add_argument(
        "--nonlinear-max-iter",
        type=int,
        default=75,
        help="Maximum nonlinear iterations per load step (default: 75)",
    )
    parser.add_argument(
        "--pseudoT",
        type=int,
        default=30,
        help="Number of pseudotime load steps (default: 30)",
    )
    parser.add_argument(
        "--dir-step",
        type=float,
        default=-0.02,
        help="Displacement increment per step for mold pin (default: -0.02)",
    )
    parser.add_argument(
        "--damp-Newton-diag",
        type=float,
        default=1.0,
        help="Newton diagonal damping coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--date-str",
        type=str,
        default=None,
        help="Date string folder name under rigid_mold/ (default: current DayMonth e.g. Nov3)",
    )
    parser.add_argument(
        "--filename-base",
        type=str,
        default=None,
        help="Base path/name for output files (auto-generated if not specified)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable SegmentProfiler timing breakdown",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _parse_cli_args()
    run_mold(
        strain_measure=cli_args.strain_measure,
        contact_model=cli_args.contact_model,
        nonlinear_max_iter=cli_args.nonlinear_max_iter,
        pseudoT=cli_args.pseudoT,
        dir_step=cli_args.dir_step,
        damp_Newton_diag=cli_args.damp_Newton_diag,
        date_str=cli_args.date_str,
        filename_base=cli_args.filename_base,
        use_profiler=not cli_args.no_profile,
    )

