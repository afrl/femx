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


def build_custom_hex(R, d, center=(0, 0), rotation=0):
    def horizontal_centers(n, d):
        if not n % 2:
            hc = [(i + 1 / 2) * d for i in range(n // 2)]
            hc = [-i for i in reversed(hc)] + hc
        else:
            hc = [(i + 1) * d for i in range(n // 2)]
            hc = [-i for i in reversed(hc)] + [0] + hc
        return hc

    H = np.vstack(
        [
            np.vstack([[hc, d * i * (np.sqrt(3) / 2)] for hc in horizontal_centers(r, d)])
            for i, r in enumerate(R)
        ]
    )
    # rotate
    R_mat = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    H = np.matmul(R_mat, H.T).T
    # recenter
    H = H - np.mean(H, axis=0) + np.array(center)
    return H


def make_boundary_conditions(fabric):
    bcs = []

    # Fix both ends of original bundle fibers.
    for i in fabric.fiber_offsets[:-1]:
        for c in range(3):
            bcs.append(DirichletBC(index=i, component=c, value=0.0, bc_type=BCType.NODE))

    for i in fabric.fiber_offsets[1:]:
        for c in range(3):
            bcs.append(DirichletBC(index=i - 1, component=c, value=0.0, bc_type=BCType.NODE))

    return bcs


def make_bc_schedule(
    fabric,
    old_fibers_n: int,
    old_points_n: int,
    fabric_n: int,
    n_load_steps: int,
    dir_step: float,
    schedule: list[str] | tuple[str, ...] = ("LOAD", "RELAX"),
):
    dyn_bcs = []
    labels = []

    for k in range(1, n_load_steps + 1):
        pin_y = k * dir_step

        for stage in schedule:
            stage = stage.upper()
            if stage not in ("LOAD", "RELAX"):
                raise ValueError(f"Unknown schedule stage: {stage}")

            dyn_bcs.append(make_boundary_conditions(fabric=fabric))
            labels.append((stage, k))

    return dyn_bcs, labels


def run_tension(
    fabric: VTMSBundle | VTMSFabric | None = None,
    fabric_path: str = "experiments/initial_single_fiber/initial_single_fiber.fab",
    contact_model: jax.tree_util.Partial | str = elastic_contact_truss_tanh,
    contact_params: ContactParams | None = None,
    strain_measure: StrainMeasure | str = StrainMeasure.LINEAR,
    nonlinear_max_iter: int = 25,
    linear_max_iter: int = 200,
    linear_solve_type: LinearSolverType = LinearSolverType.CG_JAX_SCIPY_W_INFO,
    pseudoT: int = 60,
    pre_strain: float | list[float] | np.ndarray | None = None,
    date_str: str | None = None,
    filename_base: str | None = None,
    debug_info: DebugInfo | NullDebugInfo = NULL_DEBUG_INFO,
    rigid_mold=None,
    use_profiler: bool = True,
    prof_file: str | None = None,
):
    """
    Executes the AngleInterlock tension simulation.
    """
    if fabric is None:
        fabric = read_fabric(fabric_path)

    contact_fn, contact_name = _resolve_contact_model(contact_model)
    strain_enum, strain_name = _resolve_strain_measure(strain_measure)

    fabric = refine_fabric(fabric, dX=0.25)
    fabric = refine_tow(fabric, [2, 3, 2])
    dx = fabric.diameters.min() / 2
    fabric = refine_fabric(fabric, dX=dx)
    print(f"num points: {len(fabric.points)}")

    if contact_params is None:
        contact_params = ContactParams(
            self_adjacency_block=10000,
            contact_constitutive_model=contact_fn,
            D_stiffness_to_E_ratio=8.0,
            M_stiffness_to_E_ratio=0.001 if contact_name == "tanh" else 0.0,
            M_to_D_ratio=1.0,
            C_to_D_ratio=0.9,
            contact_search_alpha=2.0,
        )

    if not isinstance(debug_info, NullDebugInfo):
        debug_info.file.attrs['contact_stiffness_model'] = contact_name
        debug_info.file.attrs['contact_D_stiffness_to_E_ratio'] = contact_params.D_stiffness_to_E_ratio
        debug_info.file.attrs['contact_M_to_D_ratio'] = contact_params.M_to_D_ratio
        debug_info.file.attrs['contact_M_stiffness_to_E_ratio'] = contact_params.M_stiffness_to_E_ratio
        debug_info.file.attrs['contact_self_adjacency_block'] = contact_params.self_adjacency_block

    if pre_strain is None:
        pre_strain = 0.141373887 * np.linspace(0, 35, pseudoT + 1)[1:]
        pre_strain = np.asarray([[ps]*3 for ps in pre_strain]).flatten()
        pseudoT = pseudoT*3

    dyn_bcs = [make_boundary_conditions(fabric)] * pseudoT
    print('dynamic boundary conditions generated!')

    E = 0.5
    tow_A = (fabric.get_diameter(0) / 2) ** 2 * np.pi

    solver_short = linear_solve_type.name.split("_")[0]
    if date_str is None:
        date_str = f"{datetime.now():%b}{datetime.now().day}"

    if filename_base is None:
        filename_base = f"AngleInterlock/{date_str}/{strain_name}/{contact_name}/NL{nonlinear_max_iter:03d}_{pseudoT:02d}steps_{solver_short}"

    if prof_file is None:
        prof_file = f"prof/AngleInterlock/{date_str}/{strain_name}/{contact_name}/NL{nonlinear_max_iter:03d}_{pseudoT:02d}steps_{solver_short}.json"

    profiler_name = f"AngleInterlock ({contact_name}, {strain_name}, NL={nonlinear_max_iter}, steps={pseudoT})"

    def _execute():
        return solve_fiber_mechanics_bvp(
            fabric=fabric,
            rigid_mold=rigid_mold,
            materials=[
                VTMSFiberMaterial(
                    id=fabric.get_material_id(i),
                    E=E,
                    A=np.pi * (fabric.get_diameter(i) / 2) ** 2,
                )
                for i in range(fabric.get_n_bundles())
            ],
            boundary_conditions=dyn_bcs,
            contact_options=contact_params,
            solver_options=SolverOptions(
                linear_solve_type=linear_solve_type,
                nonlinear_max_iter=nonlinear_max_iter,
                linear_max_iter=linear_max_iter,
                max_backtracks=20,
                max_linear_displacement=min(dx, fabric.diameters.min()) / 1,
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
        print(f"Running AngleInterlock with SegmentProfiler ('{profiler_name}')...")
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
    parser = argparse.ArgumentParser(description="Run AngleInterlock tension simulation.")
    parser.add_argument(
        "--strain-measure",
        type=str,
        default="linear",
        choices=["green_lagrange", "linear", "engineering", "logarithmic"],
        help="Strain measure formulation (default: linear)",
    )
    parser.add_argument(
        "--contact-model",
        type=str,
        default="tanh",
        choices=list(CONTACT_MODELS.keys()),
        help="Contact constitutive model (default: tanh)",
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
        default=60,
        help="Number of pseudotime load steps (default: 60)",
    )
    parser.add_argument(
        "--date-str",
        type=str,
        default=None,
        help="Date string folder name under AngleInterlock/ (default: current DayMonth e.g. Nov4)",
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
    run_tension(
        strain_measure=cli_args.strain_measure,
        contact_model=cli_args.contact_model,
        nonlinear_max_iter=cli_args.nonlinear_max_iter,
        pseudoT=cli_args.pseudoT,
        date_str=cli_args.date_str,
        filename_base=cli_args.filename_base,
        use_profiler=not cli_args.no_profile,
    )

