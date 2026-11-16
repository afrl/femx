from fe_jax.helper import *
import pytest
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from copy import deepcopy
# jax.config.update("jax_disable_jit", True)
import jax.extend


if __name__ == '__main__':
    print(jax.extend.backend.get_backend().platform)

def make_boundary_conditions(
    fabric,
    maximum_Neumann_force: float,
):
    bcs = []

    f_n = lambda z,nf : nf*(np.exp(-(4*z)**2) - np.exp(-16))/(1-np.exp(-16))
    f = lambda z: f_n(z,maximum_Neumann_force)

    # Fix both ends of original bundle fibers.
    for i in fabric.fiber_offsets[:-1]:
        for c in range(3):
            bcs.append(DirichletBC(index=i, component=c, value=0.0, bc_type=BCType.NODE))

    for i in fabric.fiber_offsets[1:]:
        for c in range(3):
            bcs.append(DirichletBC(index=i - 1, component=c, value=0.0, bc_type=BCType.NODE))

    # Apply Neumann force.
    # for nf in NeumannForce:
    #     temp_bcs =deepcopy(bcs)
    #     temp_bcs += [
    #         NeumannBC(
    #             bc_type   = BCType.NODE,
    #             component = 1,
    #             index     = fabric.fiber_offsets[nf_fiber] + i + 1,
    #             value     = -f_n(z,nf),
    #         )
    #         for i,z in enumerate(fabric.points[fabric.fiber_offsets[nf_fiber]+1:fabric.fiber_offsets[nf_fiber+1]-1,2])
    #     ]
    for i in fabric.fiber_offsets[:-1]:
        start = fabric.fiber_offsets[i]
        final = fabric.fiber_offsets[i+1]
        for i in range(start+1,final-1):
            z = fabric.points[i-1:i+2,2]
            force = (z[-1]-z[0])/6*(f(z[0]) + 4*f(z[1]) + f(z[2]))
            bcs.append(NeumannBC(index=i, component=1, value=force, bc_type=BCType.NODE))

        # midpoint_index=int((fabric.fiber_offsets[i] + fabric.fiber_offsets[i+1])/2)
        # for s in (-1,0,1):
        #     bcs.append(NeumannBC(index=midpoint_index+s, component=1, value=maximum_Neumann_force, bc_type=BCType.NODE))

    return bcs

def make_bc_schedule(
    fabric,
    neumann_loads: list | np.ndarray,
    schedule: list[str] | tuple[str, ...] = ("LOAD",),
):
    dyn_bcs = []
    labels = []

    for k,nf in enumerate(neumann_loads):

        for stage in schedule:
            stage = stage.upper()
            if stage not in ("LOAD", "RELAX"):
                raise ValueError(f"Unknown schedule stage: {stage}")

            dyn_bcs.append(make_boundary_conditions(
                fabric=fabric,
                maximum_Neumann_force=nf
            ))
            labels.append((stage, k))

    return dyn_bcs, labels

def make_single_fiber(
    n_elements: int,
    x0: tuple,
    xN: tuple,
    fiber_id: int,
    cell_shift: int
):
    points = np.vstack(
        (
            np.linspace(x0[0], xN[0], n_elements + 1),
            np.linspace(x0[1], xN[1], n_elements + 1),
            np.linspace(x0[2], xN[2], n_elements + 1),
        )
    ).T
    cells = np.array(
        [[i + cell_shift, i + cell_shift + 1] for i in range(len(points) - 1)],
        dtype=np.uint64,
    )
    fiber_ids = np.array([[fiber_id]] * len(points))
    cell_ids = np.array([[fiber_id]] * len(cells))
    y_ = np.linspace(-1,1,points.shape[0])
    points[:,1] += 0.01*(1-y_)*(1+y_)
    return points, cells, fiber_ids, cell_ids


def make_bundle(n_elements: list[int], X0: list[tuple], XN: list[tuple],NeumannForce):
    point_blocks = []
    cell_blocks = []
    point_id_blocks = []
    cell_id_blocks = []

    bcs = []

    vertex_offset = 0

    for fiber_id, (n_el, x0, xN) in enumerate(zip(n_elements, X0, XN)):
        points_i, cells_i, point_ids_i, cell_ids_i = make_single_fiber(
            n_elements=n_el,
            x0=x0,
            xN=xN,
            fiber_id=fiber_id,
            cell_shift=vertex_offset,
        )

        point_blocks.append(points_i)
        cell_blocks.append(cells_i)
        point_id_blocks.append(point_ids_i)
        cell_id_blocks.append(cell_ids_i)
        vertex_offset += points_i.shape[0]

    points = np.vstack(point_blocks)
    cells = np.vstack(cell_blocks)
    point_ids = np.vstack(point_id_blocks).reshape(-1)
    cell_ids = np.vstack(cell_id_blocks).reshape(-1)

    fiber_offsets = np.concatenate(
        [
            [0],
            np.cumsum([b.shape[0] for b in point_blocks])
        ]
    )
    # fiber_offsets = np.cumsum([b.shape[0] for b in point_blocks])
    bundle = VTMSBundle(
        name="test",
        n_fibers=len(n_elements),
        material_id=np.array([0]),
        diameter=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        # bundle_offsets=np.array([0, fiber_offsets.shape[0]]),
    )
    fabric = VTMSFabric(
        name="test",
        material_ids=np.array([0]),
        diameters=np.array([0.1]),
        points=points,
        fiber_offsets=fiber_offsets,
        bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
    )
    return fabric

def run_singleFiber(
    n_elements: list[int],
    X0: list[tuple],
    XN: list[tuple],
    contact_params: ContactParams,
    NeumannForce,
    filename_base = None,
    pre_strain = None,
    strain_measure = StrainMeasure.GREEN_LAGRANGE,
    debug_info = None,
):
    """ """
    fabric = make_bundle(n_elements=n_elements, X0=X0, XN=XN,NeumannForce=NeumannForce)
    dyn_bcs, bc_labels = make_bc_schedule(fabric, NeumannForce)
    # for nf in NeumannForce:
    #     print(nf)
    #     temp_bcs =deepcopy(bcs)
    #     for bc in temp_bcs:
    #         if isinstance(bc,NeumannBC):
    #             bc.value = nf
    #     dyn_bcs.append(temp_bcs)

    d = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_dist = d[d.nonzero()].min()
    E = 1e9
    A = (fabric.diameters[0]/2)**2*np.pi
    print(f"EA/N = {E*A/NeumannForce}")
    print(f"{min(min_dist/2,fabric.diameters[0]/2)}")
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[VTMSFiberMaterial(id=0, E=E, A=A)],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.CG_JAX_SCIPY_W_INFO,
            # linear_precond_type=PreconditionerType.JACOBI,
            # linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=300,
            # nonlinear_absolute_tol=1e-8,
            # nonlinear_relative_tol=1e-8,
            linear_absolute_tol = 1e-12,
            linear_max_iter=500,
            max_linear_displacement=min(min_dist,fabric.diameters[0])/2,
            max_backtracks = 8,
            # damp_Newton_diag=None,
        ),
        plot_convergence=False,
        filename_base=filename_base,
        pseudotime_iters=len(dyn_bcs),
        debug_info = debug_info,
        pre_strain=pre_strain,
        strain_measure=strain_measure,
        # debug_info=make_debug_info(
        #     flags = [
        #         (DebugOutputQuantities.NODE_RESIDUAL,       DebugOutputStage.NONLINEAR_SOLVE),
        #         (DebugOutputQuantities.GLOBAL_JACOBIAN_COO, DebugOutputStage.NONLINEAR_SOLVE),
        #         (DebugOutputQuantities.ELEMENT_JACOBIAN,    DebugOutputStage.NONLINEAR_SOLVE),
        #         # (DebugOutputQuantities.ELEMENT_RESIDUAL,    DebugOutputStage.NONLINEAR_SOLVE),
        #     ],
        #     filename = filename_base + '.h5'
        # ) if filename_base is not None else None
    )
    u = u.reshape((-1,3))
    #fabric.points = fabric.points + u

    D_D = np.linalg.norm(fabric.points[None,:,:]-fabric.points[:,None,:],axis=-1)
    min_d = D_D[D_D.nonzero()].min()
    return u,fabric,dyn_bcs

def get_args(incrementalLoads=10, strain_measure=StrainMeasure.GREEN_LAGRANGE, N = 40):
    args = {
        'n_elements':[N],
        'X0':[[0, 0, -1]],
        'XN':[[0, 0, 1]],
        'NeumannForce':np.linspace(1e4,1e5,incrementalLoads),
        # 'NeumannForce':[(i+1)*1e4 for i in range(10)],
        # 'NeumannForce':[1e5],
        # 'NeumannForce':[i*1e4 for i in range(10,101)],
        # 'filename_base':'ContactStiffnessModel/Linear_NeumannTest',
        # 'filename_base':  'PseudoTimeNeumann_in_y/incrementalLoad',
        'filename_base': None,
        'contact_params': ContactParams(
            self_adjacency_block       = 10000,
            contact_constitutive_model = elastic_contact_truss_piecewise_quadratic,
            D_stiffness_to_E_ratio     = 0.25,
            M_to_D_ratio               = 1.0,
            C_to_D_ratio               = 0.75,
            M_stiffness_to_E_ratio     = 0.0000001,
            contact_search_alpha       = 3,
            contact_backend            = ContactBackend.SCIPY_KDTREE,
        ),
        'pre_strain':0.0,
        'strain_measure':strain_measure,
    }
    return args


# args10 = get_args()
# args20 = get_args()
# args20['NeumannForce'] = np.linspace(1e4,1e5,20)
# X10 = run_singleFiber(**args10)
# X20 = run_singleFiber(**args20)
# print(f'drift error: {np.max(np.abs(X10[0]-X20[0]))}')

def nonlinear_solve_summary(strain_measures=('Linear','GreenLagrange', 'Logarithmic', 'Engineering'), load_steps_to_plot=(20, 40, 60),incremental_loads = range(2, 61)):
    if strain_measures is None:
        strain_measures = ('GreenLagrange', 'Logarithmic', 'Engineering')
    elif isinstance(strain_measures, str):
        strain_measures = (strain_measures,)

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    

    linestyle_pool = ['solid', 'dashed', 'dashdot', 'dotted']
    linestyles = [linestyle_pool[i % len(linestyle_pool)] for i in range(len(strain_measures))]

    stat_info = [
        ('minimum', 'tab:blue'),
        ('maximum', 'tab:orange'),
        ('median',  'tab:green'),
        ('average', 'tab:purple'),
    ]

    step_color_pool = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
    step_colors = [step_color_pool[i % len(step_color_pool)] for i in range(len(load_steps_to_plot))]

    el_data = ''

    for strain_measure, ls in zip(strain_measures, linestyles):
        min_steps = []
        max_steps = []
        med_steps = []
        avg_steps = []
        el_data = '_40elements'
        for i in incremental_loads:
            f = h5py.File(f'debug/singleFiberExternalLoad/CG_{strain_measure}_{i}loadsteps{el_data}.h5')
            nls = get_num_nonlinear_solves(f)
            min_steps.append(nls.min())
            max_steps.append(nls.max())
            med_steps.append(np.median(nls))
            avg_steps.append(np.average(nls))
            f.close()

        ax[0].semilogy(incremental_loads, min_steps, color='tab:blue', linestyle=ls)
        ax[0].semilogy(incremental_loads, max_steps, color='tab:orange', linestyle=ls)
        ax[0].semilogy(incremental_loads, med_steps, color='tab:green', linestyle=ls)
        ax[0].semilogy(incremental_loads, avg_steps, color='tab:purple', linestyle=ls)

        for step_count, c in zip(load_steps_to_plot, step_colors):
            f = h5py.File(f'debug/singleFiberExternalLoad/CG_{strain_measure}_{step_count}loadsteps{el_data}.h5')
            nls = get_num_nonlinear_solves(f)
            ax[1].semilogy(range(1, step_count + 1), nls, color=c, linestyle=ls)
            f.close()

    ax[0].set_xlabel('Number of incremental loads')
    ax[0].set_ylabel('Number on nonlinear solves')

    # Legend for Plot 1
    handles_0 = [
        Line2D([], [], color='none', label=r'$\bf{Statistics:}$'),
    ]
    for name, col in stat_info:
        handles_0.append(Line2D([0], [0], color=col, linestyle='solid', label=name))
    handles_0.append(Line2D([], [], color='none', label=''))
    handles_0.append(Line2D([], [], color='none', label=r'$\bf{Strain\ Measure:}$'))
    for sm, ls in zip(strain_measures, linestyles):
        handles_0.append(Line2D([0], [0], color='black', linestyle=ls, label=sm))
    ax[0].legend(handles=handles_0)

    ax[1].set_xlabel('Load Step')
    ax[1].set_ylabel('Number of nonlinear solves')

    # Legend for Plot 2
    handles_1 = [
        Line2D([], [], color='none', label=r'$\bf{Load\ Steps:}$'),
    ]
    for step_count, col in zip(load_steps_to_plot, step_colors):
        handles_1.append(Line2D([0], [0], color=col, linestyle='solid', label=f'{step_count} steps'))
    handles_1.append(Line2D([], [], color='none', label=''))
    handles_1.append(Line2D([], [], color='none', label=r'$\bf{Strain\ Measure:}$'))
    for sm, ls in zip(strain_measures, linestyles):
        handles_1.append(Line2D([0], [0], color='black', linestyle=ls, label=sm))
    ax[1].legend(handles=handles_1)

    plt.tight_layout()
    plt.savefig(f'scratch/CG_{el_data}_NonlinearSolves_vs_IncrementalLoad.png')
    plt.close()
# nonlinear_solve_summary()