from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np
from experiments.common.generators import create_fiber_bundle
# jax.config.update("jax_disable_jit", True)


# def make_fabric(n_elements: int):
#     x = np.linspace(0,1,n_elements+1)
#     y = x*(1-x)/2.5

#     points = np.vstack([x,y]).T
#     fiber_offsets = np.concatenate([[0],np.cumsum([points.shape[0]])])

#     fabric = VTMSFabric(
#         name="test",
#         material_ids=np.array([0]),
#         diameters=np.array([0.1]),
#         points=points,
#         fiber_offsets=fiber_offsets,
#         bundle_offsets=np.array([0, fiber_offsets.shape[0]-1]),
#     )
#     return fabric

def run_2D_flattening(
    n_elements: int,
    pseudoT: int = 1,
    filename_base: str | None = None,
    contact_params: ContactParams | None = None,
    pre_strain: float = 0.0,
    debug_info: DebugInfo | NullDebugInfo = NULL_DEBUG_INFO,
):
    """ """
    x = np.linspace(0,1,n_elements+1)
    y_shift = x * (1 - x) / 2.5
    fabric = create_fiber_bundle(
        n_elements=n_elements,
        X0=[(0.0, 0.0)],
        XN=[(1.0, 0.0)],
        y_shifts=y_shift,
    )
    dyn_bcs = [[
        DirichletBC(index=0,          component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=0,          component=1, value=0, bc_type=BCType.NODE),
        DirichletBC(index=n_elements, component=0, value=0, bc_type=BCType.NODE),
        DirichletBC(index=n_elements, component=1, value=0, bc_type=BCType.NODE),
    ]]

    E = 1e9
    A = 1
    u, _, _ = solve_fiber_mechanics_bvp(
        fabric=fabric,
        materials=[
            VTMSFiberMaterial(id=int(fabric.get_material_id(i)), E=E, A=A) for i in range(fabric.get_n_bundles())
        ],
        boundary_conditions=dyn_bcs,
        contact_options=contact_params,
        solver_options=SolverOptions(
            linear_solve_type=LinearSolverType.SPSOLVE_PYPARDISO,
            nonlinear_max_iter=50,
            linear_max_iter=500,
        ),
        pseudotime_iters=pseudoT,
        filename_base=filename_base,
        blow_up_threshold=1e6,
        plot_convergence=False,
        pre_strain=pre_strain,
        debug_info=debug_info,
    )
    if not isinstance(debug_info, NullDebugInfo): debug_info.file.close()
    return u,fabric,dyn_bcs

args = {
    'n_elements':10,
    'contact_params': None,
    'pseudoT':1,
    'filename_base':None,
    # 'debug_info':make_debug_info(
    #     flags = [
    #         (DebugOutputQuantities.NODE_SOLUTION,DebugOutputStage.NONLINEAR_SOLVE),
    #         (DebugOutputQuantities.GLOBAL_JACOBIAN_COO,DebugOutputStage.NONLINEAR_SOLVE),
    #     ],
    #     filename = 'prestrain/test_global_jac_coo.h5'
    # )
}

def plot_nodes(fabric,file):
    plt.plot(*fabric.points.T,marker='x',label = 'nl_0')
    for k in list(file['ts_0'].keys())[1:]:
        plt.plot(*(fabric.points + file['ts_0'][k]['NODE_SOLUTION']['u'][:,:]).T,marker = 'x', label = k)
    plt.legend()

if __name__ == "__main__":
    run_2D_flattening(
        n_elements=args['n_elements'],
        pseudoT=args['pseudoT'],
        filename_base=args['filename_base'],
        contact_params=args['contact_params'],
        pre_strain=0.01,
    )
