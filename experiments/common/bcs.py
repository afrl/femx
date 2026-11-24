def clamp_fiber_ends(
    fabric: VTMSFabric,
    fiber_ids: int | Sequence[int] | None = None,
    components: tuple[int, ...] = (0, 1, 2),
    value: float = 0.0,
) -> list[DirichletBC]:
    """Clamps start and end nodes for target fibers in a fabric."""
    pass

def apply_integrated_line_load(
    fabric: VTMSFabric,
    fiber_id: int,
    component: int,
    force_func: Callable[[float], float],
    axis_coord_index: int = 2,
) -> list[NeumannBC]:
    """Integrates continuous line load along interior fiber nodes via Simpson's 1/3 rule."""
    pass

def apply_point_load(
    fabric: VTMSFabric,
    fiber_id: int,
    node_fraction: float = 0.5,
    value: float = 1.0,
    component: int = 0,
    span: int = 1,
) -> list[NeumannBC]:
    """Applies discrete point force at specified fractional node position along a fiber."""
    pass