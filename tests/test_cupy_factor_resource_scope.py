"""Factor ownership tests with fake CuPy objects; no GPU allocations."""
import importlib
from contextlib import nullcontext, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from helper import *
import pytest

linear = importlib.import_module("fe_jax.sparse_linear_solve")


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(linear, "_OBJECT_STORE", {})
    monkeypatch.setattr(linear, "_RESOURCE_GROUPS", {})
    synchronize = Mock()
    monkeypatch.setattr(linear, "cp", SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda _: nullcontext(),
        runtime=SimpleNamespace(deviceSynchronize=synchronize),
    )), raising=False)
    return synchronize


def factor():
    return SimpleNamespace(L=SimpleNamespace(data=SimpleNamespace(device=SimpleNamespace(id=0))))


def test_release_only_owned_factors_and_synchronize(registry):
    linear._RESOURCE_GROUPS[42] = set()
    owned = linear.__store_object(factor(), 42)
    unrelated = linear.__store_object(factor())
    linear._release_resource_group(42)
    assert int(owned) not in linear._OBJECT_STORE
    assert int(unrelated) in linear._OBJECT_STORE
    assert 42 not in linear._RESOURCE_GROUPS
    registry.assert_called_once()


@pytest.mark.parametrize("fail", [False, True])
def test_forward_scope_releases_after_return_or_exception(registry, fail):
    settings = linear.SolverOptions(
        linear_precond_type=linear.PreconditionerType.ILU_CUPY,
        linear_solve_type=linear.LinearSolverType.GMRES_JAX_SCIPY_CSR,
    )

    @linear.with_forward_solver_resources
    def fake_bvp(solver_options=settings, _resource_owner=None):
        linear.__store_object(factor(), _resource_owner)
        if fail:
            raise FloatingPointError("nonfinite residual")
        return jnp.ones(2)

    if fail:
        with pytest.raises(FloatingPointError, match="nonfinite"):
            fake_bvp()
    else:
        np.testing.assert_array_equal(fake_bvp(), np.ones(2))
    assert not linear._OBJECT_STORE
    assert not linear._RESOURCE_GROUPS
    registry.assert_called_once()


def test_unknown_owner_does_not_leak_factor(registry):
    with pytest.raises(RuntimeError, match="outlived"):
        linear.__store_object(factor(), 999)
    assert not linear._OBJECT_STORE


def test_synchronized_callback_fences_input_and_output(monkeypatch):
    events = []

    class Output:
        device = SimpleNamespace(id=0)

        def __setitem__(self, key, value):
            events.append("write_output")

    def retrieve(handle):
        events.append("read_factor")
        return SimpleNamespace(solve=lambda rhs: events.append("solve"))

    monkeypatch.setattr(linear, "__retrieve_object", retrieve)
    monkeypatch.setattr(linear, "cp", SimpleNamespace(
        asarray=lambda value: value,
        cuda=SimpleNamespace(Device=lambda _: nullcontext(), runtime=SimpleNamespace(
            deviceSynchronize=lambda: events.append("synchronize"),
        )),
    ))
    linear.__cupy_solve_impl(None, Output(), 0, object(), synchronize=True)
    assert events == ["synchronize", "read_factor", "solve", "write_output", "synchronize"]


def test_same_stream_callback_enqueues_all_work_on_callback_stream(monkeypatch):
    events = []
    active = set()

    @contextmanager
    def scope(name):
        active.add(name)
        events.append("enter_" + name)
        yield
        active.remove(name)
        events.append("exit_" + name)

    class Output:
        def __dlpack_device__(self):
            return (2, 1)

        def __setitem__(self, key, value):
            assert active == {"device", "stream"}
            events.append("write")

    def from_external(adapter):
        assert adapter.__cuda_stream__() == (0, 1234)
        assert active == {"device"}
        return scope("stream")

    def asarray(value):
        assert active == {"device", "stream"}
        return value

    def solve(rhs):
        assert active == {"device", "stream"}
        events.append("solve")
        return rhs

    fake_factor = factor()
    fake_factor.L.data.device.id = 1
    fake_factor.solve = solve
    synchronize = Mock()
    monkeypatch.setattr(linear, "__retrieve_object", lambda _: fake_factor)
    monkeypatch.setattr(linear, "cp", SimpleNamespace(
        asarray=asarray, cuda=SimpleNamespace(
            Device=lambda device: scope("device"),
            Stream=SimpleNamespace(from_external=from_external),
            runtime=SimpleNamespace(deviceSynchronize=synchronize),
        ),
    ))
    linear.__cupy_solve_impl(SimpleNamespace(stream=1234), Output(), 0, object(), use_jax_stream=True)
    assert events == ["enter_device", "enter_stream", "solve", "write", "exit_stream", "exit_device"]
    synchronize.assert_not_called()


def test_same_stream_requires_callback_stream():
    with pytest.raises(RuntimeError, match="no stream handle"):
        linear.__JaxCudaStream(SimpleNamespace())


def test_same_stream_requires_cupy_external_stream_api(monkeypatch):
    monkeypatch.setattr(linear, "cp", SimpleNamespace(cuda=SimpleNamespace(Stream=SimpleNamespace())))
    with pytest.raises(RuntimeError, match="CuPy 14"):
        linear.__cupy_solve_impl(None, None, 0, None, use_jax_stream=True)


def test_ilu_stream_options_reject_conflicting_modes():
    with pytest.raises(ValueError, match="Select either"):
        linear.SolverOptions(ilu_synchronize=True, ilu_use_jax_stream=True)


def test_explicit_scope_keeps_factors_until_scope_exit(registry):
    with linear.cupy_factor_scope() as owner:
        handle = linear.__store_object(factor(), owner)
        assert int(handle) in linear._OBJECT_STORE
    assert not linear._OBJECT_STORE
    assert not linear._RESOURCE_GROUPS
    registry.assert_called_once()


@pytest.mark.parametrize("values, bound, expected", [
    ([1., 2.], 3., True), ([1., 2.], 1., False),
    ([float("inf")], float("inf"), False),
    ([1.], float("inf"), False), ([float("nan")], 1., False),
    ([1e200, 1e200], float("inf"), False),
])
def test_acceptance_rejects_nonfinite_norm_or_bound(values, bound, expected):
    jax.config.update("jax_enable_x64", True)
    assert bool(linear.finite_residual_accepted(jnp.asarray(values), bound)) is expected
