"""Diagnostic-only profile of PETSc versus JAX CSR CG on the saved snapshot.

First run test_microscale_linear_snapshot_oriented_JetSCI.py to create the NPZ.
Then, from fea-in-jax on Linux/CUDA:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -s -q \
    tests/test_microscale_csr_ksp_profile_JetSCI.py

This test launches a fresh worker with PETSc GPU event timing enabled. These
timings are INSTRUMENTED, not replacement headline wall times. The original
snapshot benchmark is unchanged. No algorithm/tolerance/PC tuning is applied.

FEA_JETSCI_KSP_PROFILE_REPEATS=3; FEA_JETSCI_KSP_PROFILE_TRACE=1 (JAX trace).
Set TRACE=0 explicitly if only PETSc event data is wanted. Optional external
Nsight Systems capture: nsys profile --trace=cuda,nvtx,osrt -o csr_ksp_profile
  python -m pytest -s -q tests/test_microscale_csr_ksp_profile_JetSCI.py
When using nsys, set TRACE=0 to avoid two simultaneous CUPTI collectors, and
enable tracing of child processes in the installed nsys version if necessary.
"""

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock
from time import perf_counter

import cupy as cp
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from jax.experimental.buffer_callback import buffer_callback
import numpy as np
from petsc4py import PETSc
import pytest
from scipy import sparse

import jetsci
from jetsci.lifecycle import build_linear_solver_with_reuse, destroy_linear_solver

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.slow


EVENT_GROUPS = {
    "sparse_matvec": ("MatMult", "MatMultTranspose"),
    "preconditioner_inclusive": ("PCApply", "PCApplyTranspose"),
    "vector_updates": ("VecAXPY", "VecAYPX", "VecAXPBY", "VecWAXPY", "VecMAXPY",
                       "VecScale", "VecCopy", "VecSet", "VecPointwiseMult", "VecPointwiseDivide"),
    "dot_products_and_norms": ("VecDot", "VecTDot", "VecMDot", "VecMTDot", "VecNorm", "VecDotNorm2"),
    "solve_inclusive": ("KSPSolve", "KSPSolveTranspos"),
}


@contextmanager
def profile_range(name):
    # NVTX labels are also visible to an externally launched Nsight profiler.
    cp.cuda.nvtx.RangePush(name)
    try:
        with jax.profiler.TraceAnnotation(name):
            yield
    finally:
        cp.cuda.nvtx.RangePop()


def ready(value):
    return jax.tree_util.tree_map(
        lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, value,
    )


def profile_call(name, function, arguments):
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    with profile_range(name):
        result = ready(function(*arguments))
        cp.cuda.runtime.deviceSynchronize()
    return result, perf_counter() - start


def event_info(event, stage):
    # Preserve every numeric field exposed by this installed petsc4py, rather
    # than assuming that its Python wrapper exposes GPU-time fields. The full
    # native log below contains GPU timings even if getPerfInfo does not.
    raw = event.getPerfInfo(stage.id)
    return {key: (float(value) if np.isfinite(value) else None)
            for key, value in raw.items() if isinstance(value, (int, float, np.number))}


def test_microscale_csr_ksp_profile():
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("This profile supports one MPI rank/device only.")
    if os.environ.get("FEA_JETSCI_KSP_PROFILE_WORKER") != "1":
        child_env = os.environ.copy()
        child_env["FEA_JETSCI_KSP_PROFILE_WORKER"] = "1"
        child_env["PETSC_OPTIONS"] = (child_env.get("PETSC_OPTIONS", "") +
                                      " -log_view -log_view_gpu_time").strip()
        print("KSP_PROFILE_WORKER " + json.dumps({
            "petsc_options": child_env["PETSC_OPTIONS"],
            "note": "GPU event timing is enabled only in this diagnostic worker; it can perturb performance.",
        }), flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-s", "-q", str(Path(__file__).resolve())],
            env=child_env, cwd=Path(__file__).resolve().parents[1], check=False,
        )
        assert result.returncode == 0, f"KSP profile worker failed (exit {result.returncode}); see child output above."
        return
    run_profile()


def run_profile():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("Requires CUDA JAX/PETSc; no CPU fallback.")
    snapshot_path = Path(os.environ.get("FEA_JETSCI_SNAPSHOT_PATH",
                         Path(__file__).parent / "output" / "microscale_linear_snapshot_oriented.npz")).resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Missing {snapshot_path}. Run test_microscale_linear_snapshot_oriented_JetSCI.py first.")
    output = Path(os.environ.get("FEA_JETSCI_KSP_PROFILE_DIR",
                  Path(__file__).parent / "output" / "microscale_csr_ksp_profile")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeats = int(os.environ.get("FEA_JETSCI_KSP_PROFILE_REPEATS", "3"))
    trace_flag = os.environ.get("FEA_JETSCI_KSP_PROFILE_TRACE", "1")
    if repeats < 1 or trace_flag not in ("0", "1"):
        raise ValueError("Require PROFILE_REPEATS>=1 and PROFILE_TRACE=0 or 1.")
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        metadata = json.loads(str(snapshot["metadata"]))
        host = sparse.csr_matrix((snapshot["data"].copy(), snapshot["indices"].copy(),
                                  snapshot["indptr"].copy()), shape=tuple(snapshot["shape"]))
        right_sides = {"forward": jnp.asarray(snapshot["forward_rhs"].copy()),
                       "adjoint": jnp.asarray(snapshot["adjoint_rhs"].copy())}
    diagonal_host = host.diagonal()
    if not np.isfinite(host.data).all() or np.any(diagonal_host <= 0):
        raise ValueError("Expected the corrected snapshot with finite entries and positive Jacobi diagonal; regenerate the oriented snapshot. No shift/fallback applied.")
    rtol, atol, maxiter = metadata["rtol"], metadata["atol"], metadata["maxiter"]
    csr = jsparse.CSR((jnp.asarray(host.data), jnp.asarray(host.indices, dtype=jnp.int32),
                      jnp.asarray(host.indptr, dtype=jnp.int32)), shape=host.shape)
    diagonal = jnp.asarray(diagonal_host)
    options = jetsci.LinearSolverOptions(
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_relative_tol=rtol, linear_absolute_tol=atol, linear_max_iter=maxiter,
    )
    host_coo = host.tocoo()
    solver, options = build_linear_solver_with_reuse(options, jetsci.COOData(
        shape=jnp.asarray(host.shape), rows=jnp.asarray(host_coo.row),
        cols=csr.indices, vals=csr.data,
    ))
    lock = RLock()
    try:
        solver.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        solver.ksp.setUp()
        PETSc.Log.begin()
        events = {name: PETSc.Log.Event(name) for names in EVENT_GROUPS.values() for name in names}
        print("KSP_PROFILE_CONFIG " + json.dumps({
            "snapshot": str(snapshot_path), "n": host.shape[0], "nnz": host.nnz,
            "method": solver.ksp.getType(), "pc": solver.ksp.getPC().getType(),
            "matrix_type": solver.operator_matrix.getType(), "rtol": rtol, "atol": atol,
            "maxiter": maxiter, "repeats": repeats, "output": str(output),
            "gpu_time_option_present": PETSc.Options().hasName("log_view_gpu_time"),
            "note": "Diagnostic timings include logging/GPU timing overhead. Events are inclusive; do not sum parent/child groups. JAX fusion may combine multiple CG operations.",
        }), flush=True)

        def functions_for(transpose):
            def callback(ctx, out, b):
                with lock:
                    solver.solve_from_device_buffers(out=out, rhs=b, transpose=transpose)
                    if solver.ksp.getConvergedReason() <= 0:
                        raise RuntimeError(f"Profiled PETSc CG failed: reason={solver.ksp.getConvergedReason()}, iterations={solver.ksp.getIterationNumber()}.")

            @jax.jit
            def petsc(b, operator, diag):
                return buffer_callback(callback, jax.ShapeDtypeStruct(b.shape, b.dtype), vmap_method="sequential")(b)

            @jax.jit
            def native_jax(b, operator, diag):
                A = (lambda v: operator.T @ v) if transpose else (lambda v: operator @ v)
                return jax.scipy.sparse.linalg.cg(A, b, M=lambda v: v / diag,
                                                 tol=rtol, atol=atol, maxiter=maxiter)[0]
            return {"PETSc_device_callback": petsc, "JAX_CSR": native_jax}

        prepared = {}
        for label, b in right_sides.items():
            functions = functions_for(label == "adjoint")
            arguments = (b, csr, diagonal)
            # Compile/warm everything before starting a trace or dedicated stage.
            for function in functions.values():
                for _ in range(3):
                    ready(function(*arguments))
            prepared[label] = (functions, arguments)

        # Timeline events identify each backend/system separately. GPU kernels,
        # CUDA API waits, and launch gaps are attributed using this trace or an
        # externally collected Nsight report, not inferred from wall residuals.
        trace_dir = output / f"jax_trace_{os.getpid()}"
        trace_context = (jax.profiler.trace(str(trace_dir), create_perfetto_trace=True)
                         if trace_flag == "1" else nullcontext())
        with trace_context:
            for label, (functions, arguments) in prepared.items():
                for name, function in functions.items():
                    samples = []
                    for repeat in range(repeats):
                        solution, elapsed = profile_call(f"snapshot/{label}/{name}/{repeat}", function, arguments)
                        samples.append(elapsed)
                    operator = host.T if label == "adjoint" else host
                    b_host = np.asarray(right_sides[label])
                    residual_norm = float(np.linalg.norm(operator @ np.asarray(solution) - b_host))
                    bound = max(atol, rtol * float(np.linalg.norm(b_host)))
                    assert residual_norm <= 5 * bound, (label, name, residual_norm, bound)
                    print("KSP_PROFILE_WALL " + json.dumps({
                        "system": label, "backend": name, "instrumented_samples_s": samples,
                        "instrumented_median_s": float(np.median(samples)),
                        "iterations": solver.ksp.getIterationNumber() if name.startswith("PETSc") else None,
                        "true_residual_norm": residual_norm, "requested_bound": bound,
                    }), flush=True)

        for label, b in right_sides.items():
            # Raw KSP call uses stable Vec views: no callback, wrapper creation,
            # RHS assignment, matrix update, or result conversion per solve.
            b_cp = cp.from_dlpack(b, copy=False)
            x_cp = cp.empty_like(b_cp)
            b_vec = PETSc.Vec().createWithDLPack(b_cp, size=b_cp.size)
            x_vec = PETSc.Vec().createWithDLPack(x_cp, size=x_cp.size)
            stage = PETSc.Log.Stage(f"snapshot_{label}_raw_ksp_{os.getpid()}")
            try:
                print("KSP_PROFILE_VECTORS " + json.dumps({"system": label,
                      "rhs_type": b_vec.getType(), "output_type": x_vec.getType()}), flush=True)
                samples = []
                stage.push()
                try:
                    for repeat in range(repeats):
                        cp.cuda.runtime.deviceSynchronize()
                        solver.ksp.setInitialGuessNonzero(False)
                        start = perf_counter()
                        with profile_range(f"snapshot/{label}/PETSc_raw_ksp/{repeat}"):
                            if label == "adjoint":
                                solver.ksp.solveTranspose(b_vec, x_vec)
                            else:
                                solver.ksp.solve(b_vec, x_vec)
                            cp.cuda.runtime.deviceSynchronize()
                        samples.append(perf_counter() - start)
                        assert solver.ksp.getConvergedReason() > 0, (label, solver.ksp.getConvergedReason())
                finally:
                    stage.pop()
                operator = host.T if label == "adjoint" else host
                residual_norm = float(np.linalg.norm(operator @ cp.asnumpy(x_cp) - np.asarray(b)))
                bound = max(atol, rtol * float(jnp.linalg.norm(b)))
                assert residual_norm <= 5 * bound, (label, residual_norm, bound)
                print("KSP_PROFILE_RAW " + json.dumps({"system": label, "samples_s": samples,
                      "median_s": float(np.median(samples)), "iterations": solver.ksp.getIterationNumber(),
                      "true_residual_norm": residual_norm, "requested_bound": bound}), flush=True)
                event_rows = {}
                for group, names in EVENT_GROUPS.items():
                    for name in names:
                        info = event_info(events[name], stage)
                        event_rows[name] = info
                        if info.get("count", 0):
                            print("KSP_PROFILE_EVENT " + json.dumps({"system": label,
                                  "group": group, "event": name, "stage_totals": info,
                                  "profiled_solves": repeats}), flush=True)
                solve_event = "KSPSolveTranspos" if label == "adjoint" else "KSPSolve"
                if not event_rows[solve_event].get("count", 0):
                    raise RuntimeError(f"No {solve_event} event was logged. Verify PETSc was built with logging enabled; do not interpret missing events as zero cost.")
            finally:
                x_vec.destroy()
                b_vec.destroy()
        log_path = output / f"petsc_events_{os.getpid()}.txt"
        viewer = PETSc.Viewer().createASCII(str(log_path), comm=PETSc.COMM_SELF)
        try:
            PETSc.Log.view(viewer)
        finally:
            viewer.destroy()
        print("KSP_PROFILE_FILES " + json.dumps({"petsc_log": str(log_path),
              "jax_trace": str(trace_dir) if trace_flag == "1" else None,
              "note": "Use GPU columns in the native PETSc log; event Python dictionaries may expose only host elapsed time. Missing fields are not zero. Synchronization attribution requires the GPU timeline."}), flush=True)
    finally:
        destroy_linear_solver(options.solver_key)
