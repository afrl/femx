"""Diagnose CG reductions on the saved, oriented microscale sparse snapshot.

Linux/CUDA, from fea-in-jax (no new FEA assembly required)::

  nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
    /home/alberto/venvs/mpi-gpu/bin/python -m pytest -s -q \
    tests/test_microscale_cg_reductions_JetSCI.py > cg_reductions.out 2>&1 &

Defaults: 5 repeats, 3 warmups, 100 primitive operations. Separate fresh workers
collect clean timings, PETSc event timings WITHOUT CUPTI, and short traces
WITHOUT PETSc GPU event timing. Controls:
FEA_JETSCI_CG_REPEATS, FEA_JETSCI_CG_WARMUPS, FEA_JETSCI_CG_PRIMITIVES,
FEA_JETSCI_CG_EVENTS=0 to omit event timing, FEA_JETSCI_CG_TRACE=0 to omit the
trace worker (also use TRACE=0 with external Nsight), FEA_JETSCI_CG_DIR,
FEA_JETSCI_SNAPSHOT_PATH. All output also goes into JSONL files in CG_DIR.
Only baseline and single/natural variants are traced. Each capture contains ONE
warmed raw-PETSc or JAX solve for ONE RHS; setup/calibration are never traced.
FEA_JETSCI_CG_CALIBRATION_TARGET=0.8 selects a residual safety margin for natural
norm calibration; must be in (0, 1). It does NOT change the acceptance bound.

Four CG/Jacobi configurations: ordinary/single-reduction x unpreconditioned/
natural norm. A natural norm changes the stopping test: calibrate tolerances
OUTSIDE timings, retain the original true Euclidean residual target, and print
every calibration attempt. Failed configurations are never presented as wins;
collect other configurations before failing the test. No matrix shifts, dense
conversion, automatic solver substitution, or production implementation edits.
This isolates forward/transpose LINEAR solves, not full AD or nonlinear solves.

References: https://petsc.org/release/manualpages/KSP/KSPCGUseSingleReduction/
https://petsc.org/release/manualpages/KSP/KSP_NORM_NATURAL/
https://petsc.org/release/manualpages/Log/PetscLogGpuTime/
"""

from contextlib import contextmanager
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
from jax.experimental import sparse as jsparse
from jax.experimental.buffer_callback import buffer_callback
import numpy as np
from petsc4py import PETSc
import pytest
from scipy import sparse

import jetsci
from jetsci.lifecycle import build_linear_solver_with_reuse, destroy_linear_solver
from test_microscale_csr_ksp_profile_JetSCI import (
    EVENT_GROUPS, event_info, profile_range, ready,
)

jax.config.update("jax_enable_x64", True)
pytestmark = pytest.mark.slow


def timed(function):
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    result = ready(function())
    cp.cuda.runtime.deviceSynchronize()
    return result, perf_counter() - start


@contextmanager
def log_stage(stage):
    if stage is not None:
        stage.push()
    try:
        yield
    finally:
        if stage is not None:
            stage.pop()


def test_microscale_cg_reductions():
    if PETSc.COMM_WORLD.getSize() != 1:
        pytest.skip("This diagnostic requires exactly one MPI rank and one CUDA device.")
    if os.environ.get("FEA_JETSCI_CG_WORKER"):
        run_worker(os.environ["FEA_JETSCI_CG_WORKER"])
        return
    if any(os.environ.get(key, "1") not in ("0", "1")
           for key in ("FEA_JETSCI_CG_EVENTS", "FEA_JETSCI_CG_TRACE")):
        raise ValueError("Set FEA_JETSCI_CG_EVENTS and FEA_JETSCI_CG_TRACE to 0 or 1.")
    # PETSc options must be set before worker imports/initialization. Do not
    # reuse a process once GPU event timing has been enabled.
    if os.environ.get("PETSC_OPTIONS", "").strip():
        raise ValueError("Unset PETSC_OPTIONS for controlled clean timings; this test configures PETSc explicitly.")
    failed = []
    modes = ["clean"]
    if os.environ.get("FEA_JETSCI_CG_EVENTS", "1") == "1":
        modes.append("events")
    if os.environ.get("FEA_JETSCI_CG_TRACE", "1") == "1":
        modes.append("trace")
    for mode in modes:
        env = os.environ.copy()
        env["FEA_JETSCI_CG_WORKER"] = mode
        env["PETSC_OPTIONS"] = "-log_view_gpu_time" if mode == "events" else ""
        print("CG_WORKER " + json.dumps({"mode": mode}), flush=True)
        child = subprocess.run([sys.executable, "-m", "pytest", "-s", "-q",
                                str(Path(__file__).resolve())], env=env,
                               cwd=Path(__file__).resolve().parents[1], check=False)
        if child.returncode:
            failed.append({"mode": mode, "exit": child.returncode})
    assert not failed, f"CG diagnostic workers failed: {failed}. Send cg_reductions.out and CG_DIR artifacts."


def run_worker(mode):
    if mode not in ("clean", "events", "trace"):
        raise ValueError(f"Unknown diagnostic worker mode {mode!r}.")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA JAX/PETSc required; no CPU fallback.")
    repeats = int(os.environ.get("FEA_JETSCI_CG_REPEATS", "5"))
    warmups = int(os.environ.get("FEA_JETSCI_CG_WARMUPS", "3"))
    primitives = int(os.environ.get("FEA_JETSCI_CG_PRIMITIVES", "100"))
    trace_flag = os.environ.get("FEA_JETSCI_CG_TRACE", "1")
    calibration_target = float(os.environ.get("FEA_JETSCI_CG_CALIBRATION_TARGET", "0.8"))
    if min(repeats, warmups, primitives) < 1 or trace_flag not in ("0", "1"):
        raise ValueError("Require REPEATS/WARMUPS/PRIMITIVES>=1 and TRACE=0 or 1.")
    if not 0 < calibration_target < 1:
        raise ValueError("FEA_JETSCI_CG_CALIBRATION_TARGET must be in (0, 1), e.g. 0.8; the true-residual acceptance bound remains unchanged.")
    if mode == "trace":
        repeats = 1
    output = Path(os.environ.get("FEA_JETSCI_CG_DIR", Path(__file__).parent / "output" / "cg_reductions")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = f"{mode}_{os.getpid()}"
    json_path = output / f"{stamp}.jsonl"
    failures = []
    with json_path.open("w", encoding="utf-8") as stream:
        def emit(kind, **data):
            record = {"kind": kind, "mode": mode, **data}
            line = json.dumps(record, default=str, allow_nan=False)
            print("CG_" + kind + " " + line, flush=True)
            stream.write(line + "\n")
            stream.flush()

        path = Path(os.environ.get("FEA_JETSCI_SNAPSHOT_PATH", Path(__file__).parent / "output" /
                                   "microscale_linear_snapshot_oriented.npz")).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run test_microscale_linear_snapshot_oriented_JetSCI.py first.")
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            host = sparse.csr_matrix((saved["data"].copy(), saved["indices"].copy(), saved["indptr"].copy()),
                                     shape=tuple(saved["shape"]))
            rhs_host = {name: saved[name + "_rhs"].copy() for name in ("forward", "adjoint")}
        diag_host = host.diagonal()
        if (host.shape[0] != host.shape[1] or not np.isfinite(host.data).all()
                or not np.isfinite(diag_host).all() or np.any(diag_host <= 0)):
            raise ValueError("Expected a finite square oriented snapshot with positive Jacobi diagonal; regenerate it. No repair applied.")
        asymmetry = host - host.T
        symmetry_error = float(sparse.linalg.norm(asymmetry) / sparse.linalg.norm(host))
        if symmetry_error > 1e-10:
            raise ValueError(f"CG requires symmetry; relative matrix asymmetry={symmetry_error}. Do not silently symmetrize.")
        rtol, atol, maxiter = float(meta["rtol"]), float(meta["atol"]), int(meta["maxiter"])
        csr = jsparse.CSR((jnp.asarray(host.data), jnp.asarray(host.indices, dtype=jnp.int32),
                          jnp.asarray(host.indptr, dtype=jnp.int32)), shape=host.shape)
        diag = jnp.asarray(diag_host)
        rhs = {name: jnp.asarray(b) for name, b in rhs_host.items()}
        ready((csr, diag, rhs))
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,memory.used,memory.total",
                              "--format=csv"], capture_output=True, text=True, check=False)
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                    "--format=csv"], capture_output=True, text=True, check=False)
        emit("CONFIG", diagnostic_revision=2, snapshot=str(path), n=host.shape[0], nnz=host.nnz, rtol=rtol, atol=atol,
             calibration_target=calibration_target,
             maxiter=maxiter, repeats=repeats, warmups=warmups, diagonal_min=float(diag_host.min()),
             diagonal_max=float(diag_host.max()), symmetry_error=symmetry_error,
             versions={"python": sys.version, "jax": jax.__version__, "cupy": cp.__version__,
                       "petsc": PETSc.Sys.getVersionInfo(), "jetsci_path": jetsci.__file__,
                       "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()},
             gpu_name=str(device["name"]), jax_devices=[str(d) for d in jax.devices()],
             nvidia_smi=smi.stdout if smi.returncode == 0 else smi.stderr,
             gpu_processes=processes.stdout if processes.returncode == 0 else processes.stderr,
             environment={key: os.environ.get(key) for key in ("XLA_FLAGS", "JAX_ENABLE_X64",
                          "XLA_PYTHON_CLIENT_PREALLOCATE", "CUDA_VISIBLE_DEVICES", "PETSC_OPTIONS")},
             note="Only clean worker timings are performance comparisons. No speed assertions; correctness gates every timed result.")

        def check(label, result):
            result = cp.asnumpy(result) if isinstance(result, cp.ndarray) else np.asarray(result)
            operator = host if label == "forward" else host.T
            norm = float(np.linalg.norm(operator @ result - rhs_host[label]))
            bound = max(atol, rtol * float(np.linalg.norm(rhs_host[label])))
            return {"true_residual": norm if np.isfinite(norm) else None, "bound": bound,
                    "residual_over_bound": norm / bound if np.isfinite(norm) else None,
                    "accepted": bool(np.isfinite(norm) and norm <= bound)}

        # Compile once per system; matrix and numerical operands are explicit.
        jax_functions = {}
        for label, b in rhs.items():
            transpose = label == "adjoint"

            def native(b, matrix, diagonal, transpose=transpose):
                op = (lambda v: matrix.T @ v) if transpose else (lambda v: matrix @ v)
                return jax.scipy.sparse.linalg.cg(op, b, M=lambda v: v / diagonal,
                                                 tol=rtol, atol=atol, maxiter=maxiter)[0]

            fn = jax.jit(native)
            jax_functions[label] = lambda fn=fn, b=b: fn(b, csr, diag)
            _, first = timed(jax_functions[label])
            for _ in range(warmups):
                timed(jax_functions[label])
            samples = []
            for repeat in range(repeats):
                result, elapsed = timed(jax_functions[label])
                samples.append(elapsed)
                quality = check(label, result)
                if not quality["accepted"]:
                    failures.append({"variant": "JAX_CSR", "system": label, **quality})
            emit("SOLVE", variant="JAX_CSR", system=label, first_s=first, samples_s=samples,
                 median_s=float(np.median(samples)), **quality)

        if mode == "events":
            PETSc.Log.begin()
        event_names = sorted({name for names in EVENT_GROUPS.values() for name in names})
        events = {name: PETSc.Log.Event(name) for name in event_names} if mode == "events" else {}
        for single in (False, True):
            for natural in (False, True):
                if mode == "trace" and single != natural:
                    continue
                variant = f"cg_{'single' if single else 'standard'}_{'natural' if natural else 'euclidean'}"
                run_variant(variant, single, natural, mode, stamp, output, host, csr, rhs,
                            rtol, atol, maxiter, warmups, repeats, primitives,
                            check, emit, failures, events, jax_functions, calibration_target)
        if mode == "events":
            log_path = output / f"petsc_{stamp}.txt"
            viewer = PETSc.Viewer().createASCII(str(log_path), comm=PETSc.COMM_SELF)
            try:
                PETSc.Log.view(viewer)
            finally:
                viewer.destroy()
            emit("FILES", petsc_log=str(log_path))
        emit("SUMMARY", failures=failures, jsonl=str(json_path))
    assert not failures, f"Some solves did not meet the original true-residual target: {failures}. All collected diagnostics are retained in {json_path}."


def calibrate_tolerance(ksp, solve, check, rtol, atol, maxiter, natural, target, emit):
    """Bracket/refine a measured residual threshold, then revalidate it.

    Residuals need not be strictly monotone in finite precision. The bracket is
    only a search heuristic: never accept its prediction without fresh solves.
    The natural-norm target is a safety margin inside the ORIGINAL bound.
    Standard Euclidean CG retains its requested tolerances unless they fail.
    """
    desired = target if natural else 1.0
    attempts, total_s = 0, 0.0

    def probe(factor, phase):
        nonlocal attempts, total_s
        ksp.setTolerances(rtol=rtol * factor, atol=atol * factor, max_it=maxiter)
        result, elapsed = timed(solve)
        quality = check(result)
        reason = int(ksp.getConvergedReason())
        ratio = quality["residual_over_bound"]
        ok = bool(reason > 0 and quality["accepted"] and ratio is not None and ratio <= desired)
        emit(attempt=attempts, phase=phase, factor=factor, rtol=rtol * factor,
             atol=atol * factor, elapsed_s=elapsed, reason=reason,
             iterations=int(ksp.getIterationNumber()), calibration_target=desired,
             calibration_pass=ok, **quality)
        attempts += 1
        total_s += elapsed
        return ok

    passing, failing = None, None
    for decade in range(9):
        factor = 10.0 ** (-decade)
        if probe(factor, "bracket"):
            passing = factor
            break
        failing = factor
    if passing is None:
        return {"accepted": False, "calibration_solves": attempts, "calibration_s": total_s}
    # Six geometric bisections reduce the decade-wide interval to ~3.7%.
    if natural and failing is not None:
        for _ in range(6):
            factor = (passing * failing) ** 0.5
            if probe(factor, "refine"):
                passing = factor
            else:
                failing = factor
    # Three fresh solves guard against selecting a lucky nondeterministic run.
    # If needed, tighten explicitly, log it, and repeat; never silently accept.
    for _ in range(5):
        validated = [probe(passing, "validate") for _ in range(3)]
        if all(validated):
            return {"accepted": True, "factor": passing, "rtol": rtol * passing,
                    "atol": atol * passing, "calibration_target": desired,
                    "calibration_solves": attempts, "calibration_s": total_s}
        passing *= 0.8
    return {"accepted": False, "calibration_solves": attempts, "calibration_s": total_s}


def run_variant(variant, single, natural, mode, stamp, output, host, csr, rhs,
                rtol, atol, maxiter, warmups, repeats, primitives,
                check, emit, failures, events, jax_functions, calibration_target):
    options = jetsci.LinearSolverOptions(
        linear_solver_type=jetsci.LinearSolverType.PETSC_CG,
        linear_preconditioner_type=jetsci.PreconditionerType.PETSC_JACOBI,
        linear_relative_tol=rtol, linear_absolute_tol=atol, linear_max_iter=maxiter)
    coo = host.tocoo()
    start = perf_counter()
    solver, options = build_linear_solver_with_reuse(options, jetsci.COOData(
        shape=jnp.asarray(host.shape), rows=jnp.asarray(coo.row),
        cols=jnp.asarray(coo.col), vals=jnp.asarray(coo.data)))
    try:
        # JetSci's builder has already called KSPSetUp for ordinary CG (three
        # work vectors). Single-reduction CG requires FIVE. Older PETSc versions
        # switch solve routines without invalidating that existing setup, so a
        # second setUp() can be a no-op and the first solve accesses missing
        # vectors. Configure a fresh KSP before its FIRST setup instead. Keep
        # the solver-owned matrix and the same wrapper/registry for callbacks
        # and cleanup. This replacement is untimed solve setup, not a fallback.
        previous_ksp = solver.ksp
        ksp = PETSc.KSP().create(comm=previous_ksp.comm)
        solver.ksp = ksp
        previous_ksp.destroy()
        ksp.setOperators(solver.operator_matrix, solver.operator_matrix)
        ksp.setType("cg")
        ksp.getPC().setType("jacobi")
        ksp.setTolerances(rtol=rtol, atol=atol, max_it=maxiter)
        # The prefix scopes this algorithm option to this KSP only. No global
        # PETSC_OPTIONS may quietly override the controlled experiment.
        prefix = "cgdiag_"
        ksp.setOptionsPrefix(prefix)
        db = PETSc.Options()
        db[prefix + "ksp_cg_single_reduction"] = "true" if single else "false"
        try:
            ksp.setFromOptions()
        finally:
            del db[prefix + "ksp_cg_single_reduction"]
        ksp.setNormType(PETSc.KSP.NormType.NATURAL if natural else PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setInitialGuessNonzero(False)
        ksp.setUp()
        cp.cuda.runtime.deviceSynchronize()
        assert ksp.getType() == "cg" and ksp.getPC().getType() == "jacobi"
        assert "cusparse" in solver.operator_matrix.getType(), solver.operator_matrix.getType()
        emit("SETUP", variant=variant, setup_s=perf_counter() - start, matrix_type=solver.operator_matrix.getType(),
             ksp_type=ksp.getType(), pc=ksp.getPC().getType(), norm_type=int(ksp.getNormType()),
             single_reduction_requested=single,
             setup_order="fresh KSP; configure algorithm and norm; first KSPSetUp")
        for label, b in rhs.items():
            b_cp = cp.from_dlpack(b, copy=False)
            x_cp = cp.empty_like(b_cp)
            b_vec = PETSc.Vec().createWithDLPack(b_cp, size=b_cp.size)
            x_vec = PETSc.Vec().createWithDLPack(x_cp, size=x_cp.size)
            try:
                assert "cuda" in b_vec.getType() and "cuda" in x_vec.getType()

                def raw():
                    ksp.setInitialGuessNonzero(False)
                    if label == "adjoint":
                        ksp.solveTranspose(b_vec, x_vec)
                    else:
                        ksp.solve(b_vec, x_vec)
                    return x_cp

                calibration = calibrate_tolerance(
                    ksp, raw, lambda result: check(label, result), rtol, atol, maxiter,
                    natural, calibration_target,
                    lambda **row: emit("CALIBRATION", variant=variant, system=label, **row))
                if not calibration["accepted"]:
                    failures.append({"variant": variant, "system": label, **calibration})
                    emit("REJECTED", variant=variant, system=label, note="No timing/speedup claim: calibration failed.")
                    continue
                emit("CALIBRATED", variant=variant, system=label, **calibration,
                     note="Snapshot/RHS-specific tuning, excluded from timings; not a transferable production tolerance.")
                config_path = output / f"ksp_{stamp}_{variant}_{label}.txt"
                viewer = PETSc.Viewer().createASCII(str(config_path), comm=PETSc.COMM_SELF)
                try:
                    ksp.view(viewer)
                finally:
                    viewer.destroy()
                emit("KSP_VIEW", variant=variant, system=label, path=str(config_path))
                lock = RLock()

                def callback(ctx, out, values):
                    with lock:
                        solver.solve_from_device_buffers(out=out, rhs=values, transpose=label == "adjoint")
                        if ksp.getConvergedReason() <= 0:
                            raise RuntimeError(f"{variant}/{label} callback CG failed: reason={ksp.getConvergedReason()}")

                cb = jax.jit(buffer_callback(callback, jax.ShapeDtypeStruct(b.shape, b.dtype), vmap_method="sequential"))
                functions = {"PETSc_raw": raw, "JetSci_callback": lambda: cb(b), "JAX_CSR": jax_functions[label]}
                first = {}
                for name, function in functions.items():
                    _, first[name] = timed(function)
                    for _ in range(warmups):
                        timed(function)
                if mode == "trace":
                    # Fresh CUPTI session per single warmed solve avoids filling
                    # activity buffers with compilation, calibration, or repeats.
                    # Callback overhead is already measured in the clean worker.
                    for name in ("PETSc_raw", "JAX_CSR"):
                        trace_dir = output / f"trace_{stamp}_{variant}_{label}_{name}"
                        cp.cuda.runtime.deviceSynchronize()
                        with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
                            with profile_range(f"cg/{variant}/{label}/{name}/0"):
                                result, elapsed = timed(functions[name])
                        quality = check(label, result)
                        reason = int(ksp.getConvergedReason()) if name != "JAX_CSR" else None
                        if not quality["accepted"] or (reason is not None and reason <= 0):
                            failures.append({"variant": variant, "system": label, "backend": name, **quality})
                        emit("TRACE", variant=variant, system=label, backend=name,
                             path=str(trace_dir), elapsed_s=elapsed, reason=reason, **quality)
                    continue
                samples = {name: [] for name in functions}
                histories = {name: [] for name in functions}
                # Rotate execution order: no backend always gets the first slot.
                names = list(functions)
                stages = {name: PETSc.Log.Stage(f"{stamp}_{variant}_{label}_{name}")
                          for name in names if name != "JAX_CSR"} if mode == "events" else {}
                for repeat in range(repeats):
                    order = names[repeat % len(names):] + names[:repeat % len(names)]
                    for name in order:
                        with log_stage(stages.get(name)):
                            with profile_range(f"cg/{variant}/{label}/{name}/{repeat}"):
                                result, elapsed = timed(functions[name])
                        quality = check(label, result)
                        reason = int(ksp.getConvergedReason()) if name != "JAX_CSR" else None
                        iterations = int(ksp.getIterationNumber()) if name != "JAX_CSR" else None
                        accepted = quality["accepted"] and (reason is None or reason > 0)
                        if not accepted:
                            failures.append({"variant": variant, "system": label, "backend": name, **quality})
                        samples[name].append(elapsed)
                        histories[name].append({"iterations": iterations, "reason": reason, **quality})
                jax_median = float(np.median(samples["JAX_CSR"]))
                jax_valid = all(row["accepted"] for row in histories["JAX_CSR"])
                for name in names:
                    median = float(np.median(samples[name]))
                    valid = all(row["accepted"] and (row["reason"] is None or row["reason"] > 0) for row in histories[name])
                    iteration_costs = [elapsed / row["iterations"]
                                       for elapsed, row in zip(samples[name], histories[name])
                                       if row["iterations"] and row["iterations"] > 0]
                    emit("SOLVE", variant=variant, system=label, backend=name, first_s=first[name],
                         samples_s=samples[name], median_s=median, checks=histories[name],
                         median_s_per_iteration=(float(np.median(iteration_costs)) if iteration_costs else None),
                         callback_over_raw=(median / float(np.median(samples["PETSc_raw"]))
                                            if name == "JetSci_callback" else None),
                         jax_over_petsc=(jax_median / median
                                         if mode == "clean" and valid and jax_valid and name != "JAX_CSR" else None),
                         configured_tolerances=list(ksp.getTolerances()),
                         note="Speed ratios valid only in clean mode; calibration cost excluded.")
                    if name in stages:
                        for event_name, event in events.items():
                            info = event_info(event, stages[name])
                            if info.get("count", 0):
                                emit("EVENT", variant=variant, system=label, backend=name,
                                     event=event_name, stage_totals=info, solves=repeats)
                        required = "KSPSolveTranspos" if label == "adjoint" else "KSPSolve"
                        if not event_info(events[required], stages[name]).get("count", 0):
                            failures.append({"variant": variant, "system": label, "missing_event": required})
                if mode == "clean" and variant == "cg_standard_euclidean":
                    primitive_benchmarks(solver.operator_matrix, b_vec, x_vec, b, csr, label, primitives, emit)
            finally:
                x_vec.destroy()
                b_vec.destroy()
    finally:
        destroy_linear_solver(options.solver_key)


def primitive_benchmarks(matrix, b_vec, x_vec, b, csr, label, count, emit):
    """Serial host-observed latency, NOT fused/throughput CG predictions."""
    multiply = matrix.multTranspose if label == "adjoint" else matrix.mult
    jax_mv = jax.jit(lambda a, x: a.T @ x if label == "adjoint" else a @ x)
    jax_norm = jax.jit(lambda x: jnp.linalg.norm(x))
    jax_dot = jax.jit(lambda x: jnp.vdot(x, x))
    functions = {
        "PETSc_matvec": lambda: multiply(b_vec, x_vec),
        "JAX_matvec": lambda: jax_mv(csr, b),
        "PETSc_norm_uncached_host_scalar": lambda: b_vec.norm(),
        "PETSc_norm_cached_host_scalar": lambda: b_vec.norm(),
        "JAX_norm_device_scalar": lambda: jax_norm(b),
        "JAX_norm_host_scalar": lambda: float(jax_norm(b)),
        "PETSc_dot_host_scalar": lambda: b_vec.dot(b_vec),
        "JAX_dot_device_scalar": lambda: jax_dot(b),
        "JAX_dot_host_scalar": lambda: float(jax_dot(b)),
        "device_synchronize_only": lambda: None,
    }
    for name, function in functions.items():
        # Invalidate only cached metadata, not vector data. This does not read
        # the array on the CPU, transfer it, or mutate JAX-owned input storage.
        # State increment is OUTSIDE the timed interval. Keep a cached arm to
        # expose the distinction rather than labelling cache hits kernel time.
        def prepare():
            if name == "PETSc_norm_uncached_host_scalar":
                b_vec.stateIncrease()

        if name == "PETSc_norm_cached_host_scalar":
            b_vec.norm()
        for _ in range(3):
            prepare()
            timed(function)
        samples = []
        for _ in range(count):
            prepare()
            samples.append(timed(function)[1])
        emit("PRIMITIVE", system=label, operation=name, count=count,
             median_s=float(np.median(samples)), min_s=float(min(samples)),
             p90_s=float(np.percentile(samples, 90)),
             note="Serial latency including dispatch and completion; PETSc scalars return to host. Uncached norm uses untimed stateIncrease; cached norm is metadata only. JAX device and host variants explicit. Not additive with fused CG.")
