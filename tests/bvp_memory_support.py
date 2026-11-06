"""Sample process-resident memory; JAX allocator snapshots exclude PETSc allocations."""
import gc
import json
import os
from pathlib import Path
import subprocess
import threading


def process_memory():
    status = Path('/proc/self/status').read_text()
    rss = next(int(line.split()[1]) * 1024 for line in status.splitlines()
               if line.startswith('VmRSS:'))
    output = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    gpu = 0
    found = False
    for line in output.splitlines():
        pid, used = (part.strip() for part in line.split(','))
        if int(pid) == os.getpid():
            if not used.isdigit():
                raise RuntimeError('nvidia-smi cannot report GPU memory for this process; use a supported Linux NVIDIA configuration without MPS.')
            gpu += int(used) * 1024**2
            found = True
    if not found:
        raise RuntimeError('No GPU allocation found for the BVP worker PID. Check CUDA visibility and nvidia-smi process accounting; this benchmark requires a GPU.')
    return {'rss_bytes': rss, 'gpu_process_bytes': gpu}


def measure_route(backend, solve, block, jax, validate, inputs, *, workload='forward',
                  scope='complete forward BVP including preprocessing, solver setup and cleanup; independent process per backend'):
    interval = float(os.environ.get('FEA_BVP_MEMORY_INTERVAL', '0.1'))
    if interval <= 0:
        raise ValueError('FEA_BVP_MEMORY_INTERVAL must be positive seconds.')
    repeats = int(os.environ.get('FEA_BVP_MEMORY_REPEATS', '2'))
    if repeats < 1:
        raise ValueError('FEA_BVP_MEMORY_REPEATS must be at least 1.')
    jax.effects_barrier()
    # Finish common input creation before sampling the initialized baseline.
    jax.block_until_ready(inputs)
    baseline = process_memory()

    def allocator_stats():
        return [{'device': str(device), 'stats': device.memory_stats()}
                for device in jax.devices()]

    print('BVP_MEMORY_CONFIG ' + json.dumps({
        'backend': backend, 'workload': workload, 'pid': os.getpid(), 'interval_s': interval,
        'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        'allocator': os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'default'),
        'baseline': baseline, 'jax_allocator': allocator_stats(),
        'scope': scope,
        'note': 'GPU process residency includes cached pools and CUDA context; sampled peaks are lower bounds. JAX allocator stats do not include PETSc/CuPy.',
    }), flush=True)
    result = None
    for call in range(repeats + 1):
        if result is not None:
            del result
        gc.collect()
        before = process_memory()
        samples = [before]
        failures = []
        stop = threading.Event()

        def poll():
            while not stop.wait(interval):
                try:
                    samples.append(process_memory())
                except Exception as exc:
                    failures.append(exc)
                    return

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        try:
            result = block(solve())
            jax.block_until_ready(result)
            jax.effects_barrier()
        finally:
            stop.set()
            thread.join()
        if failures:
            raise RuntimeError('Memory sampling failed; no valid peak comparison is available.') from failures[0]
        after = process_memory()
        samples.append(after)
        peak = {key: max(sample[key] for sample in samples) for key in before}
        print('BVP_MEMORY ' + json.dumps({
            'backend': backend, 'workload': workload, 'phase': 'cold' if call == 0 else 'warm',
            'repeat': call, 'before': before, 'after': after,
            'sampled_peak': peak,
            'peak_above_initialized_baseline': {key: max(0, peak[key] - baseline[key]) for key in peak},
            'samples': len(samples), 'jax_allocator': allocator_stats(),
        }), flush=True)
    # Validation runs after all memory samples, so extra residual assembly is excluded.
    validate(result)
