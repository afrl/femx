"""Memory for a prepared full nonlinear solve + one JVP, in separate workers.

Run on Linux/CUDA: python -u tests/test_microscale_bvp_jvp_memory.py
Uses the JVP benchmark's settings, including FEA_BVP_AD_MESH and solver flags.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_microscale_bvp_jvp_memory():
    for backend in ('PETSc', 'JAX_CSR'):
        environment = os.environ.copy()
        environment['FEA_BVP_JVP_MEMORY_WORKER'] = backend
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print(f'BVP_MEMORY_START backend={backend} workload=solve_plus_jvp', flush=True)
        subprocess.run(
            [sys.executable, '-u', str(Path(__file__).with_name('test_microscale_bvp_solve_jvp.py'))],
            env=environment, check=True,
        )


if __name__ == '__main__':
    test_microscale_bvp_jvp_memory()
