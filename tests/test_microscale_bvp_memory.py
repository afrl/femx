"""Run: python -u tests/test_microscale_bvp_memory.py (Linux/CUDA only).

Uses the current forward test's mesh, methods, preconditioners and tolerances.
This measures resident memory, not timing or a tally of live array bytes.
"""
import os
from pathlib import Path
import subprocess
import sys


def test_microscale_bvp_memory():
    for backend in ('PETSc', 'JAX_CSR'):
        environment = os.environ.copy()
        environment['FEA_BVP_MEMORY_WORKER'] = backend
        environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        print(f'BVP_MEMORY_START backend={backend}', flush=True)
        subprocess.run(
            [sys.executable, '-u', str(Path(__file__).with_name('test_microscale_bvp.py'))],
            env=environment, check=True,
        )


if __name__ == '__main__':
    test_microscale_bvp_memory()
