"""Time the existing JetSci damage workload against JAX and check every step.

Run on Linux/GPU: python -u tests/test_fea_solve_dmg_runtime_comparison.py
Defaults match test_fea_solve_dmg_JetSCI: 49 fibers, 500 increments.
FEA_DAMAGE_FIBERS and FEA_DAMAGE_TIMESTEPS select a shorter/smaller run.
FEA_DAMAGE_JAX_FORMAT=matrix_free (original JAX CG) or csr.
FEA_DAMAGE_METHOD=cg|gmres and FEA_DAMAGE_PC=none|jacobi|ilu select both routes.
Defaults: cg/jacobi. Per-backend overrides: FEA_DAMAGE_PETSC_METHOD/PC and
FEA_DAMAGE_JAX_METHOD/PC. PETSc additionally supports lgmres and icc.
JAX GMRES uses incremental QR and the existing true-residual stopping fix.
FEA_DAMAGE_GMRES_RESTART sets the restart for both routes (default 30,
clamped to the number of DOFs). It is applied to PETSc's live SNES KSP
when explicitly set; ordinary public differentiable_solve calls still solve.
JAX CuPy ILU uses synchronized applications and per-forward-call factor cleanup.
Temporary reference snapshots are removed when the comparison completes.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def test_damage_runtime_comparison():
    with tempfile.TemporaryDirectory(prefix='damage_compare_') as directory:
        for backend in ('PETSc', 'JAX'):
            environment = os.environ.copy()
            environment['FEA_DAMAGE_COMPARE_WORKER'] = backend
            environment['FEA_DAMAGE_COMPARE_DIR'] = directory
            environment['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            subprocess.run(
                [sys.executable, '-u', str(Path(__file__).with_name('test_fea_solve_dmg_JetSCI.py'))],
                env=environment, cwd=Path(__file__).resolve().parents[1], check=True,
            )
        petsc, jax = [json.loads((Path(directory) / f'{name}.json').read_text())
                      for name in ('PETSc', 'JAX')]
        assert petsc['config'] == jax['config'], 'Damage workload/options differ between workers.'
        assert petsc['steps'] == jax['steps'] == petsc['config']['timesteps']
        print('DAMAGE_RUNTIME_COMPARISON ' + json.dumps(dict(
            settings=petsc['config'], all_increments_agree=True,
            total_s={'PETSc': petsc['total_s'], 'JAX': jax['total_s']},
            jax_over_petsc=jax['total_s']/petsc['total_s'],
            max_u_relative_error=jax['max_u_relative_error'],
            max_state_absolute_error=jax['max_state_absolute_error'],
        )), flush=True)


if __name__ == '__main__':
    test_damage_runtime_comparison()
