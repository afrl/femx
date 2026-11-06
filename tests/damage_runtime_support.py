"""Per-increment validation and disk-backed agreement checks, outside solve timers."""
import json
from pathlib import Path
import numpy as np


class DamageRuntimeCheck:
    def __init__(self, backend, directory, rtol, atol):
        self.backend = backend
        self.directory = Path(directory)
        self.rtol, self.atol = rtol, atol
        self.times = []
        self.max_u_relative_error = 0.0
        self.max_state_absolute_error = 0.0
        self.max_damage = 0.0

    def step(self, step, elapsed, u, batches, initial_norm, final_norm, bound, on_failure=None):
        arrays = {'u': np.asarray(u)}
        arrays.update({f'state_{i}': np.asarray(batch.internal_state)
                       for i, batch in enumerate(batches)})
        if not (np.isfinite([initial_norm, final_norm, bound]).all()
                and all(np.isfinite(array).all() for array in arrays.values())
                and final_norm <= bound):
            print('DAMAGE_RUNTIME_FAILURE ' + json.dumps(dict(
                backend=self.backend, step=step, initial_residual=initial_norm,
                final_residual=final_norm, residual_bound=bound,
                residual_ratio=final_norm/initial_norm if initial_norm else None,
                nonfinite_arrays=[name for name, array in arrays.items() if not np.isfinite(array).all()],
                accepted=False,
            )), flush=True)
            if on_failure is not None:
                on_failure()
            raise AssertionError(f'{self.backend} step {step} failed equilibrium/finite-state validation: residual={final_norm}, bound={bound}. Reject runtime comparison.')
        path = self.directory / f'step_{step:06d}.npz'
        relative_error = None
        state_error = None
        if self.backend == 'PETSc':
            np.savez(path, **arrays)
        else:
            with np.load(path) as expected:
                if set(expected.files) != set(arrays):
                    raise AssertionError(f'State layout differs at step {step}.')
                for name, array in arrays.items():
                    reference = expected[name]
                    np.testing.assert_allclose(
                        array, reference, rtol=self.rtol, atol=self.atol,
                        err_msg=f'JAX vs PETSc, step {step}, {name}',
                    )
                    if name == 'u':
                        relative_error = float(np.linalg.norm(array-reference)) / max(float(np.linalg.norm(reference)), 1e-30)
                    elif array.size:
                        error = float(np.max(np.abs(array-reference)))
                        state_error = max(state_error or 0.0, error)
            path.unlink()
            self.max_u_relative_error = max(self.max_u_relative_error, relative_error)
            self.max_state_absolute_error = max(self.max_state_absolute_error, state_error or 0.0)
        damage = max((float(np.max(arrays[f'state_{i}'][..., 6]))
                      for i in (0, 1) if arrays[f'state_{i}'].size), default=0.0)
        self.max_damage = max(self.max_damage, damage)
        self.times.append(elapsed)
        print('DAMAGE_RUNTIME_STEP ' + json.dumps(dict(
            backend=self.backend, step=step, solve_s=elapsed,
            initial_residual=initial_norm, final_residual=final_norm,
            residual_bound=bound, max_matrix_damage=damage,
            u_relative_error=relative_error, state_max_abs_error=state_error,
            accepted=True,
        )), flush=True)

    def finish(self, config):
        report = dict(
            backend=self.backend, config=config, steps=len(self.times),
            first_step_s=self.times[0], total_s=sum(self.times),
            subsequent_steps_s=sum(self.times[1:]), step_samples_s=self.times,
            max_matrix_damage=self.max_damage,
            max_u_relative_error=self.max_u_relative_error if self.backend == 'JAX' else None,
            max_state_absolute_error=self.max_state_absolute_error if self.backend == 'JAX' else None,
            accepted=True,
        )
        (self.directory / f'{self.backend}.json').write_text(json.dumps(report))
        print('DAMAGE_RUNTIME_RESULT ' + json.dumps(report), flush=True)
