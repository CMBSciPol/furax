"""Margin of the fitted cell: conditioning vs wrap-around error, on the three-layer simulation.

Without a margin the periodic cell ties opposite edges of the data together (for example the
field at the start of the observation to the field at the end). A margin frees them, but its
modes see no data, which spreads the spectrum of the solve. For each margin:

- CG steps of the joint solve, with the two diagonal preconditioners,
- template error: |P a_hat - s| / |s| at the samples, s the true atmosphere signal,
- wind error: joint marginal-likelihood fit of the three drifts, started at the truth, with the
  cells fixed where they were fitted (at the true drifts) and the true prior amplitudes.

The square box (the simulation's own basis) is the reference for the template error.
"""

import runpy
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

sys.argv = ['conditioning.py', 'none']  # reuse the setup only, no solves
g = runpy.run_path(__file__.replace('margin.py', 'conditioning.py'), run_name='setup')
OMEGAS, NOISE, MARGINS = g['OMEGAS'], g['NOISE'], (0.0, 0.03, 0.1)
screen, prior, diagonal, cg = g['screen'], g['prior'], g['diagonal'], g['cg']
u, times = g['u'], g['times']
from screen_operator import fit_cell

from furax import BlockRowOperator, HomothetyOperator


def simulate(seed):
    k_atm, k_noise = jax.random.split(jax.random.key(100 + seed))
    signal = 0
    for k, omega in enumerate(OMEGAS):
        op = screen(omega)
        s = prior(op, k)
        a = jnp.sqrt(s) * jax.random.normal(jax.random.fold_in(k_atm, k), s.shape)
        signal = signal + op(a)
    return signal, signal + NOISE * jax.random.normal(k_noise, signal.shape)


def basis(margin):
    if margin is None:
        return [None] * 3, [prior(screen(w), k) for k, w in enumerate(OMEGAS)]
    cells = [fit_cell(u, times, w, margin) for w in OMEGAS]
    return cells, [prior(screen(w, *c), k) for k, (w, c) in enumerate(zip(OMEGAS, cells))]


def ops_for(omegas, cells):
    return [screen(w) if c is None else screen(w, *c) for w, c in zip(omegas, cells)]


def neg2_log_like(omegas, cells, S, d):
    """-2 ln L (no constant), dense Woodbury: fine for the few hundred modes of fitted cells."""
    phi = jnp.concatenate([op.as_matrix() for op in ops_for(omegas, cells)], axis=1) / NOISE
    s = jnp.concatenate([x.ravel() for x in S])
    ss = jnp.sqrt(s)
    chol = jnp.linalg.cholesky(ss[:, None] * (phi.T @ phi) * ss[None, :] + jnp.eye(s.size))
    w = jax.scipy.linalg.solve_triangular(chol, ss * (phi.T @ d.ravel()) / NOISE, lower=True)
    return -w @ w + 2 * jnp.sum(jnp.log(jnp.diag(chol)))


print(
    f'{"basis":>12} {"modes":>6} | {"CG steps (S / n_eff)":>20} | {"template err":>12} | '
    f'{"|omega - true| [1e-3/s]  per layer":>34}'
)
for margin in (None, *MARGINS):
    name = 'square box' if margin is None else f'margin {margin}'
    cells, S = basis(margin)
    ops = ops_for(OMEGAS, cells)
    P = BlockRowOperator(ops)
    A = P.T @ HomothetyOperator(NOISE**-2, in_structure=P.out_structure) @ P + diagonal(
        [1 / s for s in S]
    )
    n_eff = times.size * u.shape[0] / 2 / NOISE**2
    precs = [diagonal(S), diagonal([1 / (n_eff + 1 / s) for s in S])]
    steps, errs, werrs = [], [], []
    for seed in range(3):
        signal, d = simulate(seed)
        b = P.T(d / NOISE**2)
        for prec in precs:
            res = jax.jit(
                lambda b, A=A, prec=prec: cg(A, b, preconditioner=prec, rtol=1e-5, max_steps=5000)
            )(b)
            steps.append(int(res.num_steps))
        errs.append(float(jnp.linalg.norm(P(res.solution) - signal) / jnp.linalg.norm(signal)))
        if margin is not None:  # the square box is too large for the dense wind fit
            t0 = time.perf_counter()
            fit = minimize(
                lambda z, cells=cells, S=S, d=d: (
                    neg2_log_like(z.reshape(3, 2) * 1e-3, cells, S, d) / d.size
                ),
                jnp.asarray(OMEGAS.ravel() * 1e3),
                method='BFGS',
            )
            werrs.append(np.linalg.norm(np.asarray(fit.x).reshape(3, 2) - OMEGAS * 1e3, axis=1))
    m = sum(op.in_structure.shape[1] for op in ops)
    w = (
        ' '.join(f'{x:.2f}' for x in np.mean(werrs, 0)) + f'  (max {np.max(werrs):.2f})'
        if werrs
        else '-'
    )
    print(
        f'{name:>12} {m:>6} | {np.mean(steps[0::2]):8.0f} / {np.mean(steps[1::2]):<9.0f} | '
        f'{np.mean(errs):12.3f} | {w:>34}',
        flush=True,
    )
