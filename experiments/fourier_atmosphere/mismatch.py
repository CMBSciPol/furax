"""Robustness of the Fourier method to a wrong turbulence spectrum.

Simulate the three layers with the von Karman spectrum the fit assumes (slope -11/3, outer scale
L0 = 300 m) or with a different one (slope -3, L0 = 100 m), same layer rms. Then, with the
standard von Karman prior:

- fit the drifts and per-layer field variance, outer scale fixed at L0 = 300 m ('amp'),
  or also the outer scale ('amp+L0'), by the marginal likelihood, started at the true drifts;
- reconstruct the Wiener template at the fitted parameters (CG, M = S, rtol 1e-5).

The prior-free pixel reconstruction at dx = 0.04 is kept as a reference. Data: TOY_SCALE = 2,
16 detectors x 480 samples over 100 s, four times the sample density of the earlier runs.
"""

import os
import runpy
import sys

os.environ.setdefault('TOY_SCALE', '2')
sys.argv = ['conditioning.py', 'none']

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

g = runpy.run_path(__file__.replace('mismatch.py', 'pixel_reconstruction.py'), run_name='setup')
from furax import BlockRowOperator, HomothetyOperator

setup = g['g']  # the conditioning setup the pixel script loaded
OMEGAS, NOISE = g['OMEGAS'], g['NOISE']
STDS, HEIGHTS, BOX = setup['STDS'], setup['HEIGHTS'], setup['BOX']
screen, diagonal, cg, fit_cell = g['screen'], g['diagonal'], g['cg'], g['fit_cell']
u, times, pixel_template, err = g['u'], g['times'], g['pixel_template'], g['err']
MARGIN, SEEDS = 0.1, (0, 1)
SPECTRA = {'matched': (-11 / 3, 300.0), 'mismatched': (-3.0, 100.0)}  # (slope, L0 [m])
L0_FIT = 300.0
SQUARE_MODES = screen(OMEGAS[0]).modes  # the square box sets the variance normalisation


def mode_variances(op, variance, slope, q0):
    """Mode variances on the cell of `op` for a field of the given variance and spectrum."""
    shape = lambda q: (jnp.sum(q**2, -1) + q0**2) ** (slope / 2)
    s = variance * shape(op.modes) / jnp.sum(shape(SQUARE_MODES)) * BOX**2
    s = s / abs(jnp.linalg.det(op.cell))
    return jnp.stack([s, s])


def simulate(seed, slope, l0):
    k_atm, k_noise = jax.random.split(jax.random.key(300 + seed))
    signals = []
    for k, omega in enumerate(OMEGAS):
        op = screen(omega)
        s = mode_variances(op, STDS[k] ** 2, slope, 2 * np.pi * HEIGHTS[k] / l0)
        signals.append(op(jnp.sqrt(s) * jax.random.normal(jax.random.fold_in(k_atm, k), s.shape)))
    return signals, NOISE * jax.random.normal(k_noise, signals[0].shape)


def fit(layers, d, free_l0):
    """Marginal-likelihood fit of (omega, log variance[, log q0]) per layer, then the template."""
    cells = [fit_cell(u, times, OMEGAS[k], MARGIN) for k in layers]
    # start: true drifts and variances, nominal outer scale (in screen units: 2 pi h / L0)
    q0_start = [np.log(2 * np.pi * HEIGHTS[k] / L0_FIT) for k in layers]
    start = np.array(
        [[*OMEGAS[k] * 1e3, np.log(STDS[k] ** 2), q] for k, q in zip(layers, q0_start)]
    )
    npar = 4 if free_l0 else 3

    def unpack(z):
        z = z.reshape(len(layers), npar)
        q0 = jnp.exp(z[:, 3]) if free_l0 else jnp.exp(jnp.asarray(q0_start))
        return z[:, :2] * 1e-3, jnp.exp(z[:, 2]), q0

    def model(z):
        omegas, variances, q0 = unpack(z)
        ops = [screen(w, *c) for w, c in zip(omegas, cells)]
        S = [mode_variances(op, v, -11 / 3, q) for op, v, q in zip(ops, variances, q0)]
        return ops, S

    def neg2_log_like(z):
        ops, S = model(z)
        phi = jnp.concatenate([op.as_matrix() for op in ops], axis=1) / NOISE
        ss = jnp.sqrt(jnp.concatenate([s.ravel() for s in S]))
        chol = jnp.linalg.cholesky(ss[:, None] * (phi.T @ phi) * ss[None] + jnp.eye(ss.size))
        w = jax.scipy.linalg.solve_triangular(chol, ss * (phi.T @ d.ravel()) / NOISE, lower=True)
        return (-w @ w + 2 * jnp.sum(jnp.log(jnp.diag(chol)))) / d.size

    z0 = jnp.asarray(start[:, :npar].ravel())
    z = minimize(neg2_log_like, z0, method='BFGS').x
    ops, S = model(z)
    P = BlockRowOperator(ops)
    A = P.T @ HomothetyOperator(NOISE**-2, in_structure=P.out_structure) @ P + diagonal(
        [1 / s for s in S]
    )
    res = cg(A, P.T(d / NOISE**2), preconditioner=diagonal(S), rtol=1e-5, max_steps=5000)
    omegas, variances, q0 = unpack(z)
    l0 = 2 * np.pi * HEIGHTS[layers] / np.asarray(q0)
    return np.asarray(omegas), np.sqrt(np.asarray(variances)), l0, P(res.solution)


print(f'samples: {u.shape[0]} detectors x {u.shape[1]} = {u.shape[0] * u.shape[1]}')
header = (
    f'{"spectrum":>10} {"case":>8} {"method":>14} | {"wind err [1e-3/s]":>17} | '
    f'{"template err":>12} | {"chi2/N":>6} | fitted rms / L0 [m] (seed 0)'
)
print(header)
for spectrum, (slope, l0) in SPECTRA.items():
    for name, layers in (('1 layer', [0]), ('3 layers', [0, 1, 2])):
        rows = {}
        for seed in SEEDS:
            signals, noise = simulate(seed, slope, l0)
            s = sum(signals[k] for k in layers)
            d = s + noise
            for method, free_l0 in (('Wiener, amp', False), ('Wiener, amp+L0', True)):
                omegas, rms, l0_fit, t = fit(layers, d, free_l0)
                werr = np.linalg.norm(omegas - OMEGAS[layers], axis=1).mean() * 1e3
                chi2 = float(jnp.mean((d - t) ** 2)) / NOISE**2
                rows.setdefault(method, []).append((werr, err(t, s), chi2, rms, l0_fit))
                if method == 'Wiener, amp+L0':
                    omegas_ref = omegas
            t = pixel_template(omegas_ref, 0.04, d)[0]
            chi2 = float(jnp.mean((d - t) ** 2)) / NOISE**2
            rows.setdefault('pixels 0.04', []).append((np.nan, err(t, s), chi2, None, None))
        for method, r in rows.items():
            werr, terr, chi2 = (np.mean([x[i] for x in r]) for i in range(3))
            extra = (
                f'{np.round(r[0][3], 2).tolist()} / {np.round(r[0][4]).tolist()}'
                if r[0][3] is not None
                else '(at the amp+L0 winds)'
            )
            print(
                f'{spectrum:>10} {name:>8} {method:>14} | {werr:17.2f} | {terr:12.3f} | '
                f'{chi2:6.2f} | {extra}',
                flush=True,
            )
print(f'true rms: {STDS.tolist()}, true L0: matched 300, mismatched 100')
