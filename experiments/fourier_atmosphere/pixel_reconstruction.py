"""Fourier for the winds, pixels for the screens: prior-free pixel reconstruction at the fitted drifts.

1. Fit the drifts with the Fourier model (marginal likelihood with the von Karman prior, fitted
   cells with margin 0.1), started at the truth. The number of layers is given.
2. Reconstruct the screens at those drifts by prior-free least squares on pixel screens:
   `ScreenSampler` (bilinear) on a `TangentialLandscape` of unit height, so pixel sizes are in
   screen units (relative to the layer height), solved with `furax.linalg.cg` and a 1/hits
   preconditioner (zero on unhit pixels).
3. Compare the template P m_hat with the true atmosphere signal s at the samples, against the
   Fourier Wiener template at the same drifts.

Same simulation as `conditioning.py`. One layer: layer 0 alone plus noise. Three: all of them.
"""

import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

sys.argv = ['conditioning.py', 'none']  # reuse the setup only, no solves
g = runpy.run_path(__file__.replace('pixel_reconstruction.py', 'conditioning.py'), run_name='setup')
OMEGAS, NOISE = g['OMEGAS'], g['NOISE']
screen, prior, diagonal, cg = g['screen'], g['prior'], g['diagonal'], g['cg']
u, times, qbore, qdet = g['u'], g['times'], g['qbore'], g['qdet']
from screen_operator import fit_cell

from furax import BlockDiagonalOperator, BlockRowOperator, DiagonalOperator, HomothetyOperator
from furax.obs.atmosphere import ScreenSampler
from furax.obs.landscapes import TangentialLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import StokesI

MARGIN = 0.1
PIXELS = (0.08, 0.04, 0.02)  # screen units; the band limit 0.1 has Nyquist spacing 0.05
SEEDS = (0, 1)


def simulate(seed):
    """Per-layer true signals and white noise."""
    k_atm, k_noise = jax.random.split(jax.random.key(200 + seed))
    signals = []
    for k, omega in enumerate(OMEGAS):
        op = screen(omega)
        s = prior(op, k)
        signals.append(op(jnp.sqrt(s) * jax.random.normal(jax.random.fold_in(k_atm, k), s.shape)))
    return signals, NOISE * jax.random.normal(k_noise, signals[0].shape)


def fourier_fit(layers, d):
    """Drifts by the Fourier marginal likelihood, and the Wiener template at them."""
    cells = [fit_cell(u, times, OMEGAS[k], MARGIN) for k in layers]
    S = [prior(screen(OMEGAS[k], *c), k) for k, c in zip(layers, cells)]
    s_flat = jnp.concatenate([x.ravel() for x in S])

    def neg2_log_like(z):
        omegas = z.reshape(-1, 2) * 1e-3
        phi = (
            jnp.concatenate([screen(w, *c).as_matrix() for w, c in zip(omegas, cells)], axis=1)
            / NOISE
        )
        ss = jnp.sqrt(s_flat)
        chol = jnp.linalg.cholesky(ss[:, None] * (phi.T @ phi) * ss[None] + jnp.eye(ss.size))
        w = jax.scipy.linalg.solve_triangular(chol, ss * (phi.T @ d.ravel()) / NOISE, lower=True)
        return (-w @ w + 2 * jnp.sum(jnp.log(jnp.diag(chol)))) / d.size

    fit = minimize(neg2_log_like, jnp.asarray(OMEGAS[layers].ravel() * 1e3), method='BFGS')
    omegas = np.asarray(fit.x).reshape(-1, 2) * 1e-3
    P = BlockRowOperator([screen(w, *c) for w, c in zip(omegas, cells)])
    A = P.T @ HomothetyOperator(NOISE**-2, in_structure=P.out_structure) @ P + diagonal(
        [1 / s for s in S]
    )
    res = cg(A, P.T(d / NOISE**2), preconditioner=diagonal(S), rtol=1e-5, max_steps=5000)
    return omegas, P(res.solution)


def pixel_screen(omega, dx):
    """`ScreenSampler` on a unit-height landscape covering the layer's drifted samples."""
    y = np.asarray(u + times[:, None] * omega).reshape(-1, 2)
    lo, hi = y.min(0) - 2 * dx, y.max(0) + 2 * dx
    (x0, y0), (wx, wy) = (lo + hi) / 2, hi - lo
    landscape = TangentialLandscape.from_extent(wx, wy, dx, dx, 1.0, x0=x0, y0=y0)
    sampler = ScreenSampler.from_wind(qbore, qdet, jnp.asarray(omega), times, interpolate=True)
    return PointingOperator.from_sampler(landscape, sampler)


def pixel_template(omegas, dx, d):
    """Prior-free least squares over the pixel screens at fixed drifts; returns template, info."""
    P = BlockRowOperator([pixel_screen(w, dx) for w in omegas])
    tod = StokesI(d)
    hits = P.T(StokesI(jnp.ones_like(d)))
    inv_hits = [
        DiagonalOperator(
            jnp.where(h.i > 0, 1 / jnp.where(h.i > 0, h.i, 1), 0), in_structure=h.structure
        )
        for h in hits
    ]
    A = P.T @ P  # white noise: N^-1 is a scalar and cancels
    res = cg(A, P.T(tod), preconditioner=BlockDiagonalOperator(inv_hits), rtol=1e-6, max_steps=3000)
    n_hit = sum(int((h.i > 0).sum()) for h in hits)
    low = sum(int(((h.i > 0) & (h.i < 1)).sum()) for h in hits)  # under one sample's weight
    return P(res.solution).i, int(res.num_steps), bool(res.converged), n_hit, low


def err(t, s):
    return float(jnp.linalg.norm(t - s) / jnp.linalg.norm(s))


if __name__ == '__main__':  # not when another script reuses the setup
    print(
        f'{"case":>8} {"method":>16} | {"template err":>12} | {"chi2/N":>6} | {"CG steps":>8} | '
        f'{"hit px":>6} {"<1 hit":>6}'
    )
    for name, layers in (('1 layer', [0]), ('3 layers', [0, 1, 2])):
        rows = {}
        for seed in SEEDS:
            signals, noise = simulate(seed)
            s = sum(signals[k] for k in layers)
            d = s + noise
            omegas, wiener = fourier_fit(layers, d)
            werr = np.linalg.norm(omegas - OMEGAS[layers], axis=1) * 1e3
            rows.setdefault('Fourier Wiener', []).append(
                (err(wiener, s), float(jnp.mean((d - wiener) ** 2)) / NOISE**2, 0, 0, 0, werr)
            )
            for dx in PIXELS:
                t, steps, conv, n_hit, low = pixel_template(omegas, dx, d)
                chi2 = float(jnp.mean((d - t) ** 2)) / NOISE**2
                rows.setdefault(f'pixels dx={dx}', []).append(
                    (err(t, s), chi2, steps if conv else -steps, n_hit, low, werr)
                )
        for method, r in rows.items():
            r = np.array([x[:5] for x in r])
            print(
                f'{name:>8} {method:>16} | {r[:, 0].mean():12.3f} | {r[:, 1].mean():6.2f} | '
                f'{r[:, 2].mean():8.0f} | {r[:, 3].mean():6.0f} {r[:, 4].mean():6.0f}',
                flush=True,
            )
        print(
            f'{name:>8} fitted drift error [1e-3/s] per layer, seed 0: '
            f'{np.round(rows["Fourier Wiener"][0][5], 2).tolist()}'
        )
