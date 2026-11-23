"""Convergence of the amplitude solve (P^T N^-1 P + S^-1) a = P^T N^-1 d: square box vs fitted cell.

The data term P^T N^-1 P is a convolution in Fourier space by the transform of the coverage, so it
is close to diagonal only where the samples cover the periodic cell evenly. A square box around
a scan strip dragged by the wind is mostly empty; the parallelogram the strip sweeps is not.

Same three-layer simulation as `multilayer.py` (generated on its own square box), solved at the
true drifts with `furax.linalg.cg`, one layer at a time and the three jointly.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from screen_operator import FourierScreenOperator, cell_index, fit_cell, screen_coords

from furax import BlockDiagonalOperator, BlockRowOperator, DiagonalOperator, HomothetyOperator
from furax.linalg import cg
from furax.obs.coords import azel_to_quat, xieta_to_quat
from furax.tree import as_structure

jax.config.update('jax_enable_x64', True)

HEIGHTS = np.array([500.0, 1500.0, 3000.0])
WINDS = np.array([[2.0, 1.0], [-4.5, 4.5], [9.0, -9.0]])
STDS = np.array([1.0, 0.6, 0.4])
L0, NOISE = 300.0, 0.1
OMEGAS = WINDS / HEIGHTS[:, None]
# TOY_SCALE=f: f x more detectors and f x higher sampling rate over the same 100 s
SCALE = int(os.environ.get('TOY_SCALE', '1'))
NDET, NSAMP, RATE = 8 * SCALE, 240 * SCALE, 2.4 * SCALE
OMEGA_MAX = 0.006
Q_MAX = 2 * np.pi / 0.1
MARGIN = 0.1  # around the swept parallelogram, in units of the height
RTOL = 1e-5

times = jnp.arange(NSAMP) / RATE
qbore = azel_to_quat(jnp.deg2rad(10.0) * jnp.sin(2 * jnp.pi * times / 20.0), jnp.deg2rad(50.0))
k_det, k_atm, k_noise = jax.random.split(jax.random.key(1), 3)
xi, eta = jnp.deg2rad(2.0) * jax.random.uniform(k_det, (2, NDET), minval=-1, maxval=1)
qdet = xieta_to_quat(xi, eta)
u = screen_coords(qbore, qdet)
BOX = float(jnp.max(u.max((0, 1)) - u.min((0, 1)))) + OMEGA_MAX * float(times[-1]) + 0.2
SQUARE = BOX * np.eye(2)


def shape(q, h):
    """Von Karman shape of a layer at height h, in screen units (outer scale L0 / h)."""
    return (jnp.sum(q**2, -1) + (2 * np.pi * h / L0) ** 2) ** (-11 / 6)


def screen(omega, cell=SQUARE, origin=None):
    index = cell_index(cell, Q_MAX)
    return FourierScreenOperator.create(
        index, cell, qbore, qdet, times, jnp.asarray(omega), origin=origin
    )


def fitted_screen(omega):
    cell, origin = fit_cell(u, times, omega, MARGIN)
    return screen(omega, cell, origin)


def prior(op, k):
    """Mode variances of layer k on the cell of `op`: the same field rms whatever the cell."""
    sim = screen(OMEGAS[k])  # the simulation's square box sets the normalisation
    norm = STDS[k] ** 2 / jnp.sum(shape(sim.modes, HEIGHTS[k])) * BOX**2
    s = norm * shape(op.modes, HEIGHTS[k]) / abs(jnp.linalg.det(op.cell))
    return jnp.stack([s, s])


# --- simulate on the square box ---------------------------------------------------------------
signal = 0
for k, omega in enumerate(OMEGAS):
    op = screen(omega)
    s = prior(op, k)
    signal = signal + op(jnp.sqrt(s) * jax.random.normal(jax.random.fold_in(k_atm, k), s.shape))
d = signal + NOISE * jax.random.normal(k_noise, signal.shape)
Ninv = HomothetyOperator(NOISE**-2, in_structure=as_structure(d))


def diagonal(blocks):
    """Per-layer elementwise scaling of a list of (2, M) blocks."""
    return BlockDiagonalOperator(
        [DiagonalOperator(w, in_structure=as_structure(w)) for w in blocks]
    )


def coverage(op, cell_size=0.1):
    """Fraction of the cell hit, and max / median hits of the hit cells, on a ~cell_size grid."""
    xi, _ = jax.vmap(op._cell_positions)(jnp.arange(NDET))
    edges = np.linalg.norm(np.asarray(op.cell), axis=0)
    bins = np.maximum(1, np.ceil(edges / cell_size)).astype(int)
    hits, *_ = np.histogram2d(*np.asarray(xi % 1).reshape(-1, 2).T, bins=bins, range=[[0, 1]] * 2)
    seen = hits > 0
    return seen.mean(), hits.max() / np.median(hits[seen])


KINDS = {
    'square box': [screen(w) for w in OMEGAS],
    'fitted cell': [fitted_screen(w) for w in OMEGAS],
}
CASES = [('layer 0', [0]), ('layer 1', [1]), ('layer 2', [2]), ('3 layers', [0, 1, 2])]
ONLY = sys.argv[1:]  # optional: a case name, to run that case only

results = {}
for kind, ops in KINDS.items():
    for k, op in enumerate(ops):
        frac, contrast = coverage(op)
        print(
            f'{kind:>11} layer {k}: M={op.in_structure.shape[1]:4d} modes, '
            f'{frac:.0%} of the cell hit, hit contrast max/median {contrast:.1f}'
        )
    for name, layers in CASES:
        if ONLY and name != ONLY[0]:
            continue
        P = BlockRowOperator([ops[k] for k in layers])
        S = [prior(ops[k], k) for k in layers]
        A = P.T @ Ninv @ P + diagonal([1 / s for s in S])
        b = P.T(Ninv(d))
        n_eff = d.size / 2 / NOISE**2  # P^T N^-1 P = n_eff I under uniform coverage of the cell
        precs = {
            'S': diagonal(S),
            '(n_eff + 1/S)^-1': diagonal([1 / (n_eff + 1 / s) for s in S]),
        }
        for pname, prec in precs.items():
            solve = jax.jit(
                lambda b, A=A, prec=prec: cg(A, b, preconditioner=prec, rtol=RTOL, max_steps=5000)
            )
            jax.block_until_ready(solve(b))
            t0 = time.perf_counter()
            res = jax.block_until_ready(solve(b))
            dt = time.perf_counter() - t0
            results[kind, name, pname] = res
            print(
                f'{kind:>11} | {name:>8} | {pname:<17} | {int(res.num_steps):5d} steps, '
                f'converged={bool(res.converged)}, {dt:.2f}s',
                flush=True,
            )

# --- plot: the two cells of layer 0 over its samples, and the residual histories ---------------
if __name__ == '__main__':  # not when another script reuses the setup
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    y0 = np.asarray(u + times[:, None] * OMEGAS[0]).reshape(-1, 2)
    axes[0].plot(*y0.T, ',', color='k')
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    for (kind, ops), color in zip(KINDS.items(), ('C0', 'C1')):
        op = ops[0]
        # the square box is periodic, so draw it centred on the samples
        origin = y0.mean(0) - BOX / 2 if kind == 'square box' else np.asarray(op.origin)
        axes[0].plot(*(unit_square @ np.asarray(op.cell).T + origin).T, color=color, label=kind)
    axes[0].set_aspect('equal')
    axes[0].set_title('layer 0: samples on the screen')
    axes[0].legend()
    for (kind, name, pname), res in results.items():
        if name not in ('layer 0', '3 layers'):
            continue
        k = int(res.num_steps)
        axes[1].semilogy(
            np.asarray(res.residuals[: k + 1]) / float(res.residuals[0]),
            ls='-' if kind == 'fitted cell' else ':',
            label=f'{kind}, {name}, M = {pname}',
        )
    axes[1].set_xlabel('CG step')
    axes[1].set_ylabel('relative residual')
    axes[1].legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(__file__.replace('.py', '.png'))
