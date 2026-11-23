"""Pixel-space preconditioners for the Fourier amplitude solve, applied through an FFT.

On a regular (2 N1 + 1) x (2 N2 + 1) grid over a layer's cell, the cos/sin coefficients map to
grid values by one inverse FFT; scaled by sqrt(2 / G), G grid points, that map Q is orthonormal
(Q^T Q = I). A pixel-diagonal operator D then acts on the coefficients as Q^T D Q: an FFT pair.

Preconditioners, per layer (block diagonal over layers):
- 'S': the prior (baseline),
- 'pixel Jacobi': Q^T diag(QAQ^T)^-1 Q, the diagonal of A in pixels; it sees the coverage,
- 'whitened pixel Jacobi': S^1/2 Q^T diag(Q S^1/2 A S^1/2 Q^T)^-1 Q S^1/2. Unhit pixels get
  the identity (the prior), hit pixels the data weight.

The pixel diagonals are computed exactly here from small dense matrices; at scale they would come
from the hit counts binned on the grid.
"""

import runpy
import sys
from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np

sys.argv = ['conditioning.py', 'none']  # reuse the setup only, no solves
g = runpy.run_path(__file__.replace('pixel_preconditioner.py', 'conditioning.py'), run_name='setup')
OMEGAS, NOISE, d = g['OMEGAS'], g['NOISE'], g['d']
screen, prior, diagonal, cg = g['screen'], g['prior'], g['diagonal'], g['cg']
u, times = g['u'], g['times']
from screen_operator import FourierScreenOperator, fit_cell

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    DiagonalOperator,
    HomothetyOperator,
)
from furax.tree import as_structure

MARGIN = 0.1


class GridOperator(AbstractLinearOperator):
    """Orthonormal map Q from (2, M) cos/sin coefficients to the field on the cell's grid."""

    index: jax.Array  # (M, 2) integer wavevectors
    grid_shape: tuple[int, int] = field(metadata={'static': True})

    def mv(self, a):
        n1, n2 = self.grid_shape
        grid = jnp.zeros((n1, n2), complex).at[self.index[:, 0] % n1, self.index[:, 1] % n2]
        grid = grid.set(a[0] - 1j * a[1], unique_indices=True)
        # sum_n F_n exp(2 pi i n.p / n_g) = G ifft2(F); the factor sqrt(2 / G) makes Q orthonormal
        return jnp.real(jnp.fft.ifft2(grid)) * np.sqrt(2 * n1 * n2)


def grid_op(op: FourierScreenOperator) -> GridOperator:
    n1, n2 = op.n_max
    return GridOperator(op.index, (2 * n1 + 1, 2 * n2 + 1), in_structure=op.in_structure)


def preconditioners(ops, S):
    grids = [grid_op(op) for op in ops]
    Qs = [q.as_matrix() for q in grids]
    for Q in Qs[:1]:
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10), 'Q is not orthonormal'
    # share of the data each layer gets in its pixel diagonal: all of it, an equal share, or its
    # share of the total field variance (sum of its mode variances)
    variances = np.array([float(jnp.sum(s[0])) for s in S])
    shares = {
        'pixel Jacobi': np.ones(len(S)),
        'pixel Jacobi, 1/K': np.full(len(S), 1 / len(S)),
        'pixel Jacobi, var frac': variances / variances.sum(),
    }
    jacobi, whitened = {name: [] for name in shares}, []
    for k, (op, s, q, Q) in enumerate(zip(ops, S, grids, Qs)):
        phi = op.as_matrix() / NOISE
        s = s.ravel()
        A = phi.T @ phi + jnp.diag(1 / s)
        Aw = jnp.sqrt(s)[:, None] * A * jnp.sqrt(s)[None]
        data_diag = jnp.einsum('pi,ij,pj->p', Q, phi.T @ phi, Q)
        prior_diag = jnp.einsum('pi,i,pi->p', Q, 1 / s, Q)
        for name, f in shares.items():
            pix = DiagonalOperator(
                1 / (f[k] * data_diag + prior_diag).reshape(q.grid_shape),
                in_structure=q.out_structure,
            )
            jacobi[name].append(q.T @ pix @ q)
        pix_w = DiagonalOperator(
            1 / jnp.einsum('pi,ij,pj->p', Q, Aw, Q).reshape(q.grid_shape),
            in_structure=q.out_structure,
        )
        half = DiagonalOperator(
            jnp.sqrt(s).reshape(op.in_structure.shape), in_structure=op.in_structure
        )
        whitened.append(half @ q.T @ pix_w @ q @ half)
    precs = {'S': diagonal(S), 'whitened pixel Jacobi': BlockDiagonalOperator(whitened)}
    names = ['pixel Jacobi'] if len(S) == 1 else list(shares)  # sharing is moot for one layer
    return precs | {name: BlockDiagonalOperator(jacobi[name]) for name in names}


def condition(A_dense, M_dense):
    lam = np.linalg.eigvals(np.asarray(M_dense @ A_dense)).real
    return lam.max() / lam.min()


if __name__ == '__main__':  # not when another script reuses the setup
    ops_all = [screen(w, *fit_cell(u, times, w, MARGIN)) for w in OMEGAS]
    print(f'{"case":>8} {"preconditioner":>24} | {"kappa":>8} | {"CG steps":>8}')
    for name, layers in (('layer 0', [0]), ('3 layers', [0, 1, 2])):
        ops = [ops_all[k] for k in layers]
        S = [prior(ops_all[k], k) for k in layers]
        P = BlockRowOperator(ops)
        A = P.T @ HomothetyOperator(NOISE**-2, in_structure=as_structure(d)) @ P + diagonal(
            [1 / s for s in S]
        )
        b = P.T(d / NOISE**2)
        A_dense = A.as_matrix()
        for pname, M in preconditioners(ops, S).items():
            res = jax.jit(
                lambda b, M=M, A=A: cg(A, b, preconditioner=M, rtol=1e-5, max_steps=5000)
            )(b)
            kappa = f'{condition(A_dense, M.as_matrix()):8.1e}' if A_dense is not None else '-'
            print(
                f'{name:>8} {pname:>24} | {kappa:>8} | {int(res.num_steps):8d}'
                f'{"" if res.converged else " (not converged)"}',
                flush=True,
            )
