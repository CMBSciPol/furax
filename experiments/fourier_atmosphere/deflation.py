"""Two-level deflation on top of the per-layer preconditioners, for the three-layer joint solve.

The eigenpairs of the preconditioned operator M A are computed exactly (dense, small problem) and
passed to `make_two_level_preconditioner`, which maps the deflated eigenvalues to 1:
- pixel Jacobi: deflate its k smallest eigenvalues, the combinations where the layers cancel on
  the samples, which only the prior constrains;
- S: deflate its k largest, the modes the data constrain most.
At scale the eigenpairs would come from Lanczos, which finds the extreme ones first.
"""

import runpy
import sys

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sl

sys.argv = ['conditioning.py', 'none']
g = runpy.run_path(__file__.replace('deflation.py', 'pixel_preconditioner.py'), run_name='setup')
from furax import BlockRowOperator, HomothetyOperator
from furax.linalg import LowRankTerms
from furax.mapmaking.preconditioner import make_two_level_preconditioner
from furax.tree import as_structure

OMEGAS, NOISE, d, cg = g['OMEGAS'], g['NOISE'], g['d'], g['cg']
ops = [g['screen'](w, *g['fit_cell'](g['u'], g['times'], w, g['MARGIN'])) for w in OMEGAS]
S = [g['prior'](op, k) for k, op in enumerate(ops)]
P = BlockRowOperator(ops)
A = P.T @ HomothetyOperator(NOISE**-2, in_structure=as_structure(d)) @ P + g['diagonal'](
    [1 / s for s in S]
)
b = P.T(d / NOISE**2)
A_dense = np.asarray(A.as_matrix())
sizes = np.cumsum([0] + [op.in_size for op in ops])
precs = g['preconditioners'](ops, S)


def as_blocks(V):
    """Columns of V, shape (n, k), as the list of (k, 2, M) blocks the operators act on."""
    return [
        jnp.asarray(V[sizes[i] : sizes[i + 1]].T).reshape(-1, *op.in_structure.shape)
        for i, op in enumerate(ops)
    ]


print(f'{"first level":>13} {"deflated":>16} | {"kappa":>8} | {"CG steps":>8}')
for name, end in (('pixel Jacobi', 'smallest'), ('S', 'largest')):
    M = precs[name]
    lam, V = sl.eigh(A_dense, np.linalg.inv(np.asarray(M.as_matrix())))  # V^T M^-1 V = I
    for k in (0, 5, 10, 20, 40):
        if k == 0:
            M2, rest = M, lam
        else:
            idx = np.arange(k) if end == 'smallest' else np.arange(len(lam) - k, len(lam))
            pairs = LowRankTerms(jnp.asarray(lam[idx]), as_blocks(V[:, idx]))
            M2 = make_two_level_preconditioner(M, pairs, preconditioned=True)
            rest = np.concatenate([np.delete(lam, idx), [1.0]])
        res = jax.jit(lambda b, M2=M2: cg(A, b, preconditioner=M2, rtol=1e-5, max_steps=5000))(b)
        print(
            f'{name:>13} {f"{k} {end}":>16} | {rest.max() / rest.min():8.1e} | '
            f'{int(res.num_steps):8d}{"" if res.converged else " (not converged)"}',
            flush=True,
        )
