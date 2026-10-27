"""Small dense linear algebra shared by the solvers of `furax.linalg`."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Num

# Full precision, which on GPU is not the default of float32 products (TF32).
_matmul = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)


def _hermitian_part(G: Num[Array, 'k k']) -> Num[Array, 'k k']:
    return (G + G.conj().T) / 2


def _eigh_live(K: Num[Array, 'k k'], live: Bool[Array, ' k'], largest: bool) -> tuple[Array, Array]:
    """Eigendecomposition of a Hermitian matrix, ascending, with its dead directions last.

    The rows and columns of the dead directions (False in `live`) must be zero. Their spurious
    eigenvalues 0 are moved past the end of the spectrum that is selected from, the largest
    eigenvalues when `largest`, the smallest otherwise, so that they are never selected.
    """
    sentinel = (-1 if largest else 1) * (2 * jnp.linalg.norm(K) + 1)
    return jnp.linalg.eigh(K + jnp.diag(jnp.where(live, 0, sentinel)).astype(K.dtype))
