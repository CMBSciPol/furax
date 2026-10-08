from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, symmetric
from furax.core._base import structure_equal
from furax.linalg import LowRankOperator, LowRankTerms
from furax.obs.stokes import Stokes

# Pass op as an explicit argument so JAX traces its arrays as inputs rather than
# capturing them as XLA constants (which would happen with jit(op.mv) or jit(op)).
_apply = jax.jit(lambda op, x: op(x))


@symmetric
class BJPreconditioner(AbstractLinearOperator):
    """Block-diagonal (per-pixel) Jacobi preconditioner for Stokes sky maps.

    Holds one dense ``(n, n)`` block per pixel (``n = len(stokes)``), coupling the Stokes
    components at that pixel: ``blocks`` has shape ``(*sky, n, n)`` with ``blocks[..., i, j]`` the
    response of output component ``i`` to input component ``j``. Applied by an einsum over the
    Stokes axis of the map's backing array; symmetric (self-adjoint) for a symmetric operator.
    """

    blocks: Float[Array, '*sky n n']

    def __init__(
        self,
        blocks: Float[Array, '*sky n n'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'blocks', blocks)
        super().__init__(in_structure=in_structure)

    @classmethod
    def create(cls, op: AbstractLinearOperator) -> Self:
        """Assemble the per-pixel blocks from a symmetric operator acting on Stokes sky maps.

        The operator is assumed diagonal over the pixel (map) axes. Each Stokes component is probed
        with a unit map; the response is that column of every pixel's block at once.
        """
        in_struct = op.in_structure
        if not isinstance(in_struct, Stokes):
            raise TypeError('operator must act on Stokes pytrees (sky maps)')
        if not structure_equal(in_struct, op.out_structure):
            raise ValueError('operator must be square')

        stokes_cls = type(in_struct)
        n = len(in_struct.stokes)
        sky_shape = in_struct.shape
        dtype = in_struct.dtype

        columns = []
        for j in range(n):
            # unit map on component j (one everywhere on that component, zero on the others);
            # the backing array has the Stokes components on the leading axis.
            probe = stokes_cls.from_array(jnp.zeros((n, *sky_shape), dtype).at[j].set(1.0))
            columns.append(_apply(op, probe).data)  # (n_i, *sky) = column j, indexed by row i
        blocks = jnp.moveaxis(jnp.stack(columns, axis=-1), 0, -2)  # (i, *sky, j) -> (*sky, i, j)
        return cls(blocks, in_structure=in_struct)

    def mv[StokesT: Stokes](self, x: StokesT) -> StokesT:
        # einsum aligns the blocks' trailing (i, j) axes against x.data's leading Stokes axis
        # directly, without physically transposing either array.
        return type(x).from_array(jnp.einsum('...ij,j...->i...', self.blocks, x.data))

    def inverse(self) -> Self:
        # Per-pixel matrix inverse; stays a BJPreconditioner (keeps the @symmetric tag).
        return type(self)(jnp.linalg.inv(self.blocks), in_structure=self.in_structure)


def make_two_level_preconditioner(
    M: AbstractLinearOperator, eigenpairs: LowRankTerms, *, preconditioned: bool = False
) -> AbstractLinearOperator:
    r"""Two-level preconditioner deflating the eigenpairs of the system operator $A$.

    The two-level preconditioner of MAPPRAISER (https://arxiv.org/abs/2112.03370) is

    $$
    M_2 = M (I - A Q) + Q, \quad Q = Z (Z^T A Z)^{-1} Z^T,
    $$

    where the columns of $Z$ span the subspace to deflate, typically the eigenvectors with the
    smallest eigenvalues, which slow down conjugate gradient the most. $M_2 A$ is the identity on
    the span of $Z$ and coincides with $M A$ on its $A$-orthogonal complement, so the deflated
    eigenvalues no longer limit convergence. Two choices of $Z$ are supported:

    - Eigenpairs $(\Theta, Z)$ of $A$, with orthonormal $Z$. Then $A Z = Z \Theta$ and
      $M_2 = M (I - Z Z^T) + Z \Theta^{-1} Z^T$.
    - Eigenpairs $(\Theta, Z)$ of the preconditioned operator $M A$, with $M^{-1}$-orthonormal
      $Z$. Then $A Z = M^{-1} Z \Theta$ and $M_2 = M + Z (\Theta^{-1} - I) Z^T$.

    Either way, applying $M_2$ costs no product with $A$, but the simplification relies on the
    eigenpairs being converged, e.g. by [`lanczos_tr`][furax.linalg.lanczos_tr] with
    `which='SA'` (and `preconditioner=M` for the second choice).

    Args:
        M: First-level preconditioner, e.g. the inverse of a [`BJPreconditioner`][].
        eigenpairs: Eigenvalues $\Theta$ and eigenvectors $Z$ of $A$, or of $M A$.
        preconditioned: Whether `eigenpairs` are those of $M A$ rather than of $A$.

    Returns:
        The two-level preconditioner $M_2$, acting on the same unknowns as $M$.
    """
    theta, Z = eigenpairs
    if preconditioned:
        return M + LowRankOperator(LowRankTerms(1 / theta - 1, Z), in_structure=M.in_structure)

    Q = LowRankOperator(LowRankTerms(1 / theta, Z), in_structure=M.in_structure)
    projector = LowRankOperator(LowRankTerms(jnp.ones_like(theta), Z), in_structure=M.in_structure)
    identity = IdentityOperator(in_structure=M.in_structure)
    return M @ (identity - projector) + Q
