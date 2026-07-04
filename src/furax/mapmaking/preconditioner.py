import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, symmetric
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
    def create(cls, op: AbstractLinearOperator) -> 'BJPreconditioner':
        """Assemble the per-pixel blocks from a symmetric operator acting on Stokes sky maps.

        The operator is assumed diagonal over the pixel (map) axes. Each Stokes component is probed
        with a unit map; the response is that column of every pixel's block at once.
        """
        in_struct = op.in_structure
        if not isinstance(in_struct, Stokes):
            raise ValueError('operator must act on Stokes pytrees (sky maps)')
        if not in_struct == op.out_structure:
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
            resp = _apply(op, probe).array  # (n_i, *sky) = column j, indexed by output row i
            columns.append(jnp.moveaxis(resp, 0, -1))  # (*sky, n_i)
        blocks = jnp.stack(columns, axis=-1)  # (*sky, i, j)
        return cls(blocks, in_structure=in_struct)

    def mv(self, x: Stokes) -> Stokes:
        xm = jnp.moveaxis(x.array, 0, -1)  # (*sky, j)
        ym = jnp.einsum('...ij,...j->...i', self.blocks, xm)  # (*sky, i)
        return type(x).from_array(jnp.moveaxis(ym, -1, 0))  # (i, *sky)

    def inverse(self) -> 'BJPreconditioner':
        # Per-pixel matrix inverse; stays a BJPreconditioner (keeps the @symmetric tag).
        return BJPreconditioner(jnp.linalg.inv(self.blocks), in_structure=self.in_structure)

    def get_blocks(self) -> Float[Array, '*sky n n']:
        """Return the per-pixel Stokes blocks, shape ``(*sky, n, n)``."""
        return self.blocks
