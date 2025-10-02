import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Shaped, UInt8

from ._base import AbstractLinearOperator, symmetric


@symmetric
class MaskOperator(AbstractLinearOperator):
    """Operator that sets values to zero according to a boolean mask.

    The mask is bit-packed to save memory and unpacked as needed when applying to a vector.

    A True value in the boolean mask means that the data point is valid.
    """

    _mask: UInt8[Array, '...']
    """Sample mask, bit-packed along the last axis to save memory"""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    """The static input structure of the operator"""

    def __init__(
        self,
        boolean_mask: Bool[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        # Check consistency
        self._in_structure = in_structure
        if not boolean_mask.shape == in_structure.shape:
            raise ValueError('Shape mismatch')

        # Pack boolean mask along samples axis
        self._mask = jnp.packbits(boolean_mask, axis=-1)

    def mv(self, x: Shaped[Array, '*dims']) -> Shaped[Array, '*dims']:
        boolean_mask = jnp.unpackbits(self._mask, axis=-1, count=x.shape[-1])
        # True = good, False = bad
        return jnp.where(boolean_mask, x, 0)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
