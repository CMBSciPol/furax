from jax import Array
from jaxtyping import Float

from astrosim.landscapes import ValidStokesType
from astrosim.operators.qu_rotations import QURotationOperator


class HWPOperator(QURotationOperator):
    """Operator for an ideal Half-wave plate."""

    @classmethod
    def create(
        cls, shape: tuple[int, ...], stokes: ValidStokesType, hwp_angles: Float[Array, '...']
    ):
        # override to include additional factor 2 due to HWP properties
        return super().create(shape, stokes, 2 * hwp_angles)
