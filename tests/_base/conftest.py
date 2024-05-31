import jax
import numpy as np
import pytest
from jax import Array
from jax import numpy as jnp

from astrosim._base.core import (
    AbstractLinearOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
)


@pytest.fixture(params=range(3), ids=['IdentityOperator', 'HomothetyOperator', 'DiagonalOperator'])
def base_op_and_dense(request: pytest.FixtureRequest) -> (AbstractLinearOperator, Array):
    dtype = np.float32
    in_structure = (jax.ShapeDtypeStruct((2, 3), dtype), jax.ShapeDtypeStruct((), dtype))
    match request.param:
        case 0:
            return IdentityOperator(in_structure), jnp.identity(7, dtype)
        case 1:
            return HomothetyOperator(in_structure, 2.0), 2.0 * jnp.identity(7, dtype)
        case 2:
            return DiagonalOperator((jnp.arange(1, 7).reshape(2, 3), jnp.array(8))), jnp.diag(
                jnp.r_[jnp.arange(1, 7), 8]
            )
    raise Exception


@pytest.fixture
def base_op(base_op_and_dense) -> AbstractLinearOperator:
    return base_op_and_dense[0]
