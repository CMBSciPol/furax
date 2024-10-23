import string
import sys
from collections.abc import Callable
from typing import Any

if sys.version_info < (3, 12):
    from itertools import islice

    def batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch

else:
    from itertools import batched

import jax
import numpy as np
import pytest
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from furax._base.core import (
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
            return HomothetyOperator(2.0, in_structure), 2.0 * jnp.identity(7, dtype)
        case 2:
            return DiagonalOperator((jnp.arange(1, 7).reshape(2, 3), jnp.array(8))), jnp.diag(
                jnp.r_[jnp.arange(1, 7), 8]
            )
    raise Exception


@pytest.fixture
def base_op(base_op_and_dense) -> AbstractLinearOperator:
    return base_op_and_dense[0]


def pytree_dict_builder(*args: Any) -> dict[str, Any]:
    return {k: v for k, v in zip(string.ascii_lowercase, args)}


def pytree_list_builder(*args: Any) -> list[Any]:
    return list(args)


def pytree_tuple_builder(*args: Any) -> tuple[Any, ...]:
    return args


def pytree_nested_builder(*args: Any) -> dict[str, tuple[Any]]:
    return {k: v for k, v in zip(string.ascii_lowercase, batched(args, 2))}


@pytest.fixture(
    params=[pytree_dict_builder, pytree_list_builder, pytree_tuple_builder, pytree_nested_builder]
)
def pytree_builder(
    request: pytest.FixtureRequest,
) -> Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]]:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param
