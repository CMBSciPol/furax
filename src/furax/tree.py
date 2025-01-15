from typing import Any, TypeVar

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Key, Num, PyTree, ScalarLike

__all__ = [
    'as_promoted_dtype',
    'as_structure',
    'full_like',
    'is_leaf',
    'normal_like',
    'ones_like',
    'zeros_like',
]


def is_leaf(x: Any) -> bool:
    """Returns true if the input is a Pytree leaf."""
    treedef = jax.tree.structure(x)
    return jax.tree_util.treedef_is_leaf(treedef)


def dot(x: PyTree[Num[Array, '...']], y: PyTree[Num[Array, '...']]) -> Num[Array, '']:
    """Scalar product of two Pytrees.

    If one of the leaves is complex, the hermitian scalar product is returned.

    Args:
        x: The first Pytree.
        y: The second Pytree.

    Example:
        >>> import furax as fx
        >>> x = {'a': jnp.array([1., 2, 3]), 'b': jnp.array([1, 0])}
        >>> y = {'a': jnp.array([2, -1, 1]), 'b': jnp.array([2, 0])}
        >>> fx.tree.dot(x, y)
        Array(5., dtype=float32)
    """
    xy = jax.tree.map(jnp.vdot, x, y)
    return sum(jax.tree.leaves(xy), start=jnp.array(0))


P = TypeVar('P', bound=PyTree[Num[Array, '...']] | PyTree[jax.ShapeDtypeStruct])


def as_promoted_dtype(x: P) -> P:
    """Promotes the data type of the leaves of a pytree to a common data type.

    Args:
        x: The pytree to promote.

    Example:
        >>> as_promoted_dtype({'a': jnp.ones(2, jnp.float16), 'b': jnp.ones(2, jnp.float32)})
        {'a': Array([1., 1.], dtype=float32), 'b': Array([1., 1.], dtype=float32)}
    """
    leaves = jax.tree.leaves(x)
    promoted_dtype = jnp.result_type(*leaves)
    result: P = jax.tree.map(
        lambda leaf: (
            jax.ShapeDtypeStruct(leaf.shape, promoted_dtype)
            if isinstance(leaf, jax.ShapeDtypeStruct)
            else jnp.astype(leaf, promoted_dtype)
        ),
        x,
    )
    return result


def as_structure(x: P) -> P:
    """Returns the pytree of ShapedDtypeStruct leaves associated with x.

    Args:
        x: The pytree whose structure will be returned.

    Examples:
        >>> as_structure(jnp.ones(10))
        ShapeDtypeStruct(shape=(10,), dtype=float32)

        >>> as_structure({'a': [jnp.zeros((2, 3)), jnp.array(2)]})
        {'a': [ShapeDtypeStruct(shape=(2, 3), dtype=float32),
        ShapeDtypeStruct(shape=(), dtype=int32, weak_type=True)]}
    """
    result: P = jax.eval_shape(lambda _: _, x)
    return result


def zeros_like(x: P) -> P:
    """Returns a pytrees of zeros with the same structure as x.

    Args:
        x: The pytree of array-like leaves with ``shape`` and ``dtype`` attributes, whose structure
            will be used to construct the output pytree of zeros.

    Examples:
        >>> zeros_like({'a': jnp.ones(2, dtype=jnp.int32)})
        {'a': Array([0, 0], dtype=int32)}

        >>> zeros_like({'a': jax.ShapeDtypeStruct((2,), jnp.int32)})
        {'a': Array([0, 0], dtype=int32)}
    """
    return full_like(x, 0)


def ones_like(x: P) -> P:
    """Returns a pytrees of ones with the same structure as x.

    Args:
        x: The pytree of array-like leaves with ``shape`` and ``dtype`` attributes, whose structure
            will be used to construct the output pytree of ones.

    Example:
        >>> ones_like({'a': jnp.zeros(2, dtype=jnp.int32)})
        {'a': Array([1, 1], dtype=int32)}

        >>> ones_like({'a': jax.ShapeDtypeStruct((2,), jnp.int32)})
        {'a': Array([1, 1], dtype=int32)}
    """
    return full_like(x, 1)


def full_like(x: P, fill_value: ScalarLike) -> P:
    """Returns a pytrees of a specified value with the same structure as x.

    Args:
        x: The pytree of array-like leaves with ``shape`` and ``dtype`` attributes, whose structure
            will be used to construct the output pytree of the specified value..
        fill_value: The value to fill with.

    Example:
        >>> full_like({'a': jnp.array(1, jnp.int32), 'b': jnp.array(2, jnp.float32)}, 3)
        {'a': Array(3, dtype=int32), 'b': Array(3., dtype=float32)}

        >>> full_like({'a': jax.ShapeDtypeStruct((2,), jnp.int32),
        ...          'b': jax.ShapeDtypeStruct((), jnp.float32)}, 3)
        {'a': Array([3, 3], dtype=int32), 'b': Array(3., dtype=float32)}
    """
    result: P = jax.tree.map(lambda leaf: jnp.full(leaf.shape, fill_value, leaf.dtype), x)
    return result


def normal_like(x: P, key: Key[Array, '']) -> P:
    """Returns a pytrees of a normal values with the same structure as x.

    Args:
        x: The pytree of array-like leaves with ``shape`` and ``dtype`` attributes, whose structure
            will be used to construct the output pytree of pseudo-random values.
        key: The PRNGKey to use.

    Example:
        >>> normal_like({'a': jnp.array(1, jnp.float16),
        ...            'b': jnp.array(2, jnp.float32)}, jax.random.PRNGKey(0))
        {'a': Array(-1.34, dtype=float16), 'b': Array(-1.2515389, dtype=float32)}

        >>> normal_like({'a': jax.ShapeDtypeStruct((2,), jnp.float16),
        ...            'b': jax.ShapeDtypeStruct((), jnp.float32)}, jax.random.PRNGKey(0))
        {'a': Array([-1.34  ,  0.1431], dtype=float16), 'b': Array(-1.2515389, dtype=float32)}
    """
    key_leaves = jax.random.split(key, len(jax.tree.leaves(x)))
    keys = jax.tree.unflatten(jax.tree.structure(x), key_leaves)
    result: P = jax.tree.map(
        lambda leaf, key: jax.random.normal(key, leaf.shape, leaf.dtype), x, keys
    )
    return result


def uniform_like(x: P, key: Key[Array, ''], low: float = 0.0, high: float = 1.0) -> P:
    """Returns a pytrees of a uniform values with the same structure as x.

    Args:
        x: The pytree of array-like leaves with ``shape`` and ``dtype`` attributes, whose structure
            will be used to construct the output pytree of pseudo-random values.
        key: The PRNGKey to use.
        min_val: The minimum value of the uniform distribution.
        max_val: The maximum value of the uniform distribution.

    Example:
        >>> uniform_like({'a': jnp.array(1, jnp.float16),
        ...            'b': jnp.array(2, jnp.float32)}, jax.random.PRNGKey(0))
        {'a': Array(0.08984, dtype=float16), 'b': Array(0.10536897, dtype=float32)}

        >>> uniform_like({'a': jax.ShapeDtypeStruct((2,), jnp.float16),
        ...            'b': jax.ShapeDtypeStruct((), jnp.float32)}, jax.random.PRNGKey(0))
        {'a': Array([0.08984, 0.5566 ], dtype=float16),'b': Array(0.10536897, dtype=float32)}
    """
    key_leaves = jax.random.split(key, len(jax.tree.leaves(x)))
    keys = jax.tree.unflatten(jax.tree.structure(x), key_leaves)
    result: P = jax.tree.map(
        lambda leaf, key: jax.random.uniform(key, leaf.shape, leaf.dtype, low, high), x, keys
    )
    return result


def allclose(x: P, y: P, rtol=1e-05, atol=1e-08, equal_nan=False) -> bool:
    return jax.tree.all(
        jax.tree.map(
            lambda x, y: jnp.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan), x, y
        ),
        True,
    )
