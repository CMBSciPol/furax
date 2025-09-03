import operator
from collections.abc import Callable
from math import prod
from typing import Any, TypeVar

import jax
from jax import Array
from jax import numpy as jnp
from jax.tree_util import PyTreeDef
from jaxtyping import Key, Num, PyTree, ScalarLike

__all__ = [
    'as_promoted_dtype',
    'as_structure',
    'full_like',
    'normal_like',
    'ones_like',
    'zeros_like',
    'is_leaf',
    'add',
    'sub',
    'mul',
    'truediv',
    'power',
    'dot',
    'matvec',
    'vecmat',
    'matmat',
]

from furax.exceptions import StructureError

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


def is_leaf(x: Any) -> bool:
    """Returns true if the input is a Pytree leaf."""
    treedef = jax.tree.structure(x)
    return jax.tree_util.treedef_is_leaf(treedef)


def apply(
    operation: Callable[[Any, Any], Any], a: PyTree[Array], b: PyTree[Array]
) -> PyTree[Array]:
    def func_a_treedef(a_leaf, b_leaf):  # type: ignore[no-untyped-def]
        if a_leaf is None or b_leaf is None:
            return None
        if is_leaf(b_leaf):
            return operation(a_leaf, b_leaf)
        return jax.tree.map(lambda b_inner_leaf: operation(a_leaf, b_inner_leaf), b_leaf)

    def func_b_treedef(a_leaf, b_leaf):  # type: ignore[no-untyped-def]
        if a_leaf is None or b_leaf is None:
            return None
        if is_leaf(a_leaf):
            return operation(a_leaf, b_leaf)
        return jax.tree.map(lambda a_inner_leaf: operation(a_inner_leaf, b_leaf), a_leaf)

    a_leaves, treedef = jax.tree.flatten(a)
    try:
        b_leaves = treedef.flatten_up_to(b)
        func = func_a_treedef
    except ValueError:
        b_leaves, treedef = jax.tree.flatten(b)
        try:
            a_leaves = treedef.flatten_up_to(a)
        except ValueError as exc:
            raise StructureError(str(exc))
        func = func_b_treedef
    return treedef.unflatten(func(*xs) for xs in zip(a_leaves, b_leaves))


def add(a: PyTree[Array], b: PyTree[Array]) -> PyTree[Array]:
    return apply(operator.add, a, b)


def sub(a: PyTree[Array], b: PyTree[Array]) -> PyTree[Array]:
    return apply(operator.sub, a, b)


def mul(a: PyTree[Array], b: PyTree[Array]) -> PyTree[Array]:
    return apply(operator.mul, a, b)


def truediv(a: PyTree[Array], b: PyTree[Array]) -> PyTree[Array]:
    return apply(operator.truediv, a, b)


def power(a: PyTree[Array], b: PyTree[Array]) -> PyTree[Array]:
    return apply(operator.pow, a, b)


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


def matvec(
    outer_treedef: PyTreeDef | PyTree[Any], a: PyTree[Array], x: PyTree[Array]
) -> PyTree[Array]:
    """Generalized matrix-vector operation, where the matrix and the vector are pytrees.

    The structure of the generalized matrix is the tree product of an outer tree structure
    representing the rows and an inner tree structure that represents the columns. A tree product
    of two tree structures is formed by replacing each leaf of the first tree with a copy of
    the second. The leaves of the generalized matrix must be broadcastable when they belong to
    same inner tree (elements of the same row). There is no such requirement for leaves of different
    inner trees (elements of different rows).

    Args:
        outer_treedef: The outer structure of the generalized matrix.
        a: The generalized matrix, i.e. a pytree whose structure follows the tree product of
            an outer and inner tree structures, with leaves given by the elements of the
            generalized matrix.
        x: The generalized vector, with a structure matching the inner tree structure of `a`.

    Returns:
        A pytree with the same structure as the generalized matrix outer tree structure.

    Example:
        To represent with pytrees the sparse tensor
            [ a11 a12 ]
            [ a21 a22 ] of shape (2, 2, 100) where a11, a12 and a21 are arrays of 100 elements and
        a22 is zero:

        >>> from numpy.testing import assert_array_equal
        >>> a11, a12, a21 = jax.random.normal(jax.random.key(0), (3, 100))
        >>> a22 = 0
        >>> a = {
        ...     'row1': {'col1': a11, 'col2': a12},
        ...     'row2': {'col1': a21, 'col2': a22},
        ... }
        >>> x = {'col1': 1., 'col2': 2.}
        >>> y = matvec({'row1': 0, 'row2': 0}, a, x)
        >>> assert_array_equal(y['row1'], a11 * x['col1'] + a12 * x['col2'])
        >>> assert_array_equal(y['row2'], a21 * x['col1'] + a22 * x['col2'])
    """
    if not isinstance(outer_treedef, PyTreeDef):
        outer_treedef = jax.tree.structure(outer_treedef)
    outer_leaves = outer_treedef.flatten_up_to(a)
    leaves = []
    for outer_leaf in outer_leaves:
        leaf = sum(jax.tree.leaves(mul(outer_leaf, x)))
        leaves.append(leaf)
    return outer_treedef.unflatten(leaves)


def vecmat(
    x: PyTree[Array], outer_treedef: PyTreeDef | PyTree[Any], a: PyTree[Array]
) -> PyTree[Array]:
    """Generalized vector-matrix operation, where the matrix and the vector are pytrees.

    The structure of the generalized matrix is the tree product of an outer tree structure
    representing the rows and an inner tree structure that represents the columns. A tree product
    of two tree structures is formed by replacing each leaf of the first tree with a copy of
    the second. The leaves of the generalized matrix must be broadcastable when they belong to
    same inner tree (elements of the same row). There is no such requirement for leaves of different
    inner trees (elements of different rows).

    Args:
        outer_treedef: The outer structure of the generalized matrix.
        a: The generalized matrix, i.e. a pytree whose structure follows the tree product of
            an outer and inner tree structures, with leaves given by the elements of the
            generalized matrix.
        x: The generalized vector, with a structure matching the outer tree structure of `a`.

    Returns:
        A pytree with the same structure as the generalized matrix inner tree structure.

    Example:
        To represent with pytrees the sparse tensor
            [ a11 a12 ]
            [ a21 a22 ] of shape (2, 2, 100) where a11, a12 and a21 are arrays of 100 elements and
        a22 is zero:

        >>> from numpy.testing import assert_array_equal
        >>> a11, a12, a21 = jax.random.normal(jax.random.key(0), (3, 100))
        >>> a22 = 0
        >>> a = {
        ...     'row1': {'col1': a11, 'col2': a12},
        ...     'row2': {'col1': a21, 'col2': a22},
        ... }
        >>> x = {'row1': 1., 'row2': 2.}
        >>> y = vecmat(x, {'row1': 0, 'row2': 0}, a)
        >>> assert_array_equal(y['col1'], a11 * x['row1'] + a21 * x['row2'])
        >>> assert_array_equal(y['col2'], a12 * x['row1'] + a22 * x['row2'])
    """
    if not isinstance(outer_treedef, PyTreeDef):
        outer_treedef = jax.tree.structure(outer_treedef)
    inner_treedef = jax.tree.structure(outer_treedef.flatten_up_to(a)[0])
    transposed_a = jax.tree.transpose(outer_treedef, inner_treedef, a)
    return matvec(inner_treedef, transposed_a, x)


def matmat(
    a_outer_treedef: PyTreeDef | PyTree[Any],
    a: PyTree[Array],
    b_outer_treedef: PyTreeDef | PyTree[Any],
    b: PyTree[Array],
) -> PyTree[Array]:
    """Generalized matrix-matrix operation, where the matrices are pytrees.

    The structure of the generalized matrices is the tree product of an outer tree structure
    representing the rows and an inner tree structure that represents the columns. A tree product
    of two tree structures is formed by replacing each leaf of the first tree with a copy of
    the second. The leaves of the generalized matrix must be broadcastable when they belong to
    same inner tree (elements of the same row). There is no such requirement for leaves of different
    inner trees (elements of different rows).

    Args:
        a_outer_treedef: The outer structure of the first generalized matrix.
        a: The first generalized matrix, i.e. a pytree whose structure follows the tree product of
            an outer and inner tree structures, with leaves given by the elements of the
            generalized matrix.
        b_outer_treedef: The outer structure of the second generalized matrix.
        a: The second generalized matrix.

    Returns:
        A pytree whose structure is the tree product of the outer structure of `a` and the inner
        structure of `b`.
    """
    if not isinstance(a_outer_treedef, PyTreeDef):
        a_outer_treedef = jax.tree.structure(a_outer_treedef)
    if not isinstance(b_outer_treedef, PyTreeDef):
        b_outer_treedef = jax.tree.structure(b_outer_treedef)
    a_outer_leaves = a_outer_treedef.flatten_up_to(a)
    leaves = []
    for a_outer_leaf in a_outer_leaves:
        leaf = vecmat(a_outer_leaf, b_outer_treedef, b)
        leaves.append(leaf)
    return a_outer_treedef.unflatten(leaves)


def _get_outer_treedef(inner_treedef: PyTreeDef | PyTree[Any], tree: PyTree[Array]) -> PyTreeDef:
    """Given a pytree whose structure is the tree product of an outer and inner structures,
    returns the outer structure, knowing the inner structure.
    """
    if not isinstance(inner_treedef, PyTreeDef):
        inner_treedef = jax.tree.structure(inner_treedef)

    def is_inner(node: Any) -> bool:
        return jax.tree.structure(node) == inner_treedef

    outer_tree = jax.tree.map(lambda x: 0, tree, is_leaf=is_inner)
    return jax.tree.structure(outer_tree)


def _tree_to_dense(
    outer_treedef: PyTreeDef | PyTree[Any],
    inner_treedef: PyTreeDef | PyTree[Any],
    tree: PyTree[Array],
) -> Array:
    """Dense representation of a pytree matrix."""
    if not isinstance(outer_treedef, PyTreeDef):
        outer_treedef = jax.tree.structure(outer_treedef)
    if not isinstance(inner_treedef, PyTreeDef):
        inner_treedef = jax.tree.structure(inner_treedef)
    leaves = jax.tree.leaves(tree)
    promoted_dtype = jnp.result_type(*leaves)
    broadcast_shape = jnp.broadcast_shapes(*(jnp.shape(leaf) for leaf in leaves))
    tree_shape = (outer_treedef.num_leaves, inner_treedef.num_leaves)
    dense = jnp.empty(broadcast_shape + (prod(tree_shape),), dtype=promoted_dtype)
    for i, leaf in enumerate(leaves):
        dense = dense.at[..., i].set(leaf)
    dense_shape = broadcast_shape + tree_shape
    return dense.reshape(dense_shape)


def _dense_to_tree(
    outer_treedef: PyTreeDef | PyTree[Any], inner_treedef: PyTreeDef | PyTree[Any], dense: Array
) -> PyTree[Array]:
    """Pytree representation of a dense tensor."""
    if not isinstance(outer_treedef, PyTreeDef):
        outer_treedef = jax.tree.structure(outer_treedef)
    if not isinstance(inner_treedef, PyTreeDef):
        inner_treedef = jax.tree.structure(inner_treedef)
    outer_num_leaves = outer_treedef.num_leaves
    inner_num_leaves = inner_treedef.num_leaves
    return jax.tree.unflatten(
        outer_treedef,
        [
            jax.tree.unflatten(inner_treedef, [dense[..., i, j] for j in range(inner_num_leaves)])
            for i in range(outer_num_leaves)
        ],
    )
