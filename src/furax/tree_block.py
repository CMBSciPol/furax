"""Block PyTree utilities.

This module provides utilities for working with "block PyTrees" - PyTrees where each leaf
has an extra leading dimension representing k vectors. For example, a block of k vectors
where each vector is {'a': (n,), 'b': (m,)} would be represented as {'a': (k, n), 'b': (k, m)}.
"""

import itertools
from collections.abc import Sequence
from math import prod
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Key, Num, PyTree

from furax import tree as _tree

__all__ = [
    'stack',
    'unstack',
    'block_zeros_like',
    'block_normal_like',
    'block_to_array',
    'block_from_array',
    'gram',
    'block_norm',
    'matvec',
    'vecmat',
    'qr',
    'orthonormalize',
]

P = TypeVar('P', bound=PyTree[Num[Array, '...']] | PyTree[jax.ShapeDtypeStruct])


def stack(
    pytrees: Sequence[PyTree[Num[Array, '...']]], axis: int = 0
) -> PyTree[Num[Array, ' k ...']]:
    """Stack k PyTrees into a single PyTree with an extra dimension.

    Each leaf in the resulting PyTree has an extra dimension of size k inserted at `axis`.

    Args:
        pytrees: A list of k PyTrees with identical structure.
        axis: The axis in the result along which the input PyTrees are stacked. Default is 0.

    Returns:
        A PyTree where each leaf is the stack of corresponding leaves from the input PyTrees.

    Example:
        >>> import jax.numpy as jnp
        >>> p1 = {'a': jnp.array([1., 2.]), 'b': jnp.array([3.])}
        >>> p2 = {'a': jnp.array([4., 5.]), 'b': jnp.array([6.])}
        >>> block = stack([p1, p2])
        >>> block['a'].shape
        (2, 2)
        >>> block['b'].shape
        (2, 1)
    """
    if not pytrees:
        raise ValueError('Need at least one Pytree to stack')
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=axis), *pytrees)


def unstack(
    block: PyTree[Num[Array, ' k ...']], axis: int = 0
) -> tuple[PyTree[Num[Array, '...']], ...]:
    """Unstack a block PyTree into k individual PyTrees.

    Inverse operation of stack_pytrees.

    Args:
        block: A PyTree where each leaf has a dimension of size k at `axis`.
        axis: The axis along which to unstack. Default is 0.

    Returns:
        A list of k PyTrees, each with the same structure as block but with `axis` removed.

    Example:
        >>> import jax.numpy as jnp
        >>> block = {'a': jnp.array([[1., 2.], [4., 5.]]), 'b': jnp.array([[3.], [6.]])}
        >>> pytrees = unstack(block)
        >>> pytrees[0]['a']
        Array([1., 2.], dtype=float32)
        >>> pytrees[1]['b']
        Array([6.], dtype=float32)
    """
    leaves, treedef = jax.tree.flatten(block)
    k = leaves[0].shape[axis]
    return tuple(
        treedef.unflatten([jnp.take(leaf, i, axis=axis) for leaf in leaves]) for i in range(k)
    )


def block_zeros_like(x: P, k: int) -> P:
    """Create a block of k zero PyTrees with an extra leading dimension.

    Args:
        x: A PyTree of array-like leaves with ``shape`` and ``dtype`` attributes.
        k: The number of vectors in the block.

    Returns:
        A PyTree where each leaf has shape (k, ...) filled with zeros.

    Example:
        >>> import jax.numpy as jnp
        >>> x = {'a': jnp.array([1., 2.]), 'b': jnp.array([3.])}
        >>> block = block_zeros_like(x, 3)
        >>> block['a'].shape
        (3, 2)
    """
    result: P = jax.tree.map(lambda leaf: jnp.zeros((k,) + leaf.shape, leaf.dtype), x)
    return result


def block_normal_like(x: P, k: int, key: Key[Array, '']) -> P:
    """Create a block of k random normal PyTrees with an extra leading dimension.

    Args:
        x: A PyTree of array-like leaves with ``shape`` and ``dtype`` attributes.
        k: The number of vectors in the block.
        key: The PRNGKey to use.

    Returns:
        A PyTree where each leaf has shape (k, ...) filled with random normal values.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> x = {'a': jnp.array([1., 2.]), 'b': jnp.array([3.])}
        >>> block = block_normal_like(x, 3, jax.random.PRNGKey(0))
        >>> block['a'].shape
        (3, 2)
    """
    key_leaves = jax.random.split(key, len(jax.tree.leaves(x)))
    keys = jax.tree.unflatten(jax.tree.structure(x), key_leaves)
    result: P = jax.tree.map(
        lambda leaf, key: jax.random.normal(key, (k,) + leaf.shape, leaf.dtype), x, keys
    )
    return result


def block_to_array(
    X: PyTree[Num[Array, 'k ...']],
) -> tuple[Num[Array, 'k n'], jax.tree_util.PyTreeDef, list[tuple[int, ...]]]:
    """Concatenate all leaves of a block PyTree into a single (k, n) matrix.

    Each leaf's trailing dimensions are flattened and concatenated along axis 1.
    Use :func:`block_from_array` to reconstruct the original block PyTree.

    Args:
        X: A block PyTree with k vectors (each leaf has leading dimension k).

    Returns:
        X_flat: A (k, n) matrix where n is the total number of elements per vector.
        treedef: The PyTree structure, needed to reconstruct X.
        shapes: The trailing shape of each leaf, needed to reconstruct X.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.ones((3, 2)), 'b': jnp.ones((3, 1))}
        >>> X_flat, treedef, shapes = block_to_array(X)
        >>> X_flat.shape
        (3, 3)
    """
    leaves, treedef = jax.tree.flatten(X)
    k = leaves[0].shape[0]
    shapes = [leaf.shape[1:] for leaf in leaves]
    X_flat = jnp.concatenate([leaf.reshape(k, -1) for leaf in leaves], axis=1)
    return X_flat, treedef, shapes


def block_from_array(
    X_flat: Num[Array, 'k n'],
    treedef: jax.tree_util.PyTreeDef,
    shapes: list[tuple[int, ...]],
) -> PyTree[Num[Array, 'k ...']]:
    """Split a (k, n) matrix back into a block PyTree.

    Inverse of :func:`block_to_array`.

    Args:
        X_flat: A (k, n) matrix.
        treedef: The PyTree structure returned by :func:`block_to_array`.
        shapes: The trailing shapes returned by :func:`block_to_array`.

    Returns:
        A block PyTree with k vectors matching the original structure.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.ones((3, 2)), 'b': jnp.ones((3, 1))}
        >>> X_flat, treedef, shapes = block_to_array(X)
        >>> X2 = block_from_array(X_flat, treedef, shapes)
        >>> X2['a'].shape
        (3, 2)
    """
    k = X_flat.shape[0]
    sizes = [prod(s) for s in shapes]
    split_points = list(itertools.accumulate(sizes[:-1]))
    leaves = [
        chunk.reshape((k, *shape))
        for chunk, shape in zip(jnp.split(X_flat, split_points, axis=1), shapes)
    ]
    return treedef.unflatten(leaves)


def gram(X: PyTree[Num[Array, 'k ...']], Y: PyTree[Num[Array, 'm ...']]) -> Float[Array, 'k m']:
    """Compute the Gram matrix G[i,j] = dot(X[i], Y[j]) for block PyTrees.

    Args:
        X: A block PyTree with k vectors (each leaf has leading dimension k).
        Y: A block PyTree with m vectors (each leaf has leading dimension m).

    Returns:
        A (k, m) array where element [i, j] is the dot product of X[i] and Y[j].

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [0., 1.]])}  # 2 vectors
        >>> Y = {'a': jnp.array([[1., 1.], [2., 0.]])}  # 2 vectors
        >>> gram(X, Y)
        Array([[1., 2.],
               [1., 0.]], dtype=float32)
    """

    def leaf_gram(x_leaf: Array, y_leaf: Array) -> Array:
        return jnp.einsum('i...,j...->ij', x_leaf, jnp.conj(y_leaf))

    return sum(jax.tree.leaves(jax.tree.map(leaf_gram, X, Y)), start=jnp.array(0))


def block_norm(X: PyTree[Num[Array, ' k ...']]) -> Float[Array, ' k']:
    """Compute the norm of each vector in block X.

    Args:
        X: A block PyTree with k vectors.

    Returns:
        A 1D array of k norms.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[3., 4.], [1., 0.]])}  # 2 vectors
        >>> block_norm(X)
        Array([5., 1.], dtype=float32)
    """
    return jax.vmap(_tree.norm)(X)


def matvec(C: Float[Array, 'k m'], X: PyTree[Num[Array, 'm ...']]) -> PyTree[Num[Array, 'k ...']]:
    """Compute Y = C @ X for a block PyTree X.

    Args:
        C: A (k, m) matrix.
        X: A block PyTree with m vectors.

    Returns:
        A block PyTree with k vectors, where Y[i] = sum_j C[i, j] * X[j].

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [0., 1.]])}  # 2 vectors
        >>> C = jnp.array([[1., 1.], [0., 1.]])
        >>> Y = matvec(C, X)
        >>> Y['a']  # Y[0] = X[0] + X[1], Y[1] = X[1]
        Array([[1., 1.],
               [0., 1.]], dtype=float32)
    """
    return jax.tree.map(lambda leaf: jnp.einsum('km,m...->k...', C, leaf), X)


def vecmat(X: PyTree[Num[Array, 'm ...']], C: Float[Array, 'm k']) -> PyTree[Num[Array, 'k ...']]:
    """Compute Y = X @ C for a block PyTree X.

    Args:
        X: A block PyTree with m vectors.
        C: A (m, k) matrix.

    Returns:
        A block PyTree with k vectors, where Y[j] = sum_i X[i] * C[i, j].

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [0., 1.]])}  # 2 vectors
        >>> C = jnp.array([[1., 0.], [1., 1.]])
        >>> Y = vecmat(X, C)
        >>> Y['a']  # Y[0] = X[0] + X[1], Y[1] = X[1]
        Array([[1., 1.],
               [0., 1.]], dtype=float32)
    """
    return jax.tree.map(lambda leaf: jnp.einsum('mk,m...->k...', C, leaf), X)


def qr(
    X: PyTree[Num[Array, 'k ...']],
) -> tuple[PyTree[Num[Array, 'r ...']], Float[Array, 'r k']]:
    """QR decomposition for block PyTrees.

    Computes Q, R such that vecmat(Q, R) = X, where Q is orthonormal.
    If k > n (more vectors than dimension), returns r = min(k, n) orthonormal vectors.

    Args:
        X: A block PyTree with k vectors.

    Returns:
        Q: An orthonormal block PyTree with r = min(k, n) vectors.
        R: A (r, k) matrix such that vecmat(Q, R) = X.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [1., 1.]])}  # 2 vectors
        >>> Q, R = qr(X)
        >>> jnp.allclose(gram(Q, Q), jnp.eye(2), atol=1e-5)
        Array(True, dtype=bool)
    """
    X_flat, treedef, shapes = block_to_array(X)
    U, S, Vt = jnp.linalg.svd(X_flat, full_matrices=False)
    R = (U * S).T
    return block_from_array(Vt, treedef, shapes), R


def orthonormalize(X: PyTree[Num[Array, 'k ...']]) -> PyTree[Num[Array, 'r ...']]:
    """Return orthonormalized version of block PyTree X.

    Uses SVD for robust orthonormalization. If k > n (more vectors than dimension),
    returns r = min(k, n) orthonormal vectors.

    Args:
        X: A block PyTree with k vectors.

    Returns:
        Q: An orthonormal block PyTree with r = min(k, n) vectors.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [1., 1.]])}
        >>> Q = orthonormalize(X)
        >>> jnp.allclose(gram(Q, Q), jnp.eye(2), atol=1e-5)
        Array(True, dtype=bool)
    """
    Q, _ = qr(X)
    return Q
