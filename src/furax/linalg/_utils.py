"""Block PyTree linear algebra utilities for LOBPCG."""

from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Key, Num, PyTree

from furax.core import AbstractLinearOperator

P = TypeVar('P', bound=PyTree[Num[Array, '...']] | PyTree[jax.ShapeDtypeStruct])


def stack_pytrees(pytrees: list[PyTree[Num[Array, '...']]]) -> PyTree[Num[Array, ' k ...']]:
    """Stack k PyTrees into a single PyTree with an extra leading dimension.

    Each leaf in the resulting PyTree has shape (k, ...) where ... is the original leaf shape.

    Args:
        pytrees: A list of k PyTrees with identical structure.

    Returns:
        A PyTree where each leaf is the stack of corresponding leaves from the input PyTrees.

    Example:
        >>> import jax.numpy as jnp
        >>> p1 = {'a': jnp.array([1., 2.]), 'b': jnp.array([3.])}
        >>> p2 = {'a': jnp.array([4., 5.]), 'b': jnp.array([6.])}
        >>> block = stack_pytrees([p1, p2])
        >>> block['a'].shape
        (2, 2)
        >>> block['b'].shape
        (2, 1)
    """
    if not pytrees:
        raise ValueError('Cannot stack an empty list of PyTrees')
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *pytrees)


def unstack_pytree(block: PyTree[Num[Array, ' k ...']], k: int) -> list[PyTree[Num[Array, '...']]]:
    """Unstack a block PyTree into k individual PyTrees.

    Inverse operation of stack_pytrees.

    Args:
        block: A PyTree where each leaf has shape (k, ...).
        k: The number of PyTrees to unstack.

    Returns:
        A list of k PyTrees, each with the same structure as block but with the leading dimension
        removed.

    Example:
        >>> import jax.numpy as jnp
        >>> block = {'a': jnp.array([[1., 2.], [4., 5.]]), 'b': jnp.array([[3.], [6.]])}
        >>> pytrees = unstack_pytree(block, 2)
        >>> pytrees[0]['a']
        Array([1., 2.], dtype=float32)
        >>> pytrees[1]['b']
        Array([6.], dtype=float32)
    """
    leaves, treedef = jax.tree.flatten(block)
    return [treedef.unflatten([leaves[j][i] for j in range(len(leaves))]) for i in range(k)]


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


def batched_dot(
    X: PyTree[Num[Array, 'k ...']], Y: PyTree[Num[Array, 'm ...']]
) -> Float[Array, 'k m']:
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
        >>> batched_dot(X, Y)
        Array([[1., 2.],
               [1., 0.]], dtype=float32)
    """

    # X has shape (k, ...) for each leaf, Y has shape (m, ...) for each leaf
    # We want G[i, j] = sum over leaves of vdot(X_leaf[i], Y_leaf[j])
    def leaf_gram(x_leaf: Array, y_leaf: Array) -> Array:
        # x_leaf: (k, ...), y_leaf: (m, ...)
        # Flatten the non-batch dimensions
        k = x_leaf.shape[0]
        m = y_leaf.shape[0]
        x_flat = x_leaf.reshape(k, -1)  # (k, n)
        y_flat = y_leaf.reshape(m, -1)  # (m, n)
        # Compute G[i, j] = x_flat[i] @ y_flat[j].conj()
        return x_flat @ jnp.conj(y_flat).T  # (k, m)

    leaf_grams = jax.tree.map(leaf_gram, X, Y)
    leaf_list = jax.tree.leaves(leaf_grams)
    result: Array = leaf_list[0]
    for leaf in leaf_list[1:]:
        result = result + leaf
    return result


def apply_operator_block(
    A: AbstractLinearOperator, X: PyTree[Num[Array, 'k ...']]
) -> PyTree[Num[Array, 'k ...']]:
    """Apply operator A to each of k vectors in block X using vmap.

    Args:
        A: A linear operator.
        X: A block PyTree with k vectors (each leaf has leading dimension k).

    Returns:
        A block PyTree with A applied to each vector.

    Example:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([2., 3.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> X = jnp.array([[1., 1.], [0., 1.]])  # 2 vectors
        >>> apply_operator_block(A, X)
        Array([[2., 3.],
               [0., 3.]], dtype=float32)
    """
    # Get the tree structure
    leaves, treedef = jax.tree.flatten(X)
    k = leaves[0].shape[0]

    # Define a function that applies A to a single vector (represented as leaf slices)
    def apply_single(i: Array) -> list[Array]:
        single_x = treedef.unflatten([leaf[i] for leaf in leaves])
        result = A.mv(single_x)
        return jax.tree.leaves(result)

    # Vectorize over the batch dimension
    result_leaves = jax.vmap(apply_single)(jnp.arange(k))
    # result_leaves is a list of arrays, each of shape (k, ...)
    return treedef.unflatten(result_leaves)


def block_norms(X: PyTree[Num[Array, ' k ...']]) -> Float[Array, ' k']:
    """Compute the norm of each vector in block X.

    Args:
        X: A block PyTree with k vectors.

    Returns:
        A 1D array of k norms.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[3., 4.], [1., 0.]])}  # 2 vectors
        >>> block_norms(X)
        Array([5., 1.], dtype=float32)
    """

    # Compute squared norms for each leaf and sum
    def leaf_squared_norms(leaf: Array) -> Array:
        # leaf: (k, ...)
        k = leaf.shape[0]
        flat = leaf.reshape(k, -1)  # (k, n)
        return jnp.sum(jnp.abs(flat) ** 2, axis=1)  # (k,)

    leaf_list = jax.tree.leaves(jax.tree.map(leaf_squared_norms, X))
    squared_norms: Array = leaf_list[0]
    for leaf in leaf_list[1:]:
        squared_norms = squared_norms + leaf
    return jnp.sqrt(squared_norms)


def apply_rotation(
    X: PyTree[Num[Array, 'm ...']], C: Float[Array, 'm k']
) -> PyTree[Num[Array, 'k ...']]:
    """Compute linear combination Y = X @ C.

    Each column of C defines a linear combination of the m vectors in X,
    producing k output vectors.

    Args:
        X: A block PyTree with m vectors.
        C: A (m, k) coefficient matrix.

    Returns:
        A block PyTree with k vectors, where Y[j] = sum_i X[i] * C[i, j].

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [0., 1.]])}  # 2 vectors
        >>> C = jnp.array([[1., 0.], [1., 1.]])  # 2x2 rotation
        >>> Y = apply_rotation(X, C)
        >>> Y['a']  # Y[0] = X[0] + X[1], Y[1] = X[1]
        Array([[1., 1.],
               [0., 1.]], dtype=float32)
    """

    # X leaves have shape (m, ...), C has shape (m, k)
    # Output leaves should have shape (k, ...)
    def rotate_leaf(leaf: Array) -> Array:
        # leaf: (m, ...)
        m = leaf.shape[0]
        rest_shape = leaf.shape[1:]
        flat = leaf.reshape(m, -1)  # (m, n)
        # Result: C.T @ flat -> (k, n)
        result_flat = C.T @ flat
        k = C.shape[1]
        return result_flat.reshape((k,) + rest_shape)

    return jax.tree.map(rotate_leaf, X)


def qr_pytree(
    X: PyTree[Num[Array, 'k ...']],
) -> tuple[PyTree[Num[Array, 'r ...']], Float[Array, 'r k']]:
    """QR decomposition for block PyTrees.

    Computes Q, R such that X = Q @ R where Q is orthonormal and R is upper triangular.
    If k > n (more vectors than dimension), returns r = min(k, n) orthonormal vectors.

    Args:
        X: A block PyTree with k vectors.

    Returns:
        Q: An orthonormal block PyTree with r = min(k, n) vectors.
        R: A (r, k) matrix such that X = Q @ R.

    Example:
        >>> import jax.numpy as jnp
        >>> X = {'a': jnp.array([[1., 0.], [1., 1.]])}  # 2 vectors
        >>> Q, R = qr_pytree(X)
        >>> jnp.allclose(batched_dot(Q, Q), jnp.eye(2), atol=1e-5)
        Array(True, dtype=bool)
    """
    leaves, treedef = jax.tree.flatten(X)
    k = leaves[0].shape[0]

    # Flatten all leaves to get a single (k, n) matrix
    flat_leaves = [leaf.reshape(k, -1) for leaf in leaves]
    sizes = [leaf.shape[1] for leaf in flat_leaves]
    X_flat = jnp.concatenate(flat_leaves, axis=1)  # (k, total_n)
    total_n = X_flat.shape[1]

    # Use SVD for robust orthonormalization that handles k > n case
    # X_flat = U @ S @ V^T where U is (k, r), S is (r,), V is (total_n, r), r = min(k, n)
    U, S, Vt = jnp.linalg.svd(X_flat, full_matrices=False)
    # U has orthonormal columns: U^T @ U = I
    # Q_flat = V @ U^T gives orthonormal rows: Q_flat @ Q_flat^T = I

    r = min(k, total_n)
    # The orthonormalized vectors are V @ diag(sign) where sign normalizes
    # Actually, we want Q such that Q^T Q = I (orthonormal rows in our convention)
    # Q_flat = U^T would give us that, but we need to handle dimensions

    # For orthonormal vectors as rows: use Q = U (k, r) transposed view
    # But we need (r, total_n) shape to split back into leaves
    # The right singular vectors V (total_n, r) with Vt (r, total_n) are what we need
    Q_flat = Vt[:r, :]  # (r, total_n)

    # R = U[:, :r] @ diag(S[:r]) gives the coefficients: X_flat = U @ S @ Vt
    # So X_flat = (U @ diag(S)) @ Vt, meaning R.T = U @ diag(S)
    R = (U[:, :r] * S[:r]).T  # (r, k)

    # Split back into leaves with shape (r, ...)
    Q_leaves = []
    start = 0
    for i, leaf in enumerate(leaves):
        end = start + sizes[i]
        Q_leaf_flat = Q_flat[:, start:end]
        new_shape = (r,) + leaf.shape[1:]
        Q_leaves.append(Q_leaf_flat.reshape(new_shape))
        start = end

    Q = treedef.unflatten(Q_leaves)
    return Q, R


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
        >>> jnp.allclose(batched_dot(Q, Q), jnp.eye(2), atol=1e-5)
        Array(True, dtype=bool)
    """
    Q, _ = qr_pytree(X)
    return Q


def rayleigh_ritz(
    S: PyTree[Num[Array, 'm ...']], AS: PyTree[Num[Array, 'm ...']], k: int, largest: bool = False
) -> tuple[Float[Array, ' k'], PyTree[Num[Array, ' k ...']]]:
    """Perform Rayleigh-Ritz procedure to extract k Ritz pairs.

    Given a search space S and AS = A @ S, solve the reduced eigenvalue problem
    S^H A S c = lambda S^H S c and return the k smallest (or largest) eigenvalues
    and corresponding Ritz vectors.

    Args:
        S: Orthonormal search space with m vectors.
        AS: A applied to S.
        k: Number of eigenvalues to return.
        largest: If True, return largest eigenvalues; otherwise smallest.

    Returns:
        eigenvalues: The k smallest (or largest) eigenvalues.
        eigenvectors: A block PyTree with k Ritz vectors.
    """
    # Compute the reduced matrix S^H A S
    G = batched_dot(S, AS)  # (m, m)
    # Ensure Hermitian (for numerical stability)
    G = (G + jnp.conj(G.T)) / 2

    # Solve the dense eigenvalue problem
    eigenvalues, eigenvectors = jnp.linalg.eigh(G)  # eigenvalues sorted ascending

    if largest:
        # Select k largest (last k)
        idx = jnp.arange(eigenvalues.shape[0] - k, eigenvalues.shape[0])
    else:
        # Select k smallest (first k)
        idx = jnp.arange(k)

    selected_eigenvalues = eigenvalues[idx]
    selected_eigenvectors = eigenvectors[:, idx]  # (m, k)

    # Compute Ritz vectors: X = S @ C where C are the selected eigenvectors
    ritz_vectors = apply_rotation(S, selected_eigenvectors)

    return selected_eigenvalues, ritz_vectors
