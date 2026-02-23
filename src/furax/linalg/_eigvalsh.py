import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _eigvalsh_2x2(A: Float[Array, '... 2 2']) -> Float[Array, '... 2']:
    """Analytic eigenvalues of batched symmetric 2x2 matrices, sorted ascending."""
    # symmetric 2x2 matrix
    #   [ a  b ]
    #   [ b  c ]
    # characteristic polynomial
    #   λ² - (a+b)λ + (ab - c²) = 0
    # gives via the quadratic formula
    #   λ = (a+b)/2  ±  sqrt(((a-b)/2)² + c²)
    a = A[..., 0, 0]
    b = A[..., 1, 1]
    c = A[..., 0, 1]
    mean = (a + b) / 2.0
    half_diff = jnp.sqrt(((a - b) / 2.0) ** 2 + c**2)
    return jnp.stack([mean - half_diff, mean + half_diff], axis=-1)


def _eigvalsh_3x3(A: Float[Array, '... 3 3']) -> Float[Array, '... 3']:
    """Analytic eigenvalues of batched symmetric 3x3 matrices, sorted ascending.

    Uses the trigonometric method for real symmetric matrices.
    https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    """
    # 3x3 symmetric matrix
    #   [ a  d  e ]
    #   [ d  b  f ]
    #   [ e  f  c ]
    a = A[..., 0, 0]
    b = A[..., 1, 1]
    c = A[..., 2, 2]
    d = A[..., 0, 1]
    e = A[..., 0, 2]
    f = A[..., 1, 2]

    p1 = d**2 + e**2 + f**2
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    p = jnp.sqrt(p2 / 6.0)

    inv_p = 1.0 / (p + 1e-30)
    B00 = (a - q) * inv_p
    B11 = (b - q) * inv_p
    B22 = (c - q) * inv_p
    Bd = d * inv_p
    Be = e * inv_p
    Bf = f * inv_p

    detB_half = (
        B00 * (B11 * B22 - Bf * Bf) - Bd * (Bd * B22 - Bf * Be) + Be * (Bd * Bf - B11 * Be)
    ) / 2.0

    # B = (A - q*I) / p is a scaled, traceless matrix with eigenvalues in [-2, 2],
    # so det(B)/2 lies in [-1, 1] and we can safely apply arccos.
    phi = jnp.arccos(jnp.clip(detB_half, -1.0, 1.0)) / 3.0

    # since phi ∈ [0, pi/3]; the three eigenvalues correspond to cos evaluated at
    # phi, phi + 2pi/3, phi + 4pi/3, giving descending order for eig2, eig1, eig0.
    eig0 = q + 2.0 * p * jnp.cos(phi + 2.0 * jnp.pi / 3.0)
    eig2 = q + 2.0 * p * jnp.cos(phi)
    eig1 = 3.0 * q - eig0 - eig2  # since q = trace/3

    return jnp.stack([eig0, eig1, eig2], axis=-1)


def eigvalsh(A: Float[Array, '... n n'], batch_size: int = 10_000) -> Float[Array, '... n']:
    """Eigenvalues of batched symmetric matrices, sorted ascending.

    Uses an analytic closed-form for n<=3, and jax.numpy.linalg.eigvalsh for larger matrices.

    Args:
        A: Batched symmetric matrix array with shape (..., n, n).
        batch_size: Number of matrices to process per XLA kernel launch for n>3.
            When batch_size >= total number of matrices the full batch is processed
            in one launch (maximum parallelism). Reduce to limit peak memory usage.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f'eigvalsh requires (..., n, n) input, got shape {A.shape}')
    n = A.shape[-1]
    if n == 1:
        return A[..., :1, 0]
    elif n == 2:
        return _eigvalsh_2x2(A)
    elif n == 3:
        return _eigvalsh_3x3(A)
    else:
        leading = A.shape[:-2]
        flat = A.reshape(-1, n, n)
        result = jax.lax.map(jnp.linalg.eigvalsh, flat, batch_size=batch_size)
        return result.reshape(*leading, n)  # type: ignore[no-any-return]
