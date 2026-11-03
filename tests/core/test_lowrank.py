import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax import LowRankFourierOperator
from furax.obs.stokes import StokesQU

N_SAMPLES = 64
SAMPLE_RATE = 8.0
EDGES = (0.5, 2.0)
N_CHANNELS = 5
N_MODES = 2


def make_operator(key, *, n_samples=N_SAMPLES, dtype=jnp.float64, complex_modes=False):
    k_d, k_v, k_g = jax.random.split(key, 3)
    n_bins = len(EDGES) + 1
    diagonal = jax.random.uniform(k_d, (n_bins, N_CHANNELS), minval=0.5, maxval=2.0, dtype=dtype)
    modes = jax.random.normal(k_v, (n_bins, N_CHANNELS, N_MODES), dtype=dtype)
    if complex_modes:
        modes = modes + 1j * jax.random.normal(k_g, modes.shape, dtype=dtype)
    power = jax.random.uniform(k_g, (n_bins, N_MODES), minval=0.1, maxval=3.0, dtype=dtype)
    core = jax.vmap(jnp.diag)(power).astype(modes.dtype)
    structure = jax.ShapeDtypeStruct((N_CHANNELS, n_samples), dtype)
    return LowRankFourierOperator(
        diagonal, modes, core, edges=EDGES, sample_rate=SAMPLE_RATE, in_structure=structure
    )


def dense_reference(op):
    """The operator written as a dense matrix on the (channel, sample) vector, bin by bin."""
    n = op.in_structure.shape[-1]
    frequencies = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    bins = np.searchsorted(np.array(EDGES), frequencies, side='right')
    spectral = np.zeros((N_CHANNELS, N_CHANNELS, n // 2 + 1), dtype=complex)
    for i, b in enumerate(bins):
        modes = np.asarray(op.modes[b])
        spectral[..., i] = (
            np.diag(np.asarray(op.diagonal[b])) + modes @ np.asarray(op.core[b]) @ modes.conj().T
        )
    matrix = np.zeros((N_CHANNELS * n, N_CHANNELS * n))
    for j in range(N_CHANNELS * n):
        e = np.zeros(N_CHANNELS * n)
        e[j] = 1
        spectrum = np.fft.rfft(e.reshape(N_CHANNELS, n))
        out = np.einsum('abf,bf->af', spectral, spectrum)
        matrix[:, j] = np.fft.irfft(out, n=n).ravel()
    return matrix


@pytest.mark.parametrize('complex_modes', [False, True])
def test_matches_dense_reference(complex_modes):
    op = make_operator(jax.random.key(0), complex_modes=complex_modes)
    np.testing.assert_allclose(op.as_matrix(), dense_reference(op), atol=1e-10)


@pytest.mark.parametrize('n_samples', [64, 63])
def test_inverse_is_the_matrix_inverse(n_samples):
    op = make_operator(jax.random.key(1), n_samples=n_samples)
    matrix, inverse = op.as_matrix(), op.inverse().as_matrix()
    np.testing.assert_allclose(inverse @ matrix, np.eye(matrix.shape[0]), atol=1e-8)
    np.testing.assert_allclose(inverse, inverse.T, atol=1e-10)
    assert np.linalg.eigvalsh((inverse + inverse.T) / 2).min() > 0


def test_inverse_of_inverse_is_the_operator():
    op = make_operator(jax.random.key(2), complex_modes=True)
    back = op.inverse().inverse()
    np.testing.assert_allclose(back.as_matrix(), op.as_matrix(), atol=1e-8)


def test_zero_modes_leave_a_diagonal_operator():
    op = make_operator(jax.random.key(3))
    op = LowRankFourierOperator(
        op.diagonal,
        jnp.zeros_like(op.modes),
        jnp.zeros_like(op.core),
        edges=EDGES,
        sample_rate=SAMPLE_RATE,
        in_structure=op.in_structure,
    )
    x = jax.random.normal(jax.random.key(4), op.in_structure.shape)
    # the diagonal is frequency dependent, so compare in Fourier space
    n = x.shape[-1]
    bins = np.searchsorted(np.array(EDGES), np.fft.rfftfreq(n, 1 / SAMPLE_RATE), side='right')
    expected = np.fft.irfft(np.asarray(op.diagonal)[bins].T * np.fft.rfft(x), n=n)
    np.testing.assert_allclose(
        op.inverse()(x),
        np.fft.irfft(np.fft.rfft(x) / np.asarray(op.diagonal)[bins].T, n=n),
        atol=1e-10,
    )
    np.testing.assert_allclose(op(x), expected, atol=1e-10)


def test_white_noise_limit():
    structure = jax.ShapeDtypeStruct((3, 40), jnp.float64)
    sigma2 = jnp.array([1.0, 2.0, 4.0])
    op = LowRankFourierOperator(
        jnp.tile(sigma2, (2, 1)),
        jnp.zeros((2, 3, 1)),
        jnp.zeros((2, 1, 1)),
        edges=(1.0,),
        sample_rate=SAMPLE_RATE,
        in_structure=structure,
    )
    x = jax.random.normal(jax.random.key(5), structure.shape)
    np.testing.assert_allclose(op.inverse()(x), x / sigma2[:, None], atol=1e-12)


def test_pytree_input_is_stream_major():
    op = make_operator(jax.random.key(6))
    x = jax.random.normal(jax.random.key(7), op.in_structure.shape)
    split = (
        jax.ShapeDtypeStruct((2, N_SAMPLES), x.dtype),
        jax.ShapeDtypeStruct((3, N_SAMPLES), x.dtype),
    )
    tree_op = LowRankFourierOperator(
        op.diagonal, op.modes, op.core, edges=EDGES, sample_rate=SAMPLE_RATE, in_structure=split
    )
    q, u = tree_op((x[:2], x[2:]))
    np.testing.assert_allclose(jnp.concatenate([q, u]), op(x), atol=1e-12)


def test_stokes_input():
    x = StokesQU(jnp.ones((2, 32)), jnp.zeros((2, 32)))
    structure = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), x)
    op = LowRankFourierOperator(
        jnp.ones((1, 4)),
        jnp.ones((1, 4, 1)),
        jnp.ones((1, 1, 1)),
        edges=(),
        sample_rate=SAMPLE_RATE,
        in_structure=structure,
    )
    y = op(x)
    assert isinstance(y, StokesQU)
    # M = 1 + 1 1ᵀ on the (q0, q1, u0, u1) channels: the common mode of x is 2 in every channel
    np.testing.assert_allclose(y.q, jnp.full((2, 32), 3.0), atol=1e-12)
    np.testing.assert_allclose(y.u, jnp.full((2, 32), 2.0), atol=1e-12)


def test_float32_input_keeps_dtype():
    op = make_operator(jax.random.key(8), dtype=jnp.float32)
    y = op(jnp.ones(op.in_structure.shape, jnp.float32))
    assert y.dtype == jnp.float32


@pytest.mark.parametrize('edges', [(1.0,), (2.0, 0.5)], ids=['wrong-count', 'unsorted'])
def test_invalid_edges(edges):
    op = make_operator(jax.random.key(9))
    with pytest.raises(ValueError, match='bin edges'):
        LowRankFourierOperator(
            op.diagonal,
            op.modes,
            op.core,
            edges=edges,
            sample_rate=SAMPLE_RATE,
            in_structure=op.in_structure,
        )


def test_invalid_channel_count():
    op = make_operator(jax.random.key(10))
    with pytest.raises(ValueError, match='channels'):
        LowRankFourierOperator(
            op.diagonal,
            op.modes,
            op.core,
            edges=EDGES,
            sample_rate=SAMPLE_RATE,
            in_structure=jax.ShapeDtypeStruct((N_CHANNELS + 1, N_SAMPLES), jnp.float64),
        )
