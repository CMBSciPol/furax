import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax import AbstractLinearOperator, LowRankFourierOperator
from furax.obs.stokes import StokesQU

N_SAMPLES = 64
SAMPLE_RATE = 8.0
EDGES = (0.5, 2.0)
N_CHANNELS = 5
N_MODES = 2


def make_operator(
    key,
    *,
    n_samples=N_SAMPLES,
    dtype=jnp.float64,
    complex_modes=False,
    white_top=False,
    batch_size=1,
    edges=EDGES,
):
    k_d, k_v, k_g = jax.random.split(key, 3)
    n_bins = len(edges) + 1
    diagonal = jax.random.uniform(k_d, (n_bins, N_CHANNELS), minval=0.5, maxval=2.0, dtype=dtype)
    modes = jax.random.normal(k_v, (n_bins, N_CHANNELS, N_MODES), dtype=dtype)
    if complex_modes:
        modes = modes + 1j * jax.random.normal(k_g, modes.shape, dtype=dtype)
    power = jax.random.uniform(k_g, (n_bins, N_MODES), minval=0.1, maxval=3.0, dtype=dtype)
    core = jax.vmap(jnp.diag)(power).astype(modes.dtype)
    structure = jax.ShapeDtypeStruct((N_CHANNELS, n_samples), dtype)
    if white_top:
        modes, core = modes[:-1], core[:-1]
    return LowRankFourierOperator(
        diagonal,
        modes,
        core,
        edges=edges,
        sample_rate=SAMPLE_RATE,
        in_structure=structure,
        batch_size=batch_size,
    )


def probed_matrix(op):
    """The matrix found by applying `mv` to each basis vector, independently of `as_matrix`."""
    return AbstractLinearOperator.as_matrix(op)


@pytest.mark.parametrize(
    ('complex_modes', 'white_top', 'batch_size'),
    [
        (False, False, 1),
        (True, False, 1),
        (False, True, 1),
        (True, True, 2),
        (False, False, 2),
        (False, True, 5),
        (False, True, 7),
    ],
)
@pytest.mark.parametrize('n_samples', [N_SAMPLES, N_SAMPLES - 1])
def test_as_matrix_matches_probing(complex_modes, white_top, batch_size, n_samples):
    op = make_operator(
        jax.random.key(0),
        n_samples=n_samples,
        complex_modes=complex_modes,
        white_top=white_top,
        batch_size=batch_size,
    )
    np.testing.assert_allclose(op.as_matrix(), probed_matrix(op), atol=1e-10)


@pytest.mark.parametrize('edges', [(0.5, 5.0), ()], ids=['empty-white-bin', 'all-white'])
def test_white_top_bin_edge_cases(edges):
    op = make_operator(jax.random.key(12), white_top=True, edges=edges)
    matrix = op.as_matrix()
    np.testing.assert_allclose(matrix, probed_matrix(op), atol=1e-10)
    np.testing.assert_allclose(
        op.inverse().as_matrix() @ matrix, np.eye(matrix.shape[0]), atol=1e-8
    )


def test_batches_lower_the_scratch_memory():
    # many channels and a narrow correlated band, where the FFTs dominate the scratch memory
    n_channels, n_samples = 50, 1 << 14

    def scratch(batch_size):
        return (
            LowRankFourierOperator(
                jnp.ones((3, n_channels)),
                jnp.ones((2, n_channels, N_MODES)),
                jnp.tile(jnp.eye(N_MODES), (2, 1, 1)),
                edges=(0.01, 0.02),
                sample_rate=SAMPLE_RATE,
                in_structure=jax.ShapeDtypeStruct((n_channels, n_samples), jnp.float64),
                batch_size=batch_size,
            )
            .profile()
            .temp_bytes
        )

    assert scratch(1) < scratch(n_channels) / 2


@pytest.mark.parametrize('white_top', [False, True])
@pytest.mark.parametrize('n_samples', [64, 63])
def test_inverse_is_the_matrix_inverse(n_samples, white_top):
    op = make_operator(jax.random.key(1), n_samples=n_samples, white_top=white_top)
    matrix, inverse = op.as_matrix(), op.inverse().as_matrix()
    np.testing.assert_allclose(inverse @ matrix, np.eye(matrix.shape[0]), atol=1e-8)
    np.testing.assert_allclose(inverse, inverse.T, atol=1e-10)
    assert np.linalg.eigvalsh((inverse + inverse.T) / 2).min() > 0


@pytest.mark.parametrize('white_top', [False, True])
def test_inverse_with_complex_modes_away_from_zero_and_nyquist(white_top):
    op = make_operator(jax.random.key(11), complex_modes=True, white_top=white_top)
    # a white last bin holds the Nyquist frequency, otherwise modes there have to be real
    edge_bins = jnp.array([0] if white_top else [0, 2])
    modes = op.modes.at[edge_bins].set(op.modes[edge_bins].real)
    op = LowRankFourierOperator(
        op.diagonal,
        modes,
        op.core,
        edges=EDGES,
        sample_rate=SAMPLE_RATE,
        in_structure=op.in_structure,
    )
    matrix, inverse = op.as_matrix(), op.inverse().as_matrix()
    np.testing.assert_allclose(inverse @ matrix, np.eye(matrix.shape[0]), atol=1e-8)


def test_inverse_of_inverse_is_the_operator():
    op = make_operator(jax.random.key(2), complex_modes=True)
    back = op.inverse().inverse()
    np.testing.assert_allclose(back.as_matrix(), op.as_matrix(), atol=1e-8)


@pytest.mark.parametrize('white_top', [False, True])
def test_jit_and_grad(white_top):
    # batch_size=2 on 5 channels exercises both the traced loop and the remainder batch
    op = make_operator(jax.random.key(4), white_top=white_top, batch_size=2)
    x = jax.random.normal(jax.random.key(5), op.in_structure.shape)
    expected = op(x)
    np.testing.assert_allclose(jax.jit(lambda op, x: op(x))(op, x), expected, atol=1e-10)
    # the operator is real symmetric, so the gradient of x·Mx is 2Mx
    grad = jax.jit(jax.grad(lambda x: jnp.vdot(x, op(x))))(x)
    np.testing.assert_allclose(grad, 2 * expected, atol=1e-10)


@pytest.mark.parametrize('edges', [EDGES, (0.3, 2.1)], ids=['aligned-edges', 'unaligned-edges'])
def test_zero_modes_leave_a_diagonal_operator(edges):
    # an edge on a frequency of the FFT starts its bin there; otherwise the bin starts at the next
    op = make_operator(jax.random.key(3), edges=edges)
    op = LowRankFourierOperator(
        op.diagonal,
        jnp.zeros_like(op.modes),
        jnp.zeros_like(op.core),
        edges=edges,
        sample_rate=SAMPLE_RATE,
        in_structure=op.in_structure,
    )
    x = jax.random.normal(jax.random.key(4), op.in_structure.shape)
    # the diagonal is frequency dependent, so compare in Fourier space
    n = x.shape[-1]
    bins = np.searchsorted(np.array(edges), np.fft.rfftfreq(n, 1 / SAMPLE_RATE), side='right')
    diagonal = np.asarray(op.diagonal)[bins].T
    np.testing.assert_allclose(op(x), np.fft.irfft(diagonal * np.fft.rfft(x), n=n), atol=1e-10)
    np.testing.assert_allclose(
        op.inverse()(x), np.fft.irfft(np.fft.rfft(x) / diagonal, n=n), atol=1e-10
    )


@pytest.mark.parametrize('white_top', [False, True])
def test_padded_zero_modes_are_inert(white_top):
    op = make_operator(jax.random.key(13), white_top=white_top)
    # drop the second mode of every bin, or zero it in place
    kwargs = {'edges': EDGES, 'sample_rate': SAMPLE_RATE, 'in_structure': op.in_structure}
    short = LowRankFourierOperator(op.diagonal, op.modes[..., :1], op.core[:, :1, :1], **kwargs)
    padded = LowRankFourierOperator(
        op.diagonal, op.modes.at[..., 1].set(0), op.core.at[:, 1, 1].set(5.0), **kwargs
    )
    np.testing.assert_allclose(padded.as_matrix(), short.as_matrix(), atol=1e-10)
    np.testing.assert_allclose(
        padded.inverse().as_matrix(), short.inverse().as_matrix(), atol=1e-10
    )


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
    op = make_operator(jax.random.key(8))
    x = jax.random.normal(jax.random.key(9), op.in_structure.shape)
    op32 = LowRankFourierOperator(
        op.diagonal.astype(jnp.float32),
        op.modes.astype(jnp.float32),
        op.core.astype(jnp.float32),
        edges=EDGES,
        sample_rate=SAMPLE_RATE,
        in_structure=jax.ShapeDtypeStruct(x.shape, jnp.float32),
    )
    y = op32(x.astype(jnp.float32))
    assert y.dtype == jnp.float32
    # float32 matmuls may run in reduced precision on a GPU, hence the loose tolerance
    expected = op(x)
    np.testing.assert_allclose(y, expected, atol=1e-3 * np.abs(expected).max())


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


@pytest.mark.parametrize(
    ('n_mode_bins', 'n_core_bins', 'match'),
    [(1, 1, 'Modes have shape'), (3, 2, 'Core has shape')],
    ids=['modes-bins', 'core-bins'],
)
def test_invalid_modes_or_core_bins(n_mode_bins, n_core_bins, match):
    op = make_operator(jax.random.key(14))
    with pytest.raises(ValueError, match=match):
        LowRankFourierOperator(
            op.diagonal,
            op.modes[:n_mode_bins],
            op.core[:n_core_bins],
            edges=EDGES,
            sample_rate=SAMPLE_RATE,
            in_structure=op.in_structure,
        )


def test_invalid_batch_size():
    op = make_operator(jax.random.key(15))
    with pytest.raises(ValueError, match='batch_size'):
        LowRankFourierOperator(
            op.diagonal,
            op.modes,
            op.core,
            edges=EDGES,
            sample_rate=SAMPLE_RATE,
            in_structure=op.in_structure,
            batch_size=0,
        )
