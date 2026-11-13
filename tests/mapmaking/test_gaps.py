import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax import MaskOperator
from furax.mapmaking._gaps import GapSplines, gap_budget
from furax.obs.stokes import StokesQU

# the start and length of each gap of the channels with good samples
GAPS = [(5, 2), (15, 4), (25, 5), (40, 6), (70, 20)]
# the amplitudes of each gap, by knot spacing: n = ⌈L / r⌉ intervals take the n + p B-splines of
# degree p = min(3, n − 1), or one amplitude per sample when that is fewer
AMPLITUDES = {
    1: [2, 4, 5, 6, 20],  # one per sample
    2: [1, 3, 5, 5, 13],  # a constant, linear, one per sample, quadratic, cubic
    3: [1, 3, 3, 3, 10],  # a constant, three linear, cubic
}


def _mask(n: int = 120) -> np.ndarray:
    """Two channels with the gaps of every degree, and a channel with no good sample."""
    good = np.ones((3, n), bool)
    for start, length in GAPS:
        good[:2, start : start + length] = False
    good[1, 100:] = False  # a gap reaching the end
    good[2] = False
    return good


@pytest.mark.parametrize('spacing', [1, 2, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_counts(spacing: int, n_samples: int) -> None:
    """The flags past `n_samples` pad the TOD: they take no amplitudes."""
    good = _mask()
    expected = sum(AMPLITUDES[spacing])
    counts = GapSplines(spacing, budget=0).counts(jnp.asarray(good), n_samples)
    tail = AMPLITUDES[spacing][-1] if n_samples == 120 else 0  # channel 1's 20 samples at the end
    assert_array_equal(counts, [expected, expected + tail, 0])


@pytest.mark.parametrize('spacing', [1, 2, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_basis_spans_the_gaps_in_packed_slots(spacing: int, n_samples: int) -> None:
    """The amplitudes of each gap sum to one on its samples, in the first slots of the channel."""
    good = _mask()
    budget = 70
    splines = GapSplines(spacing, budget)
    structure = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=structure)
    C = splines.basis(mask, n_samples).as_matrix()
    C = np.asarray(C).reshape(*good.shape, *good.shape[:-1], budget)
    counts = np.asarray(splines.counts(jnp.asarray(good), n_samples))

    # the padding samples are zero
    flagged = ~good & good.any(axis=-1, keepdims=True) & (np.arange(good.shape[-1]) < n_samples)
    for channel, count in enumerate(counts):
        own = C[channel, :, channel]  # (samp, budget)
        assert_allclose(own.sum(axis=-1), flagged[channel], atol=1e-12)
        assert_allclose(own[:, count:], 0)  # the slots past the channel's amplitudes are unused
        assert np.linalg.matrix_rank(own[:, :count]) == count
        others = np.delete(C[channel], channel, axis=1)
        assert_allclose(others, 0)  # a channel's samples take only its own amplitudes
    if spacing == 1:
        assert_array_equal(counts[:2], flagged[:2].sum(axis=-1))


def test_basis_of_a_demodulated_tod_has_amplitudes_per_stream() -> None:
    """The streams share the mask of their detector, and each has amplitudes of its own."""
    good = jnp.asarray(_mask()[:2])
    structure = StokesQU.structure_for(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(good, in_structure=structure)
    splines = GapSplines(3, budget=70)
    C = splines.basis(mask, good.shape[-1])
    assert C.in_structure.shape == (2, *good.shape[:-1], 70)
    amplitudes = jax.random.normal(jax.random.key(0), C.in_structure.shape, jnp.float64)
    tod = C(amplitudes)
    one_stream = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    alone = splines.basis(MaskOperator.from_boolean_mask(good, in_structure=one_stream), 120)
    for k, stream in enumerate((tod.q, tod.u)):
        assert_allclose(stream, alone(amplitudes[k]), atol=1e-12)


def test_squared_basis_gives_the_diagonal_of_the_gram() -> None:
    """`C.squared().T(d)` is the diagonal of Cᵀ diag(d) C, which preconditions the amplitudes."""
    good = _mask()
    structure = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=structure)
    C = GapSplines(3, budget=40).basis(mask, good.shape[-1])
    d = jax.random.uniform(jax.random.key(0), good.shape, jnp.float64, 1, 2)
    dense = np.asarray(C.as_matrix()).reshape(good.size, -1)
    expected = np.diag(dense.T @ (np.ravel(d)[:, None] * dense))
    assert_allclose(np.ravel(C.squared().T(d)), expected, rtol=1e-12)


@pytest.mark.parametrize('spacing', [1, 4, 10])
@pytest.mark.parametrize('layout', ['singles', 'before-tail', 'short-read'])
def test_gap_budget_fits_every_mask(spacing: int, layout: str) -> None:
    """The worst masks of at most `fraction` flagged samples, read up to `n_samples`, then padded."""
    n_samples, envelope, fraction = 300, 400, 0.3
    budget = gap_budget(n_samples, fraction)
    n_read = 4 if layout == 'short-read' else n_samples
    n_flagged = int(fraction * n_read)
    good = np.zeros(envelope, bool)
    good[:n_read] = True
    if layout == 'singles':  # isolated flagged samples, one amplitude each
        good[1 : 2 * n_flagged : 2] = False
    else:  # one gap up to the padding
        good[n_read - n_flagged : n_read] = False
    count = int(GapSplines(spacing, budget).counts(jnp.asarray(good), n_read))
    assert count <= budget
