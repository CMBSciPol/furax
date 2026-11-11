import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax import MaskOperator
from furax.mapmaking._gaps import GapSplines, gap_budget, max_gap_amplitudes
from furax.obs.stokes import StokesQU

# at resolution 3: one per sample, then linear, quadratic and cubic splines
GAPS = [(5, 2), (20, 5), (40, 9), (70, 20)]


def _amplitudes(length: int, resolution: int) -> int:
    intervals = -(-length // resolution)
    return min(length, intervals + min(3, intervals - 1))


def _mask(n: int = 120) -> np.ndarray:
    """Two channels with the gaps of every degree, and a channel with no good sample."""
    good = np.ones((3, n), bool)
    for start, length in GAPS:
        good[:2, start : start + length] = False
    good[1, 100:] = False  # a gap reaching the end
    good[2] = False
    return good


@pytest.mark.parametrize('resolution', [1, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_counts(resolution: int, n_samples: int) -> None:
    """The flags past `n_samples` pad the TOD: they take no amplitudes."""
    good = _mask()
    expected = sum(_amplitudes(length, resolution) for _, length in GAPS)
    counts = GapSplines(resolution, budget=0).counts(jnp.asarray(good), n_samples)
    tail = _amplitudes(20, resolution) if n_samples == 120 else 0  # channel 1's gap at the end
    assert_array_equal(counts, [expected, expected + tail, 0])


@pytest.mark.parametrize('resolution', [1, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_basis_spans_the_gaps_in_packed_slots(resolution: int, n_samples: int) -> None:
    """The amplitudes of each gap sum to one on its samples, in the first slots of the channel."""
    good = _mask()
    budget = 70
    splines = GapSplines(resolution, budget)
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
    if resolution == 1:
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
    for k, stream in enumerate((tod.q, tod.u)):
        alone = GapSplines(3, budget=70).basis(
            MaskOperator.from_boolean_mask(
                good, in_structure=jax.ShapeDtypeStruct(good.shape, jnp.float64)
            ),
            good.shape[-1],
        )
        assert_allclose(stream, alone(amplitudes[k]), atol=1e-12)


@pytest.mark.parametrize('resolution', [1, 4, 10])
@pytest.mark.parametrize('layout', ['singles', 'before-tail', 'short-read'])
def test_gap_budget_fits_every_mask(resolution: int, layout: str) -> None:
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
    count = int(GapSplines(resolution, budget).counts(jnp.asarray(good), n_read))
    assert count <= budget


def test_max_gap_amplitudes_leaves_out_the_detectors_dropped() -> None:
    good = np.ones((2, 100), bool)
    good[0, ::2] = False  # half flagged, one amplitude each
    good[1, :10] = False
    assert max_gap_amplitudes(good, 1, max_flagged_fraction=0.5) == 50
    assert max_gap_amplitudes(good, 1, max_flagged_fraction=0.4) == 10


@pytest.mark.parametrize('resolution', [1, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_max_gap_amplitudes_counts_as_the_splines(resolution: int, n_samples: int) -> None:
    """The probe of the observation alone counts as the splines of its padded mask."""
    good = _mask()
    counts = GapSplines(resolution, 0).counts(jnp.asarray(good), n_samples)
    probed = max_gap_amplitudes(good[:, :n_samples], resolution, max_flagged_fraction=1.0)
    assert probed == int(counts.max())
