import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_allclose, assert_array_equal

from furax import DiagonalOperator, MaskOperator
from furax.core import LowRankFourierOperator
from furax.linalg import cg
from furax.mapmaking._gaps import (
    GapBlockPreconditioner,
    GapSplines,
    bridge_amplitudes,
    gap_budget,
    knot_spacing,
    max_gap_amplitudes,
)
from furax.obs.stokes import StokesQU
from tests.mapmaking.helpers import ranges

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


def test_max_gap_amplitudes_leaves_out_the_detectors_dropped() -> None:
    good = np.ones((2, 100), bool)
    good[0, ::2] = False  # half flagged, one amplitude each
    good[1, :10] = False
    assert max_gap_amplitudes(ranges(good), 1, max_flagged_fraction=0.5) == 50
    assert max_gap_amplitudes(ranges(good), 1, max_flagged_fraction=0.4) == 10
    assert max_gap_amplitudes(ranges(np.ones((2, 100), bool)), 1, max_flagged_fraction=0.4) == 0


@pytest.mark.parametrize('spacing', [1, 3])
@pytest.mark.parametrize('n_samples', [120, 100])
def test_max_gap_amplitudes_counts_as_the_splines(spacing: int, n_samples: int) -> None:
    """The count from the observation's own samples is that of the splines of its padded mask."""
    good = _mask()
    counts = GapSplines(spacing, 0).counts(jnp.asarray(good), n_samples)
    probed = max_gap_amplitudes(ranges(good[:, :n_samples]), spacing, max_flagged_fraction=1.0)
    assert probed == int(counts.max())


@pytest.mark.parametrize(
    ('resolution', 'sample_rate', 'expected'),
    [(1.25, 8.0, 10), (0.04, 100.0, 4), (0.05, 8.0, 1), (0.0, 8.0, 1)],
)
def test_knot_spacing_rounds_to_whole_samples(
    resolution: float, sample_rate: float, expected: int
) -> None:
    assert knot_spacing(resolution, sample_rate) == expected


@pytest.mark.parametrize(('spacing', 'growth'), [(3.0, 1.0), (2.0, 1.5)])
def test_bridge_spans_the_padding_in_its_own_slots(spacing: float, growth: float) -> None:
    """A partition of unity over the padding, after the gap slots, on the channels with data."""
    good, n_samples, budget = _mask(), 100, 70
    bridge = bridge_amplitudes(good.shape[-1] - n_samples, spacing, growth)
    structure = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=structure)
    C = GapSplines(3, budget, bridge, spacing, growth).basis(mask, n_samples)
    assert C.in_structure.shape == (*good.shape[:-1], budget + bridge)
    matrix = np.asarray(C.as_matrix()).reshape(*good.shape, *good.shape[:-1], budget + bridge)
    without = np.asarray(GapSplines(3, budget).basis(mask, n_samples).as_matrix())
    assert_allclose(matrix[..., :budget].reshape(without.shape), without)  # the gaps unchanged
    for channel in range(2):
        tail = matrix[channel, :, channel, budget:]  # (samp, bridge)
        assert_allclose(tail[:n_samples], 0)
        assert_allclose(tail[n_samples:].sum(axis=-1), 1, atol=1e-12)
        assert np.linalg.matrix_rank(tail[n_samples:]) == bridge
    assert_allclose(matrix[2], 0)  # no data, nothing to bridge


def test_bridge_amplitudes_grow_with_the_log_of_the_padding() -> None:
    assert bridge_amplitudes(1001, 50.0, 1.0) == 20 + 3  # uniform knots, every 50 samples
    long, short = bridge_amplitudes(10_000, 50.0, 1.5), bridge_amplitudes(1_000, 50.0, 1.5)
    assert short < long < short + 15


def _correlated_weight(
    structure: PyTree[jax.ShapeDtypeStruct], white: bool = False, n_modes: int = 0
) -> LowRankFourierOperator:
    """$N^{-1}$ of a noise of white level 1, with $1/f$ power below 2 Hz at 10 Hz unless `white`.

    With `n_modes`, the channels share that many modes below 2 Hz.
    """
    n_channels = sum(leaf.size // leaf.shape[-1] for leaf in jax.tree.leaves(structure))
    power = jnp.array([50.0, 10.0, 3.0, 1.0])[:, None] * jnp.linspace(1, 2, n_channels)
    if white:
        power = jnp.ones_like(power)
    modes = jax.random.normal(jax.random.key(1), (3, n_channels, max(n_modes, 1)))
    core = jnp.broadcast_to(
        2.0 * (n_modes > 0) * jnp.eye(modes.shape[-1]), (3, *2 * modes.shape[-1:])
    )
    noise = LowRankFourierOperator(
        power, modes, core, edges=(0.2, 0.5, 2.0), sample_rate=10.0, in_structure=structure
    )
    return noise.inverse()


def _long_gaps(n: int = 400) -> np.ndarray:
    """Three detectors: long gaps, short ones, and one without good samples."""
    good = np.ones((3, n), bool)
    good[0, 50:250] = False
    good[1, 30:150] = False
    good[1, 200:205] = False
    good[1, 300:301] = False
    good[2] = False
    return good


@pytest.mark.parametrize('white', [True, False])
@pytest.mark.parametrize(('block_size', 'bridged'), [(16, False), (100, False), (16, True)])
def test_block_preconditioner_inverts_the_blocks_of_the_gram(
    white: bool, block_size: int, bridged: bool
) -> None:
    """The inverse of Cᵀ D⁻¹ C on each block of slots, the streams of a detector each their own.

    Under white noise, its diagonal is that of the diagonal preconditioner.
    """
    good = _long_gaps()
    n_samples = 350 if bridged else good.shape[-1]
    structure = StokesQU.structure_for(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=structure)
    budget = 70
    bridge = bridge_amplitudes(good.shape[-1] - n_samples, 5.0, 1.5) if bridged else 0
    splines = GapSplines(10, budget, bridge, 5.0, 1.5)
    C = splines.basis(mask, n_samples)
    weight = _correlated_weight(structure, white)
    P = GapBlockPreconditioner.create(splines, C, weight, block_size)

    gram = np.asarray((C.T @ weight @ C).as_matrix())
    slots = np.arange(gram.shape[0])
    per_channel = budget + bridge
    same_block = (slots[:, None] // per_channel == slots[None, :] // per_channel) & (
        (slots[:, None] % per_channel) // block_size == (slots[None, :] % per_channel) // block_size
    )
    expected = np.where(same_block, gram, 0)
    expected += np.diag(np.diag(expected) == 0)  # the unused slots
    matrix = np.asarray(P.as_matrix())
    assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert_allclose(np.linalg.inv(matrix), expected, atol=1e-9 * np.abs(gram).max())
    if white:
        diagonal = np.ravel(C.squared().T(jax.tree.map(jnp.ones_like, C.T.in_structure)))
        used = diagonal > 0
        assert_allclose(np.diag(np.linalg.inv(matrix))[used], diagonal[used], rtol=1e-10)


def test_block_preconditioner_speeds_up_cg_under_correlated_modes() -> None:
    """On long gaps under a low-rank noise, far fewer steps than the diagonal, same solution."""
    good = _long_gaps(800)
    good[0, 400:700] = False
    structure = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=structure)
    splines = GapSplines(4, 160)
    C = splines.basis(mask, good.shape[-1])
    weight = _correlated_weight(structure, n_modes=2)
    used = C.squared().T(jnp.ones(good.shape)) > 0
    unused = DiagonalOperator(jnp.where(used, 0.0, 1.0), in_structure=C.in_structure)
    A = C.T @ weight @ C + unused
    b = jnp.where(used, jax.random.normal(jax.random.key(0), used.shape, jnp.float64), 0)

    # the diagonal preconditioner, from the zero-lag of the weight
    zero_lag = np.diag(np.asarray(weight.as_matrix())).reshape(good.shape)
    diagonal = C.squared().T(jnp.asarray(zero_lag))
    jacobi = DiagonalOperator(jnp.where(used, 1 / jnp.where(used, diagonal, 1), 1))
    block = GapBlockPreconditioner.create(splines, C, _correlated_weight(structure), block_size=160)

    options = {'rtol': 1e-10, 'max_steps': 1000}
    with_jacobi = cg(A, b, preconditioner=jacobi, **options)
    with_block = cg(A, b, preconditioner=block, **options)
    assert with_jacobi.converged and with_block.converged
    assert_allclose(with_block.solution, with_jacobi.solution, rtol=1e-7, atol=1e-7)
    assert 3 * with_block.num_steps < with_jacobi.num_steps
