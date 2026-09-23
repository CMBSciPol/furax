import jax
import jax.numpy as jnp
import jax_healpy as jhp
import pytest
from equinox import tree_equal
from fastquat import Quaternion
from jax.tree_util import register_static
from numpy.testing import assert_array_almost_equal, assert_array_equal

import furax.tree as ftree
from furax.core import AbstractLinearOperator, CompositionOperator, IndexOperator
from furax.math.coords import from_iso_angles, from_xieta_angles, to_polarization_angle_cos_sin
from furax.obs.landscapes import (
    CARLandscape,
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
    WCSProjection,
)
from furax.obs.operators import QURotationOperator
from furax.obs.operators._qu_rotations import rotate_qu_cs
from furax.obs.pointing import PointingOperator, SampledPointing, XSamplingOperator
from furax.obs.stokes import ValidStokesLiteral

NSIDE = 4
NDET, NSAMP = 3, 10

# Full-sphere CAR landscape (180×360 at 1°/pixel, crval at RA=180° to avoid wrap issues)
_CAR_PROJECTION = WCSProjection(crpix=(180.5, 90.5), crval=(180.0, 0.0), cdelt=(-1.0, 1.0))


def _make_landscape(landscape_type: str, stokes: ValidStokesLiteral) -> StokesLandscape:
    if landscape_type == 'healpix':
        return HealpixLandscape(NSIDE, stokes)
    return CARLandscape((180, 360), _CAR_PROJECTION, stokes)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
class TestAsExpandedOperator:
    def test_mv(self, stokes, frame, landscape_type) -> None:
        """PointingOperator.mv is equivalent to as_expanded_operator().mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = Quaternion.random(key1, (NSAMP,))
        qdet = Quaternion.random(key2, (NDET,))

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, batch_size=2)
        sky = landscape.normal(key3)

        tod_direct = pointing_op(sky)
        tod_expanded = pointing_op.as_expanded_operator()(sky)

        assert tree_equal(tod_direct, tod_expanded, rtol=1e-10, atol=0)

    def test_transpose_mv(self, stokes, frame, landscape_type) -> None:
        """PointingOperator.T.mv is equivalent to as_expanded_operator().T.mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = Quaternion.random(key1, (NSAMP,))
        qdet = Quaternion.random(key2, (NDET,))

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, batch_size=2)
        tod = pointing_op.out_structure
        tod = jax.tree.map(lambda s: jax.random.normal(key3, s.shape, s.dtype), tod)

        sky_direct = pointing_op.T(tod)
        sky_expanded = pointing_op.as_expanded_operator().T(tod)

        assert tree_equal(sky_direct, sky_expanded, rtol=1e-10, atol=0)

    def test_mv_interpolate(self, stokes, frame, landscape_type) -> None:
        """Interpolated PointingOperator.mv equals as_expanded_operator().mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = Quaternion.random(key1, (NSAMP,))
        qdet = Quaternion.random(key2, (NDET,))

        pointing_op = PointingOperator.create(
            landscape, qbore, qdet, frame=frame, batch_size=2, interpolate=True
        )
        sky = landscape.normal(key3)

        assert tree_equal(pointing_op(sky), pointing_op.as_expanded_operator()(sky), rtol=1e-10)

    def test_transpose_mv_interpolate(self, stokes, frame, landscape_type) -> None:
        """Interpolated PointingOperator.T.mv equals as_expanded_operator().T.mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = Quaternion.random(key1, (NSAMP,))
        qdet = Quaternion.random(key2, (NDET,))

        pointing_op = PointingOperator.create(
            landscape, qbore, qdet, frame=frame, batch_size=2, interpolate=True
        )
        tod = jax.tree.map(
            lambda s: jax.random.normal(key3, s.shape, s.dtype), pointing_op.out_structure
        )

        sky_direct = pointing_op.T(tod)
        sky_expanded = pointing_op.as_expanded_operator().T(tod)

        assert tree_equal(sky_direct, sky_expanded, rtol=1e-10)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
class TestInterpolate:
    def test_adjoint(self, stokes, landscape_type) -> None:
        """<P x, y> = <x, P^T y> for the interpolated operator."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(7)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        qbore = Quaternion.random(k1, (NSAMP,))
        qdet = Quaternion.random(k2, (NDET,))

        op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        sky = landscape.normal(k3)
        tod = jax.tree.map(lambda s: jax.random.normal(k4, s.shape, s.dtype), op.out_structure)

        lhs = ftree.dot(op(sky), tod)
        rhs = ftree.dot(sky, op.T(tod))
        assert_array_almost_equal(lhs, rhs, decimal=10)

    def test_uniform_sky(self, landscape_type) -> None:
        """Sampling a uniform sky returns the same constant (weights always sum to 1)."""
        landscape = _make_landscape(landscape_type, 'I')

        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        qbore = Quaternion.random(k1, (NSAMP,))
        qdet = Quaternion.random(k2, (NDET,))

        op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        tod = op(landscape.full(3.14))
        assert_array_almost_equal(tod.i, 3.14, decimal=10)


def test_expanded_interpolate_preserves_rotation_fusion() -> None:
    """The expanded interpolated operator keeps QURotation exposed so it fuses via algebra.

    An outer QURotation composed with `QURot(pa) @ XSampling @ Ravel` must reduce to a single
    QURotation (angles added), leaving one rotation in the chain rather than two.
    """
    landscape = _make_landscape('car', 'IQU')
    key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(3), 4)
    qbore = Quaternion.random(key1, (NSAMP,))
    qdet = Quaternion.random(key2, (NDET,))

    op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
    expanded = op.as_expanded_operator()
    gamma = jax.random.normal(key3, (NDET, NSAMP))
    outer = QURotationOperator(angles=gamma, in_structure=expanded.out_structure)

    reduced = (outer @ expanded).reduce()

    assert isinstance(reduced, CompositionOperator)
    leaves = jax.tree.leaves(
        reduced.operands, is_leaf=lambda x: isinstance(x, AbstractLinearOperator)
    )
    n_rotations = sum(isinstance(o, QURotationOperator) for o in leaves)
    assert n_rotations == 1

    tod = jax.tree.map(lambda s: jax.random.normal(key4, s.shape, s.dtype), expanded.out_structure)
    assert tree_equal((outer @ expanded)(op.T(tod)), reduced(op.T(tod)), rtol=1e-10)


def _nearest_quats(seed: int) -> tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(jax.random.key(seed))
    return Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))


def test_the_expanded_nearest_operator_keeps_the_index_fast_path() -> None:
    """Nearest transport is a per-sample rotation, so the gather stays an `IndexOperator`."""
    landscape = HealpixLandscape(NSIDE, 'IQU')
    qbore, qdet = _nearest_quats(20)
    op = PointingOperator.create(landscape, qbore, qdet, interpolate=False)

    reduced = op.as_expanded_operator().reduce()

    leaves = jax.tree.leaves(
        reduced.operands, is_leaf=lambda o: isinstance(o, AbstractLinearOperator)
    )
    assert any(isinstance(o, IndexOperator) for o in leaves)
    assert not any(isinstance(o, XSamplingOperator) for o in leaves)
    # the transport rotation fuses with the polarisation one rather than staying beside it
    assert sum(isinstance(o, QURotationOperator) for o in leaves) == 1


def test_the_expanded_nearest_operator_drops_samples_outside_the_map() -> None:
    """The index alone cannot express a sunk sample, so the stencil weight must ride along."""
    parent = HealpixLandscape(NSIDE, 'IQU')
    qbore, qdet = _nearest_quats(21)
    p_full = PointingOperator.create(parent, qbore, qdet, interpolate=False)
    covered = jnp.flatnonzero(p_full.T(ftree.ones_like(p_full.out_structure)).i)
    local = LocalStokesLandscape(parent, covered[::2])
    op = PointingOperator.create(local, qbore, qdet, interpolate=False)

    sky = local.normal(jax.random.key(22))
    tod = op.as_expanded_operator()(sky)
    assert tree_equal(tod, op(sky), rtol=1e-10, atol=1e-12)
    # half the pixels are unmapped, so some samples do sink: the test would pass vacuously
    assert jnp.any(op.landscape.quat2index(op.qbore * op.qdet[:, None]) == local.sink)


class TestLocalLandscape:
    """A PointingOperator on a LocalStokesLandscape matches the full-sky one."""

    @pytest.fixture(scope='class')
    @classmethod
    def keys(cls) -> jax.Array:
        return jax.random.split(jax.random.key(11), 4)

    @pytest.fixture(scope='class', params=[False, True], ids=['nearest', 'bilinear'])
    @classmethod
    def p_full(cls, request: pytest.FixtureRequest, keys: jax.Array) -> PointingOperator:
        parent = HealpixLandscape(NSIDE, 'IQU')
        qbore = Quaternion.random(keys[0], (NSAMP,))
        qdet = Quaternion.random(keys[1], (NDET,))
        return PointingOperator.create(parent, qbore, qdet, interpolate=request.param)

    @pytest.fixture(scope='class')
    @classmethod
    def sky(cls, p_full: PointingOperator, keys: jax.Array):
        return p_full.landscape.normal(keys[2])

    @pytest.fixture(scope='class')
    @classmethod
    def tod(cls, p_full: PointingOperator, keys: jax.Array):
        return ftree.normal_like(p_full.out_structure, keys[3])

    @pytest.fixture(scope='class')
    @classmethod
    def covered(cls, p_full: PointingOperator) -> jax.Array:
        # global pixels hit by the pointing: bin a ones-TOD (I accumulates 1 per hit for
        # nearest, the interpolation weights for bilinear)
        hits = p_full.T(ftree.ones_like(p_full.out_structure))
        return jnp.flatnonzero(hits.i)

    @staticmethod
    def _local_operator(
        p_full: PointingOperator, indices: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        local = LocalStokesLandscape(p_full.landscape, indices)
        p_local = PointingOperator.create(
            local, p_full.qbore, p_full.qdet, interpolate=p_full.interpolate
        )
        return local, p_local

    @pytest.fixture(scope='class')
    @classmethod
    def full_coverage(
        cls, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        return cls._local_operator(p_full, covered)

    @pytest.fixture(scope='class')
    @classmethod
    def half_coverage(
        cls, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        # samples on the dropped pixels land in the sink
        return cls._local_operator(p_full, covered[::2])

    def test_mv_matches_full_sky(self, p_full, sky, full_coverage) -> None:
        local, p_local = full_coverage
        assert tree_equal(p_local(local.restrict(sky)), p_full(sky), rtol=1e-13)

    def test_transpose_matches_full_sky(self, p_full, tod, full_coverage) -> None:
        local, p_local = full_coverage
        # uncovered pixels are zero in both the full binning and the promoted local one
        assert tree_equal(local.promote(p_local.T(tod)), p_full.T(tod), rtol=1e-12)

    def test_partial_coverage_sinks_missing_pixels(self, p_full, tod, half_coverage) -> None:
        local, p_local = half_coverage
        subset = local.global_indices
        binned_local = local.promote(p_local.T(tod)).data
        if not p_full.interpolate:
            # nearest: contributions to kept pixels are identical, sink ones are discarded
            # (bilinear renormalizes the weights over covered neighbors, so values differ)
            binned_full = p_full.T(tod).data
            assert_array_almost_equal(binned_local[:, subset], binned_full[:, subset], decimal=13)
        # nothing lands outside the subset
        assert_array_equal(binned_local.at[:, subset].set(0.0), 0.0)


def _untransported_sample(landscape: StokesLandscape, sky, qdet_full: jax.Array, interpolate: bool):
    """What the sampler would compute if it ignored the pixel frames."""
    sky_flat = sky.ravel()  # a CAR map is 2-D, and the indices address the raveled pixel axis
    if not interpolate:
        return type(sky).from_array(sky_flat.data[..., landscape.quat2index(qdet_full)])
    indices, weights = landscape.quat2interp(qdet_full)
    valid = indices >= 0
    indices = jnp.where(valid, indices, 0)
    weights = jnp.where(valid, weights, 0.0)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    unit_weights = weights / jnp.where(weight_sum > 0, weight_sum, 1.0)
    return type(sky).from_array(jnp.sum(sky_flat.data[..., indices] * unit_weights, axis=-1))


@pytest.mark.parametrize('interpolate', [False, True], ids=['nearest', 'bilinear'])
class TestTransport:
    """Sampling a polarized map always transports Q and U from the pixels it reads."""

    @staticmethod
    def _quats(seed: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        return Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))

    @pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
    def test_adjoint(self, stokes, landscape_type, interpolate) -> None:
        landscape = _make_landscape(landscape_type, stokes)
        qbore, qdet = self._quats(2)
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)
        sky = landscape.normal(jax.random.key(3))
        tod = ftree.normal_like(op.out_structure, jax.random.key(4))
        assert_array_almost_equal(ftree.dot(op(sky), tod), ftree.dot(sky, op.T(tod)), decimal=10)

    @pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
    def test_polarization_is_transported(self, landscape_type, interpolate) -> None:
        """The sampled Q differs from the value a sampler ignoring the pixel frames would give."""
        landscape = _make_landscape(landscape_type, 'IQU')
        qbore, qdet = self._quats(7)
        sky = landscape.normal(jax.random.key(8))
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)

        qdet_full = op.qbore * op.qdet[:, None]
        cos_pa, sin_pa = to_polarization_angle_cos_sin(qdet_full)
        untransported = _untransported_sample(landscape, sky, qdet_full, interpolate)
        reference = rotate_qu_cs(untransported, cos_pa, sin_pa)

        tod = op(sky)
        assert_array_almost_equal(tod.i, reference.i, decimal=13)
        assert jnp.abs(tod.q - reference.q).max() > 1e-6

    @pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
    def test_intensity_only_is_untouched(self, landscape_type, interpolate) -> None:
        """An intensity map has nothing to rotate, so it takes the plain sampling path."""
        landscape = _make_landscape(landscape_type, 'I')
        qbore, qdet = self._quats(5)
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)
        sky = landscape.normal(jax.random.key(6))
        qdet_full = op.qbore * op.qdet[:, None]
        expected = _untransported_sample(landscape, sky, qdet_full, interpolate)
        assert_array_almost_equal(op(sky).data, expected.data, decimal=13)

    def test_expanded_operator_transports_too(self, interpolate) -> None:
        """`as_expanded_operator` must not silently drop back to an untransported sampling."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(9)
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)
        sky = landscape.normal(jax.random.key(10))
        assert tree_equal(op.as_expanded_operator()(sky), op(sky), rtol=1e-10, atol=0)

    @pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
    def test_expanded_operator_adjoint(self, stokes, landscape_type, interpolate) -> None:
        """The expanded sampler has no hand-written transpose; JAX derives it from the gather."""
        landscape = _make_landscape(landscape_type, stokes)
        qbore, qdet = self._quats(15)
        op = PointingOperator.create(
            landscape, qbore, qdet, interpolate=interpolate
        ).as_expanded_operator()
        sky = landscape.normal(jax.random.key(16))
        tod = ftree.normal_like(op.out_structure, jax.random.key(17))
        assert_array_almost_equal(ftree.dot(op(sky), tod), ftree.dot(sky, op.T(tod)), decimal=10)

    @pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
    def test_expanded_transpose_matches_the_hand_written_one(
        self, landscape_type, interpolate
    ) -> None:
        """The derived transpose of the expanded sampler is the scatter the operator writes out."""
        landscape = _make_landscape(landscape_type, 'IQU')
        qbore, qdet = self._quats(18)
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)
        tod = ftree.normal_like(op.out_structure, jax.random.key(19))
        assert tree_equal(op.as_expanded_operator().T(tod), op.T(tod), rtol=1e-12, atol=1e-12)

    def test_local_landscape_adjoint(self, interpolate) -> None:
        """Samples outside the subset sink, and the centers still come from the parent."""
        parent = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(12)
        p_full = PointingOperator.create(parent, qbore, qdet, interpolate=interpolate)
        covered = jnp.flatnonzero(p_full.T(ftree.ones_like(p_full.out_structure)).i)
        local = LocalStokesLandscape(parent, covered[::2])
        op = PointingOperator.create(local, qbore, qdet, interpolate=interpolate)

        sky = local.normal(jax.random.key(13))
        tod = ftree.normal_like(op.out_structure, jax.random.key(14))
        assert_array_almost_equal(ftree.dot(op(sky), tod), ftree.dot(sky, op.T(tod)), decimal=10)


class TestTransportHooks:
    """One hook moves the pointing for a stencil sampler; the scalar index hook stands apart."""

    @staticmethod
    def _quats(seed: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        return Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))

    def test_a_subclass_moving_the_nearest_pointing_must_supply_its_own_stencil(self) -> None:
        """`_quat2index` stands beside `_quat2pointing`, so overriding it alone must raise."""

        class CustomPointingOperator(PointingOperator):
            # A real subclass moves the pointing here; delegating is enough to trip the guard.
            def _quat2index(self, qdet_full):
                return self.landscape.quat2index(qdet_full)

        qbore, qdet = self._quats(20)
        op = CustomPointingOperator.create(HealpixLandscape(NSIDE, 'IQU'), qbore, qdet)
        with pytest.raises(NotImplementedError, match='overrides _quat2index'):
            op(op.landscape.normal(jax.random.key(21)))

        # An intensity-only map never takes the transported path, so the override still works there.
        op_i = CustomPointingOperator.create(HealpixLandscape(NSIDE, 'I'), qbore, qdet)
        assert jnp.all(jnp.isfinite(op_i(op_i.landscape.normal(jax.random.key(22))).i))

    def test_one_hook_moves_the_bilinear_pointing(self) -> None:
        """Bilinear reads its pixels and its transport positions from `_quat2pointing` alone.

        There is no second hook for it to disagree with, so no guard is needed: a subclass that
        moves the pointing there moves both, and the polarized sample follows.
        """

        class ShiftedPointingOperator(PointingOperator):
            def _quat2pointing(self, qdet_full):
                theta, phi = self.landscape.quat2world(qdet_full)
                phi = phi + 0.05
                stencil = self.landscape.world2stencil(theta, phi, self._interpolation)
                return SampledPointing(stencil, theta, phi)

        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(20)
        sky = landscape.normal(jax.random.key(21))
        base = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        shifted = ShiftedPointingOperator.create(landscape, qbore, qdet, interpolate=True)

        moved = shifted(sky)
        assert jnp.all(jnp.isfinite(moved.q))
        assert float(jnp.max(jnp.abs(moved.q - base(sky).q))) > 1e-6


@register_static
class _ShiftedWorldIndexLandscape(HealpixLandscape):
    """A landscape whose two index paths disagree, as HEALPix's own do in single precision.

    `quat2index` reads the pointing axis with `vec2pix`, `world2index` reads the angles it was
    turned into with `ang2pix`, and in float32 the two land in different pixels for a handful of
    samples in a hundred thousand. Shifting one path by a whole pixel makes that rare divergence
    something a test can pin down.
    """

    def world2index(self, theta, phi):
        return (super().world2index(theta, phi) + 1) % len(self)


class TestNearestIndexAgreement:
    """The polarized nearest path reads the pixels `quat2index` counts, not the angles'."""

    @staticmethod
    def _setup(seed: int) -> tuple[PointingOperator, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        qbore, qdet = Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))
        op = PointingOperator.create(_ShiftedWorldIndexLandscape(NSIDE, 'IQU'), qbore, qdet)
        return op, op.qbore * op.qdet[:, None]

    def test_the_stencil_indexes_the_quat2index_pixel(self) -> None:
        op, qdet_full = self._setup(40)
        stencil = op._quat2pointing(qdet_full).stencil
        assert_array_equal(stencil.indices[..., 0], op.landscape.quat2index(qdet_full))

    def test_the_hit_map_of_the_polarized_operator_is_the_intensity_one(self) -> None:
        """The map the transpose fills and the map `as_stokes_i` counts must cover the same pixels.

        A sample binned into a pixel the hit map never saw is dropped by the pixel selection of a
        map-maker, so the two must agree pixel for pixel.
        """
        op, _ = self._setup(41)
        hits = op.T(ftree.ones_like(op.out_structure)).i
        op_i = op.as_stokes_i()
        expected = op_i.T(ftree.ones_like(op_i.out_structure)).i
        assert_array_equal(hits, expected)


@pytest.mark.parametrize('stokes', ['I', 'IQU'])
class TestPartialSkyNearest:
    """A nearest sample falling outside the map contributes nothing, whatever the map holds."""

    @staticmethod
    def _quats(seed: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        return Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))

    def _op(self, seed: int, stokes: ValidStokesLiteral) -> PointingOperator:
        # A 20x20 deg patch, so random pointing lands mostly outside it.
        patch = CARLandscape((20, 20), _CAR_PROJECTION, stokes)
        qbore, qdet = self._quats(seed)
        return PointingOperator.create(patch, qbore, qdet, interpolate=False)

    @staticmethod
    def _outside(op: PointingOperator, qdet_full: Quaternion) -> jax.Array:
        return op.landscape.quat2index(qdet_full) < 0

    def test_the_stencil_drops_samples_outside_the_map(self, stokes) -> None:
        op = self._op(50, stokes)
        stencil = op.landscape.index2stencil(op.landscape.quat2index(op.qbore * op.qdet[:, None]))
        weights = stencil.weights[..., 0]
        assert jnp.any(weights == 0)  # the case the mask exists for
        assert jnp.all((weights == 0) | (weights == 1))  # nearest weighs one or nothing

    def test_a_sample_outside_the_map_reads_zero(self, stokes) -> None:
        """Without the mask it would read the last pixel, which the raw index -1 wraps onto."""
        op = self._op(51, stokes)
        outside = self._outside(op, op.qbore * op.qdet[:, None])
        assert jnp.any(outside)
        sky = op.landscape.ones()
        assert_array_equal(op(sky).i[outside], 0.0)

    def test_binning_a_sample_outside_the_map_adds_nothing(self, stokes) -> None:
        """Without the mask its TOD would land on the last pixel, inflating a pixel it never hit."""
        op = self._op(52, stokes)
        outside = self._outside(op, op.qbore * op.qdet[:, None])
        assert jnp.any(outside)
        hits = op.T(ftree.ones_like(op.out_structure)).i
        assert float(hits.sum()) == pytest.approx(float((~outside).sum()))

    def test_the_expanded_operator_agrees(self, stokes) -> None:
        """The mask on the expanded sampler must drop exactly what `mv` drops."""
        op = self._op(53, stokes)
        assert jnp.any(self._outside(op, op.qbore * op.qdet[:, None]))
        sky = op.landscape.normal(jax.random.key(54))
        assert tree_equal(op.as_expanded_operator()(sky), op(sky), rtol=1e-10, atol=0)


class TestNearestTransport:
    """What the transport changes, and does not change, on the nearest-neighbour path."""

    @staticmethod
    def _quats(seed: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        return Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))

    def test_samples_on_pixel_centers_are_unchanged(self) -> None:
        """A sample sitting on its pixel center has nothing to transport."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 7)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        qbore = from_iso_angles(theta, phi, jnp.zeros_like(theta))
        qdet = Quaternion.ones((1,))  # identity: the detector points at the boresight

        op = PointingOperator.create(landscape, qbore, qdet)
        sky = landscape.normal(jax.random.key(30))
        qdet_full = op.qbore * op.qdet[:, None]
        cos_pa, sin_pa = to_polarization_angle_cos_sin(qdet_full)
        expected = rotate_qu_cs(
            _untransported_sample(landscape, sky, qdet_full, False), cos_pa, sin_pa
        )
        assert_array_almost_equal(op(sky).data, expected.data, decimal=12)

    def test_hit_counts_are_unchanged(self) -> None:
        """The transport is a rotation per sample: which pixel a sample lands in does not move."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(31)
        op = PointingOperator.create(landscape, qbore, qdet)
        hits = op.T(ftree.ones_like(op.out_structure)).i

        qdet_full = op.qbore * op.qdet[:, None]
        expected = jnp.zeros(len(landscape)).at[landscape.quat2index(qdet_full).ravel()].add(1.0)
        assert_array_equal(hits, expected)

    def test_the_system_matrix_stays_block_diagonal(self) -> None:
        """One sample still touches one pixel, so P^T P couples no two pixels."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(32)
        op = PointingOperator.create(landscape, qbore, qdet)

        # column of P^T P for one Stokes component of one hit pixel
        qdet_full = op.qbore * op.qdet[:, None]
        pixel = int(landscape.quat2index(qdet_full).ravel()[0])
        zeros = landscape.zeros()
        probe = type(zeros).from_array(zeros.data.at[:, pixel].set(1.0))
        response = op.T(op(probe)).data

        touched = jnp.flatnonzero(jnp.abs(response).sum(axis=0) > 0)
        assert_array_equal(touched, jnp.array([pixel]))


@pytest.mark.parametrize('interpolate', [False, True], ids=['nearest', 'bilinear'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
class TestOffsets:
    """A detector reads the sky at several offsets around its direction, with weights."""

    @staticmethod
    def _quats(seed: int) -> tuple[Quaternion, Quaternion]:
        # each detector has a gamma of its own: the boresight frame strips it from `qdet`, and the
        # offsets must not turn with it
        return Quaternion.random(jax.random.key(seed), (NSAMP,)), from_xieta_angles(
            jnp.array([0.0, 0.02, -0.03]),
            jnp.array([0.0, -0.01, 0.02]),
            jnp.array([0.3, 0.1, -1.2]),
        )

    @staticmethod
    def _offsets() -> Quaternion:
        return from_xieta_angles(jnp.array([0.05, -0.03]), jnp.array([0.02, 0.04]), jnp.zeros(2))

    @staticmethod
    def _per_stokes_weights(stokes: str) -> jax.Array:
        # distinct rows so that a weight applied in the wrong frame or order shows up
        rows = {'I': [0.5, 0.5], 'Q': [0.7, 0.3], 'U': [0.2, 0.8], 'V': [0.4, 0.6]}
        return jnp.array([rows[component] for component in stokes])

    def test_no_offsets_is_the_plain_operator(self, stokes, frame, interpolate) -> None:
        """`offsets=None` leaves every code path as it was."""
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(40)
        op = PointingOperator.create(landscape, qbore, qdet, frame=frame, interpolate=interpolate)
        assert op.offsets is None and op.offset_weights is None
        assert len(jax.tree.leaves(op)) == 2  # qbore, qdet: no offset leaf sneaks in

    def test_the_origin_offset_with_unit_weight_is_the_plain_operator(
        self, stokes, frame, interpolate
    ) -> None:
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(41)
        plain = PointingOperator.create(
            landscape, qbore, qdet, frame=frame, interpolate=interpolate
        )
        origin = from_xieta_angles(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1))
        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=origin,
            offset_weights=jnp.ones(1),
        )
        sky = landscape.normal(jax.random.key(42))
        tod = ftree.normal_like(op.out_structure, jax.random.key(43))
        assert tree_equal(op(sky), plain(sky), rtol=1e-12, atol=1e-12)
        assert tree_equal(op.T(tod), plain.T(tod), rtol=1e-12, atol=1e-12)

    def test_offsets_read_the_sky_where_the_physical_detector_points(
        self, frame, interpolate
    ) -> None:
        """Two offsets at equal weight average two detectors carrying those offsets.

        The reference folds each offset into the detector quaternion itself, in the detector frame,
        so it holds whatever frame the operator under test uses: the boresight frame strips the
        detector's z-rotation from `qdet`, and the offsets must stay attached to the physical
        detector regardless.
        """
        landscape = HealpixLandscape(NSIDE, 'I')
        qbore, qdet = self._quats(44)
        offsets = self._offsets()
        sky = landscape.normal(jax.random.key(45))

        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=offsets,
            offset_weights=jnp.array([0.5, 0.5]),
            batch_size=2,
        )
        parts = [
            PointingOperator.create(
                landscape, qbore, qdet * offsets[k], frame='detector', interpolate=interpolate
            )(sky)
            for k in range(2)
        ]
        expected = ftree.mul(0.5, ftree.add(parts[0], parts[1]))
        assert tree_equal(op(sky), expected, rtol=1e-12, atol=1e-12)

    def test_per_detector_offsets_match_shared_ones(self, frame, interpolate) -> None:
        landscape = HealpixLandscape(NSIDE, 'I')
        qbore, qdet = self._quats(46)
        offsets = self._offsets()
        weights = jnp.array([0.5, 0.5])
        sky = landscape.normal(jax.random.key(47))
        shared = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=offsets,
            offset_weights=weights,
        )
        identity = Quaternion.ones((NDET, 1))
        per_detector = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=identity * offsets[None, :],
            offset_weights=weights,
        )
        assert shared.offsets.shape == (NDET, 2)
        assert tree_equal(per_detector(sky), shared(sky), rtol=1e-12, atol=1e-12)

    def test_per_stokes_weights_act_on_the_output_components(
        self, stokes, frame, interpolate
    ) -> None:
        """Each row weighs its component of the output, in the frame set by `frame`.

        The reference applies the weights to the TOD of each offset read on its own, i.e. after
        the rotation by the polarization angle. Weighting before that rotation instead weighs
        the Q and U of the sky's meridian basis, and would make the response depend on the
        polarization angle as soon as the Q and U rows differ.
        """
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(55)
        offsets = self._offsets()
        weights = self._per_stokes_weights(stokes)
        sky = landscape.normal(jax.random.key(56))

        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=offsets,
            offset_weights=weights,
            batch_size=2,
        )
        parts = [
            PointingOperator.create(
                landscape,
                qbore,
                qdet,
                frame=frame,
                interpolate=interpolate,
                offsets=offsets[k : k + 1],
                offset_weights=jnp.ones(1),
            )(sky).data
            for k in range(2)
        ]
        expected = weights[:, 0, None, None] * parts[0] + weights[:, 1, None, None] * parts[1]
        assert_array_almost_equal(op(sky).data, expected, decimal=12)

    def test_adjoint_with_per_stokes_weights(self, stokes, frame, interpolate) -> None:
        """P^T is the transpose of P as matrices, with a weight of its own per Stokes component."""
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(48)
        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=self._offsets(),
            offset_weights=self._per_stokes_weights(stokes),
            batch_size=2,
        )
        assert_array_almost_equal(op.as_matrix().T, op.T.as_matrix(), decimal=12)

    def test_the_expanded_operator_carries_the_offsets(self, stokes, frame, interpolate) -> None:
        """`as_expanded_operator` must not drop the offsets, or the mapmaker loses the beam."""
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(49)
        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=self._offsets(),
            offset_weights=self._per_stokes_weights(stokes),
            batch_size=2,
        )
        expanded = op.as_expanded_operator()
        sky = landscape.normal(jax.random.key(50))
        tod = ftree.normal_like(op.out_structure, jax.random.key(51))
        assert tree_equal(expanded(sky), op(sky), rtol=1e-11, atol=1e-12)
        assert tree_equal(expanded.T(tod), op.T(tod), rtol=1e-11, atol=1e-12)
        assert_array_almost_equal(expanded.as_matrix().T, expanded.T.as_matrix(), decimal=12)

    def test_as_stokes_i_keeps_the_offsets_with_the_intensity_weights(
        self, frame, interpolate
    ) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(52)
        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=self._offsets(),
            offset_weights=self._per_stokes_weights('IQU'),
        )
        op_i = op.as_stokes_i(interpolate=not interpolate)
        assert op_i.offsets is op.offsets
        assert_array_equal(op_i.offset_weights, jnp.array([0.5, 0.5]))
        assert op_i.landscape.stokes == 'I' and op_i.interpolate is (not interpolate)

    def test_as_stokes_i_averages_the_weights_of_a_map_without_intensity(
        self, frame, interpolate
    ) -> None:
        landscape = HealpixLandscape(NSIDE, 'QU')
        qbore, qdet = self._quats(53)
        op = PointingOperator.create(
            landscape,
            qbore,
            qdet,
            frame=frame,
            interpolate=interpolate,
            offsets=self._offsets(),
            offset_weights=self._per_stokes_weights('QU'),
        )
        assert_array_almost_equal(op.as_stokes_i().offset_weights, jnp.array([0.45, 0.55]))

    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'offsets': 'given'}, 'given together'),
            ({'offset_weights': jnp.ones(2)}, 'given together'),
            ({'offsets': 'given', 'offset_weights': jnp.ones(3)}, 'offset_weights has shape'),
            ({'offsets': 'given', 'offset_weights': jnp.ones((2, 2))}, 'offset_weights has shape'),
            ({'offsets': 'wrong_ndet', 'offset_weights': jnp.ones(2)}, 'offsets has shape'),
        ],
    )
    def test_create_rejects_inconsistent_offsets(self, frame, interpolate, kwargs, match) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(54)
        offsets = self._offsets()
        if kwargs.get('offsets') == 'given':
            kwargs = {**kwargs, 'offsets': offsets}
        elif kwargs.get('offsets') == 'wrong_ndet':
            kwargs = {**kwargs, 'offsets': Quaternion.ones((NDET + 1, 1)) * offsets[None, :]}
        with pytest.raises(ValueError, match=match):
            PointingOperator.create(
                landscape, qbore, qdet, frame=frame, interpolate=interpolate, **kwargs
            )
