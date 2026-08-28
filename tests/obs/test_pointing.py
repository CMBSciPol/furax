import jax
import jax.numpy as jnp
import jax_healpy as jhp
import pytest
from equinox import tree_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

import furax.tree as ftree
from furax.core import AbstractLinearOperator, CompositionOperator
from furax.math.quaternion import from_iso_angles, qmul, to_polarization_angle_cos_sin
from furax.obs.landscapes import (
    CARLandscape,
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
    WCSProjection,
)
from furax.obs.operators import QURotationOperator
from furax.obs.operators._qu_rotations import rotate_qu_cs
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import ValidStokesLiteral

NSIDE = 4
NDET, NSAMP = 3, 10

# Full-sphere CAR landscape (180×360 at 1°/pixel, crval at RA=180° to avoid wrap issues)
_CAR_PROJECTION = WCSProjection(crpix=(180.5, 90.5), crval=(180.0, 0.0), cdelt=(-1.0, 1.0))


def _make_landscape(landscape_type: str, stokes: ValidStokesLiteral) -> StokesLandscape:
    if landscape_type == 'healpix':
        return HealpixLandscape(NSIDE, stokes)
    return CARLandscape((180, 360), _CAR_PROJECTION, stokes)


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Generate random unit quaternions."""
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
class TestAsExpandedOperator:
    def test_mv(self, stokes, frame, landscape_type) -> None:
        """PointingOperator.mv is equivalent to as_expanded_operator().mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

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
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

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
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

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
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

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
        qbore = _random_unit_quats(k1, (NSAMP,))
        qdet = _random_unit_quats(k2, (NDET,))

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
        qbore = _random_unit_quats(k1, (NSAMP,))
        qdet = _random_unit_quats(k2, (NDET,))

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
    qbore = _random_unit_quats(key1, (NSAMP,))
    qdet = _random_unit_quats(key2, (NDET,))

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


class TestLocalLandscape:
    """A PointingOperator on a LocalStokesLandscape matches the full-sky one."""

    @pytest.fixture(scope='class')
    def keys(self) -> jax.Array:
        return jax.random.split(jax.random.key(11), 4)

    @pytest.fixture(scope='class', params=[False, True], ids=['nearest', 'bilinear'])
    def p_full(self, request: pytest.FixtureRequest, keys: jax.Array) -> PointingOperator:
        parent = HealpixLandscape(NSIDE, 'IQU')
        qbore = _random_unit_quats(keys[0], (NSAMP,))
        qdet = _random_unit_quats(keys[1], (NDET,))
        return PointingOperator.create(parent, qbore, qdet, interpolate=request.param)

    @pytest.fixture(scope='class')
    def sky(self, p_full: PointingOperator, keys: jax.Array):
        return p_full.landscape.normal(keys[2])

    @pytest.fixture(scope='class')
    def tod(self, p_full: PointingOperator, keys: jax.Array):
        return ftree.normal_like(p_full.out_structure, keys[3])

    @pytest.fixture(scope='class')
    def covered(self, p_full: PointingOperator) -> jax.Array:
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
    def full_coverage(
        self, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        return self._local_operator(p_full, covered)

    @pytest.fixture(scope='class')
    def half_coverage(
        self, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        # samples on the dropped pixels land in the sink
        return self._local_operator(p_full, covered[::2])

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
        return _random_unit_quats(k1, (NSAMP,)), _random_unit_quats(k2, (NDET,))

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

        qdet_full = qmul(op.qbore, op.qdet[:, None, :])
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
        assert not op._transports

        sky = landscape.normal(jax.random.key(6))
        qdet_full = qmul(op.qbore, op.qdet[:, None, :])
        expected = _untransported_sample(landscape, sky, qdet_full, interpolate)
        assert_array_almost_equal(op(sky).data, expected.data, decimal=13)

    def test_transports_a_polarized_map_either_way(self, interpolate) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(1)
        assert PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)._transports

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
        return _random_unit_quats(k1, (NSAMP,)), _random_unit_quats(k2, (NDET,))

    def test_a_subclass_moving_the_nearest_pointing_must_supply_its_own_stencil(self) -> None:
        """`_quat2index` stands beside `_quat2stencil`, so overriding it alone must raise."""

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
        """Bilinear reads its pixels and its transport positions from `_quat2stencil` alone.

        There is no second hook for it to disagree with, so no guard is needed: a subclass that
        moves the pointing there moves both, and the polarized sample follows.
        """

        class ShiftedPointingOperator(PointingOperator):
            def _quat2stencil(self, qdet_full):
                theta, phi = self.landscape.quat2world(qdet_full)
                phi = phi + 0.05
                return self.landscape.world2stencil(theta, phi, self._stencil_order), theta, phi

        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(20)
        sky = landscape.normal(jax.random.key(21))
        base = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        shifted = ShiftedPointingOperator.create(landscape, qbore, qdet, interpolate=True)

        moved = shifted(sky)
        assert jnp.all(jnp.isfinite(moved.q))
        assert float(jnp.max(jnp.abs(moved.q - base(sky).q))) > 1e-6


class TestNearestTransport:
    """What the transport changes, and does not change, on the nearest-neighbour path."""

    @staticmethod
    def _quats(seed: int) -> tuple[jax.Array, jax.Array]:
        k1, k2 = jax.random.split(jax.random.key(seed))
        return _random_unit_quats(k1, (NSAMP,)), _random_unit_quats(k2, (NDET,))

    def test_samples_on_pixel_centers_are_unchanged(self) -> None:
        """A sample sitting on its pixel center has nothing to transport."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 7)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        qbore = from_iso_angles(theta, phi, jnp.zeros_like(theta))
        qdet = jnp.array([[0.0, 0.0, 0.0, 1.0]])  # identity: the detector points at the boresight

        op = PointingOperator.create(landscape, qbore, qdet)
        sky = landscape.normal(jax.random.key(30))
        qdet_full = qmul(op.qbore, op.qdet[:, None, :])
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

        qdet_full = qmul(op.qbore, op.qdet[:, None, :])
        expected = jnp.zeros(len(landscape)).at[landscape.quat2index(qdet_full).ravel()].add(1.0)
        assert_array_equal(hits, expected)

    def test_the_system_matrix_stays_block_diagonal(self) -> None:
        """One sample still touches one pixel, so P^T P couples no two pixels."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(32)
        op = PointingOperator.create(landscape, qbore, qdet)

        # column of P^T P for one Stokes component of one hit pixel
        qdet_full = qmul(op.qbore, op.qdet[:, None, :])
        pixel = int(landscape.quat2index(qdet_full).ravel()[0])
        zeros = landscape.zeros()
        probe = type(zeros).from_array(zeros.data.at[:, pixel].set(1.0))
        response = op.T(op(probe)).data

        touched = jnp.flatnonzero(jnp.abs(response).sum(axis=0) > 0)
        assert_array_equal(touched, jnp.array([pixel]))
