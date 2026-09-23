import jax
import jax.numpy as jnp
import jax_healpy as jhp
import pytest
from equinox import tree_equal
from fastquat import Quaternion
from jax.tree_util import register_static
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

import furax.tree as ftree
from furax.math.coords import (
    from_iso_angles,
    from_xieta_angles,
    to_polarization_angle_cos_sin,
)
from furax.obs.landscapes import (
    CARLandscape,
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
    WCSProjection,
)
from furax.obs.operators import QURotationOperator
from furax.obs.operators._qu_rotations import rotate_qu_cs
from furax.obs.pointing import PointingOperator, PointingTransposeOperator
from furax.obs.sampling import (
    AbstractSampler,
    AngleSampler,
    PointingRows,
    PrecomputedSampler,
    QuaternionSampler,
    RotatedSampler,
    SamplingKernel,
)
from furax.obs.spin2 import transport_rotation, transported_gather
from furax.obs.stencil import Interpolation
from furax.obs.stokes import Stokes, StokesIQU, ValidStokesLiteral

NSIDE = 4
NDET, NSAMP = 3, 10

# Full-sphere CAR landscape (180×360 at 1°/pixel, crval at RA=180° to avoid wrap issues)
_CAR_PROJECTION = WCSProjection(crpix=(180.5, 90.5), crval=(180.0, 0.0), cdelt=(-1.0, 1.0))


def _interpolates(op: PointingOperator) -> bool:
    return op.sampler.kernel.interpolation is Interpolation.BILINEAR


def _make_landscape(landscape_type: str, stokes: ValidStokesLiteral) -> StokesLandscape:
    if landscape_type == 'healpix':
        return HealpixLandscape(NSIDE, stokes)
    return CARLandscape((180, 360), _CAR_PROJECTION, stokes)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
@pytest.mark.parametrize('interpolate', [False, True], ids=['nearest', 'bilinear'])
@pytest.mark.parametrize('store', ['rows', 'angles'])
def test_precomputed_matches_on_the_fly(stokes, frame, landscape_type, interpolate, store) -> None:
    landscape = _make_landscape(landscape_type, stokes)
    key1, key2, key3, key4 = jax.random.split(jax.random.key(42), 4)
    qbore = Quaternion.random(key1, (NSAMP,))
    qdet = Quaternion.random(key2, (NDET,))
    op = PointingOperator.create(
        landscape, qbore, qdet, frame=frame, batch_size=2, interpolate=interpolate
    )
    precomputed = op.precomputed(store)
    sky = landscape.normal(key3)
    tod = ftree.normal_like(op.out_structure, key4)

    assert tree_equal(precomputed(sky), op(sky), rtol=1e-10, atol=1e-13)
    assert tree_equal(precomputed.T(tod), op.T(tod), rtol=1e-10, atol=1e-13)


class TestPrecomputed:
    @staticmethod
    def _op(interpolate: bool, stokes: ValidStokesLiteral = 'IQU') -> PointingOperator:
        k1, k2 = jax.random.split(jax.random.key(20))
        qbore, qdet = Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))
        landscape = HealpixLandscape(NSIDE, stokes)
        return PointingOperator.create(landscape, qbore, qdet, interpolate=interpolate)

    @pytest.mark.parametrize(
        'interpolate, expected',
        [(False, PrecomputedSampler), (True, AngleSampler)],
        ids=['nearest', 'bilinear'],
    )
    def test_the_default_store(self, interpolate, expected) -> None:
        """Nearest stores its few indices; bilinear stores angles, not four neighbours."""
        assert isinstance(self._op(interpolate).precomputed().sampler, expected)

    def test_a_map_without_polarization_stores_the_indices_alone(self) -> None:
        sampler = self._op(False, 'I').precomputed().sampler
        assert isinstance(sampler, PrecomputedSampler)
        assert sampler.stencil is None and sampler.neighbour_rotation is None
        assert sampler.nearest is not None

    def test_only_quaternions_are_stored_as_angles(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'I')
        sampler = _PointsSampler(
            kernel=SamplingKernel(), theta=jnp.array([0.5, 1.0]), phi=jnp.array([0.2, 3.0])
        )
        with pytest.raises(TypeError, match='only a QuaternionSampler'):
            PointingOperator.from_sampler(landscape, sampler).precomputed('angles')

    def test_as_stokes_i_reads_the_source_again(self) -> None:
        """The cache holds the rotations of a polarized map, so the intensity one recomputes."""
        op = self._op(False)
        op_i = op.precomputed('rows').as_stokes_i()
        assert isinstance(op_i.sampler, QuaternionSampler)
        tod = ftree.ones_like(op_i.out_structure)
        assert tree_equal(op_i.T(tod), op.as_stokes_i().T(tod))

    def test_nearest_drops_samples_outside_the_map(self) -> None:
        """A sample outside a partial map reads nothing, cached or not."""
        parent = HealpixLandscape(NSIDE, 'IQU')
        p_full = self._op(False)
        covered = jnp.flatnonzero(p_full.T(ftree.ones_like(p_full.out_structure)).i)
        local = LocalStokesLandscape(parent, covered[::2])
        op = PointingOperator.from_sampler(local, p_full.sampler)

        sky = local.normal(jax.random.key(22))
        assert tree_equal(op.precomputed()(sky), op(sky), rtol=1e-10, atol=1e-12)
        # half the pixels are unmapped, so some samples do sink: the test would pass vacuously
        rows = jnp.arange(op.sampler.shape[0])
        assert jnp.any(op.landscape.quat2index(op.sampler.quaternions(rows)) == local.sink)


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
            local, p_full.sampler.qbore, p_full.sampler.qdet, interpolate=_interpolates(p_full)
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
        if not _interpolates(p_full):
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

        qdet_full = op.sampler.quaternions()
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
        qdet_full = op.sampler.quaternions()
        expected = _untransported_sample(landscape, sky, qdet_full, interpolate)
        assert_array_almost_equal(op(sky).data, expected.data, decimal=13)

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


class _PointsSampler(AbstractSampler):
    """Reads a map at given directions, in their meridian basis: samples that are not a TOD."""

    theta: jax.Array
    phi: jax.Array

    @property
    def shape(self):
        return self.theta.shape

    def pointing_rows(self, landscape, index):
        theta, phi = self.theta[index], self.phi[index]
        stencil = landscape.world2stencil(theta, phi, self.kernel.interpolation)
        rotation = transport_rotation(stencil, theta, phi) if landscape.has_spin2 else None
        return PointingRows(stencil, rotation)


class TestCustomSampler:
    """Any sampler gets the batch loop, the transport and an exact transpose."""

    @staticmethod
    def _points(shape: tuple[int, ...], seed: int) -> _PointsSampler:
        k1, k2 = jax.random.split(jax.random.key(seed))
        theta = jax.random.uniform(k1, shape, minval=0.1, maxval=jnp.pi - 0.1)
        phi = jax.random.uniform(k2, shape, maxval=2 * jnp.pi)
        return _PointsSampler(kernel=SamplingKernel(Interpolation.BILINEAR), theta=theta, phi=phi)

    @pytest.mark.parametrize('shape', [(7,), (5, 2, 3)], ids=['points', '3d'])
    @pytest.mark.parametrize('batch_size', [3, 0], ids=['partial-batches', 'one-batch'])
    def test_samples_of_any_shape(self, shape, batch_size) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        sampler = self._points(shape, 30)
        op = PointingOperator.from_sampler(landscape, sampler, batch_size=batch_size)
        sky = landscape.normal(jax.random.key(31))

        assert op.out_structure == landscape.structure_for(shape)
        stencil = landscape.world2stencil(sampler.theta, sampler.phi, Interpolation.BILINEAR)
        expected = transported_gather(sky.ravel(), stencil, sampler.theta, sampler.phi)
        assert tree_equal(op(sky), expected, rtol=1e-12, atol=1e-12)

        tod = ftree.normal_like(op.out_structure, jax.random.key(32))
        assert_allclose(ftree.dot(op(sky), tod), ftree.dot(sky, op.T(tod)), rtol=1e-12)

    def test_the_operator_is_a_pytree(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        op = PointingOperator.from_sampler(landscape, self._points((7,), 33), batch_size=3)
        sky = landscape.normal(jax.random.key(34))
        assert tree_equal(jax.jit(lambda op, sky: op(sky))(op, sky), op(sky))


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
        return op, op.sampler.quaternions()

    def test_the_stencil_indexes_the_quat2index_pixel(self) -> None:
        op, qdet_full = self._setup(40)
        stencil = op.sampler.pointing_rows(op.landscape, jnp.arange(op.sampler.shape[0])).stencil
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
        stencil = op.landscape.index2stencil(op.landscape.quat2index(op.sampler.quaternions()))
        weights = stencil.weights[..., 0]
        assert jnp.any(weights == 0)  # the case the mask exists for
        assert jnp.all((weights == 0) | (weights == 1))  # nearest weighs one or nothing

    def test_a_sample_outside_the_map_reads_zero(self, stokes) -> None:
        """Without the mask it would read the last pixel, which the raw index -1 wraps onto."""
        op = self._op(51, stokes)
        outside = self._outside(op, op.sampler.quaternions())
        assert jnp.any(outside)
        sky = op.landscape.ones()
        assert_array_equal(op(sky).i[outside], 0.0)

    def test_binning_a_sample_outside_the_map_adds_nothing(self, stokes) -> None:
        """Without the mask its TOD would land on the last pixel, inflating a pixel it never hit."""
        op = self._op(52, stokes)
        outside = self._outside(op, op.sampler.quaternions())
        assert jnp.any(outside)
        hits = op.T(ftree.ones_like(op.out_structure)).i
        assert float(hits.sum()) == pytest.approx(float((~outside).sum()))

    def test_the_precomputed_operator_agrees(self, stokes) -> None:
        """The precomputed operator must drop exactly what `mv` drops."""
        op = self._op(53, stokes)
        assert jnp.any(self._outside(op, op.sampler.quaternions()))
        sky = op.landscape.normal(jax.random.key(54))
        assert tree_equal(op.precomputed()(sky), op(sky), rtol=1e-10, atol=1e-13)


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
        qdet_full = op.sampler.quaternions()
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

        qdet_full = op.sampler.quaternions()
        expected = jnp.zeros(len(landscape)).at[landscape.quat2index(qdet_full).ravel()].add(1.0)
        assert_array_equal(hits, expected)

    def test_the_system_matrix_stays_block_diagonal(self) -> None:
        """One sample still touches one pixel, so P^T P couples no two pixels."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(32)
        op = PointingOperator.create(landscape, qbore, qdet)

        # column of P^T P for one Stokes component of one hit pixel
        qdet_full = op.sampler.quaternions()
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
    def _per_stokes_weights(stokes: ValidStokesLiteral) -> Stokes:
        # distinct weights per component so that a weight applied in the wrong frame or order
        # shows up
        rows = {'I': [0.5, 0.5], 'Q': [0.7, 0.3], 'U': [0.2, 0.8], 'V': [0.4, 0.6]}
        return Stokes.class_for(stokes).from_array(
            jnp.array([rows[component] for component in stokes])
        )

    def test_no_offsets_is_the_plain_operator(self, stokes, frame, interpolate) -> None:
        """`offsets=None` leaves every code path as it was."""
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(40)
        op = PointingOperator.create(landscape, qbore, qdet, frame=frame, interpolate=interpolate)
        assert op.sampler.kernel.offsets is None and op.sampler.kernel.weights is None
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
        assert shared.sampler.kernel.offsets.shape == (NDET, 2)
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
        rows = weights.data
        expected = rows[:, 0, None, None] * parts[0] + rows[:, 1, None, None] * parts[1]
        assert_array_almost_equal(op(sky).data, expected, decimal=12)

    def test_equal_weights_per_component_are_shared_weights(
        self, stokes, frame, interpolate
    ) -> None:
        landscape = HealpixLandscape(NSIDE, stokes)
        qbore, qdet = self._quats(57)
        shared = jnp.array([0.3, 0.7])
        per_component = Stokes.class_for(stokes).from_array(jnp.tile(shared, (len(stokes), 1)))
        ops = [
            PointingOperator.create(
                landscape,
                qbore,
                qdet,
                frame=frame,
                interpolate=interpolate,
                offsets=self._offsets(),
                offset_weights=weights,
            )
            for weights in (shared, per_component)
        ]
        sky = landscape.normal(jax.random.key(58))
        assert tree_equal(ops[1](sky), ops[0](sky), rtol=1e-12, atol=1e-12)

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

    @pytest.mark.parametrize('store', ['rows', 'angles'])
    def test_the_precomputed_operator_carries_the_offsets(
        self, stokes, frame, interpolate, store
    ) -> None:
        """`precomputed` must not drop the offsets, or the mapmaker loses the beam."""
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
        expanded = op.precomputed(store)
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
        assert op_i.sampler.kernel.offsets is op.sampler.kernel.offsets
        assert_array_equal(op_i.sampler.kernel.weights, jnp.array([0.5, 0.5]))
        assert op_i.landscape.stokes == 'I' and _interpolates(op_i) is (not interpolate)

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
        assert_array_almost_equal(op.as_stokes_i().sampler.kernel.weights, jnp.array([0.45, 0.55]))

    def test_create_validates_the_kernel(self, frame, interpolate) -> None:
        """The offsets and weights are checked by `SamplingKernel.create`, see `test_sampling.py`."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore, qdet = self._quats(54)
        with pytest.raises(ValueError, match='offset weights have shape'):
            PointingOperator.create(
                landscape,
                qbore,
                qdet,
                frame=frame,
                interpolate=interpolate,
                offsets=self._offsets(),
                offset_weights=jnp.ones(3),
            )


class TestRotationAbsorption:
    """A QU rotation after the pointing folds into it, so the two cost a single pass."""

    @staticmethod
    def _op(store: str | None, per_stokes: bool) -> PointingOperator:
        k1, k2 = jax.random.split(jax.random.key(60))
        qbore, qdet = Quaternion.random(k1, (NSAMP,)), Quaternion.random(k2, (NDET,))
        kwargs = {}
        if per_stokes:
            offsets = from_xieta_angles(jnp.array([0.05, -0.03]), jnp.array([0.02, 0.04]), 0.0)
            weights = StokesIQU(jnp.array([0.5, 0.5]), jnp.array([0.7, 0.3]), jnp.array([0.2, 0.8]))
            kwargs = {'offsets': offsets, 'offset_weights': weights}
        landscape = HealpixLandscape(NSIDE, 'IQU')
        op = PointingOperator.create(landscape, qbore, qdet, interpolate=True, **kwargs)
        return op if store is None else op.precomputed(store)

    @pytest.mark.parametrize('per_stokes', [False, True], ids=['shared', 'per-stokes'])
    @pytest.mark.parametrize('store', [None, 'rows', 'angles'], ids=['fly', 'rows', 'angles'])
    @pytest.mark.parametrize('transposed', [False, True], ids=['R', 'R.T'])
    def test_reduces_to_one_pointing(self, store, per_stokes, transposed) -> None:
        op = self._op(store, per_stokes)
        rotation = QURotationOperator(
            angles=jax.random.normal(jax.random.key(61), (NSAMP,)), in_structure=op.out_structure
        )
        chain = (rotation.T if transposed else rotation) @ op
        reduced = chain.reduce()
        assert isinstance(reduced, PointingOperator)

        sky = op.landscape.normal(jax.random.key(62))
        tod = ftree.normal_like(op.out_structure, jax.random.key(63))
        assert tree_equal(reduced(sky), chain(sky), rtol=1e-12, atol=1e-12)
        transpose = (op.T @ (rotation if transposed else rotation.T)).reduce()
        assert isinstance(transpose, PointingTransposeOperator)
        assert tree_equal(transpose(tod), chain.T(tod), rtol=1e-12, atol=1e-12)

    def test_an_atomic_rotation_stays_apart(self) -> None:
        op = self._op(None, False)
        rotation = QURotationOperator(
            angles=jnp.ones(NSAMP), atomic=True, in_structure=op.out_structure
        )
        assert not isinstance((rotation @ op).reduce(), PointingOperator)

    def test_successive_rotations_merge(self) -> None:
        op = self._op(None, False)
        twice = op.rotated(jnp.full(NSAMP, 0.3)).rotated(jnp.full(NSAMP, 0.4))
        once = op.rotated(jnp.full(NSAMP, 0.7))
        assert isinstance(twice.sampler, RotatedSampler)
        assert isinstance(twice.sampler.source, QuaternionSampler)
        sky = op.landscape.normal(jax.random.key(64))
        assert tree_equal(twice(sky), once(sky), rtol=1e-12, atol=1e-12)
