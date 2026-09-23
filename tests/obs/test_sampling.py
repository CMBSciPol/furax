import jax
import jax.numpy as jnp
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose, assert_array_equal

from furax.math.coords import from_xieta_angles
from furax.obs.landscapes import HealpixLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.sampling import AngleSampler, PrecomputedSampler, QuaternionSampler, SamplingKernel
from furax.obs.spin2 import transported_gather
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import StokesI, StokesIQU, StokesQU

NSIDE = 4
NDET = 3


def _offsets() -> Quaternion:
    return from_xieta_angles(jnp.array([0.05, -0.03]), jnp.array([0.02, 0.04]), jnp.zeros(2))


class TestCreate:
    def test_without_offsets_holds_the_interpolation_alone(self) -> None:
        kernel = SamplingKernel.create(
            HealpixLandscape(NSIDE, 'IQU'), NDET, interpolation=Interpolation.BILINEAR
        )
        assert kernel == SamplingKernel(Interpolation.BILINEAR)
        assert jax.tree.leaves(kernel) == []

    def test_weights_take_the_map_dtype(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU', dtype=jnp.float32)
        weights = StokesIQU(*(jnp.array([0.3, 0.7], jnp.float64),) * 3)
        shared = SamplingKernel.create(landscape, NDET, offsets=_offsets(), weights=[0.3, 0.7])
        per_stokes = SamplingKernel.create(landscape, NDET, offsets=_offsets(), weights=weights)
        assert shared.weights.dtype == jnp.float32
        assert isinstance(per_stokes.weights, StokesIQU)
        assert per_stokes.weights.data.dtype == jnp.float32

    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'offsets': 'given'}, 'given together'),
            ({'weights': jnp.ones(2)}, 'given together'),
            ({'offsets': 'given', 'weights': jnp.ones(3)}, 'offset weights have shape'),
            ({'offsets': 'given', 'weights': jnp.ones((3, 2))}, 'as a Stokes'),
            (
                {'offsets': 'given', 'weights': StokesQU(jnp.ones(2), jnp.ones(2))},
                'Stokes components',
            ),
            ({'offsets': 'given', 'weights': StokesIQU(*(jnp.ones(3),) * 3)}, 'per component'),
            ({'offsets': 'wrong_ndet', 'weights': jnp.ones(2)}, 'offsets has shape'),
        ],
    )
    def test_rejects_inconsistent_offsets(self, kwargs, match) -> None:
        if kwargs.get('offsets') == 'given':
            kwargs = {**kwargs, 'offsets': _offsets()}
        elif kwargs.get('offsets') == 'wrong_ndet':
            kwargs = {**kwargs, 'offsets': Quaternion.ones((NDET + 1, 1)) * _offsets()[None, :]}
        with pytest.raises(ValueError, match=match):
            SamplingKernel.create(HealpixLandscape(NSIDE, 'IQU'), NDET, **kwargs)


class TestIntegrate:
    def test_weighs_each_direction(self) -> None:
        """Two directions of one neighbour each fold into one two-neighbour stencil."""
        stencil = Stencil.unpositioned(jnp.array([[[3], [5]]]), jnp.ones((1, 2, 1)))
        kernel = SamplingKernel(offsets=_offsets(), weights=jnp.array([0.25, 0.75]))
        integrated = kernel.integrate(stencil)
        assert_array_equal(integrated.indices, [[3, 5]])
        assert_allclose(integrated.weights, [[0.25, 0.75]])

    def test_per_stokes_weights_give_one_row_per_component(self) -> None:
        stencil = Stencil.unpositioned(jnp.array([[[3], [5]]]), jnp.ones((1, 2, 1)))
        weights = StokesQU(jnp.array([0.25, 0.75]), jnp.array([0.5, 0.5]))
        integrated = SamplingKernel(offsets=_offsets(), weights=weights).integrate(stencil)
        assert_allclose(integrated.weights, [[[0.25, 0.75]], [[0.5, 0.5]]])


class TestOffsetsFor:
    def test_selects_the_detectors_of_a_batch(self) -> None:
        offsets = from_xieta_angles(*jax.random.normal(jax.random.key(0), (3, NDET, 2)))
        kernel = SamplingKernel(offsets=offsets, weights=jnp.ones(2))
        batch = kernel.offsets_for(jnp.array([2, 0]))
        assert batch is not None
        assert_array_equal(batch.wxyz, offsets.wxyz[jnp.array([2, 0])])

    def test_shared_offsets_are_repeated_for_every_detector(self) -> None:
        kernel = SamplingKernel(offsets=_offsets(), weights=jnp.ones(2))
        batch = kernel.offsets_for(jnp.array([2, 0]))
        assert batch is not None
        assert batch.shape == (2, 2)
        assert_array_equal(batch.wxyz, jnp.stack([_offsets().wxyz] * 2))

    def test_is_none_without_offsets(self) -> None:
        assert SamplingKernel().offsets_for(jnp.arange(2)) is None


class TestIntensityOnly:
    def test_keeps_the_intensity_weights(self) -> None:
        weights = StokesIQU(jnp.array([0.5, 0.5]), jnp.array([0.7, 0.3]), jnp.array([0.2, 0.8]))
        kernel = SamplingKernel(Interpolation.BILINEAR, _offsets(), weights).intensity_only()
        assert_array_equal(kernel.weights, [0.5, 0.5])
        assert kernel.interpolation is Interpolation.BILINEAR

    def test_averages_the_components_of_a_map_without_intensity(self) -> None:
        weights = StokesQU(jnp.array([0.7, 0.3]), jnp.array([0.2, 0.8]))
        kernel = SamplingKernel(offsets=_offsets(), weights=weights).intensity_only()
        assert_allclose(kernel.weights, [0.45, 0.55])

    @pytest.mark.parametrize(
        'weights', [None, jnp.array([0.3, 0.7]), StokesI(jnp.array([0.3, 0.7]))]
    )
    def test_leaves_intensity_weights_alone(self, weights) -> None:
        kernel = SamplingKernel(offsets=None if weights is None else _offsets(), weights=weights)
        reduced = kernel.intensity_only()
        if isinstance(weights, StokesI):
            assert_array_equal(reduced.weights, weights.i)
        else:
            assert reduced is kernel


class TestAngleSampler:
    def test_without_polarization_angle_returns_the_meridian_basis(self) -> None:
        """Samples of any shape, read at given angles, as `transported_gather` reads them."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        k1, k2 = jax.random.split(jax.random.key(0))
        theta = jax.random.uniform(k1, (4, 5), minval=0.1, maxval=jnp.pi - 0.1)
        phi = jax.random.uniform(k2, (4, 5), maxval=2 * jnp.pi)
        sampler = AngleSampler(kernel=SamplingKernel(Interpolation.BILINEAR), theta=theta, phi=phi)
        sky = landscape.normal(jax.random.key(1))

        tod = PointingOperator.from_sampler(landscape, sampler, batch_size=3)(sky)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        expected = transported_gather(sky.ravel(), stencil, theta, phi)
        assert_allclose(tod.data, expected.data, rtol=1e-12, atol=1e-12)

    def test_quaternions_to_angles_reads_the_same_map(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore = Quaternion.random(jax.random.key(2), (7,))
        sampler = QuaternionSampler(
            kernel=SamplingKernel(Interpolation.BILINEAR, _offsets(), jnp.array([0.3, 0.7])),
            qbore=qbore,
            qdet=from_xieta_angles(jnp.zeros(NDET), jnp.linspace(-0.1, 0.1, NDET), jnp.ones(NDET)),
        )
        sky = landscape.normal(jax.random.key(3))
        on_the_fly = PointingOperator.from_sampler(landscape, sampler)(sky)
        from_angles = PointingOperator.from_sampler(landscape, sampler.to_angles(landscape))(sky)
        assert_allclose(from_angles.data, on_the_fly.data, rtol=1e-12, atol=1e-13)


class TestPrecomputedSampler:
    @pytest.mark.parametrize('stokes', ['I', 'IQU'])
    @pytest.mark.parametrize('interpolation', list(Interpolation))
    def test_reads_the_map_as_its_source(self, stokes, interpolation) -> None:
        landscape = HealpixLandscape(NSIDE, stokes)
        sampler = QuaternionSampler(
            kernel=SamplingKernel(interpolation),
            qbore=Quaternion.random(jax.random.key(4), (7,)),
            qdet=from_xieta_angles(jnp.zeros(NDET), jnp.linspace(-0.1, 0.1, NDET), jnp.ones(NDET)),
        )
        cached = PrecomputedSampler.from_sampler(sampler, landscape)
        sky = landscape.normal(jax.random.key(5))
        source_op = PointingOperator.from_sampler(landscape, sampler)
        cached_op = PointingOperator.from_sampler(landscape, cached, batch_size=2)
        assert_allclose(cached_op(sky).data, source_op(sky).data, rtol=1e-12, atol=1e-13)
        tod = source_op(sky)
        assert_allclose(cached_op.T(tod).data, source_op.T(tod).data, rtol=1e-12, atol=1e-13)
