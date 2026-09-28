import jax
import jax.numpy as jnp
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose, assert_array_equal

from furax.math.coords import XiEtaAngles
from furax.obs.landscapes import HealpixLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.sampling import (
    AngleSampler,
    DiscretizedBeam,
    PrecomputedSampler,
    QuaternionSampler,
    SamplingKernel,
)
from furax.obs.spin2 import transported_gather
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import StokesI, StokesIQU, StokesQU

NSIDE = 4
NDET = 3


def _nodes() -> Quaternion:
    return XiEtaAngles(
        jnp.array([0.05, -0.03]), jnp.array([0.02, 0.04]), jnp.zeros(2)
    ).to_quaternion()


class TestDiscretizedBeamCreate:
    @pytest.mark.parametrize('per_detector', [False, True], ids=['shared', 'per-detector'])
    def test_accepts_matching_nodes_and_weights(self, per_detector) -> None:
        nodes = Quaternion.ones((NDET, 1)) * _nodes()[None, :] if per_detector else _nodes()
        beam = DiscretizedBeam.create(nodes, [0.3, 0.7])
        assert_array_equal(beam.weights, [0.3, 0.7])

    @pytest.mark.parametrize(
        'nodes, weights, match',
        [
            (Quaternion.ones((1, 2, 2)), jnp.ones(2), 'beam nodes have shape'),
            (None, jnp.ones(3), r'beam weights have shape \(3,\), expected \(2,\)'),
            (None, jnp.ones((3, 2)), 'as a Stokes'),
            (None, StokesIQU(*(jnp.ones(3),) * 3), 'per component'),
        ],
    )
    def test_rejects_inconsistent_nodes_and_weights(self, nodes, weights, match) -> None:
        with pytest.raises(ValueError, match=match):
            DiscretizedBeam.create(_nodes() if nodes is None else nodes, weights)


class TestDiscretizedBeamChecked:
    def test_weights_take_the_map_dtype(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU', dtype=jnp.float32)
        weights = StokesIQU(*(jnp.array([0.3, 0.7], jnp.float64),) * 3)
        shared = DiscretizedBeam.create(_nodes(), jnp.array([0.3, 0.7])).checked(landscape, NDET)
        per_stokes = DiscretizedBeam.create(_nodes(), weights).checked(landscape, NDET)
        assert shared.weights.dtype == jnp.float32
        assert isinstance(per_stokes.weights, StokesIQU)
        assert per_stokes.weights.data.dtype == jnp.float32

    @pytest.mark.parametrize(
        'nodes, weights, match',
        [
            (
                Quaternion.ones((NDET + 1, 1)) * _nodes()[None, :],
                jnp.ones(2),
                'beam nodes have shape',
            ),
            (_nodes(), StokesQU(jnp.ones(2), jnp.ones(2)), 'Stokes components'),
        ],
    )
    def test_rejects_a_beam_that_does_not_fit(self, nodes, weights, match) -> None:
        beam = DiscretizedBeam.create(nodes, weights)
        with pytest.raises(ValueError, match=match):
            beam.checked(HealpixLandscape(NSIDE, 'IQU'), NDET)


class TestNodesFor:
    def test_selects_the_detectors_of_a_batch(self) -> None:
        nodes = XiEtaAngles(*jax.random.normal(jax.random.key(0), (3, NDET, 2))).to_quaternion()
        batch = DiscretizedBeam(nodes, jnp.ones(2)).nodes_for(jnp.array([2, 0]))
        assert_array_equal(batch.wxyz, nodes.wxyz[jnp.array([2, 0])])

    def test_shared_nodes_are_repeated_for_every_detector(self) -> None:
        batch = DiscretizedBeam(_nodes(), jnp.ones(2)).nodes_for(jnp.array([2, 0]))
        assert batch.shape == (2, 2)
        assert_array_equal(batch.wxyz, jnp.stack([_nodes().wxyz] * 2))


class TestIntegrate:
    def test_weighs_each_node(self) -> None:
        """Two nodes of one neighbour each fold into one two-neighbour stencil."""
        stencil = Stencil.unpositioned(jnp.array([[[3], [5]]]), jnp.ones((1, 2, 1)))
        integrated = DiscretizedBeam(_nodes(), jnp.array([0.25, 0.75])).integrate(stencil)
        assert_array_equal(integrated.indices, [[3, 5]])
        assert_allclose(integrated.weights, [[0.25, 0.75]])

    def test_per_stokes_weights_give_one_row_per_component(self) -> None:
        stencil = Stencil.unpositioned(jnp.array([[[3], [5]]]), jnp.ones((1, 2, 1)))
        weights = StokesQU(jnp.array([0.25, 0.75]), jnp.array([0.5, 0.5]))
        integrated = DiscretizedBeam(_nodes(), weights).integrate(stencil)
        assert_allclose(integrated.weights, [[[0.25, 0.75]], [[0.5, 0.5]]])


class TestIntensityOnly:
    def test_keeps_the_intensity_weights(self) -> None:
        weights = StokesIQU(jnp.array([0.5, 0.5]), jnp.array([0.7, 0.3]), jnp.array([0.2, 0.8]))
        beam = DiscretizedBeam(_nodes(), weights)
        kernel = SamplingKernel(Interpolation.BILINEAR, beam).intensity_only()
        assert kernel.beam is not None
        assert_array_equal(kernel.beam.weights, [0.5, 0.5])
        assert kernel.beam.nodes is beam.nodes
        assert kernel.interpolation is Interpolation.BILINEAR

    def test_averages_the_components_of_a_map_without_intensity(self) -> None:
        weights = StokesQU(jnp.array([0.7, 0.3]), jnp.array([0.2, 0.8]))
        beam = DiscretizedBeam(_nodes(), weights).intensity_only()
        assert_allclose(beam.weights, [0.45, 0.55])

    @pytest.mark.parametrize('weights', [jnp.array([0.3, 0.7]), StokesI(jnp.array([0.3, 0.7]))])
    def test_leaves_intensity_weights_alone(self, weights) -> None:
        beam = DiscretizedBeam(_nodes(), weights)
        reduced = beam.intensity_only()
        if isinstance(weights, StokesI):
            assert_array_equal(reduced.weights, weights.i)
        else:
            assert reduced is beam

    def test_a_kernel_without_beam_is_unchanged(self) -> None:
        kernel = SamplingKernel(Interpolation.BILINEAR)
        assert kernel.intensity_only() is kernel


class TestAngleSampler:
    def test_without_polarization_angle_returns_the_meridian_basis(self) -> None:
        """Samples of any shape, read at given angles, as `transported_gather` reads them."""
        landscape = HealpixLandscape(NSIDE, 'IQU')
        k1, k2 = jax.random.split(jax.random.key(0))
        theta = jax.random.uniform(k1, (4, 5), minval=0.1, maxval=jnp.pi - 0.1)
        phi = jax.random.uniform(k2, (4, 5), maxval=2 * jnp.pi)
        sampler = AngleSampler(kernel=SamplingKernel(Interpolation.BILINEAR), theta=theta, phi=phi)
        sky = landscape.normal(jax.random.key(1))

        tod = PointingOperator.from_sampler(landscape, sampler, batch_samples=3)(sky)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        expected = transported_gather(sky.ravel(), stencil, theta, phi)
        assert_allclose(tod.data, expected.data, rtol=1e-12, atol=1e-12)

    def test_quaternions_to_angles_reads_the_same_map(self) -> None:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        qbore = Quaternion.random(jax.random.key(2), (7,))
        sampler = QuaternionSampler(
            kernel=SamplingKernel(
                Interpolation.BILINEAR, DiscretizedBeam(_nodes(), jnp.array([0.3, 0.7]))
            ),
            qbore=qbore,
            qdet=XiEtaAngles(
                jnp.zeros(NDET), jnp.linspace(-0.1, 0.1, NDET), jnp.ones(NDET)
            ).to_quaternion(),
        )
        sky = landscape.normal(jax.random.key(3))
        on_the_fly = PointingOperator.from_sampler(landscape, sampler)(sky)
        from_angles = PointingOperator.from_sampler(landscape, sampler.to_angles(landscape))(sky)
        assert_allclose(from_angles.data, on_the_fly.data, rtol=1e-12, atol=1e-13)

    def test_nearest_is_not_stored_as_angles(self) -> None:
        """Angles can put a sample on a pixel boundary in another pixel than `quat2index`."""
        sampler = QuaternionSampler(
            kernel=SamplingKernel(Interpolation.NEAREST),
            qbore=Quaternion.random(jax.random.key(5), (7,)),
            qdet=Quaternion.ones((NDET,)),
        )
        with pytest.raises(ValueError, match='cannot be stored as angles'):
            sampler.to_angles(HealpixLandscape(NSIDE, 'IQU'))


class TestPrecomputedSampler:
    @pytest.mark.parametrize('stokes', ['I', 'IQU'])
    @pytest.mark.parametrize('interpolation', list(Interpolation))
    def test_reads_the_map_as_its_source(self, stokes, interpolation) -> None:
        landscape = HealpixLandscape(NSIDE, stokes)
        sampler = QuaternionSampler(
            kernel=SamplingKernel(interpolation),
            qbore=Quaternion.random(jax.random.key(4), (7,)),
            qdet=XiEtaAngles(
                jnp.zeros(NDET), jnp.linspace(-0.1, 0.1, NDET), jnp.ones(NDET)
            ).to_quaternion(),
        )
        cached = PrecomputedSampler.from_sampler(sampler, landscape)
        sky = landscape.normal(jax.random.key(5))
        source_op = PointingOperator.from_sampler(landscape, sampler)
        cached_op = PointingOperator.from_sampler(landscape, cached, batch_samples=2)
        assert_allclose(cached_op(sky).data, source_op(sky).data, rtol=1e-12, atol=1e-13)
        tod = source_op(sky)
        assert_allclose(cached_op.T(tod).data, source_op.T(tod).data, rtol=1e-12, atol=1e-13)
