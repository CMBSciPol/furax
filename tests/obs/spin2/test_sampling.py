import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.obs.landscapes import HealpixLandscape, LocalStokesLandscape, StokesLandscape
from furax.obs.spin2 import spin2_cos_sin, transported_gather, transported_scatter
from furax.obs.stokes import Stokes, ValidStokesLiteral

NSIDE = 16


def _directions(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    rng = np.random.default_rng(seed)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    return jnp.asarray(theta), jnp.asarray(phi)


def _random_tod(landscape: StokesLandscape, n: int, seed: int) -> Stokes:
    data = np.random.default_rng(seed).normal(size=(len(landscape.stokes), n))
    return Stokes.class_for(landscape.stokes).from_array(jnp.asarray(data))


def _scalar_gather(sky: Stokes, indices: jax.Array, weights: jax.Array) -> jax.Array:
    """Plain scalar interpolation, i.e. what the untransported sampler computes."""
    unit_weights = weights / weights.sum(axis=-1, keepdims=True)
    return jnp.sum(sky.data[..., indices] * unit_weights, axis=-1)


class TestTransportedGather:
    def test_matches_scalar_interpolation_on_intensity(self) -> None:
        """The transport acts on P alone, so I must be untouched."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(200, 0)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(0))

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        expected = _scalar_gather(sky, indices, weights)

        assert_allclose(np.asarray(gathered.i), np.asarray(expected[0]), atol=1e-14)
        # Q and U are where the transport bites; a no-op implementation would pass the I check.
        assert np.abs(np.asarray(gathered.q) - np.asarray(expected[1])).max() > 1e-6

    def test_stokes_i_map_is_scalar_interpolation(self) -> None:
        landscape = HealpixLandscape(NSIDE, stokes='I')
        theta, phi = _directions(200, 1)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(1))

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        assert_allclose(
            np.asarray(gathered.i), np.asarray(_scalar_gather(sky, indices, weights)[0]), atol=1e-14
        )

    def test_pixel_centers_reproduce_the_pixel_value(self) -> None:
        """At a pixel center the stencil collapses onto that pixel and no transport is left."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 37)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(2))

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        assert_allclose(np.asarray(gathered.data), np.asarray(sky.data[..., pixels]), atol=1e-12)

    def test_beats_scalar_interpolation_against_an_exact_spin2_evaluation(self) -> None:
        """Pins the transport sign against an external truth.

        `ducc0.sht.synthesis_general(spin=2)` evaluates a band-limited polarised sky off-grid
        exactly. Near the pole, where the neighbours' meridians fan out, the transported gather must
        beat scalar interpolation by a wide margin -- and the opposite sign must lose to it, which is
        what makes this a sign test rather than a smoke test.
        """
        ducc_sht = pytest.importorskip('ducc0.sht')
        healpy = pytest.importorskip('healpy')

        nside, lmax = 32, 32
        ell = np.arange(lmax + 1)
        cl_ee = np.zeros(lmax + 1)
        cl_ee[2:] = 1.0 / (ell[2:] * (ell[2:] + 1))
        zero = np.zeros(lmax + 1)
        alm_t, alm_e, alm_b = healpy.synalm([zero, cl_ee, 0.05 * cl_ee, zero], lmax=lmax, new=True)
        maps = healpy.alm2map([alm_t, alm_e, alm_b], nside=nside, lmax=lmax, pol=True)
        sky = Stokes.class_for('IQU').from_array(jnp.asarray(np.asarray(maps)))

        # a ring near the pole, offset off the pixel centers
        rng = np.random.default_rng(12)
        n = 2000
        theta_np = np.deg2rad(3.0) + rng.uniform(-0.004, 0.004, n)
        phi_np = rng.uniform(0.0, 2 * np.pi, n)
        exact = ducc_sht.synthesis_general(
            alm=np.asarray([alm_e, alm_b]),
            spin=2,
            lmax=lmax,
            loc=np.stack([theta_np, phi_np], axis=-1),
            epsilon=1e-12,
        )

        landscape = HealpixLandscape(nside, stokes='IQU')
        theta, phi = jnp.asarray(theta_np), jnp.asarray(phi_np)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        scalar = _scalar_gather(sky, indices, weights)

        def error(q: np.ndarray, u: np.ndarray) -> float:
            return float(np.sqrt(np.mean((q - exact[0]) ** 2 + (u - exact[1]) ** 2)))

        transported_error = error(np.asarray(gathered.q), np.asarray(gathered.u))
        scalar_error = error(np.asarray(scalar[1]), np.asarray(scalar[2]))
        assert transported_error < 0.5 * scalar_error

        # The same rotation applied backwards, i.e. the pair flipped before `rotate_qu`.
        cos_2delta, sin_2delta = spin2_cos_sin(
            *jhp.pix2ang(nside, jnp.where(indices >= 0, indices, 0)), theta[:, None], phi[:, None]
        )
        unit_weights = weights / weights.sum(axis=-1, keepdims=True)
        neighbors = Stokes.class_for('IQU').from_array(sky.data[..., indices])
        flipped = jnp.sum(neighbors.rotate_qu(cos_2delta, -sin_2delta).data * unit_weights, axis=-1)
        assert error(np.asarray(flipped[1]), np.asarray(flipped[2])) > scalar_error


class TestNearestStencil:
    """The one-neighbour stencil of a nearest-neighbour sampler, transported by the same kernel."""

    def test_transport_carries_the_pixel_value_to_the_sample(self) -> None:
        """The pixel's Q and U are rotated from its own meridian to the sampled direction's."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(200, 20)
        indices, weights, centers = landscape.world2nearest_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(20))

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)

        pixel = Stokes.class_for('IQU').from_array(sky.data[..., indices[..., 0]])
        cos_2delta, sin_2delta = spin2_cos_sin(*jhp.pix2ang(NSIDE, indices[..., 0]), theta, phi)
        expected = pixel.rotate_qu(cos_2delta, sin_2delta)

        assert_allclose(np.asarray(gathered.data), np.asarray(expected.data), atol=1e-14)
        # I is untouched, and Q is not: a no-op implementation would pass the I check alone
        assert_allclose(np.asarray(gathered.i), np.asarray(pixel.i), atol=1e-14)
        assert np.abs(np.asarray(gathered.q) - np.asarray(pixel.q)).max() > 1e-6

    def test_pixel_centers_reproduce_the_pixel_value(self) -> None:
        """A sample sitting on a pixel center has nothing to transport."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 37)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        indices, weights, centers = landscape.world2nearest_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(21))

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        assert_allclose(np.asarray(gathered.data), np.asarray(sky.data[..., pixels]), atol=1e-14)

    def test_beats_the_raw_pixel_value_against_an_exact_spin2_evaluation(self) -> None:
        """Pins the transport sign for the nearest stencil against an external truth.

        Same external reference as the bilinear case: `ducc0.sht.synthesis_general(spin=2)`
        evaluates the polarised sky exactly off-grid. The nearest sampler keeps the sub-pixel
        gradient error the bilinear one removes, so the transport cannot win by the same margin
        here -- but it must win, and its opposite sign must lose.
        """
        ducc_sht = pytest.importorskip('ducc0.sht')
        healpy = pytest.importorskip('healpy')

        nside, lmax = 32, 32
        ell = np.arange(lmax + 1)
        cl_ee = np.zeros(lmax + 1)
        cl_ee[2:] = 1.0 / (ell[2:] * (ell[2:] + 1))
        zero = np.zeros(lmax + 1)
        alm_t, alm_e, alm_b = healpy.synalm([zero, cl_ee, 0.05 * cl_ee, zero], lmax=lmax, new=True)
        maps = healpy.alm2map([alm_t, alm_e, alm_b], nside=nside, lmax=lmax, pol=True)
        sky = Stokes.class_for('IQU').from_array(jnp.asarray(np.asarray(maps)))

        # a ring near the pole, offset off the pixel centers
        rng = np.random.default_rng(12)
        n = 2000
        theta_np = np.deg2rad(3.0) + rng.uniform(-0.004, 0.004, n)
        phi_np = rng.uniform(0.0, 2 * np.pi, n)
        exact = ducc_sht.synthesis_general(
            alm=np.asarray([alm_e, alm_b]),
            spin=2,
            lmax=lmax,
            loc=np.stack([theta_np, phi_np], axis=-1),
            epsilon=1e-12,
        )

        landscape = HealpixLandscape(nside, stokes='IQU')
        theta, phi = jnp.asarray(theta_np), jnp.asarray(phi_np)
        indices, weights, centers = landscape.world2nearest_with_centers(theta, phi)
        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        raw = Stokes.class_for('IQU').from_array(sky.data[..., indices[..., 0]])

        def error(q: np.ndarray, u: np.ndarray) -> float:
            return float(np.sqrt(np.mean((q - exact[0]) ** 2 + (u - exact[1]) ** 2)))

        raw_error = error(np.asarray(raw.q), np.asarray(raw.u))
        assert error(np.asarray(gathered.q), np.asarray(gathered.u)) < 0.7 * raw_error

        # The same rotation applied backwards, i.e. the pair flipped before `rotate_qu`.
        cos_2delta, sin_2delta = spin2_cos_sin(*jhp.pix2ang(nside, indices[..., 0]), theta, phi)
        flipped = raw.rotate_qu(cos_2delta, -sin_2delta)
        assert error(np.asarray(flipped.q), np.asarray(flipped.u)) > raw_error


class TestAdjoint:
    """A sign error in either rotation shows up here, on a fixed stencil, with no operator built."""

    @pytest.mark.parametrize('stokes', ['I', 'QU', 'IQU'])
    def test_scatter_is_the_transpose_jax_derives(self, stokes: ValidStokesLiteral) -> None:
        """Compare against the whole transposed operator, not one random projection of it.

        The gather is linear in the sky, so `jax.linear_transpose` builds its exact adjoint. The
        hand-written scatter exists because the operator framework needs a method and the beam port
        needs a free function to vmap, not because JAX cannot derive it -- so it must agree.
        """
        landscape = HealpixLandscape(NSIDE, stokes=stokes)
        theta, phi = _directions(300, 12)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        tod = _random_tod(landscape, 300, 13)

        def gather(sky: Stokes) -> Stokes:
            return transported_gather(sky, indices, weights, centers, theta, phi)

        (derived,) = jax.linear_transpose(gather, landscape.zeros())(tod)
        written = transported_scatter(landscape.zeros(), tod, indices, weights, centers, theta, phi)
        assert_allclose(np.asarray(written.data), np.asarray(derived.data), atol=1e-14)

    @pytest.mark.parametrize('stokes', ['I', 'QU', 'IQU'])
    def test_gather_and_scatter_are_adjoint(self, stokes: ValidStokesLiteral) -> None:
        landscape = HealpixLandscape(NSIDE, stokes=stokes)
        theta, phi = _directions(500, 4)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(4))
        tod = _random_tod(landscape, 500, 5)

        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        scattered = transported_scatter(
            landscape.zeros(), tod, indices, weights, centers, theta, phi
        )
        lhs = float(jnp.sum(gathered.data * tod.data))
        rhs = float(jnp.sum(sky.data * scattered.data))
        assert_allclose(lhs, rhs, rtol=1e-12)

    def test_adjoint_on_a_subset_landscape(self) -> None:
        """Neighbours falling outside the subset go to the sink; the pair must stay adjoint.

        The subset also exercises the trap that local indices are meaningless as sky positions: the
        centers must come from the parent, and this test is wrong by 1e-1 if they do not.
        """
        parent = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(400, 6)
        covered = np.unique(np.asarray(parent.world2interp(theta, phi)[0]).ravel())
        # keep two thirds of the covered pixels, so plenty of stencils straddle the boundary
        landscape = LocalStokesLandscape(parent, covered[: 2 * len(covered) // 3])

        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        assert int((indices == landscape.sink).sum()) > 0

        sky = landscape.normal(jax.random.key(6))
        tod = _random_tod(landscape, 400, 7)
        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        scattered = transported_scatter(
            landscape.zeros(), tod, indices, weights, centers, theta, phi
        )
        assert_allclose(
            float(jnp.sum(gathered.data * tod.data)),
            float(jnp.sum(sky.data * scattered.data)),
            rtol=1e-12,
        )

    def test_scatter_accumulates_into_the_given_map(self) -> None:
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(100, 8)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        tod = _random_tod(landscape, 100, 9)

        once = transported_scatter(landscape.zeros(), tod, indices, weights, centers, theta, phi)
        twice = transported_scatter(once, tod, indices, weights, centers, theta, phi)
        assert_allclose(np.asarray(twice.data), 2 * np.asarray(once.data), rtol=1e-12)


class TestOutOfBounds:
    def test_sink_neighbours_do_not_contribute(self) -> None:
        """A neighbour sent to the sink must neither be read nor written."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(50, 10)
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)
        sky = landscape.normal(jax.random.key(10))

        # drop the last neighbour of every sample by marking it out of bounds
        masked = indices.at[..., -1].set(-1)
        gathered = transported_gather(sky, masked, weights, centers, theta, phi)

        # the same thing said differently: keep the index, zero the weight, renormalize
        kept_weights = weights.at[..., -1].set(0.0)
        expected = transported_gather(sky, indices, kept_weights, centers, theta, phi)
        assert_allclose(np.asarray(gathered.data), np.asarray(expected.data), atol=1e-14)


class TestFloat32:
    @pytest.mark.insubprocess
    def test_dtypes_are_preserved(self) -> None:
        """`double_precision=False` runs the whole pipeline in float32."""
        jax.config.update('jax_enable_x64', False)

        landscape = HealpixLandscape(NSIDE, stokes='IQU', dtype=np.float32)
        theta, phi = _directions(100, 11)
        theta, phi = theta.astype(jnp.float32), phi.astype(jnp.float32)
        for stencil in (
            landscape.world2interp_with_centers(theta, phi),
            landscape.world2nearest_with_centers(theta, phi),
        ):
            assert stencil[1].dtype == jnp.float32
            assert stencil[2].z.dtype == jnp.float32
        indices, weights, centers = landscape.world2interp_with_centers(theta, phi)

        sky = landscape.normal(jax.random.key(11))
        gathered = transported_gather(sky, indices, weights, centers, theta, phi)
        assert gathered.dtype == jnp.float32
        scattered = transported_scatter(
            landscape.zeros(), gathered, indices, weights, centers, theta, phi
        )
        assert scattered.dtype == jnp.float32
        assert np.isfinite(np.asarray(scattered.data)).all()
