import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.obs.landscapes import HealpixLandscape, LocalStokesLandscape, StokesLandscape
from furax.obs.spin2 import spin2_cos_sin, transported_gather, transported_scatter
from furax.obs.stencil import Interpolation, Stencil
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


def _polarised_sky(healpy, nside: int, lmax: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A band-limited E-dominated sky: its alm (E, B) and IQU maps, from a seeded generator.

    Drawn by hand rather than through `healpy.synalm`, which reads NumPy's global state and
    would give a different sky on every run.
    """
    ell = np.arange(lmax + 1)
    cl_ee = np.zeros(lmax + 1)
    cl_ee[2:] = 1.0 / (ell[2:] * (ell[2:] + 1))
    rng = np.random.default_rng(seed)
    l, m = healpy.Alm.getlm(lmax)

    def synalm(cl: np.ndarray) -> np.ndarray:
        sigma = np.sqrt(cl[l])
        # m = 0 coefficients are real; the others split the variance over both parts
        real = rng.normal(size=l.size) * np.where(m == 0, sigma, sigma / np.sqrt(2))
        imag = rng.normal(size=l.size) * np.where(m == 0, 0.0, sigma / np.sqrt(2))
        return real + 1j * imag

    alm = np.stack([np.zeros(l.size, complex), synalm(cl_ee), synalm(0.05 * cl_ee)])
    maps = healpy.alm2map(alm, nside=nside, lmax=lmax, pol=True)
    return alm[1:], maps


class TestTransportedGather:
    def test_matches_scalar_interpolation_on_intensity(self) -> None:
        """The transport acts on P alone, so I must be untouched."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(200, 0)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(0))

        gathered = transported_gather(sky, stencil, theta, phi)
        expected = _scalar_gather(sky, stencil.indices, stencil.weights)

        assert_allclose(gathered.i, expected[0], atol=1e-14)
        # Q and U are where the transport bites; a no-op implementation would pass the I check.
        assert jnp.abs(gathered.q - expected[1]).max() > 1e-6

    def test_stokes_i_map_is_scalar_interpolation(self) -> None:
        landscape = HealpixLandscape(NSIDE, stokes='I')
        theta, phi = _directions(200, 1)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(1))

        gathered = transported_gather(sky, stencil, theta, phi)
        expected = _scalar_gather(sky, stencil.indices, stencil.weights)
        assert_allclose(gathered.i, expected[0], atol=1e-14)

    def test_pixel_centers_reproduce_the_pixel_value(self) -> None:
        """At a pixel center the stencil collapses onto that pixel and no transport is left."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 37)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(2))

        gathered = transported_gather(sky, stencil, theta, phi)
        assert_allclose(gathered.data, sky.data[..., pixels], atol=1e-12)

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
        alm_eb, maps = _polarised_sky(healpy, nside, lmax, seed=7)
        sky = Stokes.class_for('IQU').from_array(jnp.asarray(maps))

        # a ring near the pole, offset off the pixel centers
        rng = np.random.default_rng(12)
        n = 2000
        theta_np = np.deg2rad(3.0) + rng.uniform(-0.004, 0.004, n)
        phi_np = rng.uniform(0.0, 2 * np.pi, n)
        exact = ducc_sht.synthesis_general(
            alm=alm_eb,
            spin=2,
            lmax=lmax,
            loc=np.stack([theta_np, phi_np], axis=-1),
            epsilon=1e-12,
        )

        landscape = HealpixLandscape(nside, stokes='IQU')
        theta, phi = jnp.asarray(theta_np), jnp.asarray(phi_np)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        gathered = transported_gather(sky, stencil, theta, phi)
        scalar = _scalar_gather(sky, stencil.indices, stencil.weights)

        def error(q: jax.Array, u: jax.Array) -> float:
            return float(np.sqrt(np.mean((q - exact[0]) ** 2 + (u - exact[1]) ** 2)))

        transported_error = error(gathered.q, gathered.u)
        scalar_error = error(scalar[1], scalar[2])
        assert transported_error < 0.5 * scalar_error

        # The same rotation applied backwards, i.e. the pair flipped before `rotate_qu`.
        cos_2delta, sin_2delta = spin2_cos_sin(
            *jhp.pix2ang(nside, stencil.indices), theta[:, None], phi[:, None]
        )
        neighbors = Stokes.class_for('IQU').from_array(sky.data[..., stencil.indices])
        flipped = jnp.sum(
            neighbors.rotate_qu(cos_2delta, -sin_2delta).data * stencil.weights, axis=-1
        )
        assert error(flipped[1], flipped[2]) > scalar_error


class TestNearestStencil:
    """The one-neighbour stencil of a nearest-neighbour sampler, transported by the same kernel."""

    def test_transport_carries_the_pixel_value_to_the_sample(self) -> None:
        """The pixel's Q and U are rotated from its own meridian to the sampled direction's."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(200, 20)
        stencil = landscape.world2stencil(theta, phi, Interpolation.NEAREST)
        sky = landscape.normal(jax.random.key(20))

        gathered = transported_gather(sky, stencil, theta, phi)

        pixel = Stokes.class_for('IQU').from_array(sky.data[..., stencil.indices[..., 0]])
        cos_2delta, sin_2delta = spin2_cos_sin(
            *jhp.pix2ang(NSIDE, stencil.indices[..., 0]), theta, phi
        )
        expected = pixel.rotate_qu(cos_2delta, sin_2delta)

        assert_allclose(gathered.data, expected.data, atol=1e-14)
        # I is untouched, and Q is not: a no-op implementation would pass the I check alone
        assert_allclose(gathered.i, pixel.i, atol=1e-14)
        assert jnp.abs(gathered.q - pixel.q).max() > 1e-6

    def test_pixel_centers_reproduce_the_pixel_value(self) -> None:
        """A sample sitting on a pixel center has nothing to transport."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        pixels = jnp.arange(0, 12 * NSIDE**2, 37)
        theta, phi = jhp.pix2ang(NSIDE, pixels)
        stencil = landscape.world2stencil(theta, phi, Interpolation.NEAREST)
        sky = landscape.normal(jax.random.key(21))

        gathered = transported_gather(sky, stencil, theta, phi)
        assert_allclose(gathered.data, sky.data[..., pixels], atol=1e-14)

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
        alm_eb, maps = _polarised_sky(healpy, nside, lmax, seed=8)
        sky = Stokes.class_for('IQU').from_array(jnp.asarray(maps))

        # a ring near the pole, offset off the pixel centers
        rng = np.random.default_rng(12)
        n = 2000
        theta_np = np.deg2rad(3.0) + rng.uniform(-0.004, 0.004, n)
        phi_np = rng.uniform(0.0, 2 * np.pi, n)
        exact = ducc_sht.synthesis_general(
            alm=alm_eb,
            spin=2,
            lmax=lmax,
            loc=np.stack([theta_np, phi_np], axis=-1),
            epsilon=1e-12,
        )

        landscape = HealpixLandscape(nside, stokes='IQU')
        theta, phi = jnp.asarray(theta_np), jnp.asarray(phi_np)
        stencil = landscape.world2stencil(theta, phi, Interpolation.NEAREST)
        gathered = transported_gather(sky, stencil, theta, phi)
        raw = Stokes.class_for('IQU').from_array(sky.data[..., stencil.indices[..., 0]])

        def error(q: jax.Array, u: jax.Array) -> float:
            return float(np.sqrt(np.mean((q - exact[0]) ** 2 + (u - exact[1]) ** 2)))

        raw_error = error(raw.q, raw.u)
        assert error(gathered.q, gathered.u) < 0.7 * raw_error

        # The same rotation applied backwards, i.e. the pair flipped before `rotate_qu`.
        cos_2delta, sin_2delta = spin2_cos_sin(
            *jhp.pix2ang(nside, stencil.indices[..., 0]), theta, phi
        )
        flipped = raw.rotate_qu(cos_2delta, -sin_2delta)
        assert error(flipped.q, flipped.u) > raw_error


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
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        tod = _random_tod(landscape, 300, 13)

        def gather(sky: Stokes) -> Stokes:
            return transported_gather(sky, stencil, theta, phi)

        (derived,) = jax.linear_transpose(gather, landscape.zeros())(tod)
        written = transported_scatter(landscape.zeros(), tod, stencil, theta, phi)
        assert_allclose(written.data, derived.data, atol=1e-14)

    @pytest.mark.parametrize('stokes', ['I', 'QU', 'IQU'])
    def test_gather_and_scatter_are_adjoint(self, stokes: ValidStokesLiteral) -> None:
        landscape = HealpixLandscape(NSIDE, stokes=stokes)
        theta, phi = _directions(500, 4)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(4))
        tod = _random_tod(landscape, 500, 5)

        gathered = transported_gather(sky, stencil, theta, phi)
        scattered = transported_scatter(landscape.zeros(), tod, stencil, theta, phi)
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
        covered = np.unique(parent.world2interp(theta, phi)[0].ravel())
        # keep two thirds of the covered pixels, so plenty of stencils straddle the boundary
        landscape = LocalStokesLandscape(parent, covered[: 2 * len(covered) // 3])

        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        assert int((stencil.indices == landscape.sink).sum()) > 0

        sky = landscape.normal(jax.random.key(6))
        tod = _random_tod(landscape, 400, 7)
        gathered = transported_gather(sky, stencil, theta, phi)
        scattered = transported_scatter(landscape.zeros(), tod, stencil, theta, phi)
        assert_allclose(
            float(jnp.sum(gathered.data * tod.data)),
            float(jnp.sum(sky.data * scattered.data)),
            rtol=1e-12,
        )

    def test_scatter_accumulates_into_the_given_map(self) -> None:
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(100, 8)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        tod = _random_tod(landscape, 100, 9)

        once = transported_scatter(landscape.zeros(), tod, stencil, theta, phi)
        twice = transported_scatter(once, tod, stencil, theta, phi)
        assert_allclose(twice.data, 2 * once.data, rtol=1e-12)


class TestOutOfBounds:
    def test_sink_neighbours_do_not_contribute(self) -> None:
        """A neighbour sent to the sink must neither be read nor written."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(50, 10)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(10))

        # drop the last neighbour of every sample by marking it out of bounds
        masked = stencil.reindexed(stencil.indices.at[..., -1].set(-1), stencil.weights)
        gathered = transported_gather(sky, masked, theta, phi)

        # the same thing said differently: keep the index, zero the weight, renormalize
        kept = stencil.reindexed(stencil.indices, stencil.weights.at[..., -1].set(0.0))
        expected = transported_gather(sky, kept, theta, phi)
        assert_allclose(gathered.data, expected.data, atol=1e-14)


class TestUnpositionedStencil:
    def test_transporting_a_stencil_with_no_positions_is_refused(self) -> None:
        """A stencil off the sphere cannot say what frame its Q and U are in, so it must not try."""
        landscape = HealpixLandscape(NSIDE, stokes='IQU')
        theta, phi = _directions(50, 30)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        unpositioned = Stencil.unpositioned(stencil.indices, stencil.weights)
        sky = landscape.normal(jax.random.key(30))

        with pytest.raises(ValueError, match='no sky positions'):
            transported_gather(sky, unpositioned, theta, phi)

    def test_an_intensity_map_samples_through_it_unchanged(self) -> None:
        """Intensity never asks for the positions, which is what makes such a stencil usable."""
        landscape = HealpixLandscape(NSIDE, stokes='I')
        theta, phi = _directions(50, 31)
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)
        sky = landscape.normal(jax.random.key(31))

        unpositioned = Stencil.unpositioned(stencil.indices, stencil.weights)
        assert_allclose(
            transported_gather(sky, unpositioned, theta, phi).i,
            transported_gather(sky, stencil, theta, phi).i,
            atol=1e-14,
        )


class TestFloat32:
    @pytest.mark.insubprocess
    def test_dtypes_are_preserved(self) -> None:
        """`double_precision=False` runs the whole pipeline in float32."""
        jax.config.update('jax_enable_x64', False)

        landscape = HealpixLandscape(NSIDE, stokes='IQU', dtype=np.float32)
        theta, phi = _directions(100, 11)
        theta, phi = theta.astype(jnp.float32), phi.astype(jnp.float32)
        for stencil in (
            landscape.world2stencil(theta, phi, Interpolation.BILINEAR),
            landscape.world2stencil(theta, phi, Interpolation.NEAREST),
        ):
            assert stencil.weights.dtype == jnp.float32
            assert stencil.positions.z.dtype == jnp.float32
        stencil = landscape.world2stencil(theta, phi, Interpolation.BILINEAR)

        sky = landscape.normal(jax.random.key(11))
        gathered = transported_gather(sky, stencil, theta, phi)
        assert gathered.dtype == jnp.float32
        scattered = transported_scatter(landscape.zeros(), gathered, stencil, theta, phi)
        assert scattered.dtype == jnp.float32
        assert np.isfinite(scattered.data).all()
