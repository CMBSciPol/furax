import jax.numpy as jnp
import numpy as np
import pytest
from astropy.wcs import WCS
from jax import Array
from numpy.testing import assert_array_almost_equal, assert_array_equal

from furax.obs._samplings import Sampling
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    FrequencyLandscape,
    HealpixLandscape,
    ProjectionType,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.stokes import Stokes, ValidStokesType


def test_healpix_landscape(stokes: ValidStokesType) -> None:
    nside = 64
    npixel = 12 * nside**2

    landscape = HealpixLandscape(nside, stokes)

    sky = landscape.ones()
    assert isinstance(sky, Stokes.class_for(stokes))
    assert sky.shape == (npixel,)
    for stoke in stokes:
        leaf = getattr(sky, stoke.lower())
        assert isinstance(leaf, Array)
        assert leaf.size == npixel
        assert_array_equal(leaf, 1.0)


def test_frequency_landscape(stokes: ValidStokesType) -> None:
    nside = 64
    npixel = 12 * nside**2
    frequencies = jnp.array([10, 20, 30])
    landscape = FrequencyLandscape(nside, frequencies, stokes)

    sky = landscape.ones()
    assert isinstance(sky, Stokes.class_for(stokes))
    assert sky.shape == (3, npixel)
    for stoke in stokes:
        leaf = getattr(sky, stoke.lower())
        assert isinstance(leaf, Array)
        assert leaf.size == 3 * npixel
        assert_array_equal(leaf, 1.0)


@pytest.mark.parametrize(
    'pixel, expected_index',
    [
        ((-0.5 - 1e-15, -0.5), -1),
        ((-0.5, -0.5 - 1e-15), -1),
        ((1.5 + 1e-15, -0.5), -1),
        ((1.5 - 1e-15, -0.5 - 1e-15), -1),
        ((-0.5 - 1e-15, 4.5 - 1e-15), -1),
        ((-0.5, 4.5 + 1e-15), -1),
        ((1.5 + 1e-15, 4.5 - 1e-15), -1),
        ((1.5 - 1e-15, 4.5 + 1e-15), -1),
        ((-0.5, -0.5), 0),
        ((-0.5, 0.5 - 1e-15), 0),
        ((0.5, -0.5), 0),
        ((0.5, 0.5 - 1e-15), 0),
        ((0, 0), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((1, 1), 3),
        ((0, 4), 8),
        ((1, 4), 9),
    ],
)
def test_pixel2index(pixel: tuple[float, float], expected_index: int) -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    landscape = CARStokesLandscape((5, 2), 'I')
    actual_index = landscape.pixel2index(*pixel)
    assert_array_equal(actual_index, expected_index)


def test_get_coverage() -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    samplings = Sampling(
        jnp.array([0.0, 1, 0, 1, 1, 1, 0]), jnp.array([0.0, 0, 0, 3, 0, 1, 0]), jnp.array(0.0)
    )
    landscape = CARStokesLandscape((5, 2), 'I')
    coverage = landscape.get_coverage(samplings)
    assert_array_equal(coverage, [[3, 2], [0, 1], [0, 0], [0, 1], [0, 0]])


@pytest.mark.parametrize('projection_type', list(ProjectionType))
class TestWCSLandscape:
    @staticmethod
    def _make_wcs(
        shape: tuple[int, int],
        projection_type: ProjectionType,
        res_deg: float = 0.1,
        crval: tuple[float, float] = (180.0, 0.0),
    ) -> WCS:
        proj = projection_type.name
        nrow, ncol = shape
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [f'RA---{proj}', f'DEC--{proj}']
        wcs.wcs.crpix = [ncol / 2 + 0.5, nrow / 2 + 0.5]
        wcs.wcs.crval = list(crval)
        wcs.wcs.cdelt = [-res_deg, res_deg]
        wcs.wcs.set()
        return wcs

    @pytest.mark.parametrize(
        'shape, res_deg, crval, ra_range, dec_range',
        [
            ((100, 200), 0.1, (180.0, 0.0), (176.0, 184.0), (-4.0, 4.0)),  # standard patch
            ((50, 50), 0.2, (1.0, 0.0), (0.0, 2.0), (-5.0, 5.0)),  # near RA=0
            ((50, 50), 0.2, (359.0, 0.0), (357.0, 361.0), (-5.0, 5.0)),  # near RA=360
            ((80, 120), 0.15, (45.0, 0.0), (42.0, 48.0), (-4.0, 4.0)),  # non-standard RA
            ((30, 90), 0.5, (270.0, 0.0), (264.0, 276.0), (-10.0, 10.0)),  # wide patch
        ],
    )
    def test_world2pixel_matches_astropy(
        self,
        projection_type: ProjectionType,
        shape: tuple[int, int],
        res_deg: float,
        crval: tuple[float, float],
        ra_range: tuple[float, float],
        dec_range: tuple[float, float],
    ):
        """WCSLandscape.world2pixel must agree with AstropyWCSLandscape to float precision."""
        wcs = self._make_wcs(shape, projection_type, res_deg=res_deg, crval=crval)
        landscape = WCSLandscape.from_wcs(shape, wcs, stokes='I')
        wcs_landscape = AstropyWCSLandscape(shape, wcs, stokes='I')

        rng = np.random.default_rng(0)
        dec = rng.uniform(*dec_range, 50)
        ra = rng.uniform(*ra_range, 50)
        theta = jnp.array(np.pi / 2 - np.radians(dec))
        phi = jnp.array(np.radians(ra))

        x, y = landscape.world2pixel(theta, phi)
        ref_x, ref_y = wcs_landscape.world2pixel(theta, phi)

        assert_array_almost_equal(x, ref_x, decimal=12)
        assert_array_almost_equal(y, ref_y, decimal=12)

    def test_landscape_matches_pixell(self, projection_type: ProjectionType):
        """WCSLandscape.world2pixel must agree with pixell's sky2pix to float precision."""
        pytest.importorskip('pixell')
        import pixell.enmap

        shape, wcs = pixell.enmap.geometry(
            pos=np.radians([[-5, -5], [5, 5]]),
            res=np.radians(0.1),
            proj=projection_type.name.lower(),
        )
        landscape = WCSLandscape.from_wcs(shape, wcs, stokes='I')

        rng = np.random.default_rng(42)
        dec = rng.uniform(-4.0, 4.0, 50)
        ra = rng.uniform(-4.0, 4.0, 50)
        theta = jnp.array(np.pi / 2 - np.radians(dec))
        phi = jnp.array(np.radians(ra))

        x, y = landscape.world2pixel(theta, phi)

        # pixell convention: coords=[dec, ra] in radians → returns [pix_y, pix_x]
        ref_y, ref_x = pixell.enmap.sky2pix(shape, wcs, np.array([np.radians(dec), np.radians(ra)]))

        assert_array_almost_equal(x, ref_x, decimal=12)
        assert_array_almost_equal(y, ref_y, decimal=12)

    def test_lon_wrapping(self, projection_type: ProjectionType):
        """Pixels near RA 0°/360° boundary should wrap correctly."""
        shape = (10, 360)
        proj = projection_type.name
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [f'RA---{proj}', f'DEC--{proj}']
        wcs.wcs.crpix = [180.5, 5.5]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [-1.0, 1.0]
        wcs.wcs.set()
        landscape = WCSLandscape.from_wcs(shape, wcs, stokes='I')

        # RA 359° and RA -1° (= 359°) should give the same pixel
        theta = jnp.array([np.pi / 2, np.pi / 2])
        phi_pos = jnp.array([np.radians(359.0), np.radians(-1.0)])

        x0, y0 = landscape.world2pixel(theta[0:1], phi_pos[0:1])
        x1, y1 = landscape.world2pixel(theta[1:2], phi_pos[1:2])
        assert_array_almost_equal(x0, x1, decimal=12)
        assert_array_almost_equal(y0, y1, decimal=12)
