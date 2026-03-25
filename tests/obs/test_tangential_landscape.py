import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from furax.math.quaternion import from_iso_angles
from furax.obs.landscapes import TangentialLandscape

# A small landscape used across most tests:
#   shape (4, 3) → n_y=4, n_x=3  →  pixel_shape = (3, 4)
#   dx=dy=100, height=1000
#   x covered: [-150, 150],  pixel centers at x = -100, 0, 100
#   y covered: [-200, 200],  pixel centers at y = -150, -50, 50, 150
SHAPE = (4, 3)
DX = 100.0
DY = 100.0
HEIGHT = 1000.0


@pytest.fixture
def landscape() -> TangentialLandscape:
    return TangentialLandscape(SHAPE, DX, DY, HEIGHT, stokes='I')


def test_non_i_stokes_raises() -> None:
    with pytest.raises(NotImplementedError):
        TangentialLandscape(SHAPE, DX, DY, HEIGHT, stokes='IQU')


class TestFromExtent:
    def test_shape_from_exact_multiple(self) -> None:
        """from_extent with exact multiples produces the expected shape."""
        ls = TangentialLandscape.from_extent(300.0, 400.0, 100.0, 100.0, HEIGHT)
        assert ls.shape == (4, 3)
        assert ls.dx == 100.0
        assert ls.dy == 100.0

    def test_shape_rounds_up(self) -> None:
        """from_extent rounds up to whole pixels when size is not an exact multiple."""
        ls = TangentialLandscape.from_extent(250.0, 350.0, 100.0, 100.0, HEIGHT)
        assert ls.shape == (4, 3)  # ceil(250/100)=3, ceil(350/100)=4

    def test_edges_cover_requested_extent(self) -> None:
        """The actual map extent must be >= the requested size."""
        x_size, y_size = 300.0, 400.0
        ls = TangentialLandscape.from_extent(x_size, y_size, 100.0, 100.0, HEIGHT)
        n_x, n_y = ls.pixel_shape
        x_extent, y_extent = ls.extent
        assert x_extent >= x_size
        assert y_extent >= y_size


class TestXy2Pixel:
    def test_origin_maps_to_map_center(self, landscape: TangentialLandscape) -> None:
        """(x, y) = (0, 0) must map to the center of the pixel grid."""
        # n_x=3: center between pix 1 and 2 → pix_x = 1.0
        # n_y=4: center between pix 1 and 2 → pix_y = 1.5
        pix_x, pix_y = landscape.xy2pixel(jnp.array(0.0), jnp.array(0.0))
        assert_array_almost_equal(pix_x, 1.0)
        assert_array_almost_equal(pix_y, 1.5)

    def test_pixel_centers(self, landscape: TangentialLandscape) -> None:
        """Pixel centers must map to integer pixel coordinates."""
        # pixel centers: x = -100, 0, 100  →  pix_x = 0, 1, 2
        xs = jnp.array([-100.0, 0.0, 100.0])
        ys = jnp.array([-150.0, -50.0, 50.0])  # pix_y = 0, 1, 2
        pix_x, pix_y = landscape.xy2pixel(xs, ys)
        assert_array_almost_equal(pix_x, [0.0, 1.0, 2.0])
        assert_array_almost_equal(pix_y, [0.0, 1.0, 2.0])

    def test_map_edges(self, landscape: TangentialLandscape) -> None:
        """Left/bottom map edge maps to -0.5; right/top edge maps to n - 0.5."""
        # x_min = -n_x/2 * dx = -150, x_max = +150
        pix_x_min, _ = landscape.xy2pixel(jnp.array(-150.0), jnp.array(0.0))
        pix_x_max, _ = landscape.xy2pixel(jnp.array(150.0), jnp.array(0.0))
        assert_array_almost_equal(pix_x_min, -0.5)
        assert_array_almost_equal(pix_x_max, 2.5)


class TestOffsetCenter:
    """Tests for the x0, y0 center offset."""

    def test_center_maps_to_map_center(self) -> None:
        """(x0, y0) must map to the geometric center of the pixel grid."""
        ls = TangentialLandscape(SHAPE, DX, DY, HEIGHT, x0=500.0, y0=-300.0)
        pix_x, pix_y = ls.xy2pixel(jnp.array(500.0), jnp.array(-300.0))
        assert_array_almost_equal(pix_x, 1.0)  # same as (0, 0) on centred map
        assert_array_almost_equal(pix_y, 1.5)

    def test_pixel_centers_shifted(self) -> None:
        """Pixel centers must be offset by (x0, y0) relative to the default map."""
        x0, y0 = 500.0, -300.0
        ls = TangentialLandscape(SHAPE, DX, DY, HEIGHT, x0=x0, y0=y0)
        # pixel centers are at x0 + {-100, 0, 100}
        xs = jnp.array([x0 - 100.0, x0, x0 + 100.0])
        ys = jnp.array([y0 - 150.0, y0 - 50.0, y0 + 50.0])
        pix_x, pix_y = ls.xy2pixel(xs, ys)
        assert_array_almost_equal(pix_x, [0.0, 1.0, 2.0])
        assert_array_almost_equal(pix_y, [0.0, 1.0, 2.0])


class TestWorld2Pixel:
    def test_consistency_with_xy2pixel(self, landscape: TangentialLandscape) -> None:
        """world2pixel must agree with the direct gnomonic formula via xy2pixel."""
        rng = np.random.default_rng(42)
        theta = jnp.array(rng.uniform(0.01, 0.12, 30))
        phi = jnp.array(rng.uniform(0.0, 2 * np.pi, 30))

        px_w, py_w = landscape.world2pixel(theta, phi)

        x = HEIGHT * jnp.sin(theta) * jnp.cos(phi) / jnp.cos(theta)
        y = HEIGHT * jnp.sin(theta) * jnp.sin(phi) / jnp.cos(theta)
        px_ref, py_ref = landscape.xy2pixel(x, y)

        assert_array_almost_equal(px_w, px_ref)
        assert_array_almost_equal(py_w, py_ref)


class TestQuat2Pixel:
    def test_consistency_with_world2pixel(self, landscape: TangentialLandscape) -> None:
        """quat2pixel must agree with world2pixel for the same direction."""
        rng = np.random.default_rng(7)
        theta = jnp.array(rng.uniform(0.01, 0.12, 30))
        phi = jnp.array(rng.uniform(0.0, 2 * np.pi, 30))
        quats = from_iso_angles(theta, phi, jnp.zeros(30))

        px_q, py_q = landscape.quat2pixel(quats)
        px_w, py_w = landscape.world2pixel(theta, phi)

        assert_array_almost_equal(px_q, px_w)
        assert_array_almost_equal(py_q, py_w)

    def test_zenith_pointing_maps_to_origin(self, landscape: TangentialLandscape) -> None:
        """Identity quaternion (pointing straight up) maps to (x, y) = (0, 0)."""
        quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        x, y = landscape.quat2xy(quat)
        assert_array_almost_equal(x, 0.0)
        assert_array_almost_equal(y, 0.0)
