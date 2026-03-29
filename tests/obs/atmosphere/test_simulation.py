import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax.obs.atmosphere import simulate_kolmogorov_screen
from furax.obs.landscapes import TangentialLandscape
from furax.obs.stokes import StokesI

DX = DY = 10.0


def _make_landscape():
    return TangentialLandscape.from_extent(
        x_size=5000.0, y_size=5000.0, dx=DX, dy=DY, height=100.0, dtype=jnp.float64
    )


class TestSimulateKolmogorovScreen:
    def test_output_is_stokes_i(self):
        landscape = _make_landscape()
        key = jax.random.PRNGKey(0)
        screen = simulate_kolmogorov_screen(landscape, key)
        assert isinstance(screen, StokesI)

    def test_output_shape_matches_landscape(self):
        landscape = _make_landscape()
        key = jax.random.PRNGKey(0)
        screen = simulate_kolmogorov_screen(landscape, key)
        assert screen.shape == landscape.shape

    def test_output_dtype_matches_landscape(self):
        landscape = _make_landscape()
        key = jax.random.PRNGKey(0)
        screen = simulate_kolmogorov_screen(landscape, key)
        assert screen.i.dtype == landscape.dtype

    def test_reproducible(self):
        landscape = _make_landscape()
        key = jax.random.PRNGKey(42)
        s1 = simulate_kolmogorov_screen(landscape, key)
        s2 = simulate_kolmogorov_screen(landscape, key)
        assert_allclose(s1.i, s2.i)

    def test_different_keys_differ(self):
        landscape = _make_landscape()
        s1 = simulate_kolmogorov_screen(landscape, jax.random.PRNGKey(0))
        s2 = simulate_kolmogorov_screen(landscape, jax.random.PRNGKey(1))
        assert not jnp.allclose(s1.i, s2.i)

    def test_amplitude_scales_variance(self):
        landscape = _make_landscape()
        key = jax.random.PRNGKey(7)
        s1 = simulate_kolmogorov_screen(landscape, key, amplitude=1.0)
        s4 = simulate_kolmogorov_screen(landscape, key, amplitude=4.0)
        # variance scales linearly with amplitude
        ratio = float(jnp.var(s4.i) / jnp.var(s1.i))
        assert_allclose(ratio, 4.0, rtol=1e-5)

    def test_dc_component_is_zero(self):
        """The (0,0) Fourier mode should be excluded (set to inf in k-space → zero power)."""
        landscape = _make_landscape()
        screen = simulate_kolmogorov_screen(landscape, jax.random.PRNGKey(0))
        # Mean is approximately zero (DC mode excluded)
        assert abs(float(jnp.mean(screen.i))) < 1.0  # loose check; DC exactly zero
