import jax
import jax.numpy as jnp
import pytest

from furax.obs.coords import ZAXIS, AzElAngles, azel_to_vec, tangent_basis, vec_to_azel


@pytest.mark.parametrize(
    'az, el, expected',
    [
        (0.0, 0.0, [1.0, 0.0, 0.0]),  # North
        (jnp.pi / 2, 0.0, [0.0, -1.0, 0.0]),  # East
        (jnp.pi, 0.0, [-1.0, 0.0, 0.0]),  # South
        (-jnp.pi / 2, 0.0, [0.0, 1.0, 0.0]),  # West
        (0.3, jnp.pi / 2, [0.0, 0.0, 1.0]),  # Zenith
    ],
)
def test_cardinal_directions(az, el, expected):
    assert jnp.allclose(azel_to_vec(az, el), jnp.array(expected), atol=1e-12)
    boresight = AzElAngles(az, el, 0.7).to_quaternion()
    assert jnp.allclose(boresight.rotate_vector(ZAXIS), jnp.array(expected), atol=1e-12)


def test_azel_vec_roundtrip():
    az = jnp.linspace(-3.0, 3.0, 7)
    el = jnp.linspace(-1.5, 1.5, 7)
    az_out, el_out = vec_to_azel(azel_to_vec(az, el))
    assert jnp.allclose(az_out, az, atol=1e-12)
    assert jnp.allclose(el_out, el, atol=1e-12)


def test_vec_to_azel_unnormalized():
    az_out, el_out = vec_to_azel(3.0 * azel_to_vec(0.4, 0.2))
    assert jnp.allclose(az_out, 0.4) and jnp.allclose(el_out, 0.2)


def test_tangent_basis():
    az, el = jnp.meshgrid(jnp.linspace(-3.0, 3.0, 5), jnp.linspace(-1.4, 1.4, 5))
    v = azel_to_vec(az, el)
    e_xel, e_el = tangent_basis(az, el)
    basis = jnp.stack([e_xel, e_el, v], axis=-2)
    gram = basis @ jnp.swapaxes(basis, -1, -2)
    assert jnp.allclose(gram, jnp.eye(3), atol=1e-12)
    assert jnp.allclose(jnp.cross(e_el, e_xel), v, atol=1e-12)

    # the basis vectors are the directions of increasing azimuth and elevation
    d_az = jax.vmap(jax.vmap(jax.jacfwd(azel_to_vec, argnums=0)))(az, el)
    d_el = jax.vmap(jax.vmap(jax.jacfwd(azel_to_vec, argnums=1)))(az, el)
    assert jnp.allclose(d_az, jnp.cos(el)[..., None] * e_xel, atol=1e-12)
    assert jnp.allclose(d_el, e_el, atol=1e-12)


def test_tangent_basis_broadcasts():
    e_xel, e_el = tangent_basis(jnp.zeros(3), 0.5)
    assert e_xel.shape == e_el.shape == (3, 3)
