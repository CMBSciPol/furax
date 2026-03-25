import jax
import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal

from furax.mapmaking.atmosphere import AtmospherePointingOperator
from furax.math.quaternion import qmul
from furax.obs.landscapes import TangentialLandscape
from furax.obs.stokes import StokesI

# Small problem size for fast tests
NDET = 4
NSAMP = 20
HEIGHT = 100.0
DX = DY = 10.0


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


WIND_VELOCITY = jnp.array([1.0, 0.5])


def _make_operator(
    wind_velocity=None,
    times=None,
    ndet=NDET,
    nsamp=NSAMP,
    chunk_size=2,
    seed=0,
) -> tuple[AtmospherePointingOperator, TangentialLandscape, StokesI]:
    """Return (operator, landscape, atm_map) for use in tests."""
    if wind_velocity is None:
        wind_velocity = WIND_VELOCITY
    if times is None:
        times = jnp.arange(nsamp, dtype=jnp.float64)

    # Map large enough to cover the pointing + wind drift
    landscape = TangentialLandscape.from_extent(
        x_size=5000.0, y_size=5000.0, dx=DX, dy=DY, height=HEIGHT, stokes='I'
    )

    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    # Small zenith angles so all detectors land inside the map
    qbore = _random_unit_quats(k1, (nsamp,))
    qbore = qbore.at[:, 1:3].multiply(0.05)  # keep mostly near zenith
    qbore = qbore / jnp.linalg.norm(qbore, axis=-1, keepdims=True)

    qdet = _random_unit_quats(k2, (ndet,))
    qdet = qdet.at[:, 1:3].multiply(0.01)
    qdet = qdet / jnp.linalg.norm(qdet, axis=-1, keepdims=True)

    atm_map = landscape.normal(k3)

    op = AtmospherePointingOperator.from_wind(
        landscape, qbore, qdet, wind_velocity, times, chunk_size=chunk_size
    )
    return op, landscape, atm_map


class TestAtmosphereOperatorMv:
    def test_output_shape(self) -> None:
        op, _, atm = _make_operator()
        tod = op(atm)
        assert isinstance(tod, StokesI)
        assert tod.shape == (NDET, NSAMP)

    def test_no_wind_matches_direct_indexing(self) -> None:
        """With zero wind, mv must equal direct nearest-neighbour indexing."""
        op, landscape, atm = _make_operator(wind_velocity=jnp.zeros(2))

        tod = op(atm)

        flat = atm.ravel()
        for d in range(NDET):
            qdet_full = qmul(op.qbore, op.qdet[d : d + 1, None, :])  # (1, samp, 4)
            x, y = landscape.quat2xy(qdet_full[0])  # (samp,)
            idx = landscape.pixel2index(*landscape.xy2pixel(x, y))
            expected = flat.i[idx]
            assert_array_almost_equal(tod.i[d], expected)

    def test_wind_shifts_sample_positions(self) -> None:
        """With non-zero wind, samples must come from wind-displaced positions."""
        wind_velocity = WIND_VELOCITY
        times = jnp.arange(NSAMP, dtype=jnp.float64)
        op, landscape, atm = _make_operator(wind_velocity=wind_velocity, times=times)

        tod = op(atm)

        flat = atm.ravel()
        for d in range(NDET):
            qdet_full = qmul(op.qbore, op.qdet[d : d + 1, None, :])  # (1, samp, 4)
            x, y = landscape.quat2xy(qdet_full[0])  # (samp,)
            x_shifted = x + times * wind_velocity[0]
            y_shifted = y + times * wind_velocity[1]
            idx = landscape.pixel2index(*landscape.xy2pixel(x_shifted, y_shifted))
            expected = flat.i[idx]
            assert_array_almost_equal(tod.i[d], expected)


class TestAtmosphereTranspose:
    def test_adjoint(self) -> None:
        """<A x, y> == <x, A^T y> to numerical precision."""
        op, _, atm = _make_operator()
        key = jax.random.PRNGKey(7)
        tod_rand = StokesI(i=jax.random.normal(key, (NDET, NSAMP)))

        tod = op(atm)
        atm_back = op.T(tod_rand)

        # <A @ atm, tod_rand>
        lhs = float(jnp.sum(tod.i * tod_rand.i))
        # <atm, A^T @ tod_rand>
        rhs = float(jnp.sum(atm.i * atm_back.i))

        assert abs(lhs - rhs) / (abs(lhs) + 1e-12) < 1e-10

    def test_transpose_output_shape(self) -> None:
        op, landscape, _ = _make_operator()
        key = jax.random.PRNGKey(3)
        tod = StokesI(i=jax.random.normal(key, (NDET, NSAMP)))
        atm_back = op.T(tod)
        assert isinstance(atm_back, StokesI)
        assert atm_back.shape == landscape.shape
