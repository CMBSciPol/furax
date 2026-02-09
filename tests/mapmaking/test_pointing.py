import jax
import jax.numpy as jnp
import numpy as np

from furax.mapmaking.pointing import PointingOperator
from furax.obs.landscapes import HealpixLandscape, LocalStokesLandscape
from furax.obs.stokes import Stokes, StokesI


def _make_pointing_operator(landscape, ndet=2, nsamp=5):
    """Helper to build a PointingOperator with simple deterministic quaternions."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    # Random unit quaternions for boresight (nsamp, 4) and detectors (ndet, 4)
    qbore = jax.random.normal(k1, (nsamp, 4))
    qbore = qbore / jnp.linalg.norm(qbore, axis=-1, keepdims=True)
    qdet = jax.random.normal(k2, (ndet, 4))
    qdet = qdet / jnp.linalg.norm(qdet, axis=-1, keepdims=True)
    stokes_cls = Stokes.class_for(landscape.stokes)
    return PointingOperator(
        landscape=landscape,
        qbore=qbore,
        qdet=qdet,
        _in_structure=landscape.structure,
        _out_structure=stokes_cls.structure_for((ndet, nsamp), dtype=landscape.dtype),
        chunk_size=0,
    )


class TestPointingWithLocalLandscape:
    """Verify PointingOperator works with LocalStokesLandscape as a drop-in."""

    def test_forward_stokes_i(self):
        """Forward (map->tod) with StokesI using the full pixel set should match parent."""
        nside = 4
        parent = HealpixLandscape(nside=nside, stokes='I')
        # Use all pixels so results should be identical
        all_indices = np.arange(12 * nside**2)
        local = LocalStokesLandscape(parent, all_indices)

        P_parent = _make_pointing_operator(parent)
        P_local = _make_pointing_operator(local)

        sky_parent = parent.ones()
        sky_local = local.ones()

        tod_parent = P_parent.mv(sky_parent)
        tod_local = P_local.mv(sky_local)
        np.testing.assert_allclose(tod_local.i, tod_parent.i, atol=1e-12)

    def test_transpose_stokes_i(self):
        """Transpose (tod->map) with StokesI using the full pixel set should match parent."""
        nside = 4
        parent = HealpixLandscape(nside=nside, stokes='I')
        all_indices = np.arange(12 * nside**2)
        local = LocalStokesLandscape(parent, all_indices)

        P_parent = _make_pointing_operator(parent)
        P_local = _make_pointing_operator(local)

        tod = StokesI(jnp.ones((2, 5)))
        map_parent = P_parent.T.mv(tod)
        map_local = P_local.T.mv(tod)
        np.testing.assert_allclose(map_local.i, map_parent.i, atol=1e-12)

    def test_forward_subset(self):
        """Forward with a pixel subset: local sky is smaller, tod should still work."""
        nside = 4
        npix = 12 * nside**2
        parent = HealpixLandscape(nside=nside, stokes='I')
        # Take only even-indexed pixels
        subset = np.arange(0, npix, 2)
        local = LocalStokesLandscape(parent, subset)

        P_local = _make_pointing_operator(local)
        sky_local = local.ones()

        # Should produce a valid tod without errors
        tod = P_local.mv(sky_local)
        assert tod.i.shape == (2, 5)

    def test_transpose_subset(self):
        """Transpose with a pixel subset produces a local-shaped output."""
        nside = 4
        npix = 12 * nside**2
        parent = HealpixLandscape(nside=nside, stokes='I')
        subset = np.arange(0, npix, 2)
        local = LocalStokesLandscape(parent, subset)

        P_local = _make_pointing_operator(local)
        tod = StokesI(jnp.ones((2, 5)))
        sky = P_local.T.mv(tod)
        assert sky.i.shape == local.shape

    def test_forward_iqu(self):
        """Forward with StokesIQU using the full pixel set should match parent."""
        nside = 4
        parent = HealpixLandscape(nside=nside, stokes='IQU')
        all_indices = np.arange(12 * nside**2)
        local = LocalStokesLandscape(parent, all_indices)

        P_parent = _make_pointing_operator(parent)
        P_local = _make_pointing_operator(local)

        sky_parent = parent.ones()
        sky_local = local.ones()

        tod_parent = P_parent.mv(sky_parent)
        tod_local = P_local.mv(sky_local)
        np.testing.assert_allclose(tod_local.i, tod_parent.i, atol=1e-12)
        np.testing.assert_allclose(tod_local.q, tod_parent.q, atol=1e-12)
        np.testing.assert_allclose(tod_local.u, tod_parent.u, atol=1e-12)

    def test_roundtrip_forward_transpose(self):
        """P^T P x should have the correct shape for a local landscape."""
        nside = 4
        npix = 12 * nside**2
        parent = HealpixLandscape(nside=nside, stokes='I')
        subset = np.arange(0, npix, 3)
        local = LocalStokesLandscape(parent, subset)

        P = _make_pointing_operator(local)
        sky = local.ones()
        result = P.T.mv(P.mv(sky))
        assert result.i.shape == local.shape
