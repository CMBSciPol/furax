"""Tests for Map2Alm, Alm2Map, and SHTRule."""

import jax
import jax.numpy as jnp
import pytest

from furax import IdentityOperator
from furax.core._base import CompositionOperator
from furax.math.sht import Alm2Map, Map2Alm
from furax.obs.stokes import StokesIQU

jax.config.update('jax_enable_x64', True)

NSIDE = 16
LMAX = 2 * NSIDE - 1  # 31
NPIX = 12 * NSIDE**2  # 3072
NALM_ROWS = LMAX + 1  # 32  — l-axis of jax_healpy 2-D alm layout
NALM_COLS = 2 * LMAX + 1  # 63  — m-axis (m = -lmax … lmax)
NFREQ = 3


@pytest.fixture(scope='module')
def map_structure():
    """StokesIQU structure for (nfreq, npix) maps."""
    return StokesIQU.structure_for((NFREQ, NPIX), jnp.float64)


@pytest.fixture(scope='module')
def map2alm(map_structure):
    """Map2Alm operator with multi-frequency map input."""
    return Map2Alm(lmax=LMAX, nside=NSIDE, in_structure=map_structure)


@pytest.fixture(scope='module')
def alm_structure(map2alm):
    """StokesIQU structure for (nfreq, lmax+1, 2*lmax+1) alms derived from map2alm."""
    return map2alm.out_structure


@pytest.fixture(scope='module')
def alm2map(alm_structure):
    """Alm2Map operator with multi-frequency alm input."""
    return Alm2Map(lmax=LMAX, nside=NSIDE, in_structure=alm_structure)


@pytest.fixture(scope='module')
def random_maps():
    """Random StokesIQU map with shape (NFREQ, NPIX)."""
    key = jax.random.PRNGKey(0)
    return StokesIQU(
        i=jax.random.normal(key, (NFREQ, NPIX)),
        q=jax.random.normal(jax.random.fold_in(key, 1), (NFREQ, NPIX)),
        u=jax.random.normal(jax.random.fold_in(key, 2), (NFREQ, NPIX)),
    )


class TestMap2Alm:
    """Tests for Map2Alm analysis operator."""

    def test_output_shape(self, map2alm, random_maps):
        """Output leaves should have shape (nfreq, lmax+1, 2*lmax+1)."""
        alms = map2alm(random_maps)
        assert isinstance(alms, StokesIQU)
        for leaf in jax.tree.leaves(alms):
            assert leaf.shape == (NFREQ, NALM_ROWS, NALM_COLS)

    def test_output_dtype_is_complex(self, map2alm, random_maps):
        """Output leaves should be complex (alm coefficients are complex-valued)."""
        alms = map2alm(random_maps)
        for leaf in jax.tree.leaves(alms):
            assert jnp.issubdtype(leaf.dtype, jnp.complexfloating)

    def test_atleast_2d_promotes_1d_input(self):
        """A 1-D map leaf (single frequency) should be promoted and produce (1, lmax+1, 2*lmax+1)."""
        struct = StokesIQU.structure_for((NPIX,), jnp.float64)
        op = Map2Alm(lmax=LMAX, nside=NSIDE, in_structure=struct)
        maps = StokesIQU(
            i=jnp.ones(NPIX),
            q=jnp.ones(NPIX),
            u=jnp.ones(NPIX),
        )
        alms = op(maps)
        for leaf in jax.tree.leaves(alms):
            assert leaf.shape == (1, NALM_ROWS, NALM_COLS)

    def test_transpose_returns_alm2map(self, map2alm):
        """Transpose of Map2Alm should return an Alm2Map instance."""
        assert isinstance(map2alm.T, Alm2Map)

    def test_transpose_in_structure_matches_out_structure(self, map2alm):
        """Transpose operator's in_structure must equal Map2Alm's out_structure."""
        assert map2alm.T.in_structure == map2alm.out_structure

    def test_inverse_returns_alm2map(self, map2alm):
        """Inverse of Map2Alm should return an Alm2Map instance."""
        assert isinstance(map2alm.I, Alm2Map)

    def test_inverse_in_structure_matches_out_structure(self, map2alm):
        """Inverse operator's in_structure must equal Map2Alm's out_structure."""
        assert map2alm.I.in_structure == map2alm.out_structure


class TestAlm2Map:
    """Tests for Alm2Map synthesis operator."""

    def test_output_shape(self, alm2map, map2alm, random_maps):
        """Output leaves should have shape (nfreq, npix)."""
        alms = map2alm(random_maps)
        maps = alm2map(alms)
        assert isinstance(maps, StokesIQU)
        for leaf in jax.tree.leaves(maps):
            assert leaf.shape == (NFREQ, NPIX)

    def test_output_dtype_is_float(self, alm2map, map2alm, random_maps):
        """Output leaves should be real-valued (pixel maps are real)."""
        alms = map2alm(random_maps)
        maps = alm2map(alms)
        for leaf in jax.tree.leaves(maps):
            assert jnp.issubdtype(leaf.dtype, jnp.floating)

    def test_reshape_promotes_2d_single_freq_input(self):
        """A 2-D alm leaf (lmax+1, 2*lmax+1) should be reshaped to (1, lmax+1, 2*lmax+1)."""
        struct = StokesIQU.structure_for((NALM_ROWS, NALM_COLS), jnp.complex128)
        op = Alm2Map(lmax=LMAX, nside=NSIDE, in_structure=struct)
        alms = StokesIQU(
            i=jnp.zeros((NALM_ROWS, NALM_COLS), dtype=jnp.complex128),
            q=jnp.zeros((NALM_ROWS, NALM_COLS), dtype=jnp.complex128),
            u=jnp.zeros((NALM_ROWS, NALM_COLS), dtype=jnp.complex128),
        )
        maps = op(alms)
        for leaf in jax.tree.leaves(maps):
            assert leaf.shape == (1, NPIX)

    def test_transpose_returns_map2alm(self, alm2map):
        """Transpose of Alm2Map should return a Map2Alm instance."""
        assert isinstance(alm2map.T, Map2Alm)

    def test_transpose_in_structure_matches_out_structure(self, alm2map):
        """Transpose operator's in_structure must equal Alm2Map's out_structure."""
        assert alm2map.T.in_structure == alm2map.out_structure

    def test_inverse_returns_map2alm(self, alm2map):
        """Inverse of Alm2Map should return a Map2Alm instance."""
        assert isinstance(alm2map.I, Map2Alm)

    def test_inverse_in_structure_matches_out_structure(self, alm2map):
        """Inverse operator's in_structure must equal Alm2Map's out_structure."""
        assert alm2map.I.in_structure == alm2map.out_structure


class TestSHTRule:
    """Tests for the SHTRule algebraic simplification."""

    def test_map2alm_at_alm2map_reduces_to_identity(self, map2alm, alm2map):
        """(Map2Alm @ Alm2Map).reduce() should yield an IdentityOperator."""
        comp = map2alm @ alm2map
        assert isinstance(comp, CompositionOperator)
        assert isinstance(comp.reduce(), IdentityOperator)

    def test_reduced_identity_has_alm_in_structure(self, map2alm, alm2map):
        """The reduced IdentityOperator must live in alm space (input of Alm2Map)."""
        identity = (map2alm @ alm2map).reduce()
        assert identity.in_structure == alm2map.in_structure

    def test_alm2map_at_map2alm_does_not_reduce(self, map2alm, alm2map):
        """(Alm2Map @ Map2Alm) is a projection and must not simplify to identity."""
        comp = alm2map @ map2alm
        reduced = comp.reduce()
        assert not isinstance(reduced, IdentityOperator)


class TestRoundTrip:
    """Numerical round-trip tests for the SHT operators."""

    def test_alm_round_trip(self, map2alm, alm2map, random_maps):
        """map2alm(alm2map(map2alm(m))) ≈ map2alm(m) within iter=0 accuracy.

        Applies analysis → synthesis → analysis and checks that the second
        analysis recovers the first alm coefficients to within ~1e-2, which
        is the expected accuracy of the iterationless map2alm quadrature.
        """
        alms_first = map2alm(random_maps)
        maps_synth = alm2map(alms_first)
        alms_second = map2alm(maps_synth)

        for leaf_first, leaf_second in zip(
            jax.tree.leaves(alms_first), jax.tree.leaves(alms_second)
        ):
            max_residual = float(jnp.max(jnp.abs(leaf_second - leaf_first)))
            assert max_residual < 1e-2, f'Round-trip residual {max_residual} exceeds 1e-2'
