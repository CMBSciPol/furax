"""Tests for BeamOperator, BeamOperatorIQU, BeamRule, and BeamIQURule."""

import jax
import jax.numpy as jnp
import pytest

from furax.core._base import CompositionOperator
from furax.math.sht import Alm2Map, Map2Alm
from furax.obs.operators._beam_operator import (
    BeamIQURule,
    BeamOperator,
    BeamOperatorIQU,
    BeamRule,
)
from furax.obs.stokes import StokesIQU

jax.config.update('jax_enable_x64', True)

NSIDE = 16
LMAX = 2 * NSIDE - 1  # 31
NPIX = 12 * NSIDE**2  # 3072
NFREQ = 2


@pytest.fixture(scope='module')
def map_structure():
    """StokesIQU structure for (NFREQ, NPIX) maps."""
    return StokesIQU.structure_for((NFREQ, NPIX), jnp.float64)


@pytest.fixture(scope='module')
def flat_beam_fl():
    """Flat (all-ones) beam transfer function, shape (NFREQ, LMAX+1)."""
    return jnp.ones((NFREQ, LMAX + 1))


@pytest.fixture(scope='module')
def half_beam_fl():
    """Uniform half-power beam transfer function, shape (NFREQ, LMAX+1).

    Well-conditioned for deconvolution: inverse simply doubles all modes.
    """
    return jnp.full((NFREQ, LMAX + 1), 0.5)


@pytest.fixture(scope='module')
def beam_op(flat_beam_fl, map_structure):
    """BeamOperator with a flat (identity) beam."""
    return BeamOperator(lmax=LMAX, beam_fl=flat_beam_fl, in_structure=map_structure)


@pytest.fixture(scope='module')
def half_beam_op(half_beam_fl, map_structure):
    """BeamOperator with a uniform half-power beam."""
    return BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)


@pytest.fixture(scope='module')
def beam_iqu_op(flat_beam_fl, map_structure):
    """BeamOperatorIQU with identical flat beams for all Stokes components."""
    beam_fl_iqu = StokesIQU(i=flat_beam_fl, q=flat_beam_fl, u=flat_beam_fl)
    return BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl_iqu, in_structure=map_structure)


@pytest.fixture(scope='module')
def random_maps():
    """Random StokesIQU map with shape (NFREQ, NPIX)."""
    key = jax.random.PRNGKey(42)
    return StokesIQU(
        i=jax.random.normal(key, (NFREQ, NPIX)),
        q=jax.random.normal(jax.random.fold_in(key, 1), (NFREQ, NPIX)),
        u=jax.random.normal(jax.random.fold_in(key, 2), (NFREQ, NPIX)),
    )


@pytest.fixture(scope='module')
def sht_roundtrip_maps(map_structure, random_maps):
    """Maps projected through Map2Alm → Alm2Map (band-limited version of random_maps).

    Used as reference for tests where the identity beam should be a no-op.
    The identity beam computes alm2map(map2alm(x)); comparing its output to
    this reference avoids sensitivity to the SHT projection accuracy.
    """
    m2a = Map2Alm(lmax=LMAX, nside=NSIDE, in_structure=map_structure)
    alms = m2a(random_maps)
    return Alm2Map(lmax=LMAX, nside=NSIDE, in_structure=m2a.out_structure)(alms)


class TestBeamOperator:
    """Tests for BeamOperator."""

    def test_output_shape(self, beam_op, random_maps):
        """Output leaves must have the same shape as the input leaves."""
        smoothed = beam_op(random_maps)
        assert isinstance(smoothed, StokesIQU)
        for leaf in jax.tree.leaves(smoothed):
            assert leaf.shape == (NFREQ, NPIX)

    def test_output_dtype_is_float(self, beam_op, random_maps):
        """Output leaves must be real-valued (pixel maps are real)."""
        smoothed = beam_op(random_maps)
        for leaf in jax.tree.leaves(smoothed):
            assert jnp.issubdtype(leaf.dtype, jnp.floating)

    def test_identity_beam_matches_sht_roundtrip(self, beam_op, random_maps, sht_roundtrip_maps):
        """An identity beam (fl=1) must match alm2map(map2alm(x)) to machine precision.

        The beam operator internally computes alm2map(almxfl(map2alm(x), 1)) which
        is identical to the SHT round-trip reference; they should agree exactly.
        """
        smoothed = beam_op(random_maps)
        for ref, out in zip(jax.tree.leaves(sht_roundtrip_maps), jax.tree.leaves(smoothed)):
            assert jnp.allclose(ref, out, atol=1e-12), (
                f'Identity beam deviates from SHT round-trip by {float(jnp.max(jnp.abs(out - ref)))}'
            )

    def test_1d_beam_fl_is_broadcast(self, map_structure, random_maps, sht_roundtrip_maps):
        """A 1-D beam_fl (shape lmax+1) must broadcast to match the nfreq axis.

        jnp.atleast_2d promotes the 1-D input to (1, L); the operator then
        broadcasts it to (nfreq, L) before scanning over frequencies.
        """
        fl_1d = jnp.ones(LMAX + 1)
        fl_2d = jnp.ones((NFREQ, LMAX + 1))
        op_1d = BeamOperator(lmax=LMAX, beam_fl=fl_1d, in_structure=map_structure)
        op_2d = BeamOperator(lmax=LMAX, beam_fl=fl_2d, in_structure=map_structure)
        out_1d = op_1d(random_maps)
        out_2d = op_2d(random_maps)
        for leaf_1d, leaf_2d in zip(jax.tree.leaves(out_1d), jax.tree.leaves(out_2d)):
            assert jnp.allclose(leaf_1d, leaf_2d, atol=1e-12)

    def test_symmetric_operator_is_self_adjoint(self, beam_op, random_maps):
        """BeamOperator is symmetric, so B and B.T must produce identical outputs."""
        x = random_maps
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.allclose(a, b, atol=1e-12),
                beam_op(x),
                beam_op.T(x),
            )
        )

    def test_inverse_beam_fl_is_reciprocal(self, half_beam_fl, map_structure):
        """Inverse operator must have beam_fl equal to 1 / original beam_fl."""
        op = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        inv_op = op.I
        assert isinstance(inv_op, BeamOperator)
        assert jnp.allclose(inv_op.beam_fl, 1.0 / half_beam_fl)

    def test_inverse_beam_rule_gives_unity_fl(self, half_beam_fl, map_structure):
        """B.I @ B must reduce (via BeamRule) to a BeamOperator with fl = ones."""
        op = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        reduced = (op.I @ op).reduce()
        assert isinstance(reduced, BeamOperator)
        assert jnp.allclose(reduced.beam_fl, jnp.ones_like(reduced.beam_fl))

    def test_inverse_returns_beam_operator(self, beam_op):
        """BeamOperator.I must return a BeamOperator instance."""
        assert isinstance(beam_op.I, BeamOperator)

    def test_inverse_preserves_in_structure(self, beam_op):
        """Inverse operator's in_structure must equal the original's in_structure."""
        assert beam_op.I.in_structure == beam_op.in_structure

    def test_inverse_preserves_lmax(self, beam_op):
        """Inverse operator must carry the same lmax."""
        assert beam_op.I.lmax == beam_op.lmax


class TestBeamOperatorIQU:
    """Tests for BeamOperatorIQU."""

    def test_output_shape(self, beam_iqu_op, random_maps):
        """Output leaves must have the same shape as the input leaves."""
        smoothed = beam_iqu_op(random_maps)
        assert isinstance(smoothed, StokesIQU)
        for leaf in jax.tree.leaves(smoothed):
            assert leaf.shape == (NFREQ, NPIX)

    def test_output_dtype_is_float(self, beam_iqu_op, random_maps):
        """Output leaves must be real-valued."""
        smoothed = beam_iqu_op(random_maps)
        for leaf in jax.tree.leaves(smoothed):
            assert jnp.issubdtype(leaf.dtype, jnp.floating)

    def test_identity_beam_matches_sht_roundtrip(
        self, beam_iqu_op, random_maps, sht_roundtrip_maps
    ):
        """Flat per-Stokes beams must produce output identical to the SHT round-trip."""
        smoothed = beam_iqu_op(random_maps)
        for ref, out in zip(jax.tree.leaves(sht_roundtrip_maps), jax.tree.leaves(smoothed)):
            assert jnp.allclose(ref, out, atol=1e-12), (
                f'Identity beam deviates from SHT round-trip by {float(jnp.max(jnp.abs(out - ref)))}'
            )

    def test_symmetric_operator_is_self_adjoint(self, beam_iqu_op, random_maps):
        """BeamOperatorIQU is symmetric, so B and B.T must produce identical outputs."""
        x = random_maps
        assert jax.tree.all(
            jax.tree.map(
                lambda a, b: jnp.allclose(a, b, atol=1e-12),
                beam_iqu_op(x),
                beam_iqu_op.T(x),
            )
        )

    def test_inverse_beam_fl_is_reciprocal(self, map_structure):
        """Inverse operator must have per-Stokes beam_fl leaves equal to 1 / originals."""
        fl = jnp.full((NFREQ, LMAX + 1), 2.0)
        beam_fl = StokesIQU(i=fl, q=fl * 0.5, u=fl * 0.25)
        op = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl, in_structure=map_structure)
        inv_op = op.I
        assert isinstance(inv_op, BeamOperatorIQU)
        for orig_leaf, inv_leaf in zip(
            jax.tree.leaves(op.beam_fl), jax.tree.leaves(inv_op.beam_fl)
        ):
            assert jnp.allclose(inv_leaf, 1.0 / orig_leaf)

    def test_inverse_returns_beam_operatoriqu(self, beam_iqu_op):
        """BeamOperatorIQU.I must return a BeamOperatorIQU instance."""
        assert isinstance(beam_iqu_op.I, BeamOperatorIQU)

    def test_agrees_with_beam_operator_when_beams_are_equal(
        self, map_structure, random_maps, flat_beam_fl
    ):
        """BeamOperatorIQU with equal per-Stokes beams must match BeamOperator output."""
        op_shared = BeamOperator(lmax=LMAX, beam_fl=flat_beam_fl, in_structure=map_structure)
        beam_fl_iqu = StokesIQU(i=flat_beam_fl, q=flat_beam_fl, u=flat_beam_fl)
        op_iqu = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl_iqu, in_structure=map_structure)
        out_shared = op_shared(random_maps)
        out_iqu = op_iqu(random_maps)
        for a, b in zip(jax.tree.leaves(out_shared), jax.tree.leaves(out_iqu)):
            assert jnp.allclose(a, b, atol=1e-12)


class TestBeamRule:
    """Tests for the BeamRule algebraic simplification."""

    def test_beam_op_at_beam_op_reduces_to_beam_op(self, half_beam_op):
        """(BeamOperator @ BeamOperator).reduce() must yield a BeamOperator."""
        comp = half_beam_op @ half_beam_op
        assert isinstance(comp, CompositionOperator)
        reduced = comp.reduce()
        assert isinstance(reduced, BeamOperator)

    def test_combined_beam_fl_is_product(self, half_beam_fl, map_structure):
        """Reduced beam_fl must equal the element-wise product of both beam_fls."""
        op1 = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        op2 = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl * 0.5, in_structure=map_structure)
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, BeamOperator)
        expected_fl = half_beam_fl * (half_beam_fl * 0.5)
        assert jnp.allclose(reduced.beam_fl, expected_fl)

    def test_no_reduction_for_mismatched_lmax(self, flat_beam_fl, map_structure):
        """Two BeamOperators with different lmax must not reduce."""
        lmax2 = LMAX - 2
        op1 = BeamOperator(lmax=LMAX, beam_fl=flat_beam_fl, in_structure=map_structure)
        op2 = BeamOperator(
            lmax=lmax2, beam_fl=jnp.ones((NFREQ, lmax2 + 1)), in_structure=map_structure
        )
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, CompositionOperator)

    def test_reduced_in_structure_matches_right_operator(self, half_beam_fl, map_structure):
        """Reduced operator's in_structure must equal the right operand's in_structure."""
        op1 = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        op2 = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, BeamOperator)
        assert reduced.in_structure == op2.in_structure

    def test_inverse_beam_rule_unity_fl(self, half_beam_fl, map_structure):
        """B.I @ B must reduce to a BeamOperator with all-ones beam_fl."""
        op = BeamOperator(lmax=LMAX, beam_fl=half_beam_fl, in_structure=map_structure)
        reduced = (op.I @ op).reduce()
        assert isinstance(reduced, BeamOperator)
        assert jnp.allclose(reduced.beam_fl, jnp.ones_like(reduced.beam_fl))


class TestBeamIQURule:
    """Tests for the BeamIQURule algebraic simplification."""

    def test_beam_iqu_at_beam_iqu_reduces_to_beam_iqu(self, beam_iqu_op):
        """(BeamOperatorIQU @ BeamOperatorIQU).reduce() must yield a BeamOperatorIQU."""
        comp = beam_iqu_op @ beam_iqu_op
        assert isinstance(comp, CompositionOperator)
        reduced = comp.reduce()
        assert isinstance(reduced, BeamOperatorIQU)

    def test_combined_beam_fl_is_product_per_stokes(self, flat_beam_fl, map_structure):
        """Reduced per-Stokes beam_fl leaves must be element-wise products."""
        fl_half = flat_beam_fl * 0.5
        beam_fl1 = StokesIQU(i=flat_beam_fl, q=flat_beam_fl, u=flat_beam_fl)
        beam_fl2 = StokesIQU(i=fl_half, q=fl_half, u=fl_half)
        op1 = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl1, in_structure=map_structure)
        op2 = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl2, in_structure=map_structure)
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, BeamOperatorIQU)
        for orig1, orig2, combined in zip(
            jax.tree.leaves(beam_fl1),
            jax.tree.leaves(beam_fl2),
            jax.tree.leaves(reduced.beam_fl),
        ):
            assert jnp.allclose(combined, orig1 * orig2)

    def test_no_reduction_for_mismatched_lmax(self, flat_beam_fl, map_structure):
        """Two BeamOperatorIQU with different lmax must not reduce."""
        lmax2 = LMAX - 2
        fl2 = jnp.ones((NFREQ, lmax2 + 1))
        beam_fl1 = StokesIQU(i=flat_beam_fl, q=flat_beam_fl, u=flat_beam_fl)
        beam_fl2 = StokesIQU(i=fl2, q=fl2, u=fl2)
        op1 = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl1, in_structure=map_structure)
        op2 = BeamOperatorIQU(lmax=lmax2, beam_fl=beam_fl2, in_structure=map_structure)
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, CompositionOperator)

    def test_reduced_in_structure_matches_right_operator(self, flat_beam_fl, map_structure):
        """Reduced operator's in_structure must equal the right operand's in_structure."""
        beam_fl = StokesIQU(i=flat_beam_fl, q=flat_beam_fl, u=flat_beam_fl)
        op1 = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl, in_structure=map_structure)
        op2 = BeamOperatorIQU(lmax=LMAX, beam_fl=beam_fl, in_structure=map_structure)
        reduced = (op1 @ op2).reduce()
        assert isinstance(reduced, BeamOperatorIQU)
        assert reduced.in_structure == op2.in_structure
