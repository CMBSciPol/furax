"""Beam operators for applying instrumental transfer functions in spherical harmonic space.

The two core operators are:

- :class:`BeamOperator`: applies a single beam transfer function shared across all
  Stokes components.
- :class:`BeamOperatorIQU`: applies an independent beam transfer function per Stokes
  component.

Both operators round-trip input pixel maps through a spherical harmonic transform
(Map2Alm → per-multipole scaling → Alm2Map).  :class:`BeamRule` and
:class:`BeamIQURule` register algebraic simplifications so that
``BeamOperator @ BeamOperator`` collapses to a single operator whose transfer
function is the product of both beams at construction time.
"""

import jax
import jax.numpy as jnp
import jax_healpy as jhp
from jaxtyping import Array, Float, Inexact, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.core.rules import AbstractBinaryRule, NoReduction
from furax.math.sht import Alm2Map, Map2Alm


def _apply_fl_to_alm_leaf(
    alm_leaf: Float[Array, 'nfreq lp1 alm_cols'],
    fl: Float[Array, 'nfreq lp1'],
) -> Float[Array, 'nfreq lp1 alm_cols']:
    """Apply a per-frequency beam transfer function to a single alm leaf.

    Iterates over the frequency axis with :func:`jax.lax.scan`, calling
    :func:`jax_healpy.almxfl` on each ``(lmax+1, 2*lmax+1)`` slice independently.
    The shared ``nfreq`` and ``lp1`` dimensions are validated by jaxtyping at call
    time, ensuring the frequency and multipole axes of *alm_leaf* and *fl* agree.

    Args:
        alm_leaf: Spherical harmonic coefficients for a single Stokes component,
            shape ``(nfreq, lmax+1, 2*lmax+1)`` (jax-healpy 2-D alm layout).
        fl: Beam transfer function values at each multipole for each frequency,
            shape ``(nfreq, lmax+1)``.

    Returns:
        Beam-filtered alm coefficients with the same shape as *alm_leaf*.
    """

    def scan_fn(_, pair: tuple) -> tuple:
        alm_i, fl_i = pair
        return None, jhp.almxfl(alm_i, fl_i, healpy_ordering=False)

    return jax.lax.scan(scan_fn, None, (alm_leaf, fl))[1]


@symmetric
class BeamOperator(AbstractLinearOperator):
    """Beam operator with a single transfer function shared across all Stokes components.

    Input PyTree leaves are expected to have shape ``(nfreq, npix)``.  They are
    round-tripped through a spherical harmonic transform and the beam is applied
    in alm space as a per-multipole, per-frequency scaling.

    The operator is symmetric (self-adjoint): applying it twice is equivalent to
    squaring the transfer function.  Use :attr:`inverse` to obtain the deconvolution
    operator (``beam_fl`` replaced by its reciprocal).

    Algebraic rule: ``BeamOperator @ BeamOperator`` reduces to a single
    :class:`BeamOperator` whose ``beam_fl`` is the element-wise product of both
    transfer functions (see :class:`BeamRule`).

    Attributes:
        lmax: Maximum spherical harmonic degree.
        beam_fl: Beam transfer function, shape ``(nfreq, lmax+1)`` for a
            per-frequency beam, or ``(lmax+1,)`` for a beam shared across all
            frequencies (promoted to 2-D via :func:`jnp.atleast_2d` before
            broadcasting over frequencies).

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.obs.stokes import StokesIQU
        >>> from furax.obs.operators.beam_operator import BeamOperator
        >>> nside, lmax, nfreq = 16, 31, 2
        >>> npix = 12 * nside ** 2
        >>> structure = StokesIQU.structure_for((nfreq, npix), jnp.float64)
        >>> beam_fl = jnp.ones((nfreq, lmax + 1))
        >>> op = BeamOperator(lmax=lmax, beam_fl=beam_fl, in_structure=structure)
    """

    lmax: int
    beam_fl: Float[Array, 'ab']

    def mv(self, x: PyTree[Inexact[Array, 'nfreq npix']]) -> PyTree[Inexact[Array, 'nfreq npix']]:
        """Apply the beam to input sky maps.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, npix)``.

        Returns:
            Beam-smoothed PyTree with the same structure and leaf shapes as *x*.
        """
        first_leaf = jax.tree_util.tree_leaves(x)[0]
        nside = jhp.npix2nside(first_leaf.shape[-1])

        map2alm = Map2Alm(lmax=self.lmax, nside=nside, in_structure=self.in_structure)
        alm = map2alm.mv(x)

        nfreq = first_leaf.shape[0]
        fl = jnp.broadcast_to(jnp.atleast_2d(self.beam_fl), (nfreq, self.lmax + 1))
        alm_beam = jax.tree.map(lambda alm_leaf: _apply_fl_to_alm_leaf(alm_leaf, fl), alm)

        return Alm2Map(lmax=self.lmax, nside=nside, in_structure=map2alm.out_structure).mv(alm_beam)

    def inverse(self) -> 'BeamOperator':
        """Return a :class:`BeamOperator` with the reciprocal transfer function.

        Returns:
            A new :class:`BeamOperator` whose ``beam_fl`` equals ``1 / self.beam_fl``.
        """
        return BeamOperator(
            lmax=self.lmax, beam_fl=1.0 / self.beam_fl, in_structure=self.in_structure
        )


@symmetric
class BeamOperatorIQU(AbstractLinearOperator):
    """Beam operator with an independent transfer function per Stokes component.

    Input PyTree leaves are expected to have shape ``(nfreq, npix)``.  Each leaf
    is smoothed with its own transfer function taken from the corresponding leaf
    of ``beam_fl``, enabling different beams for I, Q, and U (or any other
    Stokes combination).

    The operator is symmetric (self-adjoint). Use :attr:`inverse` to obtain the
    deconvolution operator (each ``beam_fl`` leaf replaced by its reciprocal).

    Algebraic rule: ``BeamOperatorIQU @ BeamOperatorIQU`` reduces to a single
    operator whose per-leaf transfer functions are the element-wise products of
    both (see :class:`BeamIQURule`).

    Attributes:
        lmax: Maximum spherical harmonic degree.
        beam_fl: PyTree with the same structure as the input maps; each leaf has
            shape ``(nfreq, lmax+1)`` and carries the per-frequency beam transfer
            function for that Stokes component.

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.obs.stokes import StokesIQU
        >>> from furax.obs.operators.beam_operator import BeamOperatorIQU
        >>> nside, lmax, nfreq = 16, 31, 2
        >>> npix = 12 * nside ** 2
        >>> structure = StokesIQU.structure_for((nfreq, npix), jnp.float64)
        >>> fl = jnp.ones((nfreq, lmax + 1))
        >>> beam_fl = StokesIQU(i=fl, q=fl, u=fl)
        >>> op = BeamOperatorIQU(lmax=lmax, beam_fl=beam_fl, in_structure=structure)
    """

    lmax: int
    beam_fl: PyTree[Float[Array, 'ab']]

    def mv(self, x: PyTree[Inexact[Array, 'nfreq npix']]) -> PyTree[Inexact[Array, 'nfreq npix']]:
        """Apply per-Stokes beams to input sky maps.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, npix)``.  The
                PyTree structure must match that of ``self.beam_fl``.

        Returns:
            Beam-smoothed PyTree with the same structure and leaf shapes as *x*.
        """
        first_leaf = jax.tree_util.tree_leaves(x)[0]
        nside = jhp.npix2nside(first_leaf.shape[-1])

        map2alm = Map2Alm(lmax=self.lmax, nside=nside, in_structure=self.in_structure)
        alm = map2alm.mv(x)

        alm_beam = jax.tree.map(_apply_fl_to_alm_leaf, alm, self.beam_fl)
        return Alm2Map(lmax=self.lmax, nside=nside, in_structure=map2alm.out_structure).mv(alm_beam)

    def inverse(self) -> 'BeamOperatorIQU':
        """Return a :class:`BeamOperatorIQU` with per-Stokes reciprocal transfer functions.

        Returns:
            A new :class:`BeamOperatorIQU` whose ``beam_fl`` leaves equal
            ``1 / leaf`` for each leaf in ``self.beam_fl``.
        """
        inv_beam = jax.tree.map(lambda fl: 1.0 / fl, self.beam_fl)
        return BeamOperatorIQU(lmax=self.lmax, beam_fl=inv_beam, in_structure=self.in_structure)


class BeamRule(AbstractBinaryRule):
    """Algebraic rule reducing ``BeamOperator @ BeamOperator`` to a single operator.

    When two :class:`BeamOperator` instances with the same ``lmax`` are composed,
    the composition is replaced by a single :class:`BeamOperator` whose
    ``beam_fl`` is the element-wise product of both transfer functions.

    Raises:
        NoReduction: If the operator types do not match or their ``lmax`` values
            differ.
    """

    left_operator_class = BeamOperator
    right_operator_class = BeamOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        """Combine two beam operators into one with multiplied transfer functions.

        Args:
            left: Left operator in the composition (must be :class:`BeamOperator`).
            right: Right operator in the composition (must be :class:`BeamOperator`).

        Returns:
            A single-element list containing a :class:`BeamOperator` whose
            ``beam_fl`` is ``left.beam_fl * right.beam_fl``.

        Raises:
            NoReduction: If either operator is not a :class:`BeamOperator` or
                their ``lmax`` values differ.
        """
        if not (isinstance(left, BeamOperator) and isinstance(right, BeamOperator)):
            raise NoReduction
        if left.lmax != right.lmax:
            raise NoReduction
        return [
            BeamOperator(
                lmax=left.lmax,
                beam_fl=left.beam_fl * right.beam_fl,
                in_structure=right.in_structure,
            )
        ]


class BeamIQURule(AbstractBinaryRule):
    """Algebraic rule reducing ``BeamOperatorIQU @ BeamOperatorIQU`` to a single operator.

    When two :class:`BeamOperatorIQU` instances with the same ``lmax`` are
    composed, the composition is replaced by a single :class:`BeamOperatorIQU`
    whose per-Stokes ``beam_fl`` leaves are the element-wise products of the
    corresponding leaves from both operators.

    Raises:
        NoReduction: If the operator types do not match or their ``lmax`` values
            differ.
    """

    left_operator_class = BeamOperatorIQU
    right_operator_class = BeamOperatorIQU

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        """Combine two per-Stokes beam operators into one with multiplied transfer functions.

        Args:
            left: Left operator in the composition (must be :class:`BeamOperatorIQU`).
            right: Right operator in the composition (must be :class:`BeamOperatorIQU`).

        Returns:
            A single-element list containing a :class:`BeamOperatorIQU` whose
            per-leaf ``beam_fl`` entries are ``left_leaf * right_leaf``.

        Raises:
            NoReduction: If either operator is not a :class:`BeamOperatorIQU` or
                their ``lmax`` values differ.
        """
        if not (isinstance(left, BeamOperatorIQU) and isinstance(right, BeamOperatorIQU)):
            raise NoReduction
        if left.lmax != right.lmax:
            raise NoReduction
        combined_beam = jax.tree.map(jnp.multiply, left.beam_fl, right.beam_fl)
        return [
            BeamOperatorIQU(
                lmax=left.lmax,
                beam_fl=combined_beam,
                in_structure=right.in_structure,
            )
        ]
