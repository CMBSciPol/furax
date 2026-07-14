"""Beam operators for applying instrumental transfer functions in spherical harmonic space.

The two core operators are:

- [`BeamOperator`][]: applies a single beam transfer function shared across all
  Stokes components.
- [`BeamOperatorIQU`][]: applies an independent beam transfer function per Stokes
  component.

Both operators round-trip input pixel maps through a spherical harmonic transform
(Map2Alm → per-multipole scaling → Alm2Map).  `BeamRule` and
`BeamIQURule` register algebraic simplifications so that
``BeamOperator @ BeamOperator`` collapses to a single operator whose transfer
function is the product of both beams at construction time.
"""

import jax
import jax.numpy as jnp
import jax_healpy as jhp
from jaxtyping import Array, Float, Inexact, PyTree

import furax.tree as fxtree
from furax import AbstractLinearOperator, symmetric
from furax.core.rules import AbstractCompositionRule, NoReduction
from furax.math.sht import Alm2Map, Map2Alm


@symmetric
class BeamOperator(AbstractLinearOperator):
    """Beam operator with a single transfer function shared across all Stokes components.

    Input PyTree leaves are expected to have shape ``(nfreq, npix)``.  They are
    round-tripped through a spherical harmonic transform and the beam is applied
    in alm space as a per-multipole, per-frequency scaling.

    The operator is symmetric (self-adjoint): applying it twice is equivalent to
    squaring the transfer function.  Use [`inverse`][] to obtain the deconvolution
    operator (``beam_fl`` replaced by its reciprocal).

    Algebraic rule: ``BeamOperator @ BeamOperator`` reduces to a single
    [`BeamOperator`][] whose ``beam_fl`` is the element-wise product of both
    transfer functions (see `BeamRule`).

    Attributes:
        lmax: Maximum spherical harmonic degree.
        beam_fl: Beam transfer function, shape ``(nfreq, lmax+1)`` for a
            per-frequency beam, or ``(lmax+1,)`` for a beam shared across all
            frequencies (promoted to 2-D via `jnp.atleast_2d` before
            broadcasting over frequencies).

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.obs.stokes import StokesIQU
        >>> from furax.obs.operators import BeamOperator
        >>> nside, lmax, nfreq = 16, 31, 2
        >>> npix = 12 * nside ** 2
        >>> structure = StokesIQU.structure_for((nfreq, npix), jnp.float64)
        >>> beam_fl = jnp.ones((nfreq, lmax + 1))
        >>> op = BeamOperator(lmax=lmax, beam_fl=beam_fl, in_structure=structure)
    """

    lmax: int
    beam_fl: Float[Array, '...']

    def mv(self, x: PyTree[Inexact[Array, 'nfreq npix']]) -> PyTree[Inexact[Array, 'nfreq npix']]:
        """Apply the beam to input sky maps.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, npix)`` or ``(npix,)``
                for a single-frequency input.

        Returns:
            Beam-smoothed PyTree with the same structure and leaf shapes as *x*.
        """
        first_leaf = jax.tree_util.tree_leaves(x)[0]
        input_is_1d = first_leaf.ndim == 1
        first_leaf = jnp.atleast_2d(first_leaf)  # (nfreq, npix)
        nside = jhp.npix2nside(first_leaf.shape[-1])

        map2alm = Map2Alm(lmax=self.lmax, nside=nside, in_structure=self.in_structure)
        alm = map2alm.mv(x)

        # Apply the per-multipole transfer function, broadcasting it over every leading axis of the
        # alm leaf (frequency and, for a Stokes-backed map, the leading Stokes axis).
        def apply_beam(alm_leaf: jax.Array) -> jax.Array:
            fl = jnp.broadcast_to(self.beam_fl, alm_leaf.shape[:-2] + (self.lmax + 1,))
            return jnp.asarray(jhp.almxfl(alm_leaf, fl, healpy_ordering=False))

        alm_beam = jax.tree.map(apply_beam, alm)

        out = Alm2Map(lmax=self.lmax, nside=nside, in_structure=map2alm.out_structure).mv(alm_beam)
        if input_is_1d:
            out = jax.tree.map(lambda leaf: jnp.squeeze(leaf, axis=0), out)
        return out

    def inverse(self) -> 'BeamOperator':
        """Return a [`BeamOperator`][] with the reciprocal transfer function.

        Returns:
            A new [`BeamOperator`][] whose ``beam_fl`` equals ``1 / self.beam_fl``.
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

    The operator is symmetric (self-adjoint). Use [`inverse`][] to obtain the
    deconvolution operator (each ``beam_fl`` leaf replaced by its reciprocal).

    Algebraic rule: ``BeamOperatorIQU @ BeamOperatorIQU`` reduces to a single
    operator whose per-leaf transfer functions are the element-wise products of
    both (see `BeamIQURule`).

    Attributes:
        lmax: Maximum spherical harmonic degree.
        beam_fl: PyTree with the same structure as the input maps; each leaf has
            shape ``(nfreq, lmax+1)`` and carries the per-frequency beam transfer
            function for that Stokes component.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.obs.stokes import StokesIQU
        >>> from furax.obs.operators import BeamOperatorIQU
        >>> nside, lmax, nfreq = 16, 31, 2
        >>> npix = 12 * nside ** 2
        >>> structure = StokesIQU.structure_for((nfreq, npix), jnp.float64)
        >>> fl = jnp.ones((nfreq, lmax + 1))
        >>> beam_fl = StokesIQU(i=fl, q=fl, u=fl)
        >>> op = BeamOperatorIQU(lmax=lmax, beam_fl=beam_fl, in_structure=structure)
    """

    lmax: int
    beam_fl: PyTree[Float[Array, 'nfreq lp1']]

    def mv(self, x: PyTree[Inexact[Array, 'nfreq npix']]) -> PyTree[Inexact[Array, 'nfreq npix']]:
        """Apply per-Stokes beams to input sky maps.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, npix)`` or ``(npix,)``
                for a single-frequency input. The PyTree structure must match that
                of ``self.beam_fl``.

        Returns:
            Beam-smoothed PyTree with the same structure and leaf shapes as *x*.
        """
        first_leaf = jax.tree_util.tree_leaves(x)[0]
        input_is_1d = first_leaf.ndim == 1
        first_leaf = jnp.atleast_2d(first_leaf)  # (nfreq, npix)
        nside = jhp.npix2nside(first_leaf.shape[-1])

        map2alm = Map2Alm(lmax=self.lmax, nside=nside, in_structure=self.in_structure)
        alm = map2alm.mv(x)

        alm_beam = jax.tree.map(
            lambda alm_leaf, fl: jhp.almxfl(alm_leaf, fl, healpy_ordering=False),
            alm,
            self.beam_fl,
        )
        out = Alm2Map(lmax=self.lmax, nside=nside, in_structure=map2alm.out_structure).mv(alm_beam)
        if input_is_1d:
            out = jax.tree.map(lambda leaf: jnp.squeeze(leaf, axis=0), out)
        return out

    def inverse(self) -> 'BeamOperatorIQU':
        """Return a [`BeamOperatorIQU`][] with per-Stokes reciprocal transfer functions.

        Returns:
            A new [`BeamOperatorIQU`][] whose ``beam_fl`` leaves equal
            ``1 / leaf`` for each leaf in ``self.beam_fl``.
        """
        inv_beam = jax.tree.map(lambda fl: 1.0 / fl, self.beam_fl)
        return BeamOperatorIQU(lmax=self.lmax, beam_fl=inv_beam, in_structure=self.in_structure)


class BeamRule(AbstractCompositionRule):
    """Algebraic rule reducing ``BeamOperator @ BeamOperator`` to a single operator.

    When two [`BeamOperator`][] instances with the same ``lmax`` are composed,
    the composition is replaced by a single [`BeamOperator`][] whose
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
            left: Left operator in the composition (must be [`BeamOperator`][]).
            right: Right operator in the composition (must be [`BeamOperator`][]).

        Returns:
            A single-element list containing a [`BeamOperator`][] whose
            ``beam_fl`` is ``left.beam_fl * right.beam_fl``.

        Raises:
            NoReduction: If either operator is not a [`BeamOperator`][] or
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


class BeamIQURule(AbstractCompositionRule):
    """Algebraic rule reducing ``BeamOperatorIQU @ BeamOperatorIQU`` to a single operator.

    When two [`BeamOperatorIQU`][] instances with the same ``lmax`` are
    composed, the composition is replaced by a single [`BeamOperatorIQU`][]
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
            left: Left operator in the composition (must be [`BeamOperatorIQU`][]).
            right: Right operator in the composition (must be [`BeamOperatorIQU`][]).

        Returns:
            A single-element list containing a [`BeamOperatorIQU`][] whose
            per-leaf ``beam_fl`` entries are ``left_leaf * right_leaf``.

        Raises:
            NoReduction: If either operator is not a [`BeamOperatorIQU`][] or
                their ``lmax`` values differ.
        """
        if not (isinstance(left, BeamOperatorIQU) and isinstance(right, BeamOperatorIQU)):
            raise NoReduction
        if left.lmax != right.lmax:
            raise NoReduction
        combined_beam = fxtree.mul(left.beam_fl, right.beam_fl)
        return [
            BeamOperatorIQU(
                lmax=left.lmax,
                beam_fl=combined_beam,
                in_structure=right.in_structure,
            )
        ]
