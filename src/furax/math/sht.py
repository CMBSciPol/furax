"""Operators for spherical harmonic transforms backed by jax-healpy.

The two core operators are:

- :class:`Map2Alm`: analysis transform (pixel map → spherical harmonic coefficients).
- :class:`Alm2Map`: synthesis transform (spherical harmonic coefficients → pixel map).

Both operators accept PyTree inputs whose leaves are 2-D arrays of shape
``(nfreq, npix)`` or ``(nfreq, lmax+1, 2*lmax+1)``.  A 1-D leaf is promoted to 2-D via
``jnp.atleast_2d`` before processing so that single-frequency inputs work
without special-casing.

:class:`SHTRule` registers an algebraic simplification so that the composition
``Map2Alm @ Alm2Map`` (synthesis followed by analysis) is reduced to an
:class:`~furax.IdentityOperator` at operator-construction time.
"""

import jax
import jax.numpy as jnp
import jax_healpy as jhp
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, IdentityOperator
from furax.core.rules import AbstractBinaryRule, NoReduction


class Map2Alm(AbstractLinearOperator):
    """Analysis spherical harmonic transform: pixel map → alm coefficients.

    Each leaf of the input PyTree must be a 2-D array of shape
    ``(nfreq, npix)``.  A 1-D leaf is silently promoted to ``(1, npix)`` via
    ``jnp.atleast_2d``.  The transform are applied to all frequency rows, 
    via ``jhp.map2alm`` producing an output leaf of shape
    ``(nfreq, lmax+1, 2*lmax+1)``.

    Attributes:
        lmax: Maximum spherical harmonic degree.
        nside: HEALPix resolution parameter, carried so that
            :meth:`inverse` can construct
            :class:`Alm2Map` with the correct resolution.
        iter: Number of iterations for the map2alm solver; more iterations,
            which are more expensive, yield more accurate alm coefficients.
            (default is 3)

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.math.sht import Map2Alm
        >>> from furax.obs.stokes import StokesIQU
        >>> nside, lmax, nfreq = 32, 63, 3
        >>> npix = 12 * nside ** 2
        >>> structure = StokesIQU.structure_for((nfreq, npix), jnp.float64)
        >>> op = Map2Alm(lmax=lmax, nside=nside, in_structure=structure)
    """

    lmax: int
    nside: int
    iter: int = 3

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        """Apply the analysis SHT to every leaf of *x*.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, npix)`` or
                ``(npix,)`` for a single frequency.

        Returns:
            PyTree with the same structure as *x*; each leaf has shape
            ``(nfreq, lmax+1, 2*lmax+1)`` and complex dtype.
        """

        def func(value: jax.Array) -> jax.Array:
            value = jnp.atleast_2d(value)  # (nfreq, npix)
            return jhp.map2alm(
                value, iter=self.iter, lmax=self.lmax, pol=False
            )  # (nfreq, lmax+1, 2*lmax+1)

        return jax.tree.map(func, x)

    def inverse(self) -> 'Alm2Map':
        """Return the pseudo-inverse operator :class:`Alm2Map`.

        Returns:
            An :class:`Alm2Map` whose ``in_structure`` matches the output
            structure of this operator (i.e. alm space).
        """
        return Alm2Map(
            lmax=self.lmax, nside=self.nside, iter=self.iter, in_structure=self.out_structure
        )


class Alm2Map(AbstractLinearOperator):
    """Synthesis spherical harmonic transform: alm coefficients → pixel map.

    Each leaf of the input PyTree must be a 2-D array of shape
    ``(nfreq, lmax+1, 2*lmax+1)``.  A 2-D leaf ``(lmax+1, 2*lmax+1)`` is
    silently reshaped to ``(1, lmax+1, 2*lmax+1)`` via ``reshape(-1, ...)``,
    handling the single-frequency case without branching.  The transform are applied 
    to all frequency rows, via ``jhp.alm2map`` producing an output leaf of shape
    ``(nfreq, npix)`` where ``npix = 12 * nside ** 2``.

    Attributes:
        lmax: Maximum spherical harmonic degree.
        nside: HEALPix resolution parameter used by the synthesis step.
        iter: Number of iterations for the map2alm solver used by
            the inverse. (Default is 3)

    Example:
        >>> import jax.numpy as jnp
        >>> from furax.math.sht import Alm2Map
        >>> from furax.obs.stokes import StokesIQU
        >>> nside, lmax, nfreq = 32, 63, 3
        >>> structure = StokesIQU.structure_for((nfreq, lmax+1, 2*lmax+1), jnp.complex128)
        >>> op = Alm2Map(lmax=lmax, nside=nside, in_structure=structure)
    """

    lmax: int
    nside: int
    iter: int = 3

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        """Apply the synthesis SHT to every leaf of *x*.

        Args:
            x: Input PyTree whose leaves have shape ``(nfreq, lmax+1, 2*lmax+1)`` or
                ``(lmax+1, 2*lmax+1,)`` for a single frequency.

        Returns:
            PyTree with the same structure as *x*; each leaf has shape
            ``(nfreq, npix)`` and real dtype.
        """

        def func(value: jax.Array) -> jax.Array:
            value = value.reshape(-1, *value.shape[-2:])  # (nfreq, lmax+1, 2*lmax+1)
            return jnp.real(jhp.alm2map(value, nside=self.nside, lmax=self.lmax, pol=False))

        return jax.tree.map(func, x)

    def inverse(self) -> 'Map2Alm':
        """Return the pseudo-inverse operator :class:`Map2Alm`.

        Returns:
            A :class:`Map2Alm` whose ``in_structure`` matches the output
            structure of this operator (i.e. map space).
        """
        return Map2Alm(
            lmax=self.lmax, nside=self.nside, iter=self.iter, in_structure=self.out_structure
        )


class SHTRule(AbstractBinaryRule):
    """Algebraic rule reducing ``Map2Alm @ Alm2Map`` to an identity.

    The composition *analysis after synthesis* (``Map2Alm @ Alm2Map``) is
    exact when the band-limit ``lmax`` is consistent with the HEALPix
    resolution ``nside``, so the rule collapses it to an
    :class:`~furax.IdentityOperator` on the alm input space.

    The reverse composition ``Alm2Map @ Map2Alm`` is a band-limited
    projection, **not** an identity, and is therefore not reduced.
    """

    left_operator_class = Map2Alm
    right_operator_class = Alm2Map

    def apply(
        self, op1: AbstractLinearOperator, op2: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        """Reduce ``op1 @ op2`` to an identity when the pair is Map2Alm/Alm2Map.

        Args:
            op1: Left operator in the composition (must be :class:`Map2Alm`).
            op2: Right operator in the composition (must be :class:`Alm2Map`).

        Returns:
            A single-element list containing an
            :class:`~furax.IdentityOperator` on the alm input space.

        Raises:
            NoReduction: If the operator pair does not match the expected types
            or if the transforms use different ``lmax`` values.
        """
        if isinstance(op1, Map2Alm) and isinstance(op2, Alm2Map):
            if op1.lmax != op2.lmax:
                raise NoReduction
            return [IdentityOperator(in_structure=op2.in_structure)]
        raise NoReduction
