r"""Pomme: implicit deprojection of the mean of every short interval of the TOD.

Pomme fits and removes one offset per detector and per interval of `tau` consecutive samples. It
is a template in the sense of [`furax.mapmaking.templates`][], with the indicator template $Z$
(one column per detector and interval, an [`OffsetBasis`][furax.mapmaking.bases.OffsetBasis]),
whose matrix is never formed: with a diagonal weight $W$ that is constant on each interval, the
marginalised weight $W' = W D_Z$ reduces to $W F$ with $F = I - Z (Z^\top Z)^{-1} Z^\top$ the plain
interval-mean remover, [`PommeProjectionOperator`][].

The constant-weight requirement is what the mapmaker's mask widening provides: every interval with
a flagged sample is flagged whole (and so is the partial tail interval), so per-detector white
noise gives a weight that commutes with $F$. Then $W F$ is symmetric positive semidefinite and
the normal equations $P^\top W F P\, m = P^\top W F\, d$ are a valid GLS system.

Other templates $T$ combine with Pomme through the effective weight $W F$: the joint deprojector
of $[Z\ T]$ factorises exactly as $D_{FT} D_Z$ (block Gram-Schmidt in the $W$ metric), so the
only new ingredient is the Gram of the Pomme-filtered templates $T^\top W F T$, which
[`furax.mapmaking.gram`][] assembles by Schur-eliminating $Z$.
"""

from dataclasses import field

import jax
from jax import Array
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, square

from .bases import OffsetBasis

__all__ = [
    'PommeProjectionOperator',
]


@square
class PommeProjectionOperator(AbstractLinearOperator):
    r"""Remove the mean of every `tau`-sample interval of a `(det, samp)` TOD.

    This is the Pomme deprojector $F = I - Z (Z^\top Z)^{-1} Z^\top$ for the template $Z$ whose
    columns are the indicators of the consecutive `tau`-sample intervals (one amplitude per
    detector and interval): an unweighted projector, the same for every detector. The samples
    past the last whole interval (the tail) are left untouched.
    """

    tau: int = field(metadata={'static': True})

    def __init__(
        self,
        tau: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'tau', tau)
        object.__setattr__(self, 'in_structure', in_structure)

    @property
    def template(self) -> OffsetBasis:
        """The Pomme template $Z$, shared by every detector."""
        return OffsetBasis(self.tau, self.in_structure.shape[-1], self.in_structure.dtype)

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det samp']:
        z = self.template
        # Zᵀ Z = tau on every interval: each interval's mean, spread back over its samples
        return x - z(z.T(x) / self.tau)
