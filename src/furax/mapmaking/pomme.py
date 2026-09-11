r"""Pomme: implicit deprojection of the mean of every short interval of the TOD.

Pomme fits and removes one offset per detector and per interval of `tau` consecutive samples. It
is a template in the sense of [`furax.mapmaking.templates`][], with the indicator template $Z$
(one column per detector and interval), but the template matrix is never formed: with a diagonal
weight $W$ that is constant on each interval, the marginalised weight $W' = W D_Z$ reduces to
$W F$ with $F = I - Z (Z^\top Z)^{-1} Z^\top$ the plain interval-mean remover,
[`PommeProjectionOperator`][].

The constant-weight requirement is what the mapmaker's mask widening provides: every interval with
a flagged sample is flagged whole (and so is the partial tail interval), so per-detector white
noise gives a weight that commutes with $F$. Then $W F$ is symmetric positive semidefinite and
the normal equations $P^\top W F P\, m = P^\top W F\, d$ are a valid GLS system.

Other templates $T$ combine with Pomme through the effective weight $W F$: the joint deprojector
of $[Z\ T]$ factorises exactly as $D_{FT} D_Z$ (block Gram-Schmidt in the $W$ metric), so the
only new ingredient is the Gram of the Pomme-filtered templates $T^\top W F T$, which
[`furax.mapmaking.gram`][] assembles by Schur-eliminating $Z$ with [`PommeIntervals`][].
"""

from dataclasses import field
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PyTree

from furax import AbstractLinearOperator, square

from .templates import BasisColumns

__all__ = [
    'PommeProjectionOperator',
    'PommeIntervals',
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

    @staticmethod
    def interval_ids(n_samp: int, tau: int) -> tuple[Int[Array, ' samp'], int]:
        """The interval each sample belongs to, and the number of whole intervals `n_int`.

        Tail samples (past the last whole interval) get the id `n_int`, one past the last real
        interval, so a consumer can keep or drop them with one extra row.

        Examples:
            >>> ids, n_int = PommeProjectionOperator.interval_ids(7, 3)
            >>> ids.tolist(), n_int
            ([0, 0, 0, 1, 1, 1, 2], 2)
        """
        n_int = n_samp // tau
        ids = jnp.minimum(jnp.arange(n_samp) // tau, n_int)
        return ids.astype(jnp.int32), n_int

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det samp']:
        n_det, n_samp = self.in_structure.shape
        n_int, n_rem = divmod(n_samp, self.tau)
        y = x[:, : n_int * self.tau].reshape(n_det, n_int, self.tau)
        y = y - jnp.mean(y, axis=-1, keepdims=True)
        y = y.reshape(n_det, n_int * self.tau)
        if n_rem == 0:
            return y
        return jnp.concatenate([y, x[:, -n_rem:]], axis=1)


class PommeIntervals(eqx.Module):
    r"""The Pomme intervals of a TOD, for Schur-eliminating the Pomme template out of a Gram.

    For a template basis $B$ and a diagonal weight $W$ (one detector's weights), the Gram of the
    Pomme-filtered basis is

    $$ \tilde G = B^\top W B - C^\top D^{-1} C, \qquad C = Z^\top W B, \quad D = Z^\top W Z, $$

    with $D$ the weighted sample count of each interval ([`counts`][PommeIntervals.counts]) and
    $C$ the weighted sum of each basis column over each interval ([`sums`][PommeIntervals.sums]).
    Intervals of zero weight (masked whole) have zero rows in $C$ and drop out.

    `ids` is each sample's interval, tail samples getting the extra id `n_int` (a row that is
    accumulated and then dropped, so the tail never enters the correction: the Pomme projector
    leaves it untouched).
    """

    ids: Int[Array, ' samp']
    n_int: int = eqx.field(static=True)

    @classmethod
    def create(cls, n_points: int, tau: int) -> Self:
        """The intervals of a TOD of `n_points` samples, `tau` samples each."""
        ids, n_int = PommeProjectionOperator.interval_ids(n_points, tau)
        return cls(ids, n_int)

    def counts(self, weights: Float[Array, ' samp']) -> Float[Array, ' n_int']:
        """`D = Zᵀ W Z`: the weighted sample count of each interval."""
        return jnp.zeros(self.n_int + 1, weights.dtype).at[self.ids].add(weights)[: self.n_int]

    def inverse_counts(self, weights: Float[Array, ' samp']) -> Float[Array, ' n_int']:
        """`D⁻¹`, zero on intervals of zero weight (whose `C` rows vanish too)."""
        d = self.counts(weights)
        return jnp.where(d > 0, 1.0 / jnp.where(d > 0, d, 1.0), 0.0)

    def sums(self, view: BasisColumns, weights: Float[Array, ' samp']) -> Float[Array, 'n_int K']:
        """`C = Zᵀ W B`: the weighted sum of every basis column over each interval, dense."""
        k = view.values.shape[0]
        wv = (view.values * weights[None, :]).T  # (samp, k)
        sums = jnp.zeros((self.n_int + 1, view.n_blocks, k), view.values.dtype)
        for slot in range(view.blocks.shape[1]):  # window width is static
            sums = sums.at[self.ids, view.blocks[:, slot]].add(view.taps[:, slot][:, None] * wv)
        return sums[: self.n_int].reshape(self.n_int, view.n_blocks * k)

    def first_blocks(self, view: BasisColumns, w1: int) -> Int[Array, ' n_int1']:
        """The first block each interval touches, checking that it touches at most `w1`.

        An interval couples every block it touches, and a corrected band holds `w1` of them from
        the first. A wider span means a block shorter than an interval, which the band layout
        cannot represent; it is reported as a runtime error. The returned array has the extra tail
        row of `ids`.
        """
        n_blocks = view.n_blocks
        first = jnp.full(self.n_int + 1, n_blocks, jnp.int32).at[self.ids].min(view.blocks[:, 0])
        last = jnp.full(self.n_int + 1, -1, jnp.int32).at[self.ids].max(view.blocks[:, -1])
        span = (last - first)[: self.n_int]
        return eqx.error_if(
            first,
            jnp.any(span >= w1),
            'a Pomme interval spans more template blocks than the Gram band can hold: '
            'template blocks (scan intervals, spline knots) must be longer than pomme_tau',
        )

    def band_correction(
        self,
        view: BasisColumns,
        first: Int[Array, ' n_int1'],
        weights: Float[Array, ' samp'],
        w1: int,
    ) -> Float[Array, 'n_blocks w1 k k']:
        """`Cᵀ D⁻¹ C` in the band layout of [`banded_cholesky`][furax.linalg.banded_cholesky].

        Interval `i` touches the blocks `first[i] + s`, `s < w1`; its contribution is the outer
        product of its per-slot sums, scattered into the band at `(first[i] + s, s' − s)`.
        Cost `O(n_samp · window · k)` for the sums and `O(n_int · w1² · k²)` for the products,
        without any `(samp, k, k)` intermediate.
        """
        k = view.values.shape[0]
        n_blocks = view.n_blocks
        wv = (view.values * weights[None, :]).T  # (samp, k)
        c = jnp.zeros((self.n_int + 1, w1, k), view.values.dtype)
        for slot in range(view.blocks.shape[1]):
            s = jnp.clip(view.blocks[:, slot] - first[self.ids], 0, w1 - 1)
            c = c.at[self.ids, s].add(view.taps[:, slot][:, None] * wv)
        c = c[: self.n_int]
        inv_d = self.inverse_counts(weights)
        bands = jnp.zeros((n_blocks, w1, k, k), view.values.dtype)
        for s in range(w1):
            for s2 in range(s, w1):
                contrib = jnp.einsum('ia,ib,i->iab', c[:, s], c[:, s2], inv_d)
                # an interval touching fewer than w1 blocks has zero rows there: clip the index
                rows = jnp.clip(first[: self.n_int] + s, 0, n_blocks - 1)
                bands = bands.at[rows, s2 - s].add(contrib)
        return bands
