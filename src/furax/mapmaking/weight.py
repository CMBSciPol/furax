from dataclasses import field
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Inexact, PyTree

from furax import AbstractLinearOperator, MaskOperator, symmetric, tree
from furax.core import DiagonalOperator, SymmetricBandToeplitzOperator

from .config import NestedConfig


@symmetric
class WeightOperator(AbstractLinearOperator):
    """Masked noise weight `M W M` as a single operator."""

    weight: AbstractLinearOperator  # symmetric
    mask: MaskOperator

    @classmethod
    def create(cls, weight: AbstractLinearOperator, mask: MaskOperator) -> Self:
        return cls(weight, mask, in_structure=mask.in_structure)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        W, M = self.weight, self.mask
        if W.is_diagonal:
            return W(M(x))
        return M(W(M(x)))


@symmetric
class MarginalizedWeightOperator(AbstractLinearOperator):
    r"""A weight with the amplitudes of some templates marginalized.

    Marginalizing the amplitudes of templates $T$ turns the weight $W$ into

    $$ W' = W - W T G^{-1} T^\top W, $$

    with $G^{-1}$ the inverse of their Gram $T^\top W T$, possibly regularized. When $G^{-1}$ is
    exact, $W'$ is blind to the templates: $W' T = 0$.
    """

    weight: AbstractLinearOperator
    """The weight $W$."""
    templates: AbstractLinearOperator
    """The templates $T$ whose amplitudes are marginalized."""
    gram_inverse: AbstractLinearOperator
    """The inverse $G^{-1}$ of their Gram under $W$."""

    @classmethod
    def create(
        cls,
        weight: AbstractLinearOperator,
        templates: AbstractLinearOperator,
        gram_inverse: AbstractLinearOperator,
    ) -> Self:
        return cls(weight, templates, gram_inverse, in_structure=weight.in_structure)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        W, T = self.weight, self.templates
        wx = W(x)
        return tree.sub(wx, W(T(self.gram_inverse(T.T(wx)))))


@symmetric
class NestedWeightOperator(AbstractLinearOperator):
    r"""Minimum-variance inverse-noise weighting for gappy data, using an iterative algorithm.

    The minimum-variance weight for gappy data is the inverse of the *good-good block* of the noise
    covariance, $W_\text{exact} = S^\top N_{gg}^{-1} S$, with $S$ selecting the good samples. The
    noise models provide $N^{-1}$ as an operator that is cheap to apply, but not $N_{gg}^{-1}$:
    applying it would take an iterative solve with $N_{gg}$ over every good sample, at each
    application. Instead, the weight is written with $N^{-1}$ and a solve over the flagged samples
    only:

    $$ W = M \left[ N^{-1} - N^{-1} C \left( C^\top N^{-1} C \right)^{-1} C^\top N^{-1} \right] M, $$

    where $M$ is the mask of the good samples and $C$ selects the flagged ones. Order the samples
    good/flagged and split $N^{-1}$ into $2 \times 2$ blocks: the Schur complement of its flagged
    block, $(N^{-1})_{gg} - (N^{-1})_{gf} \left( (N^{-1})_{ff} \right)^{-1} (N^{-1})_{fg}$, is
    exactly $N_{gg}^{-1}$: it is the good-good block of the bracket above, which $M$ selects.

    **Inner solve.** $W$ must be a constant linear operator, identical on the RHS and on every
    application in the system, so $A^{-1}$, with $A = C^\top N^{-1} C$, is applied by a fixed
    polynomial whose coefficients do not depend on the vector $b$ it acts on: `inner_steps` steps
    of Chebyshev iteration, preconditioned by $P$, the inverse of the diagonal of $D$ = `white`, a
    diagonal approximation of $N^{-1}$ such as the white-noise weight. The iteration needs an
    interval $[a, b]$ holding the spectrum of $P A$. $N^{-1}$ is a banded Toeplitz matrix per
    channel, so the spectrum of $A$, a principal submatrix of it, lies within the range of its
    frequency response: with $P$ the inverse white level of each channel, $[a, b]$ is the range of
    the frequency response over the white level, over the channels, whatever the gaps. Its
    residual polynomial $r(\lambda) = 1 - \lambda p(\lambda)$, with $p$ the polynomial applied, is
    scaled to stay nonnegative on $[a, b]$, which keeps $W$ positive semidefinite for any number of
    steps: $M N^{-1} M \succeq W \succeq W_\text{exact}$. Only $b$ must bound the spectrum: an
    eigenvalue below $a$ just converges more slowly, so $a$ is raised to what the steps can
    resolve. A tolerance-based inner solve, or CG, would make $W$ depend on its input.

    **Static shapes.** Under jit the flagged subspace must have a static size, but the flag count
    is a runtime value. The unknowns of the inner solve therefore live on the samples of the TOD:
    those of the good samples act as the identity and solve to 0. The flags are read from the mask
    at each application, so the weight holds nothing the size of the TOD beyond $N^{-1}$. A
    fully-masked input gives 0. The TOD is a single array with the samples last, plain or in a
    [`Stokes`][furax.obs.stokes.Stokes] container, as [`TODFormat`][furax.obs.tod.TODFormat]
    describes it.
    """

    ninv: SymmetricBandToeplitzOperator
    """The inverse noise covariance $N^{-1}$, a banded Toeplitz matrix per channel, symmetric
    positive semidefinite."""
    mask: MaskOperator
    """The mask $M$ of the good samples, which also sets the flagged ones."""
    white: AbstractLinearOperator
    """The diagonal approximation $D$ of $N^{-1}$ that preconditions the inner solve."""
    spectrum: Float[Array, '2']
    """The interval $[a, b]$ holding the spectrum of $P A$."""
    inner_steps: int = field(metadata={'static': True})
    """The number of Chebyshev steps of the inner solve."""

    @classmethod
    def create(
        cls,
        ninv: SymmetricBandToeplitzOperator,
        mask: MaskOperator,
        config: NestedConfig,
        white: AbstractLinearOperator,
    ) -> Self:
        r"""Build the weight.

        Args:
            ninv: The inverse noise covariance $N^{-1}$, a banded Toeplitz matrix per channel.
            mask: The mask $M$ of the good samples.
            config: The inner-solve options.
            white: A diagonal approximation $D$ of $N^{-1}$, such as the white-noise weight,
                under which the inner solve is preconditioned.
        """
        spectrum = _spectrum(ninv, white, mask.in_structure, config.inner_steps)
        return cls(
            ninv,
            mask,
            white,
            spectrum,
            inner_steps=config.inner_steps,
            in_structure=mask.in_structure,
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        V, inner, preconditioner = _inner_system(self.ninv, self.mask, self.white)
        x = self.mask(x)  # the flagged samples carry no data
        nx = self.ninv(x)
        y = _chebyshev(inner, V(nx), self.spectrum, self.inner_steps, preconditioner)
        return self.mask(tree.sub(nx, self.ninv(y)))


def _inner_system(
    ninv: AbstractLinearOperator, mask: MaskOperator, white: AbstractLinearOperator
) -> tuple[MaskOperator, AbstractLinearOperator, AbstractLinearOperator]:
    """The mask `V` of the unknowns in use, the inner operator, and its preconditioner `P`.

    The unknowns live on the samples of the TOD. `V` marks those in use, the flagged samples of
    the live channels, and the inner operator is `V N⁻¹ V + (I − V)`: on them, it is
    `A = Cᵀ N⁻¹ C`, and the other unknowns act as the identity and solve to 0. A dead channel,
    without white weight, carries no data, so its flagged samples are left out too. `P` is the
    inverse of the diagonal of `Cᵀ D C`: the inverse white level of each channel, held per channel
    rather than per sample. It is 1 on a dead channel, whose unknowns solve to 0 whatever `P`.
    """
    flagged = jax.tree.leaves(mask.to_boolean_mask())[0] == 0
    level = _white_levels(white, mask.in_structure)
    live = level > 0
    V = MaskOperator.from_boolean_mask(flagged & live, in_structure=mask.in_structure)
    inner = V @ ninv @ V + V.complement()
    inverse = jnp.where(live, 1 / jnp.where(live, level, 1), 1)
    return V, inner.reduce(), DiagonalOperator(inverse, in_structure=mask.in_structure)


def _white_levels(
    white: AbstractLinearOperator, tod_structure: PyTree[jax.ShapeDtypeStruct]
) -> Float[Array, '*channels 1']:
    """The white level $d$ of each channel of the TOD, the diagonal of `white`."""
    return jax.tree.leaves(white(tree.ones_like(tod_structure)))[0][..., :1]


def _chebyshev(
    operator: AbstractLinearOperator,
    b: PyTree[Array],
    interval: Float[Array, '2'],
    steps: int,
    preconditioner: AbstractLinearOperator | None = None,
) -> PyTree[Array]:
    r"""A fixed polynomial approximation of $A^{-1} b$: `steps` steps of Chebyshev iteration.

    The iteration starts from 0 and applies $A$ = `operator` `steps` times. Its result is
    $p(PA) P b$, with $P$ = `preconditioner` (the identity by default) and $p$ a polynomial set by
    `interval` $= [a, b]$ and `steps` alone, not by the vector $b$: the result is linear in $b$,
    and symmetric in it when $A$ and $P$ are, unlike a CG solve stopped early.

    The residual polynomial $r(\lambda) = 1 - \lambda p(\lambda)$ is that of Chebyshev iteration,
    $T_k(z) / T_k(z_0)$ with $k$ = `steps` + 1, $z$ mapping $[a, b]$ to $[1, -1]$ and $z_0$ the
    image of 0, shifted and scaled into $(1 + T_k(z)) / (1 + T_k(z_0))$. It converges at the same
    rate, and $0 \le r(\lambda) < 1$ for $0 < \lambda \le b$: if the spectrum of $PA$ lies in
    $[0, b]$, then $p(PA) P$ is symmetric positive definite whenever $A$ is symmetric positive
    semidefinite and $P$ symmetric positive definite, so it can precondition CG.

    The upper end `b` must bound the spectrum of $PA$. The lower end `a` need not: an eigenvalue
    below it only converges more slowly. In $k$ steps, eigenvalues down to about $b / k^2$ can be
    resolved.

    Args:
        operator: The operator $A$.
        b: The right-hand side.
        interval: The interval $[a, b]$ of the spectrum of $PA$, with $0 < a < b$, `(2,)`.
        steps: The number of applications of $A$.
        preconditioner: The preconditioner $P$; the identity by default.

    Returns:
        $p(PA) P b$, an approximation of $A^{-1} b$.
    """

    def precondition(x):
        return x if preconditioner is None else preconditioner(x)

    lower, upper = interval[0], interval[1]
    center, half_width = (upper + lower) / 2, (upper - lower) / 2
    sigma = center / half_width
    rho = 1 / sigma
    d = tree.mul(1 / center, precondition(b))

    def step(_, carry):
        y, r, d, rho = carry
        y = tree.add(y, d)
        r = tree.sub(r, operator(d))
        rho_next = 1 / (2 * sigma - rho)
        d = tree.add(
            tree.mul(rho_next * rho, d), tree.mul(2 * rho_next / half_width, precondition(r))
        )
        return y, r, d, rho_next

    y, _, d, _ = jax.lax.fori_loop(0, steps, step, (tree.zeros_like(b), b, d, rho))
    # T_k(z_0) for z_0 = sigma > 1; 1 / T_k(z_0) underflows to 0 rather than overflowing
    inverse_t = 1 / jnp.cosh((steps + 1) * jnp.arccosh(sigma))
    return tree.mul(1 / (1 + inverse_t), tree.add(y, d))


def _spectrum(
    ninv: SymmetricBandToeplitzOperator,
    white: AbstractLinearOperator,
    tod_structure: PyTree[jax.ShapeDtypeStruct],
    steps: int,
) -> Array:
    r"""The interval `[a, b]` of the Chebyshev iteration for `P A`, `(2,)`, of `steps` steps.

    The eigenvalues of a principal submatrix of a symmetric Toeplitz matrix lie within the range of
    its frequency response $s(\omega) = b_0 + 2 \sum_k b_k \cos k\omega$. Under `P`, the inverse
    white level $d$ of each channel, the interval is the range of $s / d$ over the live channels.
    The response is read on a fine frequency grid; `b` gets a margin for its maximum between the
    grid points, since an eigenvalue above it would make the weight indefinite.

    `a` need not hold the spectrum: an eigenvalue below it only converges more slowly. In $k$
    steps, Chebyshev iteration resolves the spectrum down to about $b / k^2$, so a lower `a`
    spreads the polynomial over eigenvalues it cannot reach anyway: `a` is raised to
    $b \, (c / k)^2$.
    """
    band = ninv.band_values
    n_lags = band.shape[-1]
    response = 2 * jnp.fft.rfft(band, n=_OVERSAMPLING * 2 * n_lags).real - band[..., :1]
    level = _white_levels(white, tod_structure)[..., 0]
    shape = jnp.broadcast_shapes(response.shape[:-1], level.shape)
    response = jnp.broadcast_to(response, (*shape, response.shape[-1]))
    level = jnp.broadcast_to(level, shape)[..., None]
    live = level > 0  # a dead channel takes no part in the inner solve
    ratio = response / jnp.where(live, level, 1)
    lower = jnp.min(jnp.where(live, ratio, jnp.inf))
    upper = jnp.max(jnp.where(live, ratio, -jnp.inf))
    upper = _MARGIN * upper
    lower = jnp.maximum(lower, upper * (_RESOLVED / max(steps, 1)) ** 2)
    bounds = jnp.stack([jnp.minimum(lower, upper / 2), upper])
    return jnp.where(jnp.any(live), bounds, jnp.array([0.0, 1.0], bounds.dtype))


_OVERSAMPLING = 8  # frequencies per lag of the band, twice over, reading its frequency response
_MARGIN = 1.01  # over the largest response on the grid, below its maximum between grid points
_RESOLVED = 3  # c: the lowest eigenvalue worth resolving in k steps is b (c / k)²
