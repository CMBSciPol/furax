r"""Goodness-of-fit statistics of the solution of the mapmaker, one detector tile at a time.

The residual of a tile is $r = d - J \hat x$, with $J$ and the data $d$ as the normal system takes
them (see [`Tile.joint`][furax.mapmaking._tile.Tile.joint]), and $\hat x$ every unknown solved
for. Its $\chi^2 = r^\top W' r$ under the weight $W'$ of the system splits over the channels (the
streams of each detector) and, by Parseval, over frequency bands:

$$ \chi^2_c = \frac1n \sum_k w_k \, \mathrm{Re}\big(\hat r_c(f_k)^* \, \widehat{W' r}_c(f_k)\big), $$

over the rfft frequencies $f_k$ of the $n$ samples, with $w_k = 1$ at zero and Nyquist and 2
elsewhere. For a stationary weight equal to the inverse noise, each frequency contributes about
$w_k$ times the fraction of good samples, which normalizes each band.
"""

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from furax import tree
from furax.core._lowrank import rfft_folds

from ._tile import Tile
from .config import DiagnosticsConfig
from .noise import LowRankNoiseModel
from .noise.binned import _bin_bounds


def tile_statistics(
    tile: Tile, tod: PyTree[Array], unknowns: dict[str, Any], diagnostics: DiagnosticsConfig
) -> dict[str, Array]:
    r"""The residual statistics of one detector tile.

    Args:
        tile: The tile.
        tod: Its TOD.
        unknowns: What was solved for, as the tile reads it: the `'map'` on the whole sky, and its
            amplitudes of each kind it has.
        diagnostics: The statistics to compute.

    Returns:
        `good`, the number of good samples, `(stream, det)`, and those of the statistics turned
        on: `chi2`, `(stream, det)`; `bands`, the $\chi^2$ of each band, `(band, stream, det)`,
        and `band_samples`, its expected value; `map`, the residual binned under the white-noise
        weight, before its normalization by the white pixel system. Under a low-rank noise model,
        the $\chi^2$ are those under the diagonal weight, and `modes` and `low` give their
        correction (see [`statistics_correction`][]).
    """
    J, weight = tile.joint(), tile.system_weight()
    # with explicit gaps, the gap amplitudes stand for the flagged samples, under N⁻¹ unmasked
    data = tod if tile.gaps is None else tile.M(tod)
    residual = tree.sub(data, J(unknowns))
    fmt = tile.tod_format
    out = {'good': jnp.sum(fmt.as_array(tile.M(fmt.ones())) != 0, axis=-1)}
    if diagnostics.residual_map:
        out['map'] = tile.H.T(tile.white_weight()(residual))
    if not (diagnostics.chi2 or diagnostics.bands):
        return out
    if isinstance(tile.noise_model, LowRankNoiseModel):
        # under the low-rank weight P D⁻¹ P, the statistics pair P r with D⁻¹ P r, as its
        # correction does (the χ² is that of the weight, but the frequencies of the flagged
        # samples are not)
        T, inverse_diagonal, P = tile.low_rank_weight()
        residual = P(residual)
        weighted = inverse_diagonal(residual)
        # the correlated modes are marginalized once the whole observation is summed (see
        # `statistics_correction`)
        out['modes'] = T.T(weighted)
        out['low'] = jnp.fft.rfft(fmt.as_array(residual))[..., : T.in_structure.shape[0]]
    else:
        weighted = weight(residual)
    r, wr = fmt.as_array(residual), fmt.as_array(weighted)
    if diagnostics.chi2:
        out['chi2'] = jnp.sum(r * wr, axis=-1)
    if diagnostics.bands:
        n = r.shape[-1]
        product = jnp.real(jnp.conj(jnp.fft.rfft(r)) * jnp.fft.rfft(wr))
        out['bands'] = _band_sums(product, tile.sample_rate, n, diagnostics.bands)
        # the expected χ² of a band: its frequencies, times the fraction of the samples that are good
        ones = jnp.ones(n // 2 + 1)
        share = _band_sums(ones, tile.sample_rate, n, diagnostics.bands)[:, None, None]
        out['band_samples'] = share * out['good']
    return out


def statistics_correction(
    tile: Tile, low: Array, z: Array, diagnostics: DiagnosticsConfig
) -> dict[str, Array]:
    r"""What the correlated modes of a low-rank noise model take from the statistics of a tile.

    Under $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$, the $\chi^2$ of a residual $r$ splits into
    that of the residual less the fitted modes, $(P r - T z)^\top D^{-1} (P r - T z)$, which is per
    channel, and that of the modes, $z^\top z$ (see [`modes_statistics`][]), with
    $z = K T^\top D^{-1} P r$ summed over the tiles of the observation. The first is the $\chi^2$
    under $D^{-1}$ less $(2 P r - T z)^\top D^{-1} T z$, which lives below the last coupled bin
    edge, where `low` holds the coefficients of $P r$.

    Args:
        tile: The tile.
        low: The rfft coefficients of $P r$ below the last coupled bin edge, `(stream, det, freq)`.
        z: $K$ of the sums of the observation, on the amplitudes of its modes.
        diagnostics: The statistics to correct.

    Returns:
        The corrections of `chi2`, `(stream, det)`, and of the band sums `bands`, `(band, stream,
        det)`, of those turned on.
    """
    T, inverse_diagonal, _ = tile.low_rank_weight()
    fmt = tile.tod_format
    n_low = low.shape[-1]
    # the TOD T z keeps only the real parts of its coefficients at the zero and Nyquist frequencies
    real_only = rfft_folds(fmt.n_samples, n_low) == 1
    fitted = T.coefficients(z)
    fitted = jnp.where(real_only, fitted.real, fitted).reshape(low.shape)
    # D⁻¹ is diagonal in each bin: its product with T z is that of its modes
    weighted = T.weighted_by(inverse_diagonal.diagonal).coefficients(z)
    weighted = jnp.where(real_only, weighted.real, weighted).reshape(low.shape)
    product = jnp.real(jnp.conj(2 * low - fitted) * weighted)
    sums = _band_sums(product, tile.sample_rate, fmt.n_samples, diagnostics.bands)
    out = {}
    if diagnostics.chi2:
        out['chi2'] = jnp.sum(sums, axis=0)
    if diagnostics.bands:
        out['bands'] = sums
    return out


def modes_statistics(
    z: Float[Array, 'segment frequency mode 2'],
    core: Float[Array, 'segment coupled mode mode'],
    model: LowRankNoiseModel,
    n_samples: int,
) -> dict[str, Array]:
    r"""The $\chi^2$ of the correlated modes of a low-rank noise model, per observation.

    The amplitudes $z$ of the modes fitted to the residual (see [`statistics_correction`][]) have
    the identity for prior covariance, so their $\chi^2$ is $z^\top z$. With that of the channels,
    once the modes are fitted out of them, it makes the $\chi^2$ of the observation under $N^{-1}$.

    Args:
        z: The amplitudes of the modes of each observation.
        core: The mode covariance $G_b$ of each observation.
        model: The model of the observations, for its bins.
        n_samples: The number of samples of their TODs.

    Returns:
        `modes_chi2`, and `modes_samples`, the number of real amplitudes of the modes of nonzero
        power, which bounds its expected value from above; each `(segment,)`.
    """
    n_coupled = core.shape[1]
    bounds = _bin_bounds(model.edges, model.sample_rate, n_samples)[: n_coupled + 1]
    # a real and an imaginary part per frequency, but at the zero and Nyquist frequencies
    folds = rfft_folds(n_samples, bounds[-1])
    amplitudes_per_mode = np.array([folds[a:b].sum() for a, b in itertools.pairwise(bounds)])
    active = jnp.diagonal(core, axis1=-2, axis2=-1) > 0  # (segment, coupled, mode)
    # TODO: normalize by the expected value, tr(1 - K), rather than by the number of amplitudes
    return {
        'modes_chi2': jnp.sum(z**2, axis=(1, 2, 3)),
        'modes_samples': jnp.sum(active, axis=-1) @ amplitudes_per_mode,
    }


def _band_sums(product: Array, sample_rate: Array, n: int, bands: tuple[float, ...]) -> Array:
    r"""The $\chi^2$ of each band, `(band, ...)`, from the products of the rfft coefficients.

    The products are those of the first frequencies of `n` samples, by Parseval.
    """
    n_frequencies = product.shape[-1]
    frequencies = jnp.arange(n_frequencies) * sample_rate / n
    weights = jnp.asarray(rfft_folds(n, n_frequencies), product.dtype)
    band = jnp.searchsorted(jnp.asarray(bands), frequencies, side='right')
    in_band = jax.nn.one_hot(band, len(bands) + 1, dtype=product.dtype) * weights[:, None]
    return jnp.moveaxis(product @ in_band, -1, 0) / n
