r"""Goodness-of-fit statistics of the solution of the mapmaker, one detector tile at a time.

The residual of a tile is $r = d - J \hat x$, with $J$ and the data $d$ as the normal system takes
them (see [`furax.mapmaking._bucket`][]), and $\hat x$ every unknown solved for. Its
$\chi^2 = r^\top W' r$ under the weight $W'$ of the system splits over the channels (the streams of
each detector) and, by Parseval, over frequency bands:

$$ \chi^2_c = \frac1n \sum_k w_k \, \mathrm{Re}\big(\hat r_c(f_k)^* \, \widehat{W' r}_c(f_k)\big), $$

over the rfft frequencies $f_k$ of the $n$ samples, with $w_k = 1$ at zero and Nyquist and 2
elsewhere. For a stationary weight equal to the inverse noise, each frequency contributes about
$w_k$ times the fraction of good samples, which normalizes each band.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from furax.core import BlockRowOperator

from ._model import ObservationModel, TileTemplates
from .config import DiagnosticsConfig
from .weight import WeightOperator


def tile_statistics(
    obs: ObservationModel,
    templates: TileTemplates | None,
    tod: PyTree[Array],
    sky: Any,
    explicit: PyTree[Array] | None,
    common: PyTree[Array] | None,
    gaps: Array | None,
    diagnostics: DiagnosticsConfig,
) -> dict[str, Array]:
    r"""The residual statistics of one detector tile.

    Args:
        obs: The tile's model.
        templates: Its templates, if any.
        tod: Its TOD.
        sky: The map solved for, on the whole sky.
        explicit: The tile's own explicit template amplitudes, if any.
        common: Its observation's shared template amplitudes, if any.
        gaps: The amplitudes of its explicit gaps, if any.
        diagnostics: The statistics to compute.

    Returns:
        `good`, the number of good samples, `(stream, det)`, and those of the statistics turned
        on: `chi2`, `(stream, det)`; `bands`, the reduced $\chi^2$ of each band, `(band, stream,
        det)`; `map`, the residual binned under the white-noise weight, before its normalization
        by the white pixel system.
    """
    operators = {} if templates is None else templates.operators
    blocks, inputs = [obs.H], [sky]
    for kind, amplitudes in (('explicit', explicit), ('common', common)):
        if amplitudes is not None:
            operator = operators[kind]
            assert operator is not None  # solved for, so built
            blocks.append(operator)
            inputs.append(amplitudes)
    data = tod
    if gaps is not None:
        # the gap amplitudes stand for the flagged samples, under N⁻¹ unmasked
        blocks = [obs.M @ block for block in blocks]
        blocks.append(obs.gap_basis())
        inputs.append(gaps)
        data = obs.M(tod)
        assert isinstance(obs.W, WeightOperator)  # explicit gaps take the plain weight
        weight = obs.W.weight
    else:
        implicit = None if templates is None else templates.implicit
        weight = obs.deprojected_weight(implicit)
    residual = jax.tree.map(jnp.subtract, data, BlockRowOperator(blocks)(inputs))
    fmt = obs.tod_format
    out = {'good': jnp.sum(fmt.as_array(obs.M(fmt.ones())) != 0, axis=-1)}
    if diagnostics.residual_map:
        out['map'] = obs.H.T(obs.diag_W()(residual))
    if not (diagnostics.chi2 or diagnostics.bands):
        return out
    r, wr = fmt.as_array(residual), fmt.as_array(weight(residual))
    if diagnostics.chi2:
        out['chi2'] = jnp.sum(r * wr, axis=-1)
    if diagnostics.bands:
        out['bands'] = _band_chi2(r, wr, out['good'], obs.sample_rate, diagnostics.bands)
    return out


def _band_chi2(
    r: Array, wr: Array, good: Array, sample_rate: Array, bands: tuple[float, ...]
) -> Array:
    r"""The reduced $\chi^2$ of each band, `(band, stream, det)`, by Parseval."""
    n = r.shape[-1]
    product = jnp.real(jnp.conj(jnp.fft.rfft(r)) * jnp.fft.rfft(wr))
    frequencies = jnp.arange(product.shape[-1]) * sample_rate / n
    weights = jnp.full(product.shape[-1], 2.0).at[0].set(1.0)
    if n % 2 == 0:
        weights = weights.at[-1].set(1.0)
    band = jnp.searchsorted(jnp.asarray(bands), frequencies, side='right')
    n_bands = len(bands) + 1
    in_band = jax.nn.one_hot(band, n_bands, dtype=product.dtype) * weights[:, None]  # (freq, band)
    band_chi2 = jnp.moveaxis(product @ in_band, -1, 0) / n  # (band, stream, det)
    # the expected χ² of a band: its frequencies, times the fraction of the samples that are good
    expected = jnp.sum(in_band, axis=0)[:, None, None] * good / n
    return band_chi2 / jnp.where(expected > 0, expected, 1)
