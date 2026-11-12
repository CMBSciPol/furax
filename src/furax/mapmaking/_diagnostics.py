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
from .lowrank import LowRankNoiseModel
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
        on: `chi2`, `(stream, det)`; `bands`, the $\chi^2$ of each band, `(band, stream, det)`,
        and `band_samples`, its expected value; `map`, the residual binned under the white-noise
        weight, before its normalization by the white pixel system. Under a low-rank noise model,
        the $\chi^2$ are those under the diagonal weight, and `modes` and `low` give their
        correction (see [`modes_correction`][]).
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
    if gaps is not None:
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
    low_rank = isinstance(obs.noise_model, LowRankNoiseModel)
    # under a low-rank model, the weight is P D⁻¹ P, P the mask unless the gaps are explicit
    masked = residual if gaps is not None else obs.M(residual)
    if diagnostics.bands:
        n = r.shape[-1]
        if low_rank:
            # the masked residual against D⁻¹ of it, as its correction pairs them (the total is
            # the same as with the weight, but the frequencies of the flagged samples are not)
            assert isinstance(obs.W, WeightOperator)
            r, wr = fmt.as_array(masked), fmt.as_array(obs.W.weight(masked))
        product = jnp.real(jnp.conj(jnp.fft.rfft(r)) * jnp.fft.rfft(wr))
        out['bands'] = _band_sums(product, obs.sample_rate, n, diagnostics.bands)
        # the expected χ² of a band: its frequencies, times the fraction of the samples that are good
        ones = jnp.ones(n // 2 + 1)
        share = _band_sums(ones, obs.sample_rate, n, diagnostics.bands)[:, None, None]
        out['band_samples'] = share * out['good']
    if low_rank:
        # the correlated modes are marginalized once the whole observation is summed (see
        # `modes_correction`)
        model = obs.noise_model
        assert isinstance(model, LowRankNoiseModel) and isinstance(obs.W, WeightOperator)
        modes = model.modes_operator(fmt.structure)
        out['modes'] = modes.T(obs.W.weight(masked))
        n_low = modes.in_structure.shape[0]
        out['low'] = jnp.fft.rfft(fmt.as_array(masked))[..., :n_low]
    return out


def modes_correction(
    obs: ObservationModel,
    low: Array,
    z: Array,
    diagnostics: DiagnosticsConfig,
) -> dict[str, Array]:
    r"""What the correlated modes of a low-rank noise model take from the statistics of a tile.

    Under $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$, the $\chi^2$ of a residual $r$ is that under
    $D^{-1}$ less $(P r)^\top D^{-1} T z$, with $z = K T^\top D^{-1} P r$ summed over the tiles of
    the observation. It lives below the last coupled bin edge, where `low` holds the coefficients
    of $P r$.

    Args:
        obs: The tile's model.
        low: The rfft coefficients of $P r$ below the last coupled bin edge, `(stream, det, freq)`.
        z: $K$ of the sums of the observation, on the amplitudes of its modes.
        diagnostics: The statistics to correct.

    Returns:
        The corrections of `chi2`, `(stream, det)`, and of the band sums `bands`, `(band, stream,
        det)`, of those turned on.
    """
    model = obs.noise_model
    assert isinstance(model, LowRankNoiseModel)
    assert isinstance(obs.W, WeightOperator)
    fmt = obs.tod_format
    n = fmt.n_samples
    modes = obs.W.weight(model.modes_operator(fmt.structure)(z))
    product = jnp.real(jnp.conj(low) * jnp.fft.rfft(fmt.as_array(modes))[..., : low.shape[-1]])
    sums = _band_sums(product, obs.sample_rate, n, diagnostics.bands)
    out = {}
    if diagnostics.chi2:
        out['chi2'] = jnp.sum(sums, axis=0)
    if diagnostics.bands:
        out['bands'] = sums
    return out


def _band_sums(product: Array, sample_rate: Array, n: int, bands: tuple[float, ...]) -> Array:
    r"""The $\chi^2$ of each band, `(band, ...)`, from the products of the rfft coefficients.

    The products are those of the first frequencies of `n` samples, by Parseval.
    """
    n_frequencies = product.shape[-1]
    frequencies = jnp.arange(n_frequencies) * sample_rate / n
    weights = jnp.full(n_frequencies, 2.0).at[0].set(1.0)
    if n % 2 == 0 and n_frequencies == n // 2 + 1:
        weights = weights.at[-1].set(1.0)
    band = jnp.searchsorted(jnp.asarray(bands), frequencies, side='right')
    in_band = jax.nn.one_hot(band, len(bands) + 1, dtype=product.dtype) * weights[:, None]
    return jnp.moveaxis(product @ in_band, -1, 0) / n
