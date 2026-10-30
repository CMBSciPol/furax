"""The templates of an observation, and their bases, built from its data.

The basis builders return the [`Basis`][furax.mapmaking.templates.bases.Basis] of functions of time
of one kind of template: polynomials per scan interval, scan- and HWP-synchronous signals,
temperature-to-polarization leakage. The template builders make the
[`Template`][furax.mapmaking.templates.Template]s of one kind from its configuration and the
observation's data, and [`build_templates`][furax.mapmaking.templates.build_templates] all those a
[`TemplatesConfig`][furax.mapmaking.config.TemplatesConfig] enables.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax import Array
from jax.scipy.signal import detrend
from jaxtyping import Complex, DTypeLike, Float

from furax.math import bspline
from furax.math.coords import XiEtaAngles
from furax.math.legendre import legvander
from furax.math.zernike import zernike_vander
from furax.obs import TODFormat

from .._observation import ReaderField, _sample_rate
from ..config import (
    AzimuthHWPSynchronousConfig,
    AzimuthSynchronousConfig,
    BinnedAzimuthHWPSynchronousConfig,
    BinsConfig,
    CommonModeConfig,
    HWPSynchronousConfig,
    PolynomialOrders,
    SplineDriftConfig,
    SplineHWPSynchronousConfig,
    T2PConfig,
    TemplatesConfig,
    _per_stream_groups,
)
from ._operator import Template
from .bases import (
    Basis,
    DecimatedBasis,
    KroneckerBasis,
    SplineBasis,
    TensorBasis,
    WindowedBasis,
)

__all__ = [
    'build_templates',
    'required_reader_fields',
    'spline_drift_templates',
    'azimuth_synchronous_templates',
    'hwp_synchronous_templates',
    'azimuth_hwp_synchronous_templates',
    'binned_azimuth_hwp_synchronous_templates',
    'spline_hwp_synchronous_templates',
    't2p_templates',
    'common_mode_templates',
    'zernike_couplings',
    'winding_couplings',
    'polarization_field_couplings',
]


def build_templates(
    config: TemplatesConfig,
    data: Mapping[str, Any],
    tod_format: TODFormat,
    *,
    temperature: Float[Array, 'det samp'] | None = None,
) -> tuple[dict[str, Template], dict[str, Template]]:
    """The templates `config` enables for an observation, explicit and implicit.

    Args:
        config: The templates to build.
        data: The observation's fields, as the reader returns them (see
            [`required_reader_fields`][furax.mapmaking.templates.required_reader_fields]).
        tod_format: The observation's TOD, on which the templates are seen.
        temperature: The I timestream, from which the T2P template is built; by default, the I
            stream of the sample data.

    Returns:
        The templates whose amplitudes are solved for with the map, and those marginalized, by
        name.

    Raises:
        ValueError: If `config` enables no template.
        NotImplementedError: If `config` enables the ground template.
    """
    if config.ground is not None:
        raise NotImplementedError(
            'Ground templates are not supported in the multi-observation path.'
        )
    explicit: dict[str, Template] = {}
    implicit: dict[str, Template] = {}
    builders = {
        'spline_drift': spline_drift_templates,
        'azimuth_synchronous': azimuth_synchronous_templates,
        'hwp_synchronous': hwp_synchronous_templates,
        'azimuth_hwp_synchronous': azimuth_hwp_synchronous_templates,
        'binned_azimuth_hwp_synchronous': binned_azimuth_hwp_synchronous_templates,
        'spline_hwp_synchronous': spline_hwp_synchronous_templates,
    }
    for name, build in builders.items():
        if (section := getattr(config, name)) is not None:
            (explicit if section.explicit else implicit).update(build(section, data, tod_format))
    if config.t2p is not None:
        explicit.update(t2p_templates(config.t2p, data, tod_format, temperature=temperature))
    if config.common_mode is not None:
        explicit.update(common_mode_templates(config.common_mode, data, tod_format))
    if not explicit and not implicit:
        raise ValueError('config.templates is set but no template is active.')
    return explicit, implicit


def required_reader_fields(config: TemplatesConfig) -> set[str]:
    """The fields of an observation the templates `config` enables are built from."""
    fields: set[str] = set()
    if config.spline_drift is not None:
        fields |= {ReaderField.TIMESTAMPS, ReaderField.VALID_SCANNING_MASKS}
    if config.azimuth_synchronous is not None:
        fields |= {ReaderField.AZIMUTH}
    if config.hwp_synchronous is not None:
        fields |= {ReaderField.HWP_ANGLES}
    if config.azimuth_hwp_synchronous is not None:
        fields |= {ReaderField.AZIMUTH, ReaderField.HWP_ANGLES}
    if any(
        section is not None and section.split_scans
        for section in (config.azimuth_synchronous, config.azimuth_hwp_synchronous)
    ):
        fields |= {ReaderField.LEFT_SCAN_MASK, ReaderField.RIGHT_SCAN_MASK}
    if config.binned_azimuth_hwp_synchronous is not None:
        fields |= {ReaderField.AZIMUTH, ReaderField.HWP_ANGLES}
    if config.spline_hwp_synchronous is not None:
        fields |= {ReaderField.TIMESTAMPS, ReaderField.HWP_ANGLES}
    if config.t2p is not None:
        fields |= {ReaderField.SAMPLE_DATA, ReaderField.TIMESTAMPS}
        if config.t2p.zernike_order is not None or config.t2p.quadrupole_order is not None:
            fields |= {ReaderField.DETECTOR_QUATERNIONS}
    if config.common_mode is not None:
        fields |= {
            ReaderField.TIMESTAMPS,
            ReaderField.VALID_SCANNING_MASKS,
            ReaderField.DETECTOR_QUATERNIONS,
        }
    if config.ground is not None:
        raise NotImplementedError(
            'Ground templates are not supported in the multi-observation path.'
        )
    return fields


def spline_drift_templates(
    config: SplineDriftConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """Cubic B-splines of time over the whole observation.

    One template per group of streams fitted with the same knot spacing (see `_stream_groups`).
    """
    basis = partial(_spline_drift_basis, data, dtype=tod_format.dtype)
    return _stream_groups('spline_drift', config.samples_per_knot, basis, tod_format)


def azimuth_synchronous_templates(
    config: AzimuthSynchronousConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """Functions of the azimuth, by scan direction if `config.split_scans`.

    Bins or a cubic B-spline, whichever `config` sets. One template per group of streams fitted
    alike (see `_stream_groups`).
    """
    azimuth, dtype = data[ReaderField.AZIMUTH], tod_format.dtype
    scan_masks = None
    if config.split_scans:
        scan_masks = (data[ReaderField.LEFT_SCAN_MASK], data[ReaderField.RIGHT_SCAN_MASK])
    if config.bins is not None:
        return _stream_groups(
            'azimuth_synchronous',
            config.bins,
            lambda bins: azimuth_bins_basis(bins, azimuth, dtype, scan_masks=scan_masks),
            tod_format,
        )
    assert config.n_knots is not None  # exactly one alternative is set
    return _stream_groups(
        'azimuth_synchronous',
        config.n_knots,
        lambda n_knots: azimuth_spline_basis(n_knots, azimuth, dtype, scan_masks=scan_masks),
        tod_format,
    )


def hwp_synchronous_templates(
    config: HWPSynchronousConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """Harmonics of the HWP angle, on every stream."""
    basis = hwp_synchronous_basis(
        config.n_harmonics, data[ReaderField.HWP_ANGLES], tod_format.dtype
    )
    return {'hwp_synchronous': _on_every_stream(basis, tod_format)}


def azimuth_hwp_synchronous_templates(
    config: AzimuthHWPSynchronousConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """HWP harmonics modulated by Legendre polynomials of the azimuth, on every stream.

    One template for each scan direction if `config.split_scans`.
    """
    azimuth, hwp_angles = data[ReaderField.AZIMUTH], data[ReaderField.HWP_ANGLES]

    def template(scan_mask: Array | None = None) -> Template:
        basis = azimuth_hwp_synchronous_basis(
            config.legendre,
            config.n_harmonics,
            azimuth,
            hwp_angles,
            tod_format.dtype,
            scan_mask=scan_mask,
        )
        return _on_every_stream(basis, tod_format)

    if not config.split_scans:
        return {'azimuth_hwp_synchronous': template()}
    return {
        'azimuth_hwp_synchronous_left': template(data[ReaderField.LEFT_SCAN_MASK]),
        'azimuth_hwp_synchronous_right': template(data[ReaderField.RIGHT_SCAN_MASK]),
    }


def binned_azimuth_hwp_synchronous_templates(
    config: BinnedAzimuthHWPSynchronousConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """HWP harmonics in each azimuth bin, on every stream."""
    basis = binned_azimuth_hwp_synchronous_basis(
        config.bins,
        config.n_harmonics,
        data[ReaderField.AZIMUTH],
        data[ReaderField.HWP_ANGLES],
        tod_format.dtype,
    )
    return {'binned_azimuth_hwp_synchronous': _on_every_stream(basis, tod_format)}


def spline_hwp_synchronous_templates(
    config: SplineHWPSynchronousConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """HWP harmonics whose amplitudes are cubic B-splines in time, on every stream."""
    times = data[ReaderField.TIMESTAMPS]
    basis = spline_hwp_synchronous_basis(
        times,
        data[ReaderField.HWP_ANGLES],
        config.resolve_n_knots(times.size),
        config.harmonics,
        tod_format.dtype,
    )
    return {'spline_hwp_synchronous': _on_every_stream(basis, tod_format)}


def t2p_templates(
    config: T2PConfig,
    data: Mapping[str, Any],
    tod_format: TODFormat,
    *,
    temperature: Float[Array, 'det samp'] | None = None,
) -> dict[str, Template]:
    """The temperature-to-polarization leakage on the Q and U streams of the TOD, none without.

    Each detector's leakage is its own temperature stream, `temperature`, by default the I stream
    of the sample data. Q and U each fit their own amplitudes per detector, or, with
    `config.zernike_order` or `config.quadrupole_order`, a few amplitudes shared by the detectors:
    the leakage follows a polarization field over the focal plane, from the detector quaternions
    in `data`.
    """
    streams = _qu_streams(tod_format)
    if not streams:
        return {}
    if temperature is None:
        temperature = data[ReaderField.SAMPLE_DATA].i
    basis = t2p_basis(
        temperature,
        tod_format.dtype,
        fit_band=config.fit_band,
        sample_rate=_sample_rate(data[ReaderField.TIMESTAMPS]),
        decimation_factor=config.decimation_factor,
    )
    if config.zernike_order is None and config.quadrupole_order is None:
        return {'t2p': Template(basis, tod_format, streams=streams)}
    angles = _focal_plane_angles(data)
    joint = config.joint_qu
    if config.quadrupole_order is None:
        assert config.zernike_order is not None  # guaranteed by the test above
        coupling = zernike_couplings(
            angles.xi,
            angles.eta,
            angles.gamma,
            config.zernike_order,
            config.zernike_radius,
            streams=streams if joint else None,
        )
    else:
        coupling = winding_couplings(
            angles.xi,
            angles.eta,
            angles.gamma,
            config.quadrupole_order,
            config.zernike_radius,
            decentred=config.decentred,
            uniform=config.decentred,
            streams=streams if joint else None,
        )
    coupling = coupling.astype(tod_format.dtype)
    # one common field for the detectors; its own for each stream unless Q and U share one
    template = Template(
        basis,
        tod_format,
        streams=streams,
        per=() if joint else ('stream',),
        coupling=coupling if joint else coupling[None],
    )
    return {'t2p': template}


def common_mode_templates(
    config: CommonModeConfig, data: Mapping[str, Any], tod_format: TODFormat
) -> dict[str, Template]:
    """The polarization common to the detectors on the Q and U streams of the TOD, none without.

    The detectors see one polarization pattern over the focal plane, `winding_couplings` from the
    detector quaternions in `data`, with a time dependence they share: a cubic B-spline of time
    over the whole observation for each mode of the pattern.
    """
    streams = _qu_streams(tod_format)
    if not streams:
        return {}
    basis = _spline_drift_basis(data, config.samples_per_knot, tod_format.dtype)
    angles = _focal_plane_angles(data)
    coupling = winding_couplings(
        angles.xi,
        angles.eta,
        angles.gamma,
        config.quadrupole_order,
        config.radius,
        decentred=config.decentred,
        uniform=config.uniform,
        streams=streams,
    ).astype(tod_format.dtype)
    template = Template(basis, tod_format, streams=streams, per=(), coupling=coupling)
    return {'common_mode': template}


def zernike_couplings(
    xi: Float[Array, ' det'],
    eta: Float[Array, ' det'],
    gamma: Float[Array, ' det'],
    order: int,
    radius: float,
    *,
    streams: str | None = None,
) -> Float[Array, '*stream det mode']:
    r"""Couplings of a leakage that follows a polarization field over the focal plane.

    The field $P = \sum_k c_k Z_k(\xi / R, \eta / R)$, with complex $c_k$ over the Zernike
    polynomials up to radial order `order`, reaches detector $i$, of polarization angle $\gamma_i$,
    as the leakage coefficient $e^{2i\gamma_i} P(\xi_i, \eta_i)$. On Q or U alone, that is a real
    combination of $\cos 2\gamma_i Z_k$ and $\sin 2\gamma_i Z_k$: the couplings, `(det, mode)`
    with the modes ordered by Zernike polynomial in the standard (ANSI) order, each with its
    $(\cos, \sin)$ pair.

    With `streams`, such as `'qu'`, the couplings are those of Q and U together, `(stream, det,
    mode)`: the modes are then $(\mathrm{Re}\, c_k, \mathrm{Im}\, c_k)$, common to the streams,
    seen as $\mathrm{Re}(c_k e^{2i\gamma})$ on Q and $\mathrm{Im}(c_k e^{2i\gamma})$ on U.

    Args:
        xi: The detectors' first focal-plane coordinate, in radians.
        eta: The detectors' second focal-plane coordinate, in radians.
        gamma: The detectors' polarization angle, in radians.
        order: The highest radial order of the field.
        radius: The focal-plane radius $R$, in radians.
        streams: The streams `'q'` and `'u'` to couple together; `None` for the couplings of one
            stream.

    Raises:
        ValueError: If `streams` names a stream other than `'q'` and `'u'`.
    """
    x, y = xi / radius, eta / radius
    return polarization_field_couplings(zernike_vander(x, y, order), gamma, streams=streams)


def winding_couplings(
    xi: Float[Array, ' det'],
    eta: Float[Array, ' det'],
    gamma: Float[Array, ' det'],
    radial_order: int,
    radius: float,
    *,
    decentred: bool = True,
    uniform: bool = True,
    streams: str | None = None,
) -> Float[Array, '*stream det mode']:
    r"""Couplings of a leakage that follows a quadrupolar polarization field over the focal plane.

    With $\zeta = (-\eta - i\xi) / R$ the complex position of a detector, the field of optics
    symmetric about their axis is $P = \zeta^2 g(|\zeta|^2)$: exactly two turns of the
    polarization around the axis. Its modes are the monomials $\zeta^{j+2}\bar\zeta^{j}$, for $j$
    below `radial_order`. If `decentred`, the first-order terms of an axis shifted from the origin
    are added: $\zeta^{j+1}\bar\zeta^{j}$ and, from $j = 1$, $\zeta^{j+2}\bar\zeta^{j-1}$.
    If `uniform`, so is the constant. The modes are ordered by degree and then by number of turns,
    each with the couplings of [`polarization_field_couplings`][].

    Args:
        xi: The detectors' first focal-plane coordinate, in radians.
        eta: The detectors' second focal-plane coordinate, in radians.
        gamma: The detectors' polarization angle, in radians.
        radial_order: The number of terms of the axisymmetric field, at least 1.
        radius: The focal-plane radius $R$, in radians.
        decentred: Whether to include the terms of a shifted axis.
        uniform: Whether to include the constant mode.
        streams: The streams `'q'` and `'u'` to couple together; `None` for the couplings of one
            stream.

    Raises:
        ValueError: If `radial_order` is below 1, or `streams` names a stream other than `'q'` and
            `'u'`.
    """
    if radial_order < 1:
        raise ValueError(f"'radial_order' must be >= 1, got {radial_order}")
    zeta = (-eta - 1j * xi) / radius
    fields = jnp.stack(
        [
            zeta**p * jnp.conj(zeta) ** q
            for p, q in _winding_exponents(radial_order, decentred, uniform)
        ],
        axis=-1,
    )
    return polarization_field_couplings(fields, gamma, streams=streams)


def polarization_field_couplings(
    fields: Complex[Array, 'det field'],
    gamma: Float[Array, ' det'],
    *,
    streams: str | None = None,
) -> Float[Array, '*stream det mode']:
    r"""Couplings of a leakage that follows a polarization field over the focal plane.

    The field $P = \sum_k c_k F_k$, with complex $c_k$ over the functions $F_k$ whose values at the
    detectors are `fields`, reaches detector $i$, of polarization angle $\gamma_i$, as the leakage
    coefficient $e^{2i\gamma_i} P_i$. With $w_{ik} = F_k(i) e^{2i\gamma_i}$, a stream alone sees
    $a_k \mathrm{Re}\, w_{ik} + b_k \mathrm{Im}\, w_{ik}$, a real combination: the couplings,
    `(det, mode)` with the modes $(a_k, b_k)$ of each function in turn.

    With `streams`, such as `'qu'`, the couplings are those of Q and U together, `(stream, det,
    mode)`: the modes are then $(\mathrm{Re}\, c_k, \mathrm{Im}\, c_k)$, common to the streams,
    seen as $\mathrm{Re}(c_k w_{ik})$ on Q and $\mathrm{Im}(c_k w_{ik})$ on U.

    Args:
        fields: The values of the functions at the detectors, `(det, field)`.
        gamma: The detectors' polarization angle, in radians.
        streams: The streams `'q'` and `'u'` to couple together; `None` for the couplings of one
            stream.

    Raises:
        ValueError: If `streams` names a stream other than `'q'` and `'u'`.
    """
    if streams is not None and not set(streams) <= set('qu'):
        raise ValueError(f"'streams' must name Q and U streams only, got {streams!r}")
    w = fields * jnp.exp(2j * gamma)[:, None]  # (det, field)
    if streams is None:
        couplings = jnp.stack([w.real, w.imag], axis=-1)  # (det, field, 2)
    else:
        # Q sees (Re, -Im) of the complex amplitude, U sees (Im, Re)
        couplings = jnp.stack(
            [
                jnp.stack([w.real, -w.imag] if s == 'q' else [w.imag, w.real], axis=-1)
                for s in streams
            ]
        )  # (stream, det, field, 2)
    return couplings.reshape(*couplings.shape[:-2], -1)


def _winding_exponents(radial_order: int, decentred: bool, uniform: bool) -> list[tuple[int, int]]:
    r"""The powers `(p, q)` of the monomials $\zeta^p \bar\zeta^q$ of `winding_couplings`."""
    exponents = [(j + 2, j) for j in range(radial_order)]
    if decentred:
        exponents += [(j + 1, j) for j in range(radial_order)]
        exponents += [(j + 2, j - 1) for j in range(1, radial_order)]
    if uniform:
        exponents.append((0, 0))
    return sorted(exponents, key=lambda pq: (sum(pq), pq[0] - pq[1]))


def _on_every_stream(basis: Basis, tod_format: TODFormat) -> Template:
    """A template on every stream of `tod_format`, with its own amplitudes on each."""
    return Template(
        basis,
        tod_format,
        streams=None if tod_format.stokes is None else tod_format.stokes.lower(),
    )


def _stream_groups[T](
    name: str,
    per_stream: T | dict[str, T],
    basis: Callable[[T], Basis],
    tod_format: TODFormat,
) -> dict[str, Template]:
    """One template per group of streams fitted with the same option, such as Legendre orders.

    It is named `name` if every stream of the TOD is fitted alike, `<name>_<streams>` otherwise,
    such as `spline_drift_qu`. Options per stream need a demodulated TOD.
    """
    if tod_format.stokes is None:
        assert not isinstance(per_stream, dict)  # per-stream options need demodulation
        return {name: Template(basis(per_stream), tod_format)}
    streams = tod_format.stokes.lower()
    return {
        name if group == streams else f'{name}_{group}': Template(
            basis(value), tod_format, streams=group
        )
        for group, value in _per_stream_groups(per_stream, streams).items()
    }


def _qu_streams(tod_format: TODFormat) -> str:
    """The Q and U streams of the TOD, as `TODFormat.stokes` names them, none without."""
    if tod_format.stokes is None:
        return ''
    return ''.join(s for s in tod_format.stokes.lower() if s in 'qu')


def _focal_plane_angles(data: Mapping[str, Any]) -> XiEtaAngles:
    """The detectors' positions and orientations on the focal plane, from their quaternions."""
    return XiEtaAngles.from_quaternion(
        Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS])
    )


def _spline_drift_basis(data: Mapping[str, Any], samples_per_knot: int, dtype: DTypeLike) -> Basis:
    """A cubic B-spline of time over the observation, with a knot every `samples_per_knot`."""
    times = data[ReaderField.TIMESTAMPS]
    return spline_drift_basis(
        max(2, times.size // samples_per_knot),
        times=times,
        dtype=dtype,
        valid_mask=data[ReaderField.VALID_SCANNING_MASKS],
    )


def _bin_weights(
    x: Float[Array, ' samp'],
    n_bins: int,
    interpolate: bool,
    smooth: bool,
    dtype: DTypeLike,
) -> Float[Array, '{n_bins} samp']:
    """Assign each sample to a bin of `x`.

    Splits the range of `x` into `n_bins` equal bins. Entry `[j, s]` is how much sample `s` belongs
    to bin `j`. With `interpolate=False` this is one-hot: 1 for the bin the sample falls in, 0
    elsewhere. With `interpolate=True` a sample near a bin edge is shared with its neighbour
    (triangular weights, or sin² hats if `smooth`), each sample's weights summing to 1.

    Each row is then one basis function of a binned template (a bin's time profile).
    """
    n_samps = x.size
    lo = jnp.min(x)
    hi = jnp.max(x) + 1e-8  # nudge so the global max falls in the last bin
    edges = jnp.linspace(lo, hi, n_bins + 1)

    if not interpolate:
        # Hard assignment: each sample contributes 1.0 to exactly its bin.
        sample_bin = jnp.digitize(x, edges[1:])
        bins = jnp.zeros((n_bins, n_samps)).at[sample_bin, jnp.arange(n_samps)].set(1.0)
        return bins.astype(dtype)

    # Soft assignment: triangular weights peaking at each bin centre, falling
    # linearly to zero one bin-width (`delta`) away. Shape (n_bins, n_samps).
    centres = 0.5 * (edges[:-1] + edges[1:])
    delta = (hi - lo) / n_bins
    triangular = jnp.clip(1 - jnp.abs(x[None, :] - centres[:, None]) / delta, min=0)

    if smooth:
        # sin^2 reshaping of the triangle, renormalised so weights sum to 1 per sample.
        bins = jnp.sin((jnp.pi / 2) * triangular) ** 2
        bins /= jnp.sum(bins, axis=0)[None, :]
    else:
        # Linear interpolation; clamp samples beyond the end centres to the edge bins.
        bins = triangular.at[0, x < centres[0]].set(1.0).at[-1, x > centres[-1]].set(1.0)

    return bins.astype(dtype)


def _legendre_values(
    u: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, '{max_order-min_order+1} samp']:
    """Legendre polynomials of orders `min_order..max_order` (inclusive).

    Evaluated on `u`, already rescaled to `[-1, 1]`.
    """
    return legvander(u, max_order)[:, min_order:].T.astype(dtype)


def _legendre(
    x: Float[Array, ' samp'],
    min_order: int,
    max_order: int,
    dtype: DTypeLike,
) -> Float[Array, '{max_order-min_order+1} samp']:
    """Legendre polynomials of orders `min_order..max_order` (inclusive).

    Evaluated on `x`, rescaled to `[-1, 1]` over its global range.
    """
    u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
    return _legendre_values(u, min_order, max_order, dtype)


def _by_scan(
    values: Float[Array, 'k samp'], scan_masks: Sequence[Array] | None
) -> Float[Array, 'k2 samp']:
    """`values` times each scan mask in turn, stacked: a set of functions per mask."""
    if scan_masks is None:
        return values
    return jnp.concatenate([values * mask.astype(values.dtype) for mask in scan_masks])


def _harmonics(
    angles: Float[Array, ' samp'],
    harmonics: int | Sequence[int],
    dtype: DTypeLike,
    *,
    dc: bool,
) -> Float[Array, ' rows samp']:
    """Harmonic basis `sin(k·angle), cos(k·angle)` for each harmonic `k`.

    Optionally prepended with a constant (DC) row when `dc` is set.

    `harmonics` is either an int `n` (the harmonics `1..n`) or an explicit sequence of harmonic
    orders.
    """
    h = jnp.arange(1, harmonics + 1) if isinstance(harmonics, int) else jnp.asarray(harmonics)
    sines = jnp.sin(h[:, None] * angles[None, :])
    cosines = jnp.cos(h[:, None] * angles[None, :])
    parts = ([jnp.ones((1, angles.size), dtype=dtype)] if dc else []) + [sines, cosines]
    return jnp.concatenate(parts, axis=0).astype(dtype)


def spline_drift_basis(
    n_knots: int,
    times: Float[Array, ' samp'],
    dtype: DTypeLike,
    valid_mask: Float[Array, ' samp'] | None = None,
) -> Basis:
    """Basis for a drift template, a cubic B-spline of time over the whole observation.

    One amplitude per knot, `n_knots + 2` in all. A knot that no valid sample sees has an exactly
    zero row in the Gram. `valid_mask` optionally zeroes flagged samples (1 = keep, 0 = drop) so
    they neither carry template signal nor constrain the fitted amplitudes.
    """
    offset, taps = bspline.spline_window(times, n_knots)  # taps (samp, 4)
    if valid_mask is not None:
        taps = taps * valid_mask[:, None].astype(taps.dtype)
    lengths = _spline_block_lengths(times.size, n_knots)
    return SplineBasis(offset, taps.astype(dtype), n_knots + 2, block_lengths=lengths)


def _spline_block_lengths(n_samples: int, n_knots: int) -> tuple[int, int] | None:
    """The `block_lengths` of a spline of `n_knots` knots over evenly spaced samples.

    The samples lie `Δ = (n_samples − 1) / (n_knots − 2)` apart from knot to knot, so a window
    starts at the same knot for `⌊Δ⌋` or `⌈Δ⌉` of them. One sample of slack either way leaves room
    for the jitter of real timestamps. `None` when knots are closer than samples, so that some
    knot starts no window.
    """
    n_intervals = n_knots - 2
    if n_intervals < 1:  # a single run
        return 1, n_samples
    if n_samples - 1 < n_intervals:
        return None
    fewest = (n_samples - 1) // n_intervals
    most = -(-(n_samples - 1) // n_intervals)
    return max(fewest - 1, 1), most + 1


_T2P_BATCH_SIZE = 32
"""Detectors band-passed at once by [`t2p_basis`][]."""


def t2p_basis(
    temperature: Float[Array, 'det samp'],
    dtype: DTypeLike,
    fit_band: tuple[float, float] | None = None,
    sample_rate: Float[Array, ''] | float = 1.0,
    decimation_factor: int = 1,
) -> Basis:
    """Per-detector basis for a temperature-to-polarization leakage template.

    Each detector's basis is just its own temperature stream, so fitting one amplitude per detector
    estimates how much temperature leaks into its polarization.

    `fit_band=(f0, f1)` restricts the temperature basis to that frequency band (Hz), so the leakage
    is both estimated and removed only there, keeping the template a clean linear operator. The
    band-pass treats the stream as periodic, so each stream is first detrended (its least-squares
    line removed): the mismatch between the two ends of a drifting stream would otherwise ring
    into the band.

    `decimation_factor=q` stores the basis on a `q`-times coarser grid to cut memory; the
    coarse-grid Nyquist frequency `sample_rate / 2q` must stay above `f1`. As a rule of thumb keep
    it at a few times `f1` (`q ≲ sample_rate / 6·f1`): the fractional error on the fitted amplitude
    grows like `(f1 / (sample_rate / 2q))²`.

    Assumes `temperature` is already deglitched/gap-filled upstream: a glitch left in it would smear
    across the band and bias the fitted amplitude.
    """
    n_full = temperature.shape[-1]
    q = decimation_factor
    n_dec = -(-n_full // q)  # ceil

    band = None
    if fit_band is not None:
        f0, f1 = fit_band
        freqs = jnp.fft.rfftfreq(n_full, d=1.0 / sample_rate)
        band = (freqs > f0) & (freqs < f1)

    def filter_and_decimate(t: Float[Array, ' samp']) -> Float[Array, ' dec']:
        if band is not None:
            t = jnp.fft.irfft(jnp.fft.rfft(detrend(t)) * band, n=n_full)
        if q > 1:
            # Block-average onto a q-times coarser grid: pad the tail to a whole block,
            # reshape (n_dec, q) and mean. `DecimatedBasis` hold-upsamples back to `n_full` in
            # synthesis (band-limits above sample_rate / 2q).
            t = jnp.pad(t, (0, n_dec * q - n_full)).reshape(n_dec, q).mean(axis=-1)
        return t.astype(dtype)

    t = jax.lax.map(filter_and_decimate, temperature, batch_size=_T2P_BATCH_SIZE)
    basis = TensorBasis(t[:, None, :], batch_shape=t.shape[:1])  # (det, k=1, dec)
    return basis if q == 1 else DecimatedBasis(basis, q, n_full)


def azimuth_bins_basis(
    bins: BinsConfig,
    azimuth: Float[Array, ' samp'],
    dtype: DTypeLike,
    *,
    scan_masks: Sequence[Array] | None = None,
) -> Basis:
    """One amplitude per azimuth bin, one set per scan mask if given."""
    weights = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
    return TensorBasis(_by_scan(weights, scan_masks))


def azimuth_spline_basis(
    n_knots: int,
    azimuth: Float[Array, ' samp'],
    dtype: DTypeLike,
    *,
    scan_masks: Sequence[Array] | None = None,
) -> Basis:
    r"""Cubic B-splines of the azimuth, one set per scan mask, without the constant.

    `n_knots` uniform cubic B-splines $B_k$ cover the azimuth range of the observation, summing to
    one over it. With `scan_masks`, each mask has its own set, $B_k$ times the mask. The functions
    $f_j$ then sum to one on the samples of the masks: that constant is removed by taking the
    differences $f_j - f_{\text{last}}$ to the last function, which span the combinations whose
    coefficients sum to zero. A set less one function remains: `n_knots - 1` functions, or
    `len(scan_masks) * n_knots - 1`. Each is local but for the last function it carries; dropping
    that function instead would leave the constant too close to the span.
    """
    # the last of the `n_knots + 1` functions of `spline_basis` is zero over the azimuth range
    splines = bspline.spline_basis(azimuth, n_knots - 1)[:-1].astype(dtype)
    values = _by_scan(splines, scan_masks)
    return TensorBasis(values[:-1] - values[-1:])


def hwp_synchronous_basis(
    n_harmonics: int,
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """HWP-synchronous basis: harmonics of the HWP angle, `k = 1..n_harmonics`."""
    matrix = _harmonics(hwp_angles, n_harmonics, dtype, dc=False)
    return TensorBasis(matrix)


def azimuth_hwp_synchronous_basis(
    legendre: PolynomialOrders,
    n_harmonics: int,
    azimuth: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
    scan_mask: Float[Array, ' samp'] | None = None,
) -> Basis:
    """Azimuth-Legendre x HWP-harmonic basis (Kronecker product of the two).

    `scan_mask` optionally zeroes the azimuth factor on flagged samples (e.g. to fit separate
    amplitudes per scan direction).
    """
    poly = _legendre(azimuth, legendre.min_order, legendre.max_order, dtype)
    if scan_mask is not None:
        poly = scan_mask[None, :] * poly
    harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
    return KroneckerBasis((poly, harm))


def binned_azimuth_hwp_synchronous_basis(
    bins: BinsConfig,
    n_harmonics: int,
    azimuth: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    dtype: DTypeLike,
) -> Basis:
    """Azimuth-binned x HWP-harmonic basis (azimuth is always binned)."""
    bin_basis = _bin_weights(azimuth, bins.n_bins, bins.interpolate, bins.smooth, dtype)
    harm = _harmonics(hwp_angles, n_harmonics, dtype, dc=True)
    return KroneckerBasis((bin_basis, harm))


def spline_hwp_synchronous_basis(
    times: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    n_knots: int,
    harmonics: int | Sequence[int],
    dtype: DTypeLike,
) -> Basis:
    """Spline-based HWP synchronous basis.

    A cubic B-spline models the slowly time-varying amplitude of the HWP-synchronous signal: knot
    `j` carries a `(sin kχ, cos kχ)` pair for each harmonic `k`, so the amplitudes have shape `(K,
    2*n_harmonics)` with `K = n_knots + 2`.

    Assumes `times` is non-decreasing.
    """
    offset, weights = bspline.spline_window(times, n_knots)  # weights (samp, 4)
    sub_values = _harmonics(hwp_angles, harmonics, dtype, dc=False).astype(dtype)
    lengths = _spline_block_lengths(times.size, n_knots)
    return WindowedBasis(
        offset, weights.astype(dtype), sub_values, n_blocks=n_knots + 2, block_lengths=lengths
    )
