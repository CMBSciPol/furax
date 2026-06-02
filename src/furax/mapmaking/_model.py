from dataclasses import dataclass
from typing import Any, Self, TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.core import BlockDiagonalOperator, BlockRowOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import StokesIQU

from .acquisition import build_acquisition_operator
from .basis_templates import PerDetectorTemplate
from .config import LegendreOrders, MapMakingConfig, Methods, NoiseSource, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .templates import ATOPProjectionOperator


@register_dataclass
@dataclass
class ObservationModel:
    """Holds the operators and data for one or more observations.

    When stacked via ``jax.lax.scan``, the array fields carry a leading batch
    dimension over observations while static fields (structures, chunk sizes)
    remain shared.
    """

    H: AbstractLinearOperator
    """Acquisition operator"""

    W: AbstractLinearOperator
    """Weighting operator"""

    masker: AbstractLinearOperator
    """Sample masking operator"""

    noise_model: PyTree[NoiseModel]
    """Noise model"""

    sample_rate: Array
    """Data sampling rate"""

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        fields = {
            'boresight_quaternions',
            'detector_quaternions',
            'valid_sample_masks',
            'timestamps',
        }
        if not config.demodulated:
            # FIXME: this does not handle the case of a telescope without HWP
            fields.add('hwp_angles')
        if config.scanning_mask:
            fields.add('valid_scanning_masks')
        if config.weighting.mode != WeightingMode.IDENTITY:
            if config.weighting.source == NoiseSource.FIT:
                fields.update({'sample_data', 'hwp_angles'})
            else:
                fields.add('noise_model_fits')
        if config.gaps.fill and not config.binned:
            fields.add('metadata')

        return fields

    @classmethod
    def create(
        cls, data: Any, padding: Any, config: MapMakingConfig, landscape: StokesLandscape
    ) -> Self:
        H = build_acquisition_operator(
            landscape,
            data['boresight_quaternions'],
            data['detector_quaternions'],
            data.get('hwp_angles'),
            demodulated=config.demodulated,
            pointing_chunk_size=config.pointing.chunk_size,
            pointing_on_the_fly=config.pointing.on_the_fly,
            pointing_interpolate=config.pointing.interpolation == 'bilinear',
            dtype=config.dtype,
        )
        masker = _mask_projector(
            _sample_mask(data, config),
            data.get('valid_scanning_masks'),
            structure=H.out_structure,
        )
        sample_rate = _sample_rate(data['timestamps'])
        F_T = _template_deprojector(config, H.out_structure)

        # W weights the data after F_T, so a data-fit noise PSD must be estimated on the
        # deprojected TOD (otherwise N⁻¹ mismatches the post-deprojection residual).
        deproj_data = data
        if (
            F_T is not None
            and config.weighting.mode != WeightingMode.IDENTITY
            and config.weighting.source == NoiseSource.FIT
        ):
            deproj_data = {**data, 'sample_data': F_T(data['sample_data'])}

        noise_model, _ = _noise_model(deproj_data, config, tod_structure=H.out_structure)
        W = _noise_operator(
            noise_model,
            H.out_structure,
            sample_rate,
            config.weighting.correlation_length,
            inverse=True,
        )
        if F_T is not None:
            W = W @ F_T
        return cls(H, W, masker, noise_model, sample_rate)

    @property
    def tod_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.out_structure

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    def noise_operator(
        self, correlation_length: int, *, inverse: bool = True
    ) -> AbstractLinearOperator:
        """Build the (inverse) noise covariance operator."""
        return _noise_operator(
            self.noise_model,
            self.tod_structure,
            self.sample_rate,
            correlation_length,
            inverse=inverse,
        )

    def diag_W(self) -> AbstractLinearOperator:
        """Build the inverse white noise covariance operator."""
        operator_tree = jax.tree.map(
            lambda noise, s: noise.to_white_noise_model().inverse_operator(s),
            self.noise_model,
            self.tod_structure,
            is_leaf=lambda nm: isinstance(nm, NoiseModel),
        )
        return BlockDiagonalOperator(operator_tree)

    def _get_indexer(self) -> IndexOperator:
        """Get the IndexOperator for gap-filling"""
        if isinstance(self.masker, MaskOperator):
            mask = self.masker.to_boolean_mask()
        elif isinstance(self.masker, IdentityOperator):
            mask = tree.ones_like(self.tod_structure).astype(bool)
        else:
            raise NotImplementedError
        return IndexOperator(mask, in_structure=self.tod_structure)

    def pad(self, n_pad: int) -> 'ObservationModel':
        """Pad with ``n_pad`` dummy observations that contribute nothing to sums.

        Copies of the last real observation with their masker zeroed out.
        """
        if n_pad == 0:
            return self
        padded = _pad_stacked(self, n_pad)
        zero_tail_masker = jax.tree.map(lambda m: m.at[-n_pad:].set(0), padded.masker)
        return ObservationModel(
            H=padded.H,
            W=padded.W,
            masker=zero_tail_masker,
            noise_model=padded.noise_model,
            sample_rate=padded.sample_rate,
        )


_T = TypeVar('_T')


def _pad_stacked(stacked: _T, n_pad: int) -> _T:
    """Repeat the last entry ``n_pad`` times and append along the leading axis."""
    if n_pad == 0:
        return stacked
    last = jax.tree.map(lambda a: jnp.repeat(a[-1:], n_pad, axis=0), stacked)
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), stacked, last)  # type: ignore[no-any-return]


def _noise_model(
    data: Any,
    config: MapMakingConfig,
    tod_structure: jax.ShapeDtypeStruct | None = None,
) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data['timestamps'])
    if config.weighting.mode == WeightingMode.IDENTITY:
        if tod_structure is None:
            raise ValueError('tod_structure is required when config.weighting.mode is IDENTITY')
        noise_model = jax.tree.map(
            lambda s: WhiteNoiseModel(sigma=jnp.ones(s.shape[0], dtype=s.dtype)),
            tod_structure,
        )
        return noise_model, fs
    if config.weighting.source == NoiseSource.FIT:
        fit_config = config.weighting.fitting
        noise_model_class = WhiteNoiseModel if config.binned else AtmosphericNoiseModel
        fhwp = _hwp_frequency(data['timestamps'], data['hwp_angles'])

        def _compute_Pxx_and_fit(tod):  # type: ignore[no-untyped-def]
            f, Pxx = jax.scipy.signal.welch(tod, fs=fs, nperseg=fit_config.nperseg)
            return noise_model_class.fit_psd_model(
                f,
                Pxx,
                sample_rate=fs,
                hwp_frequency=fhwp,
                config=fit_config,
            )

        noise_model = jax.tree.map(_compute_Pxx_and_fit, data['sample_data'])
    else:
        noise_model = jax.tree.map(lambda x: AtmosphericNoiseModel(*x.T), data['noise_model_fits'])
        if config.binned:
            noise_model = jax.tree.map(
                lambda m: m.to_white_noise_model(),
                noise_model,
                is_leaf=lambda x: isinstance(x, AtmosphericNoiseModel),
            )
    return noise_model, fs


def _sample_mask(data: Any, config: MapMakingConfig) -> Array:
    """Get the sample mask from data.

    For ATOP mapmaker, extra pixels may be masked depending on atop_tau.
    """

    mask = data['valid_sample_masks']

    if config.method == Methods.ATOP:
        tau = config.atop_tau
        F = ATOPProjectionOperator(config.atop_tau, in_structure=tree.as_structure(mask))
        # Mask all tau-intervals that are partially masked
        interval_mask = jnp.abs(F(mask)) < 0.5 / tau
        mask = jnp.logical_and(mask, interval_mask)
        # The partial interval at the end is unchanged by ATOP operator
        # -> True samples get interval_mask = False (since 1 > 0.5/tau)
        # -> False samples have mask = False
        # in both cases the logical and eliminates the tail

    return mask  # type: ignore[no-any-return]


def _noise_operator(
    noise_model: NoiseModel,
    tod_structure: jax.ShapeDtypeStruct,
    sample_rate: Array,
    correlation_length: int,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """Build the (inverse) noise covariance operator for this block."""
    operator_tree = jax.tree.map(
        lambda model, struct: (model.inverse_operator if inverse else model.operator)(
            struct, sample_rate=sample_rate, correlation_length=correlation_length
        ),
        noise_model,
        tod_structure,
        is_leaf=lambda x: isinstance(x, NoiseModel),
    )
    return BlockDiagonalOperator(operator_tree)


def _template_deprojector(
    config: MapMakingConfig,
    tod_structure: jax.ShapeDtypeStruct,
) -> AbstractLinearOperator | None:
    """Build a fixed deprojection operator folded into ``W`` (``W = W @ F_T``).

    Only ATOP uses this. T→P leakage is handled as a fitted template (see
    :meth:`ObservationTemplates.create`), not deprojected here.
    """
    if config.method == Methods.ATOP:
        return ATOPProjectionOperator(config.atop_tau, in_structure=tod_structure)
    return None


def _sample_rate(timestamps: Float[Array, '...']) -> Float[Array, '']:
    # Note that the reader extrapolates timestamps in the padded region, keeping sample rate constant
    return (timestamps.size - 1) / jnp.ptp(timestamps)


def _hwp_frequency(
    timestamps: Float[Array, '...'], hwp_angles: Float[Array, '...']
) -> Float[Array, '']:
    # Note that the reader extrapolates hwp_angles in the padded region, keeping hwp freq constant
    return (jnp.unwrap(hwp_angles)[-1] - hwp_angles[0]) / jnp.ptp(timestamps) / (2 * jnp.pi)


def _mask_projector(
    *valid_masks: Array | None, structure: jax.ShapeDtypeStruct
) -> AbstractLinearOperator:
    """Mask operator built from a series of boolean masks."""

    def _masker(valid_mask: Array | None) -> AbstractLinearOperator:
        if valid_mask is None:
            return IdentityOperator(in_structure=structure)
        return MaskOperator.from_boolean_mask(valid_mask, in_structure=structure)

    combined_masker = CompositionOperator([_masker(valid_mask) for valid_mask in valid_masks])
    return combined_masker.reduce()


@register_dataclass
@dataclass
class ObservationTemplates:
    """Per-observation template operators, stackable across observations via ``jax.lax.scan``.

    The active families are partitioned by their ``explicit`` config flag:

    - ``operator`` — the *explicit* templates (``explicit=True``), whose amplitudes are solved
      for in the two-step CG and returned. ``None`` when every family is marginalised.
    - ``marginal`` — the *marginalised* templates (``explicit=False``), folded into the
      weighting ``W_m`` and never explicitly solved (see :mod:`._marginalize`). ``None`` when
      every family is explicit (then ``W_m = W``, the classic two-step).
    """

    operator: BlockRowOperator | None
    marginal: BlockRowOperator | None = None

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build an :class:`ObservationTemplates` via :meth:`create`."""
        tcfg = config.templates
        if tcfg is None:
            return set()
        fields: set[str] = set()
        if tcfg.polynomial is not None:
            # ``timestamps`` supplies the (shift/scale-invariant) Legendre abscissa;
            # ``scanning_intervals`` defines the per-interval polynomial blocks;
            # ``valid_scanning_masks`` zeroes turnaround samples in the basis.
            fields.update({'scanning_intervals', 'timestamps', 'valid_scanning_masks'})
        if tcfg.scan_synchronous is not None:
            fields.add('azimuth')
        if tcfg.binaz_synchronous is not None:
            fields.add('azimuth')
        if tcfg.hwp_synchronous is not None:
            fields.add('hwp_angles')
        if tcfg.azhwp_synchronous is not None:
            fields.update({'azimuth', 'hwp_angles'})
            if tcfg.azhwp_synchronous.split_scans:
                fields.update({'left_scan_mask', 'right_scan_mask'})
        if tcfg.binazhwp_synchronous is not None:
            fields.update({'azimuth', 'hwp_angles'})
        if config.demodulated and tcfg.t2p is not None:
            # the t2p template basis is each detector's temperature (demodulated I leg)
            fields.add('sample_data')
            if tcfg.t2p.fit_band is not None:  # band-limited: need timestamps for sample rate
                fields.add('timestamps')
        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-obs two-step path: a '
                'shared HorizonLandscape across observations is required.'
            )
        return fields

    @classmethod
    def create(
        cls,
        data: Any,
        config: MapMakingConfig,
        tod_structure: jax.ShapeDtypeStruct,
    ) -> Self:
        if (tcfg := config.templates) is None:
            raise ValueError('templates config required to build template operator')
        n_dets, n_samps = tod_structure.shape
        dtype = config.dtype
        blocks: dict[str, AbstractLinearOperator] = {}
        explicit_flags: dict[str, bool] = {}

        # In the demodulated case the TOD is a StokesIQU (I, Q, U) per detector, so each
        # template acts independently on the three legs via a Stokes-keyed block diagonal:
        # I uses ``op_i``, Q and U use ``op_qu`` (defaulting to ``op_i``). This is where
        # per-Stokes orders (e.g. polynomial ``legendre`` for I vs ``legendre_qu`` for Q/U)
        # take effect. HWP-coupled templates are rejected upstream (see _check_config).
        demod = config.demodulated

        def per_stokes(op_i: AbstractLinearOperator, op_qu: AbstractLinearOperator | None = None):  # type: ignore[no-untyped-def]
            if not demod:
                return op_i
            op_qu = op_i if op_qu is None else op_qu
            return BlockDiagonalOperator(StokesIQU(op_i, op_qu, op_qu))  # type: ignore[arg-type]

        if (poly := tcfg.polynomial) is not None:
            # Per-interval Legendre polynomials. ``timestamps`` is a valid abscissa
            # because ``polynomial`` rescales each interval to [-1, 1] (shift/scale
            # invariant). Padded (degenerate) intervals contribute a zero basis.
            intervals = data['scanning_intervals']
            times = data['timestamps']
            valid_mask = data['valid_scanning_masks']

            def make_poly(orders: LegendreOrders) -> AbstractLinearOperator:
                return PerDetectorTemplate.polynomial(
                    max_poly_order=orders.max_order,
                    intervals=intervals,
                    times=times,
                    n_dets=n_dets,
                    dtype=dtype,
                    valid_mask=valid_mask,
                )

            poly_i = make_poly(poly.legendre)
            # In the demodulated case Q/U use ``legendre_qu`` when given (e.g. degree 1
            # for Q/U vs degree 9 for I), falling back to the I orders otherwise.
            poly_qu = make_poly(poly.legendre_qu) if (demod and poly.legendre_qu) else None
            blocks['polynomial'] = per_stokes(poly_i, poly_qu)
            explicit_flags['polynomial'] = poly.explicit

        if (sss := tcfg.scan_synchronous) is not None:
            scan = PerDetectorTemplate.scan_synchronous(
                legendre=sss.legendre,
                azimuth=data['azimuth'],
                n_dets=n_dets,
                dtype=dtype,
            )
            blocks['scan_synchronous'] = per_stokes(scan)
            explicit_flags['scan_synchronous'] = sss.explicit

        if (binazss := tcfg.binaz_synchronous) is not None:
            binaz = PerDetectorTemplate.binaz_synchronous(
                bins=binazss.bins,
                azimuth=data['azimuth'],
                n_dets=n_dets,
                dtype=dtype,
            )
            blocks['binaz_synchronous'] = per_stokes(binaz)
            explicit_flags['binaz_synchronous'] = binazss.explicit

        if (hwpss := tcfg.hwp_synchronous) is not None:
            blocks['hwp_synchronous'] = PerDetectorTemplate.hwp_synchronous(
                n_harmonics=hwpss.n_harmonics,
                hwp_angles=data['hwp_angles'],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['hwp_synchronous'] = hwpss.explicit

        if (azhwpss := tcfg.azhwp_synchronous) is not None:
            if azhwpss.split_scans:
                blocks['azhwp_synchronous_left'] = PerDetectorTemplate.azhwp_synchronous(
                    legendre=azhwpss.legendre,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data['azimuth'],
                    hwp_angles=data['hwp_angles'],
                    n_dets=n_dets,
                    dtype=dtype,
                    scan_mask=data['left_scan_mask'],
                )
                blocks['azhwp_synchronous_right'] = PerDetectorTemplate.azhwp_synchronous(
                    legendre=azhwpss.legendre,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data['azimuth'],
                    hwp_angles=data['hwp_angles'],
                    n_dets=n_dets,
                    dtype=dtype,
                    scan_mask=data['right_scan_mask'],
                )
                explicit_flags['azhwp_synchronous_left'] = azhwpss.explicit
                explicit_flags['azhwp_synchronous_right'] = azhwpss.explicit
            else:
                blocks['azhwp_synchronous'] = PerDetectorTemplate.azhwp_synchronous(
                    legendre=azhwpss.legendre,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data['azimuth'],
                    hwp_angles=data['hwp_angles'],
                    n_dets=n_dets,
                    dtype=dtype,
                )
                explicit_flags['azhwp_synchronous'] = azhwpss.explicit

        if (binazhwpss := tcfg.binazhwp_synchronous) is not None:
            blocks['binazhwp_synchronous'] = PerDetectorTemplate.binazhwp_synchronous(
                bins=binazhwpss.bins,
                n_harmonics=binazhwpss.n_harmonics,
                azimuth=data['azimuth'],
                hwp_angles=data['hwp_angles'],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['binazhwp_synchronous'] = binazhwpss.explicit

        if demod and tcfg.t2p is not None:
            # T→P leakage as a fitted template (not a deprojection folded into W): each
            # detector's basis is its own temperature stream T_d (the demodulated I leg),
            # acting on Q and U only — I left untouched via an empty (k=0) leg. The two
            # fitted amplitudes per detector are the leakage coefficients lamQ, lamU.
            temperature = data['sample_data'].i
            fit_band = tcfg.t2p.fit_band
            sample_rate = _sample_rate(data['timestamps']) if fit_band is not None else 1.0
            blocks['t2p'] = per_stokes(
                PerDetectorTemplate.none(n_dets, n_samps, dtype),
                PerDetectorTemplate.temperature(
                    temperature,
                    dtype,
                    fit_band=fit_band,
                    sample_rate=sample_rate,
                    decimate=tcfg.t2p.decimate,
                ),
            )
            explicit_flags['t2p'] = tcfg.t2p.explicit

        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-obs two-step path: a '
                'shared HorizonLandscape across observations is required.'
            )

        if not blocks:
            raise ValueError('config.templates is set but no template is active.')

        explicit = {k: v for k, v in blocks.items() if explicit_flags[k]}
        marginal = {k: v for k, v in blocks.items() if not explicit_flags[k]}

        # Marginalising several coupled families jointly needs their full (non-block-diagonal)
        # joint Gram — not yet supported. A single family keeps a block-diagonal Gram that is
        # inverted in closed form (see :mod:`._marginalize`). ``split_scans`` produces two
        # blocks from one family, so count distinct families, not blocks.
        marginal_families = {k.removesuffix('_left').removesuffix('_right') for k in marginal}
        if len(marginal_families) > 1:
            raise NotImplementedError(
                'At most one template family may be marginalised (explicit=False); got '
                f'{sorted(marginal_families)}. Mark all but one as explicit=True, or see '
                'the joint-marginalisation TODO.'
            )

        operator = BlockRowOperator(blocks=explicit) if explicit else None
        marginal_op = BlockRowOperator(blocks=marginal) if marginal else None
        return cls(operator, marginal_op)

    def pad(self, n_pad: int) -> 'ObservationTemplates':
        """Pad with ``n_pad`` copies of the last entry."""
        return _pad_stacked(self, n_pad)
