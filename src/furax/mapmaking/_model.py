from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Self

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, symmetric, tree
from furax.core import BlockDiagonalOperator, IndexOperator
from furax.obs import TODFormat
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._observation import ReaderField, _sample_rate
from .acquisition import build_acquisition_operator
from .config import (
    GapTreatment,
    MapMakingConfig,
    Methods,
    NoiseSource,
    TemplatesConfig,
    WeightingMode,
)
from .layout import Bucket, SlotPlacement
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel, padding_aware_welch
from .streaming import StreamLayout, StreamOperator
from .templates import TemplateOperator, build_templates, required_reader_fields
from .templates.gram import BlockGram, factor, template_gram
from .templates.pomme import PommeProjectionOperator
from .weight import NestedWeightOperator, WeightOperator


@register_dataclass
@dataclass
class ObservationModel:
    """The operators of one observation, or of one batch of its detectors, and its noise model.

    The mapmaker builds one for each entry of a bucket as it reads the data, then stacks them over
    the entries: the array fields gain a leading axis, while the static fields (structures, chunk
    sizes) stay shared. [`BucketModel`][] holds the stack.
    """

    H: AbstractLinearOperator
    """Acquisition operator"""

    W: WeightOperator | NestedWeightOperator
    """Weighting operator (noise weights + mask)"""

    F: AbstractLinearOperator
    """Pomme filter/deprojector (identity when not enabled)"""

    noise_model: PyTree[NoiseModel]
    """Noise model"""

    sample_rate: Array
    """Data sampling rate"""

    @classmethod
    def create(
        cls, data: Any, padding: Any, config: MapMakingConfig, landscape: StokesLandscape
    ) -> Self:
        H = acquisition_operator(data, config, landscape)
        tod_format = _tod_format(H)
        M = _mask_projector(_sample_mask(data, config), tod_format=tod_format)
        noise_model, sample_rate = _noise_model(data, config, tod_format, padding=padding)
        Ninv = _noise_operator(
            noise_model,
            tod_format,
            sample_rate,
            config.weighting.correlation_length,
            inverse=True,
        )
        W: WeightOperator | NestedWeightOperator
        if config.gaps.treatment == GapTreatment.NESTED and not config.binned:
            # Minimum-variance correlated-noise weight using iterative solve.
            cov = None
            if config.gaps.nested.precondition:
                # Precondition inner flagged-subspace CG using covariance N
                cov = _noise_operator(
                    noise_model,
                    tod_format,
                    sample_rate,
                    config.weighting.correlation_length,
                    inverse=False,
                )
            W = NestedWeightOperator.create(Ninv, M, config.gaps.nested, cov=cov)
        else:
            # Plain masked weights, only exact for diagonal W
            W = WeightOperator.create(Ninv, M)
        F = (
            PommeProjectionOperator(config.pomme_tau, in_structure=tod_format.structure)
            if config.method == Methods.POMME
            else IdentityOperator(in_structure=tod_format.structure)
        )
        return cls(H, W, F, noise_model, sample_rate)

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build an [`ObservationModel`][] via [`create`][]."""
        fields: set[str] = {
            ReaderField.BORESIGHT_QUATERNIONS,
            ReaderField.DETECTOR_QUATERNIONS,
            ReaderField.VALID_SAMPLE_MASKS,
            ReaderField.TIMESTAMPS,
        }
        if not config.demodulated:
            # FIXME: this does not handle the case of a telescope without HWP
            fields.add(ReaderField.HWP_ANGLES)
        if config.scanning_mask:
            fields.add(ReaderField.VALID_SCANNING_MASKS)
        if config.weighting.mode != WeightingMode.IDENTITY:
            if config.weighting.source == NoiseSource.FIT:
                fields.update({ReaderField.SAMPLE_DATA, ReaderField.HWP_ANGLES})
            else:
                fields.add(ReaderField.NOISE_MODEL_FITS)
        if config.gaps.treatment == GapTreatment.FILL and not config.binned:
            fields.add(ReaderField.METADATA)
        return fields

    @property
    def tod_format(self) -> TODFormat:
        """The format of the observation's TOD."""
        return _tod_format(self.H)

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    @property
    def M(self) -> MaskOperator:
        return self.W.mask

    @M.setter
    def M(self, mask: MaskOperator) -> None:
        # rebuild the weight around the new mask (W is the only holder of M)
        self.W = self.W.with_mask(mask)

    @property
    def rhs_operator(self) -> AbstractLinearOperator:
        return (self.H.T @ self.W @ self.F).reduce()

    @property
    def rhs_operator_prefilled(self) -> AbstractLinearOperator:
        """RHS operator for gap-filled data: the data-side mask is dropped.

        Gap-filling already replaced the flagged samples with a constrained realization so that
        ``N⁻¹`` applies cleanly across the gaps; re-zeroing them with the inner mask of ``W`` would
        defeat the fill. Keep the outer mask (applied after ``N⁻¹``) and skip the inner one.

        Only reached under ``GapTreatment.FILL``, where ``W`` is the plain inner-mask weight.
        """
        assert isinstance(self.W, WeightOperator)
        return (self.H.T @ self.M @ self.W.weight @ self.F).reduce()

    def noise_operator(
        self, correlation_length: int, *, inverse: bool = True
    ) -> AbstractLinearOperator:
        """Build the (inverse) noise covariance operator."""
        return _noise_operator(
            self.noise_model,
            self.tod_format,
            self.sample_rate,
            correlation_length,
            inverse=inverse,
        )

    def white_weight(self) -> WeightOperator:
        """The weight `W` without the noise correlations: that of the white part of the noise.

        Assumes a model that is not stacked: its noise model and TOD format are one entry's.
        """
        white = self.noise_model.to_white_noise_model()
        inv = _noise_operator(white, self.tod_format, self.sample_rate, inverse=True)
        return WeightOperator.create(inv, self.M)

    def deprojected_weight(
        self,
        implicit: TemplateBundle | None = None,
        lift: Callable[[AbstractLinearOperator], AbstractLinearOperator] = lambda op: op,
    ) -> AbstractLinearOperator:
        r"""The weight of the normal system, from `W` and the Pomme deprojector.

        The implicit templates $T$, if there are some, are marginalised over:
        $W' = W F - W F T G^{-1} T^\top W F$, with $G$ their Gram.

        Args:
            implicit: The implicit templates, if any.
            lift: Applied to each operator of the formula before they are combined, such as
                [`StreamLayout.diagonal`][] for a model stacked over the entries of a bucket.
        """
        W = (lift(self.W) @ lift(self.F)).reduce()
        if implicit is None:
            return W
        T, G = lift(implicit.operator), lift(implicit.gram_inverse)
        return (W - W @ T @ G @ T.T @ W).reduce()


def acquisition_operator(
    data: Any, config: MapMakingConfig, landscape: StokesLandscape
) -> AbstractLinearOperator:
    """The acquisition operator of an observation, from the sky of `landscape` to its TOD."""
    return build_acquisition_operator(
        landscape,
        Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
        Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
        data.get(ReaderField.HWP_ANGLES),
        demodulated=config.demodulated,
        pointing_on_the_fly=config.pointing.on_the_fly,
        pointing_batch_samples=config.pointing.batch_samples,
        pointing_interpolate=config.pointing.interpolation == 'bilinear',
        dtype=config.dtype,
    )


def _noise_model(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    padding: Any | None = None,
) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data[ReaderField.TIMESTAMPS])
    nsamp = tod_format.n_samples

    # One noise model covers every stream of a demodulated TOD: its per-detector parameters carry
    # the leading stream axis.
    lead = (
        tod_format.det_shape
        if tod_format.stokes is None
        else (tod_format.n_streams, *tod_format.det_shape)
    )
    if config.weighting.mode == WeightingMode.IDENTITY:
        return WhiteNoiseModel(sigma=jnp.ones(lead, dtype=tod_format.dtype)), fs
    if config.weighting.source == NoiseSource.FIT:
        fit_config = config.weighting.fitting
        noise_model_class = WhiteNoiseModel if config.binned else AtmosphericNoiseModel
        fhwp = _hwp_frequency(data[ReaderField.TIMESTAMPS], data[ReaderField.HWP_ANGLES])

        samples = tod_format.as_array(data[ReaderField.SAMPLE_DATA]).reshape(-1, nsamp)
        sample_padding = 0
        if padding is not None:  # one padding per axis of the TOD's array, the samples last
            sample_padding = jax.tree.leaves(padding[ReaderField.SAMPLE_DATA])[0][-1]
        # the segments that hold a flagged sample are left out of the estimate, for every stream
        valid = jnp.broadcast_to(_sample_mask(data, config), (*lead, nsamp)).reshape(-1, nsamp)
        f, Pxx = padding_aware_welch(
            samples,
            sample_padding,
            fs=fs,
            nperseg=fit_config.nperseg,
            valid=valid,
        )
        flat_model = noise_model_class.fit_psd_model(
            f, Pxx, sample_rate=fs, hwp_frequency=fhwp, config=fit_config
        )
        # restore the leading axes on each (flattened-detector) parameter array
        noise_model = jax.tree.map(lambda p: p.reshape(lead + p.shape[1:]), flat_model)
    else:
        fits = data[ReaderField.NOISE_MODEL_FITS]  # (*lead, 4), already a plain array
        noise_model = AtmosphericNoiseModel(*jnp.moveaxis(fits, -1, 0))
        if config.binned:
            noise_model = noise_model.to_white_noise_model()
    return noise_model, fs


def _sample_mask(data: Any, config: MapMakingConfig) -> Array:
    """The valid-sample mask, `(ndet, nsamp)`, combining the sample and scanning masks.

    For Pomme, every tau-interval with a masked sample is masked whole, and so is the partial
    interval at the end. The weight is then constant on each interval, which makes it commute with
    the Pomme projector (see [`PommeProjectionOperator`][]). The widening applies to the combined
    mask: an interval straddling a scan boundary is dropped too.
    """
    mask = data[ReaderField.VALID_SAMPLE_MASKS]
    if (scanning := data.get(ReaderField.VALID_SCANNING_MASKS)) is not None:
        mask = jnp.logical_and(mask, scanning)

    if config.method == Methods.POMME:
        tau = config.pomme_tau
        F = PommeProjectionOperator(config.pomme_tau, in_structure=tree.as_structure(mask))
        # Mask all tau-intervals that are partially masked
        interval_mask = jnp.abs(F(mask)) < 0.5 / tau
        mask = jnp.logical_and(mask, interval_mask)
        # The partial interval at the end is unchanged by Pomme operator
        # -> True samples get interval_mask = False (since 1 > 0.5/tau)
        # -> False samples have mask = False
        # in both cases the logical and eliminates the tail

    return mask


def _noise_operator(
    noise_model: NoiseModel,
    tod_format: TODFormat,
    sample_rate: Array,
    correlation_length: int | None = None,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """Build the (inverse) noise covariance operator for this block.

    ``correlation_length`` sets the Toeplitz band for correlated (atmospheric) models; it is unused
    by white-noise models and may be omitted for them.
    """
    build = noise_model.inverse_operator if inverse else noise_model.operator
    return build(
        tod_format.structure, sample_rate=sample_rate, correlation_length=correlation_length
    )


def _hwp_frequency(
    timestamps: Float[Array, ' ...'], hwp_angles: Float[Array, ' ...']
) -> Float[Array, '']:
    # Note that the reader extrapolates hwp_angles in the padded region, keeping hwp freq constant
    return (jnp.unwrap(hwp_angles)[-1] - hwp_angles[0]) / jnp.ptp(timestamps) / (2 * jnp.pi)


def _mask_projector(*valid_masks: Array | None, tod_format: TODFormat) -> MaskOperator:
    """Mask operator combining a series of boolean masks (logical AND)."""
    masks = [mask for mask in valid_masks if mask is not None]
    combined = functools.reduce(jnp.logical_and, masks) if masks else jnp.array(True)
    # A per-sample mask (ndet, nsamp) broadcasts right-aligned over a demodulated Stokes TOD's
    # leading Stokes axis (n, ndet, nsamp), and the sample axis stays last (packed by MaskOperator).
    return MaskOperator.from_boolean_mask(combined, in_structure=tod_format.structure)


def _tod_format(H: AbstractLinearOperator) -> TODFormat:
    """The format of the TOD the acquisition operator `H` maps the sky to."""
    structure = H.out_structure
    if isinstance(structure, Stokes):
        return TODFormat(structure.shape, structure.dtype, structure.stokes)
    return TODFormat(structure.shape, structure.dtype)


@register_dataclass
@dataclass
class TemplateBundle:
    r"""One template operator $T$, its Gram inverse, and any configured ridge.

    The effective weight $W_\text{eff}$ is the diagonal weight $W$, or $W F$ with $F$ the Pomme
    deprojector when the templates are combined with Pomme. The ridge uses the amplitude structure
    and scaling of the Gram factorization.
    """

    operator: TemplateOperator
    gram_inverse: AbstractLinearOperator
    ridge: AbstractLinearOperator | None

    @classmethod
    def create(
        cls,
        operator: TemplateOperator,
        weight: AbstractLinearOperator,
        config: TemplatesConfig,
        *,
        bandwidth: int = 0,
        pomme_tau: int | None = None,
    ) -> Self:
        r"""Pair a template operator $T$ with its Gram inverse given a weight $W$.

        Args:
            operator: The template operator.
            weight: The weight matrix.
            config: Supplies the Gram regularization and batch size.
            bandwidth: The largest distance, in samples, between two samples that $W$ couples; 0
                for a diagonal $W$.
            pomme_tau: Pomme interval length, when the templates are combined with Pomme; the
                Gram is then taken under the effective weight $W F$.
        """
        gram = template_gram(
            operator,
            weight,
            bandwidth=bandwidth,
            pomme_tau=pomme_tau,
            batch_size=config.gram_batch_size,
        )
        ginv, ridge = factor(gram, regularization=config.regularization)
        return cls(operator, ginv, ridge)


@register_dataclass
@dataclass
class SharedTemplates:
    r"""Templates whose amplitudes the detectors share, over one batch of an observation's detectors.

    Such amplitudes are common to the batches of an observation, so no batch can factor their Gram
    alone: each holds its share $T^\top W T$ of it, the observation's Gram being their sum.
    """

    operator: TemplateOperator
    gram: BlockGram
    """The Gram of this batch, assembled but not factored."""

    @classmethod
    def create(
        cls,
        operator: TemplateOperator,
        weight: AbstractLinearOperator,
        config: TemplatesConfig,
        *,
        bandwidth: int = 0,
        pomme_tau: int | None = None,
    ) -> Self:
        """Pair a template operator with its Gram under a weight, as for [`TemplateBundle`][]."""
        gram = template_gram(
            operator,
            weight,
            bandwidth=bandwidth,
            pomme_tau=pomme_tau,
            batch_size=config.gram_batch_size,
        )
        return cls(operator, gram)


@register_dataclass
@dataclass
class ObservationTemplates:
    """One observation's templates, stackable across observations via ``jax.lax.scan``.

    Active templates are partitioned by their ``explicit`` config flag when built, because the two
    kinds enter the map-making system in different places. The explicit ones further split by who
    sees their amplitudes: each detector alone, or detectors in common.
    """

    own: TemplateBundle | None
    """Explicit templates whose amplitudes are each detector's own, solved jointly with the map."""

    implicit: TemplateBundle | None
    """Templates whose amplitudes are marginalised over."""

    shared: SharedTemplates | None = None
    """Explicit templates whose amplitudes are shared by detectors, solved jointly with the map."""

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build the active templates via [`create`][]."""
        if config.templates is None:
            return set()
        return required_reader_fields(config.templates)

    @classmethod
    def create(
        cls,
        data: Any,
        config: MapMakingConfig,
        model: ObservationModel,
        temperature: Float[Array, 'det samp'] | None = None,
    ) -> Self:
        """Build one observation's templates.

        Args:
            data: The fields read for the observation.
            config: The mapmaking configuration, whose templates are built.
            model: The observation's model.
            temperature: The I timestream, from which the T2P template is built; by default, the
                I stream of the sample data.
        """
        if (tcfg := config.templates) is None:
            raise ValueError('templates config required to build template operators')
        explicit_templates, implicit_templates = build_templates(
            tcfg, data, model.tod_format, temperature=temperature
        )

        # The Grams are taken under the effective weight `W F` of the observation's model, from
        # `model.W` and the Pomme interval length.
        pomme_tau = config.pomme_tau if config.method == Methods.POMME else None
        # the Toeplitz band of a correlated weight holds lags 0 to `correlation_length - 1`
        bandwidth = 0 if config.binned else config.weighting.correlation_length - 1
        gram_options = {'bandwidth': bandwidth, 'pomme_tau': pomme_tau}

        implicit = None
        if implicit_templates:
            if not all(template.per_detector for template in implicit_templates.values()):
                raise NotImplementedError(
                    'implicit templates need amplitudes of each detector alone: those shared '
                    'by detectors would couple the batches of an observation'
                )
            op = TemplateOperator(implicit_templates)
            implicit = TemplateBundle.create(op, model.W, tcfg, **gram_options)

        own = shared = None
        per_detector = {n: t for n, t in explicit_templates.items() if t.per_detector}
        common = {n: t for n, t in explicit_templates.items() if not t.per_detector}
        if per_detector:
            op = TemplateOperator(per_detector)
            own = TemplateBundle.create(op, model.W, tcfg, **gram_options)
        if common:
            op = TemplateOperator(common)
            shared = SharedTemplates.create(op, model.W, tcfg, **gram_options)

        return cls(own=own, implicit=implicit, shared=shared)


def restrict_streams(
    data: dict[str, Any], tod: Stokes, stokes: ValidStokesLiteral
) -> tuple[dict[str, Any], Stokes]:
    """An observation's sample data, noise fits and TOD, restricted to the streams `stokes`.

    The reader may load more demodulated streams than the map has (see
    [`MapMakingConfig.read_stokes`][furax.mapmaking.config.MapMakingConfig.read_stokes]).
    """
    index = jnp.array([tod.stokes.index(s) for s in stokes])

    def select(x: Stokes) -> Stokes:
        return Stokes.class_for(stokes).from_array(x.data[index])

    data = {**data, ReaderField.SAMPLE_DATA: select(data[ReaderField.SAMPLE_DATA])}
    if (fits := data.get(ReaderField.NOISE_MODEL_FITS)) is not None:
        data[ReaderField.NOISE_MODEL_FITS] = fits[index]  # one row per stream
    return data, select(tod)


@register_dataclass
@dataclass(frozen=True)
class BucketModel:
    """The model of one bucket: the models of its entries, and the amplitudes solved for.

    It is assembled once the bucket is read. Its operators stream over the entries.
    """

    streams: StreamLayout = field(metadata={'static': True})
    """How the operators stacked over the entries of the bucket form one stream."""
    entries: ObservationModel
    """The model of every entry, stacked over the entries (what the detector batches of a slot
    share, once per slot)."""
    implicit: TemplateBundle | None
    """Per-entry implicit templates, marginalised over (None without them)."""
    amplitudes: tuple[BucketAmplitudes, ...]
    """The explicit amplitudes of each kind present: those of each detector alone, then those
    shared by the detectors of an observation."""

    @classmethod
    def create(
        cls,
        bucket: Bucket,
        entries: ObservationModel,
        templates: ObservationTemplates | None,
        own_rhs: PyTree[Array] | None,
        shared_rhs: PyTree[Array] | None,
        placement: SlotPlacement,
        config: TemplatesConfig | None,
    ) -> BucketModel:
        """The model of a bucket, from what its accumulation stacked over its entries.

        The shared data of the entries of a slot, its detector batches, is stacked once per slot.

        Args:
            bucket: The bucket.
            entries: The model of every entry.
            templates: The templates, if any.
            own_rhs: The right-hand side of the explicit amplitudes of each detector alone.
            shared_rhs: The right-hand side of the explicit amplitudes shared by the detectors.
            placement: Where the slots of the bucket lie.
            config: The templates configuration.
        """
        streams = bucket.stream_layout
        if templates is None:
            return cls(streams, entries, None, ())
        amplitudes = []
        if (own := templates.own) is not None:
            # the detector axis follows the stream axis, if any (see `Template`), after the axis
            # of entries
            axes = tuple(
                (name, 1 + t.per.index('det')) for name, t in own.operator.templates.items()
            )
            amplitudes.append(
                BucketAmplitudes(
                    streams.diagonal(own.operator),
                    streams.diagonal(own.gram_inverse),
                    None if own.ridge is None else streams.diagonal(own.ridge),
                    own_rhs,
                    axes,
                )
            )
        if templates.shared is not None:
            assert config is not None  # shared templates are configured
            amplitudes.append(
                _shared_amplitudes(
                    bucket, templates.shared, shared_rhs, placement, config.regularization
                )
            )
        return cls(streams, entries, templates.implicit, tuple(amplitudes))

    @property
    def H(self) -> StreamOperator:
        """The sky pointing, from the sky to the TOD of every entry."""
        return self.streams.column(self.entries.H)

    @property
    def W(self) -> StreamOperator:
        """The weight of every entry."""
        return self.streams.diagonal(self.entries.W)

    def white_weight(self) -> StreamOperator:
        """The weight of every entry, without the noise correlations.

        It is the plain masked weight of the white part of the noise, even under the nested gap
        treatment.
        """
        if isinstance(self.entries.noise_model, WhiteNoiseModel) and isinstance(
            self.entries.W, WeightOperator
        ):
            return self.W
        # Map only the inverse noise: the mask is W's own, used as it is.
        weight = self.streams.vmap(lambda m: m.white_weight().weight, self.entries)
        return self.streams.diagonal(WeightOperator.create(weight, self.entries.M))

    @property
    def F(self) -> StreamOperator:
        """The Pomme deprojector of every entry."""
        return self.streams.diagonal(self.entries.F)

    @property
    def deprojected_weight(self) -> AbstractLinearOperator:
        """The weight of the normal system, from `W` and the Pomme deprojector.

        The implicit templates, if there are some, are marginalised over (see
        [`ObservationModel.deprojected_weight`][]).
        """
        return self.entries.deprojected_weight(self.implicit, lift=self.streams.diagonal)


@register_dataclass
@dataclass(frozen=True)
class BucketAmplitudes:
    """One kind of explicit-template amplitudes of one bucket.

    The counterpart of [`TemplateBundle`][] over the entries of a bucket, with the right-hand side.
    """

    operator: StreamOperator
    """The templates, from the amplitudes to the TOD of every entry of the bucket."""
    gram_inverse: AbstractLinearOperator
    """The inverse of the Gram of the amplitudes, which preconditions them."""
    ridge: AbstractLinearOperator | None
    """The ridge the Gram inverse was factored with, if any."""
    rhs: PyTree[Array]
    """The right-hand side of the amplitudes."""
    detector_axes: tuple[tuple[str, int | None], ...] = field(metadata={'static': True})
    """For each template, the detector axis of its amplitudes, after the leading axis of entries
    or slots; `None` for amplitudes without one."""


def _shared_amplitudes(
    bucket: Bucket,
    shared: SharedTemplates,
    rhs: PyTree[Array],
    placement: SlotPlacement,
    regularization: float,
) -> BucketAmplitudes:
    """The amplitudes shared by the detectors of each slot of a bucket.

    The entries of a slot, its detector batches, see its amplitudes alone: of the stack
    `(n_slots, ...)` every entry reads its own slot's, its selection being stacked once per slot
    like the other data the batches share, so that their transpose adds the batches' projections
    up into the slot's amplitudes. Likewise, a slot's Gram and right-hand side are the
    sums of those of its detector batches.

    Args:
        bucket: The bucket.
        shared: The shared templates, stacked over the entries of the bucket.
        rhs: The right-hand side of each entry.
        placement: Where the slots of the bucket lie.
        regularization: The regularization the Gram is factored with.
    """
    structure = {
        name: jax.ShapeDtypeStruct((bucket.n_slots, *s.shape), s.dtype)
        for name, s in shared.operator.in_structure.items()
    }

    def select(slot):
        return BlockDiagonalOperator(
            {name: IndexOperator(slot, in_structure=s) for name, s in structure.items()}
        )

    templates = bucket.stream_layout.column(
        shared.operator @ jax.vmap(select)(jnp.arange(bucket.n_slots))
    )
    rhs = placement.per_slot(rhs, bucket.sum_detector_batches)
    gram = placement.per_slot(shared.gram, bucket.sum_detector_batches)
    inverse, ridge = jax.vmap(functools.partial(factor, regularization=regularization))(gram)
    inverse, ridge = placement.replicate((inverse, ridge))
    return BucketAmplitudes(
        templates,
        _PerSlotOperator(inverse, in_structure=structure),
        None if ridge is None else _PerSlotOperator(ridge, in_structure=structure),
        rhs,
        tuple((name, None) for name in structure),  # one set per slot, no detector axis
    )


@symmetric
class _PerSlotOperator(AbstractLinearOperator):
    """A symmetric operator of its own for each slot of a leading axis: `x[s] -> operator[s](x[s])`.

    `operator` is a stack of operators over the slots, as the stream operators take them.
    """

    operator: AbstractLinearOperator

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return jax.vmap(lambda operator, v: operator(v))(self.operator, x)
