import functools
from dataclasses import dataclass, field
from typing import Any, Final, Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PyTree

from furax import (
    AbstractLinearOperator,
    IdentityOperator,
    MaskOperator,
    SymmetricBandToeplitzOperator,
    tree,
)
from furax.obs import TODFormat
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._gaps import GapSplineOperator, GapSplines
from ._observation import ReaderField, _sample_rate, real_extent
from .acquisition import build_acquisition_operator
from .config import GapTreatment, MapMakingConfig, NoiseSource, TemplatesConfig, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel, padding_aware_welch
from .templates import TemplateOperator, build_templates, required_reader_fields
from .templates.gram import BlockGram, factor, template_gram
from .templates.pomme import PommeProjectionOperator
from .weight import MarginalizedWeightOperator, NestedWeightOperator, WeightOperator

DROP_REASONS = ('over_flagged', 'over_gap_budget')
"""Why a detector is dropped: each names the field of [`TileModel`][] marking the detectors
dropped for it, and the columns that report them in `MapMakingResults`."""


@register_dataclass
@dataclass
class TileModel:
    """The model of the TOD of one detector tile: a subset of the detectors of an observation.

    It holds the acquisition operator, the weight with the mask of the good samples, the noise
    model, and the detectors dropped. Stacked over the tiles of a bucket, the array fields carry a
    leading tile axis, while the static fields (structures, chunk sizes) are shared.
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

    over_flagged: Bool[Array, ' det']
    """Per detector, whether it is dropped for having too many of its samples flagged"""

    over_gap_budget: Bool[Array, ' det']
    """Per detector, whether it is dropped for needing more gap amplitudes than `gaps` holds"""

    n_samples: Int[Array, '']
    """The number of samples of the observation, before those padding it to its bucket"""

    gaps: GapSplines | None = field(default=None, metadata={'static': True})
    """The spline model of the gaps, whose amplitudes are solved for with the map; `None`
    unless the gaps are explicit"""

    @classmethod
    def create(
        cls,
        data: Any,
        padding: Any,
        config: MapMakingConfig,
        landscape: StokesLandscape,
        use: Bool[Array, ''] | bool = True,
        gaps: GapSplines | None = None,
    ) -> Self:
        """Build the model of a tile from the fields read for its detectors.

        Args:
            data: The fields read for the tile's detectors.
            padding: Their padding widths, as returned by the reader.
            config: The mapmaking configuration.
            landscape: The landscape of the maps.
            use: Whether the tile contributes: false for a padding tile or a tile of a failed
                observation, whose samples are then all masked, and none of its detectors
                reported dropped.
            gaps: The spline model of the gaps, when they are explicit; a detector needing more
                amplitudes than its budget is dropped.
        """
        H = acquisition_operator(data, config, landscape)
        tod_format = _tod_format(H)
        mask = _sample_mask(data, config) & jnp.asarray(use)
        n_detectors, n_samples = real_extent(padding, mask.shape)
        # a padding tile or a failed observation has no detectors to drop
        n_detectors = jnp.where(use, n_detectors, 0)
        over_flagged = _over_flagged(mask, n_detectors, n_samples, config.gaps.max_flagged_fraction)
        over_gap_budget = jnp.zeros_like(over_flagged)
        if gaps is not None:
            kept = mask & ~over_flagged[:, None]
            over_gap_budget = ~over_flagged & (gaps.counts(kept, n_samples) > gaps.budget)
        valid_detectors = ~(over_flagged | over_gap_budget)[:, None]
        M = _mask_projector(mask, valid_detectors, tod_format=tod_format)
        noise_model, sample_rate = _noise_model(data, config, tod_format, n_samples)
        Ninv = _noise_operator(
            noise_model,
            tod_format,
            sample_rate,
            config.weighting.correlation_length,
            inverse=True,
        )
        W: WeightOperator | NestedWeightOperator
        if config.gap_treatment == GapTreatment.NESTED:
            # Minimum-variance correlated-noise weight; the white-noise weight preconditions its
            # linear inner solve
            white = noise_model.to_white_noise_model().inverse_operator(tod_format.structure)
            # the correlated noise models build a banded inverse
            assert isinstance(Ninv, SymmetricBandToeplitzOperator)
            W = NestedWeightOperator.create(Ninv, M, config.gaps.nested, white)
        else:
            # Plain masked weights, only exact for diagonal W. With explicit gaps, the normal
            # system takes N⁻¹ unmasked from it, the gap amplitudes standing for the flagged samples
            W = WeightOperator.create(Ninv, M)
        F = (
            PommeProjectionOperator(config.pomme_tau, in_structure=tod_format.structure)
            if config.pomme_tau is not None
            else IdentityOperator(in_structure=tod_format.structure)
        )
        return cls(
            H, W, F, noise_model, sample_rate, over_flagged, over_gap_budget, n_samples, gaps
        )

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build an [`TileModel`][] via [`create`][]."""
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
        if config.gap_treatment == GapTreatment.FILL:
            fields.add(ReaderField.METADATA)
        return fields

    @property
    def tod_format(self) -> TODFormat:
        """The format of the tile's TOD."""
        return _tod_format(self.H)

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    @property
    def M(self) -> MaskOperator:
        return self.W.mask

    def gap_basis(self) -> GapSplineOperator:
        """The basis $C$ of the explicit gaps, from the mask."""
        assert self.gaps is not None
        return self.gaps.basis(self.M, self.n_samples)

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

    def diag_W(self) -> WeightOperator:
        """The weight of the tile under the white part of its noise, with its mask."""
        white = self.noise_model.to_white_noise_model()
        inv = _noise_operator(white, self.tod_format, self.sample_rate, inverse=True)
        return WeightOperator.create(inv, self.M)

    def deprojected_weight(
        self, implicit: 'TemplateBundle | None' = None
    ) -> AbstractLinearOperator:
        r"""The weight of the normal system: $W F$, with $F$ the Pomme deprojector.

        With implicit templates $T$, their amplitudes are marginalized:
        $W' = W F - W F T G^{-1} T^\top W F$, with $G^{-1}$ the inverse of their Gram (see
        [`MarginalizedWeightOperator`][furax.mapmaking.weight.MarginalizedWeightOperator]).

        Args:
            implicit: The implicit templates, if any.
        """
        weight = (self.W @ self.F).reduce()
        if implicit is None:
            return weight
        return MarginalizedWeightOperator.create(weight, implicit.operator, implicit.gram_inverse)


def acquisition_operator(
    data: Any, config: MapMakingConfig, landscape: StokesLandscape
) -> AbstractLinearOperator:
    """The acquisition operator of a tile, from the sky of `landscape` to its TOD."""
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
    n_samples: Int[Array, ''] | int | None = None,
) -> tuple[PyTree[NoiseModel], Array]:
    """The noise model and the sample rate of a tile.

    The samples past the first `n_samples`, padding, are left out of a fit; all count without it.
    """
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
        sample_padding = 0 if n_samples is None else nsamp - n_samples
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

    if (tau := config.pomme_tau) is not None:
        F = PommeProjectionOperator(tau, in_structure=tree.as_structure(mask))
        # Mask all tau-intervals that are partially masked
        interval_mask = jnp.abs(F(mask)) < 0.5 / tau
        mask = jnp.logical_and(mask, interval_mask)
        # The partial interval at the end is unchanged by Pomme operator
        # -> True samples get interval_mask = False (since 1 > 0.5/tau)
        # -> False samples have mask = False
        # in both cases the logical and eliminates the tail

    return mask


def _over_flagged(
    mask: Bool[Array, 'det samp'],
    n_detectors: Int[Array, ''] | int,
    n_samples: Int[Array, ''] | int,
    max_flagged_fraction: float | None,
) -> Bool[Array, ' det']:
    """The detectors with more than `max_flagged_fraction` of their samples flagged.

    Only the first `n_detectors` and `n_samples` are the observation's, those after them padding.
    """
    if max_flagged_fraction is None:
        return jnp.zeros(mask.shape[0], bool)
    flagged = jnp.sum(~mask & (jnp.arange(mask.shape[-1]) < n_samples), axis=-1)
    return (jnp.arange(mask.shape[0]) < n_detectors) & (flagged > max_flagged_fraction * n_samples)


def _noise_operator(
    noise_model: NoiseModel,
    tod_format: TODFormat,
    sample_rate: Array,
    correlation_length: int | None = None,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """The (inverse) noise covariance operator of a tile.

    ``correlation_length`` sets the Toeplitz band for correlated (atmospheric) models; it is unused
    by white-noise models and may be omitted for them.
    """
    build = noise_model.inverse_operator if inverse else noise_model.operator
    return build(
        tod_format.structure, sample_rate=sample_rate, correlation_length=correlation_length
    )


def _hwp_frequency(
    timestamps: Float[Array, '...'], hwp_angles: Float[Array, '...']
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
        pomme_tau: int | None = None,
    ) -> Self:
        r"""Pair a template operator $T$ with its Gram inverse given a weight $W$.

        Args:
            operator: The template operator.
            weight: The diagonal weight matrix.
            config: Supplies the Gram regularization and batch size.
            pomme_tau: Pomme interval length, when the templates are combined with Pomme; the
                Gram is then taken under the effective weight $W F$.
        """
        gram = template_gram(
            operator,
            weight,
            pomme_tau=pomme_tau,
            batch_size=config.gram_batch_size,
        )
        ginv, ridge = factor(gram, regularization=config.regularization)
        return cls(operator, ginv, ridge)


TemplateKind = Literal['templates', 'shared_templates']
"""A kind of template amplitudes solved for with the map: a key of the unknowns of the system."""

TEMPLATES: Final = 'templates'
"""The amplitudes of the explicit templates of each detector, among the unknowns of the system."""

SHARED_TEMPLATES: Final = 'shared_templates'
"""The amplitudes of the explicit templates the detectors of an observation share, among the
unknowns of the system."""

Holder = Literal['tile', 'observation']
"""Where amplitudes are held: in each detector tile, for those of its detectors, or once per
observation, for those its detectors share, their Gram and right-hand side summed over its tiles."""

TEMPLATE_KINDS: dict[TemplateKind, Holder] = {TEMPLATES: 'tile', SHARED_TEMPLATES: 'observation'}
"""The template amplitudes solved for with the map, and where they are held."""


class UnfactoredTemplates(NamedTuple):
    """The templates of one detector tile, as built, before their Grams are factored."""

    implicit: 'TemplateBundle | None'
    """The implicit templates whose amplitudes are each detector's own, factored: the weight of the
    tile needs them."""
    operators: dict[TemplateKind, TemplateOperator | None]
    """The operator of the explicit templates of each of `TEMPLATE_KINDS`, `None` for those
    absent."""
    grams: dict[TemplateKind, BlockGram | None]
    """The Gram of each, assembled but not factored: the tile's share of it for the templates the
    observation's detectors share."""
    weighted_tod: PyTree[Array]
    """The TOD under the tile's weight, its own implicit templates marginalized."""


@register_dataclass
@dataclass
class TileTemplates:
    """The templates of one detector tile, their Grams factored.

    Active templates are partitioned by their ``explicit`` config flag when built, because the two
    enter the map-making system in different places: the amplitudes of the implicit ones are
    marginalized in the weight, those of the explicit ones solved for with the map. The explicit
    ones further split by who sees their amplitudes: each detector alone, or the detectors of the
    observation together.
    """

    implicit: TemplateBundle | None
    """Implicit templates, whose amplitudes are each detector's own, marginalized."""

    explicit: dict[TemplateKind, TemplateBundle | None]
    """Explicit templates, solved for with the map, by the kind of their amplitudes (see
    `TEMPLATE_KINDS`): each detector's own, or shared by the detectors of the observation; `None`
    for a kind absent."""

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build the active templates via [`assemble`][]."""
        if config.templates is None:
            return set()
        return required_reader_fields(config.templates)

    @classmethod
    def from_inverses(
        cls,
        implicit: 'TemplateBundle | None',
        operators: dict[TemplateKind, TemplateOperator | None],
        inverses: dict[
            TemplateKind, tuple[AbstractLinearOperator, AbstractLinearOperator | None] | None
        ],
    ) -> Self:
        """A tile's templates, from those of [`UnfactoredTemplates`][] and their Gram inverses.

        Args:
            implicit: The implicit templates whose amplitudes are each detector's own.
            operators: The operator of each of `TEMPLATE_KINDS`, `None` for those absent.
            inverses: The inverse of the Gram of each, and the ridge added, as
                [`factor`][furax.mapmaking.templates.gram.factor] gives them.
        """
        explicit = {}
        for kind, operator in operators.items():
            explicit[kind] = None
            if operator is not None:
                inverse = inverses[kind]
                assert inverse is not None  # every template present has its Gram factored
                explicit[kind] = TemplateBundle(operator, *inverse)
        return cls(implicit=implicit, explicit=explicit)

    @staticmethod
    def assemble(
        data: Any,
        config: MapMakingConfig,
        model: TileModel,
        tod: PyTree[Array],
        temperature: Float[Array, 'det samp'] | None = None,
    ) -> UnfactoredTemplates:
        """Build one tile's templates, and weight its TOD.

        The implicit templates are factored here and marginalized from the weight, which needs
        their Gram inverse. The Grams of the explicit ones are only assembled: for the templates
        the observation's detectors share, the tile's share of a Gram summed over its tiles.
        [`from_inverses`][] makes the tile's templates once they are factored. The
        templates the detectors share are taken under the marginalized weight, those of each
        detector under `W`.

        Args:
            data: The fields read for the tile's detectors.
            config: The mapmaking configuration, whose templates are built.
            model: The tile's model.
            tod: The TOD to weight.
            temperature: The I timestream, from which the T2P template is built; by default, the
                I stream of the sample data.
        """
        if (tcfg := config.templates) is None:
            raise ValueError('templates config required to build template operators')

        explicit_templates, implicit_templates = build_templates(
            tcfg, data, model.tod_format, temperature=temperature
        )

        # The Grams are taken under the weight of the system (see `deprojected_weight`), from
        # `model.W` and `tau`, when it is diagonal. A correlated weight couples samples, which the
        # Grams cannot take. The templates are then all explicit (see `MapMakingConfig`), so their
        # Grams only precondition the solve and set its ridge: those under the zero-lag white
        # weight serve.
        pomme_tau = config.pomme_tau
        gram_weight = model.W if config.binned else model.diag_W()

        implicit = None
        weight = gram_weight
        if implicit_templates:
            if not all(template.per_detector for template in implicit_templates.values()):
                raise NotImplementedError(
                    'implicit templates need amplitudes of each detector alone: those shared '
                    'by detectors would couple the tiles of an observation'
                )
            op = TemplateOperator(implicit_templates)
            implicit = TemplateBundle.create(op, model.W, tcfg, pomme_tau=pomme_tau)
            weight = MarginalizedWeightOperator.create(model.W, op, implicit.gram_inverse)

        def by_detectors(templates, per_detector):
            chosen = {n: t for n, t in templates.items() if t.per_detector == per_detector}
            return TemplateOperator(chosen) if chosen else None

        operators: dict[TemplateKind, TemplateOperator | None] = {
            TEMPLATES: by_detectors(explicit_templates, True),
            SHARED_TEMPLATES: by_detectors(explicit_templates, False),
        }
        weights = {TEMPLATES: gram_weight, SHARED_TEMPLATES: weight}
        grams: dict[TemplateKind, BlockGram | None] = {
            kind: None
            if op is None
            else template_gram(
                op,
                weights[kind],
                pomme_tau=pomme_tau,
                batch_size=tcfg.gram_batch_size,
            )
            for kind, op in operators.items()
        }
        wd = model.deprojected_weight(implicit)(tod)
        return UnfactoredTemplates(implicit, operators, grams, wd)


def restrict_streams(
    data: dict[str, Any], tod: Stokes, stokes: ValidStokesLiteral
) -> tuple[dict[str, Any], Stokes]:
    """A tile's sample data, noise fits and TOD, restricted to the streams `stokes`.

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
