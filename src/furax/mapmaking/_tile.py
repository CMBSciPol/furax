r"""A detector tile of an observation: the model of its TOD, its templates, and its operators.

The operators are those of the tile's block of the normal system: the joint operator $J_t$ from
the unknowns to the TOD, the weight $W'_t$, and their products.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, Self

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PyTree

from furax import (
    AbstractLinearOperator,
    DiagonalOperator,
    IdentityOperator,
    MaskOperator,
    SymmetricBandToeplitzOperator,
    tree,
)
from furax.core import BlockRowOperator
from furax.obs import TODFormat
from furax.obs.landscapes import LocalStokesLandscape, StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._gaps import GapBasisOperator, GapSplines
from ._observation import ReaderField, _sample_rate, real_extent
from .acquisition import _acquisition_factors
from .config import GapTreatment, MapMakingConfig, NoiseSource, TemplatesConfig, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel, padding_aware_welch
from .templates import TemplateOperator, build_templates
from .templates.gram import BlockGram, factor, template_gram
from .templates.pomme import PommeProjectionOperator
from .weight import MarginalizedWeightOperator, NestedWeightOperator, WeightOperator

TemplateKind = Literal['templates', 'shared_templates']
"""A kind of template amplitudes solved for with the map."""

AmplitudeKind = TemplateKind | Literal['gaps']
"""A kind of amplitudes solved for with the map: a key of the unknowns besides the `'map'`."""

TEMPLATES: Final = 'templates'
"""The amplitudes of the explicit templates of each detector."""

SHARED_TEMPLATES: Final = 'shared_templates'
"""The amplitudes of the explicit templates the detectors of an observation share."""

GAPS: Final = 'gaps'
"""The gap amplitudes of each detector."""

AMPLITUDE_STRIDES: Final[dict[AmplitudeKind, Literal[1, 'segment']]] = {
    TEMPLATES: 1,
    SHARED_TEMPLATES: 'segment',
    GAPS: 1,
}
"""How the tiles hold each kind of amplitudes: per tile (stride 1), or per observation
(`'segment'`)."""

DROP_REASONS: Final = ('over_flagged', 'over_gap_budget')
"""Why a detector is dropped: the fields of [`Tile`][] marking the detectors dropped for each."""


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """A detector tile of an observation: the model of its TOD, and its templates.

    The templates are partitioned by their `explicit` config flag: the amplitudes of the implicit
    ones are marginalized in the weight, those of the explicit ones solved for with the map.
    """

    R: AbstractLinearOperator
    """The response $R$ of the detectors, from the TOD the pointing gives to theirs."""

    P: PointingOperator
    """The pointing $P$, from the sky to the TOD; the acquisition is $H = R P$."""

    W: WeightOperator | NestedWeightOperator
    """The weight $W$: the inverse noise, with the mask $M$ of the good samples."""

    F: AbstractLinearOperator
    """The Pomme deprojector $F$; the identity without Pomme."""

    noise_model: PyTree[NoiseModel]
    """The noise model of the TOD."""

    sample_rate: Array
    """The sample rate of the TOD."""

    over_flagged: Bool[Array, ' det']
    """Per detector, whether it is dropped for having too many of its samples flagged."""

    over_gap_budget: Bool[Array, ' det']
    """Per detector, whether it is dropped for needing more gap amplitudes than `gaps` holds."""

    n_samples: Int[Array, '']
    """The number of samples of the observation; those after them are padding."""

    gaps: GapSplines | None = field(default=None, metadata={'static': True})
    """The spline model of the gaps, whose amplitudes are solved for with the map; `None` unless
    the gaps are explicit."""

    gap_preconditioner: DiagonalOperator | None = None
    """The inverse of the diagonal of the white-noise Gram $C^\top D C$ of the gap amplitudes;
    `None` unless the gaps are explicit."""

    implicit: TemplateBundle | None = None
    """The implicit templates, whose amplitudes are each detector's own, marginalized."""

    explicit: dict[TemplateKind, TemplateBundle] = field(default_factory=dict)
    """The explicit templates, solved for with the map, by the kind of their amplitudes: only the
    kinds the tile has."""

    H_S: AbstractLinearOperator | None = None
    r"""The acquisition $H S^\top$ from the selected pixels of the sky, the unknowns of the map;
    `None` until they are selected (see [`selected`][furax.mapmaking._tile.Tile.selected])."""

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
        """The tile of the fields read for its detectors, without templates.

        Args:
            data: The fields read for the tile's detectors.
            padding: Their padding widths.
            config: The mapmaking configuration.
            landscape: The landscape of the maps.
            use: Whether the tile contributes; if not, all its samples are masked, and none of its
                detectors is dropped.
            gaps: The spline model of the gaps, when they are explicit; a detector needing more
                amplitudes than its budget is dropped.
        """
        R, P = acquisition_factors(data, config, landscape)
        tod_format = _tod_format(R.out_structure)
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
        # a per-sample mask (det, samp) broadcasts over the leading stream axis of a demodulated TOD
        M = MaskOperator.from_boolean_mask(
            mask & valid_detectors, in_structure=tod_format.structure
        )
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
        tile = cls(
            R, P, W, F, noise_model, sample_rate, over_flagged, over_gap_budget, n_samples, gaps
        )
        if gaps is None:
            return tile
        # computed once: it is amplitude-sized, while the basis it comes from is TOD-sized
        return replace(tile, gap_preconditioner=tile._gap_preconditioner())

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """The reader fields [`create`][furax.mapmaking._tile.Tile.create] needs."""
        fields: set[str] = {
            ReaderField.Q_SKY_BORE,
            ReaderField.Q_BORE_DET,
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

    def selected(self, pixels: LocalStokesLandscape) -> Self:
        r"""The tile whose map is that of the selected `pixels`, read by $H_t S^\top$.

        Its pointing reads and bins those pixels alone, so that no map of the full sky is made in
        the normal system. The acquisition from the full sky is kept, for a sky given on it.
        """
        return replace(self, H_S=self.R @ self.P.with_landscape(pixels))

    def joint(self) -> AbstractLinearOperator:
        r"""The joint operator $J_t$, from the unknowns to the TOD.

        The unknowns are a dictionary of the `'map'` and of the amplitudes of each kind the tile
        has (see [`kinds`][furax.mapmaking._tile.Tile.kinds]): $J_t = [H_t, T_t, T_{s,t}]$, the
        map that of the selected pixels once they are (see
        [`selected`][furax.mapmaking._tile.Tile.selected]), $H_t S^\top$ in place of $H_t$, and
        on the full sky before. With explicit gaps, whose amplitudes stand for the flagged
        samples, the other blocks are masked: $J_t = [M_t H_t, M_t T_t, M_t T_{s,t}, C_t]$. Its
        weight in the normal system is [`system_weight`][furax.mapmaking._tile.Tile.system_weight].
        """
        H = self.H if self.H_S is None else self.H_S
        blocks = {'map': H} | {kind: bundle.operator for kind, bundle in self.explicit.items()}
        if self.gaps is not None:
            blocks = {kind: self.M @ block for kind, block in blocks.items()}
            blocks[GAPS] = GapBasisOperator.create(self.gaps, self.M, self.n_samples)
        return BlockRowOperator(blocks)

    def system_weight(self) -> AbstractLinearOperator:
        r"""The weight $W'_t$ of the tile in the normal system.

        It is $W F$, with $F$ the Pomme deprojector. With implicit templates $T$, their
        amplitudes are marginalized: $W' = W F - W F T G^{-1} T^\top W F$, with $G^{-1}$ the
        inverse of their Gram (see
        [`MarginalizedWeightOperator`][furax.mapmaking.weight.MarginalizedWeightOperator]). With
        explicit gaps, it is the inverse noise $N^{-1}$ unmasked: the gap amplitudes stand for
        the flagged samples.
        """
        if self.gaps is not None:
            # explicit gaps need a correlated weight, which neither Pomme nor implicit templates
            # allow
            assert self.implicit is None and isinstance(self.F, IdentityOperator)
            return self.W.ninv
        weight = (self.W @ self.F).reduce()
        if self.implicit is None:
            return weight
        implicit = self.implicit
        assert implicit.gram_inverse is not None  # implicit templates are factored per tile
        return MarginalizedWeightOperator.create(weight, implicit.operator, implicit.gram_inverse)

    @property
    def H(self) -> AbstractLinearOperator:
        """The acquisition operator $H$, from the sky to the TOD."""
        return self.R @ self.P

    def masked_acquisition(self) -> AbstractLinearOperator:
        """$M_t H_t$: from a sky on the full grid to the good samples of the TOD it gives."""
        return self.M @ self.H

    def map_block(self) -> AbstractLinearOperator:
        r"""$H_t^\top W_t F_t H_t$: the map block of the system, without the implicit templates."""
        return self.H.T @ (self.W @ self.F).reduce() @ self.H

    def white_map_block(self) -> AbstractLinearOperator:
        r"""The [`map_block`][furax.mapmaking._tile.Tile.map_block] under the white weight."""
        return self.H.T @ (self.white_weight() @ self.F).reduce() @ self.H

    @property
    def kinds(self) -> tuple[AmplitudeKind, ...]:
        """The kinds of amplitudes the tile has, each a key of the unknowns besides the `'map'`."""
        has_gaps = self.gaps is not None
        return tuple(k for k in AMPLITUDE_STRIDES if k in self.explicit or (k == GAPS and has_gaps))

    def templates(self, kind: AmplitudeKind) -> TemplateBundle | None:
        r"""The templates of the amplitudes of `kind`; `None` if the tile has none.

        Their bundle holds the operator from them to the TOD, $T_t$, $T_{s,t}$ or $C_t$, the
        inverse of their Gram, and its ridge. The gap amplitudes are preconditioned by
        [`gap_preconditioner`][furax.mapmaking._tile.Tile.gap_preconditioner], with no ridge, and
        their basis is built from the mask when applied.
        """
        if kind != GAPS:
            return self.explicit.get(kind)
        if self.gaps is None:
            return None
        basis = GapBasisOperator.create(self.gaps, self.M, self.n_samples)
        return TemplateBundle(basis, self.gap_preconditioner, None)

    def covariance(self, correlation_length: int) -> AbstractLinearOperator:
        """The noise covariance $N$ of the TOD.

        Args:
            correlation_length: The band of the Toeplitz operator of a correlated noise model.
        """
        return _noise_operator(
            self.noise_model, self.tod_format, self.sample_rate, correlation_length, inverse=False
        )

    def white_weight(self) -> WeightOperator:
        """The weight of the tile under the white part of its noise, with its mask."""
        white = self.noise_model.to_white_noise_model()
        inv = _noise_operator(white, self.tod_format, self.sample_rate)
        return WeightOperator.create(inv, self.M)

    @property
    def tod_format(self) -> TODFormat:
        """The format of the tile's TOD."""
        return _tod_format(self.H.out_structure)

    @property
    def M(self) -> MaskOperator:
        """The mask $M$ of the good samples."""
        return self.W.mask

    def _gap_preconditioner(self) -> DiagonalOperator:
        assert self.gaps is not None
        C = self.gaps.basis(self.M, self.n_samples)
        diagonal = C.squared().T(self.white_weight().ninv(self.tod_format.ones()))
        usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
        inverse = jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)
        return DiagonalOperator(inverse, in_structure=C.in_structure)


@register_dataclass
@dataclass
class TemplateBundle:
    r"""A kind of amplitudes of a tile: their operator, the inverse of their Gram, and its ridge.

    The gap amplitudes have a bundle too: the gap basis $C$, the inverse of the diagonal of their
    white-noise Gram, and no ridge (see [`Tile.templates`][furax.mapmaking._tile.Tile.templates]).
    """

    operator: AbstractLinearOperator
    """The operator $T$ from the amplitudes to the TOD."""
    gram_inverse: AbstractLinearOperator | None
    """The inverse of their Gram $T^\top W T$; `None` until it is factored."""
    ridge: AbstractLinearOperator | None
    """The ridge the Gram was factored with, a term of the system; `None` without
    regularization."""

    @classmethod
    def create(
        cls,
        operator: TemplateOperator,
        weight: AbstractLinearOperator,
        config: TemplatesConfig,
        *,
        pomme_tau: int | None = None,
    ) -> Self:
        r"""The bundle of the templates `operator`, their Gram under `weight` factored.

        Args:
            operator: The template operator $T$.
            weight: The weight $W$ the Gram $T^\top W T$ is taken under.
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


def assemble_templates(
    data: Any,
    config: MapMakingConfig,
    tile: Tile,
    temperature: Float[Array, 'det samp'] | None = None,
) -> tuple[TemplateBundle | None, dict[TemplateKind, TemplateBundle], BlockGram | None]:
    """The templates of a tile.

    The Grams of the implicit templates, and of the explicit ones of each detector, are factored.
    That of the templates the detectors of the observation share is only assembled: it is the
    tile's share of theirs, taken under the weight with the implicit templates marginalized.

    Args:
        data: The fields read for the tile's detectors.
        config: The mapmaking configuration, whose templates are built.
        tile: The tile, without templates.
        temperature: The I timestream, from which the T2P template is built; by default, the
            I stream of the sample data.

    Returns:
        The implicit templates, `None` if absent; the explicit ones by the kind of their
        amplitudes, those the detectors share without their Gram inverse; and the tile's share of
        the Gram of the latter, `None` without them.
    """
    if (tcfg := config.templates) is None:
        raise ValueError('templates config required to build template operators')

    explicit_templates, implicit_templates = build_templates(
        tcfg, data, tile.tod_format, temperature=temperature
    )

    # The Grams are taken under the weight of the system (see `Tile.system_weight`), from
    # `tile.W` and `tau`, when it is diagonal. A correlated weight couples samples, which the
    # Grams cannot take. The templates are then all explicit (see `MapMakingConfig`), so their
    # Grams only precondition the solve and set its ridge: those under the zero-lag white
    # weight serve.
    pomme_tau = config.pomme_tau
    gram_weight = tile.W if config.binned else tile.white_weight()

    implicit = None
    weight = gram_weight
    if implicit_templates:
        if not all(template.per_detector for template in implicit_templates.values()):
            raise NotImplementedError(
                'implicit templates need amplitudes of each detector alone: those shared '
                'by detectors would couple the tiles of an observation'
            )
        op = TemplateOperator(implicit_templates)
        implicit = TemplateBundle.create(op, tile.W, tcfg, pomme_tau=pomme_tau)
        assert implicit.gram_inverse is not None  # set by `create`
        weight = MarginalizedWeightOperator.create(tile.W, op, implicit.gram_inverse)

    def by_detectors(per_detector):
        chosen = {n: t for n, t in explicit_templates.items() if t.per_detector == per_detector}
        return TemplateOperator(chosen) if chosen else None

    explicit: dict[TemplateKind, TemplateBundle] = {}
    shared_gram = None
    if (op := by_detectors(True)) is not None:
        explicit[TEMPLATES] = TemplateBundle.create(op, gram_weight, tcfg, pomme_tau=pomme_tau)
    if (op := by_detectors(False)) is not None:
        shared_gram = template_gram(
            op, weight, pomme_tau=pomme_tau, batch_size=tcfg.gram_batch_size
        )
        explicit[SHARED_TEMPLATES] = TemplateBundle(op, None, None)
    return implicit, explicit, shared_gram


def acquisition_factors(
    data: Any, config: MapMakingConfig, landscape: StokesLandscape
) -> tuple[AbstractLinearOperator, PointingOperator]:
    """The response $R$ and pointing $P$ of the acquisition $H = R P$ of a tile."""
    return _acquisition_factors(
        landscape,
        Quaternion.from_array(data[ReaderField.Q_SKY_BORE]),
        Quaternion.from_array(data[ReaderField.Q_BORE_DET]),
        data.get(ReaderField.HWP_ANGLES),
        demodulated=config.demodulated,
        pointing_on_the_fly=config.pointing.on_the_fly,
        pointing_batch_samples=config.pointing.batch_samples,
        pointing_interpolate=config.pointing.interpolation == 'bilinear',
        dtype=config.dtype,
    )


def restrict_streams(
    data: dict[str, Any], tod: Stokes, stokes: ValidStokesLiteral
) -> tuple[dict[str, Any], Stokes]:
    """A tile's sample data, noise fits and TOD, restricted to the streams `stokes`."""
    index = jnp.array([tod.stokes.index(s) for s in stokes])

    def select(x: Stokes) -> Stokes:
        return Stokes.class_for(stokes).from_array(x.data[index])

    data = {**data, ReaderField.SAMPLE_DATA: select(data[ReaderField.SAMPLE_DATA])}
    if (fits := data.get(ReaderField.NOISE_MODEL_FITS)) is not None:
        data[ReaderField.NOISE_MODEL_FITS] = fits[index]  # one row per stream
    return data, select(tod)


def _tod_format(structure: PyTree[jax.ShapeDtypeStruct]) -> TODFormat:
    """The format of a TOD of structure `structure`."""
    if isinstance(structure, Stokes):
        return TODFormat(structure.shape, structure.dtype, structure.stokes)
    return TODFormat(structure.shape, structure.dtype)


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


def _hwp_frequency(
    timestamps: Float[Array, ' ...'], hwp_angles: Float[Array, ' ...']
) -> Float[Array, '']:
    """The rotation frequency of the HWP, from its angles at `timestamps`."""
    # Note that the reader extrapolates hwp_angles in the padded region, keeping hwp freq constant
    return (jnp.unwrap(hwp_angles)[-1] - hwp_angles[0]) / jnp.ptp(timestamps) / (2 * jnp.pi)


def _noise_operator(
    noise_model: NoiseModel,
    tod_format: TODFormat,
    sample_rate: Array,
    correlation_length: int | None = None,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """The (inverse) noise covariance operator of a tile.

    `correlation_length` sets the Toeplitz band of a correlated (atmospheric) model; a white-noise
    model ignores it.
    """
    build = noise_model.inverse_operator if inverse else noise_model.operator
    return build(
        tod_format.structure, sample_rate=sample_rate, correlation_length=correlation_length
    )
