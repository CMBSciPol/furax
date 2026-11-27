r"""A detector tile of an observation: the model of its TOD, its templates, and its operators.

The operators are those of the tile's block of the normal system: the joint operator $J_t$ from
the unknowns to the TOD, the weight $W'_t$, and their products.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal, NamedTuple, Self

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
from furax.core import BlockRowOperator, LowRankFourierOperator
from furax.linalg import CGSolver
from furax.obs import TODFormat
from furax.obs.landscapes import LocalStokesLandscape, StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._gaps import GapBasisOperator, GapBlockPreconditioner, GapSplines
from ._observation import ReaderField, _sample_rate, real_extent
from .acquisition import _acquisition_factors
from .config import (
    ExplicitGapsConfig,
    GapTreatment,
    MapMakingConfig,
    NoiseSource,
    TemplatesConfig,
    WeightingMode,
)
from .noise import (
    LowRankModesOperator,
    LowRankNoiseModel,
    ModesCore,
    NoiseModel,
    OneOverFNoiseModel,
    WhiteNoiseModel,
)
from .noise.lowrank import _low_rank_spectra
from .templates import GLOBAL_TEMPLATE_NAMES, Template, TemplateOperator, build_templates
from .templates.fit import TemplateFit
from .templates.gram import BlockGram, factor, template_gram, template_gram_diagonal
from .templates.pomme import PommeProjectionOperator
from .weight import MarginalizedWeightOperator, NestedWeightOperator, WeightOperator

TemplateKind = Literal['templates', 'shared_templates', 'global_templates']
"""A kind of template amplitudes solved for with the map."""

AmplitudeKind = TemplateKind | Literal['gaps']
"""A kind of amplitudes solved for with the map: a key of the unknowns besides the `'map'`."""

TEMPLATES: Final = 'templates'
"""The amplitudes of the explicit templates of each detector."""

SHARED_TEMPLATES: Final = 'shared_templates'
"""The amplitudes of the explicit templates the detectors of an observation share."""

GLOBAL_TEMPLATES: Final = 'global_templates'
"""The amplitudes of the explicit templates every observation of the run shares."""

GAPS: Final = 'gaps'
"""The gap amplitudes of each detector."""

AMPLITUDE_STRIDES: Final[dict[AmplitudeKind, Literal[1, 'segment'] | None]] = {
    TEMPLATES: 1,
    SHARED_TEMPLATES: 'segment',
    GLOBAL_TEMPLATES: None,
    GAPS: 1,
}
"""How the tiles hold each kind of amplitudes: per tile (stride 1), per observation
(`'segment'`), or once for the run, read whole by every tile as the map is (`None`)."""

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

    gap_preconditioner: AbstractLinearOperator | None = None
    """The preconditioner of the gap amplitudes, an approximate inverse of their Gram (see
    `ExplicitGapsConfig.preconditioner`); `None` unless the gaps are explicit."""

    implicit: TemplateBundle | None = None
    """The implicit templates, whose amplitudes are each detector's own, marginalized."""

    explicit: dict[TemplateKind, TemplateBundle] = field(default_factory=dict)
    """The explicit templates, solved for with the map, by the kind of their amplitudes: only the
    kinds the tile has."""

    H_S: AbstractLinearOperator | None = None
    r"""The acquisition $H S^\top$ from the selected pixels of the sky, the unknowns of the map;
    `None` until they are selected (see [`selected`][furax.mapmaking._tile.Tile.selected])."""

    modes_core: ModesCore | None = None
    r"""Under a low-rank noise model, the core $K$ of the correlated modes of its observation (see
    [`modes`][furax.mapmaking._tile.Tile.modes]); `None` otherwise."""

    @classmethod
    def create(
        cls,
        data: Any,
        padding: Any,
        config: MapMakingConfig,
        landscape: StokesLandscape,
        use: Bool[Array, ''] | bool = True,
        gaps: GapSplines | None = None,
        noise_model: PyTree[NoiseModel] | None = None,
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
            noise_model: The noise model of its detectors, fitted beforehand, as the low-rank
                one is, from their whole observation; fitted from the tile by default.
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
        if noise_model is None:
            noise_model, sample_rate = _noise_model(data, config, tod_format, n_samples)
        else:
            sample_rate = _sample_rate(data[ReaderField.TIMESTAMPS])
        # correlations across detectors, if any, are marginalized over the whole observation, not
        # weighted in a tile (see `Tile.modes`)
        Ninv = noise_model.decoupled().inverse_covariance(tod_format.structure)
        W: WeightOperator | NestedWeightOperator
        if config.gap_treatment == GapTreatment.NESTED:
            # Minimum-variance correlated-noise weight; the white-noise weight preconditions its
            # linear inner solve
            white = noise_model.white().inverse_covariance(tod_format.structure)
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
        # computed once: it is held per amplitude, while the basis it comes from is TOD-sized
        return replace(tile, gap_preconditioner=tile._gap_preconditioner(config.gaps.explicit))

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

    def modes(self) -> AbstractLinearOperator:
        r"""$T_t^\top D_t^{-1} P_t$: from the TOD to the amplitudes of the correlated modes.

        Under a low-rank noise model $N = D + V G V^\dagger$, the weight of a tile is that of the
        diagonal $D$, which couples no detectors (see
        [`system_weight`][furax.mapmaking._tile.Tile.system_weight]), and the correlated modes
        $T$ (see [`LowRankModesOperator`][furax.mapmaking.noise.LowRankModesOperator]) couple
        the tiles of the observation: $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$, $K$ the
        [`modes_core`][furax.mapmaking._tile.Tile.modes_core]. $P_t$ is the mask $M_t$, the
        identity with explicit gaps, which leave the weight unmasked. The amplitudes are
        `(frequency, mode, 2)`, summed over the tiles of the observation.
        """
        T, inverse_diagonal, P = self.low_rank_weight()
        # D⁻¹ T as modes of their own: one FFT, rather than D⁻¹'s pair and then T's
        return T.weighted_by(inverse_diagonal.diagonal).T @ P

    def low_rank_weight(
        self,
    ) -> tuple[LowRankModesOperator, LowRankFourierOperator, AbstractLinearOperator]:
        r"""The modes $T_t$, $D_t^{-1}$ and $P_t$ of the low-rank weight $W'_t = P_t D_t^{-1} P_t$.

        See [`modes`][furax.mapmaking._tile.Tile.modes].
        """
        assert isinstance(self.noise_model, LowRankNoiseModel)
        # the low-rank weight is never nested, and has neither implicit templates nor Pomme
        assert isinstance(self.W, WeightOperator) and self.implicit is None
        assert isinstance(self.F, IdentityOperator)
        inverse_diagonal = self.W.ninv
        assert isinstance(inverse_diagonal, LowRankFourierOperator)
        T = self.noise_model.modes_operator(self.tod_format.structure)
        P = self.M if self.gaps is None else IdentityOperator(in_structure=self.W.in_structure)
        return T, inverse_diagonal, P

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

    def covariance(self) -> AbstractLinearOperator:
        """The noise covariance of the TOD, decoupled across detectors as the weight is."""
        return self.noise_model.decoupled().covariance(self.tod_format.structure)

    def white_weight(self) -> WeightOperator:
        """The weight of the tile under the white part of its noise, with its mask."""
        inv = self.noise_model.white().inverse_covariance(self.tod_format.structure)
        return WeightOperator.create(inv, self.M)

    @property
    def tod_format(self) -> TODFormat:
        """The format of the tile's TOD."""
        return _tod_format(self.H.out_structure)

    @property
    def M(self) -> MaskOperator:
        """The mask $M$ of the good samples."""
        return self.W.mask

    def _gap_preconditioner(self, config: ExplicitGapsConfig) -> AbstractLinearOperator:
        assert self.gaps is not None
        C = self.gaps.basis(self.M, self.n_samples)
        weight = self.system_weight()
        if config.preconditioner == 'block' and isinstance(weight, LowRankFourierOperator):
            return GapBlockPreconditioner.create(self.gaps, C, weight, config.block_size)
        diagonal = C.squared().T(self.white_weight().ninv(self.tod_format.ones()))
        usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
        inverse = jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)
        return DiagonalOperator(inverse, in_structure=C.in_structure)


@register_dataclass
@dataclass
class TemplateBundle:
    r"""A kind of amplitudes of a tile: their operator, the inverse of their Gram, and its ridge.

    The gap amplitudes have a bundle too: the gap basis $C$, their preconditioner, and no ridge
    (see [`Tile.templates`][furax.mapmaking._tile.Tile.templates]).
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
) -> tuple[
    TemplateBundle | None,
    dict[TemplateKind, TemplateBundle],
    BlockGram | None,
    PyTree[Array] | None,
]:
    """The templates of a tile.

    The Grams of the implicit templates, and of the explicit ones of each detector, are factored.
    That of the templates the detectors of the observation share is only assembled: it is the
    tile's share of theirs, taken under the weight with the implicit templates marginalized. Of
    the templates every observation shares, the tile gives its share of the diagonal of their Gram
    (see [`template_gram_diagonal`][furax.mapmaking.templates.gram.template_gram_diagonal]).

    Args:
        data: The fields read for the tile's detectors.
        config: The mapmaking configuration, whose templates are built.
        tile: The tile, without templates.
        temperature: The I timestream, from which the T2P template is built; by default, the
            I stream of the sample data.

    Returns:
        The implicit templates, `None` if absent; the explicit ones by the kind of their
        amplitudes, those the detectors or the observations share without their Gram inverse; the
        tile's share of the Gram of the templates the detectors of the observation share, and of
        the diagonal of that of the templates every observation shares, under the weight without
        the implicit templates nor Pomme; `None` without them.
    """
    if (tcfg := config.templates) is None:
        raise ValueError('templates config required to build template operators')

    explicit_templates, implicit_templates = build_templates(
        tcfg, data, tile.tod_format, temperature=temperature
    )

    # The Grams are taken under the weight of the system (see `Tile.system_weight`), from
    # `tile.W` and `tau`, when it is diagonal. A correlated weight couples samples or detectors,
    # which the Grams cannot take. The templates are then all explicit (see `MapMakingConfig`), so their
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

    of_kind = _by_kind(explicit_templates)
    explicit: dict[TemplateKind, TemplateBundle] = {}
    shared_gram = global_diagonal = None
    if (op := of_kind.get(TEMPLATES)) is not None:
        explicit[TEMPLATES] = TemplateBundle.create(op, gram_weight, tcfg, pomme_tau=pomme_tau)
    if (op := of_kind.get(SHARED_TEMPLATES)) is not None:
        shared_gram = template_gram(
            op, weight, pomme_tau=pomme_tau, batch_size=tcfg.gram_batch_size
        )
        explicit[SHARED_TEMPLATES] = TemplateBundle(op, None, None)
    if (op := of_kind.get(GLOBAL_TEMPLATES)) is not None:
        # Too many amplitudes for a dense Gram: its diagonal under the weight without the implicit
        # templates nor Pomme preconditions them
        global_diagonal = template_gram_diagonal(op, gram_weight)
        explicit[GLOBAL_TEMPLATES] = TemplateBundle(op, None, None)
    return implicit, explicit, shared_gram, global_diagonal


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
    # One noise model covers every stream of a demodulated TOD: its per-detector parameters carry
    # the leading stream axis.
    lead = (
        tod_format.det_shape
        if tod_format.stokes is None
        else (tod_format.n_streams, *tod_format.det_shape)
    )
    if config.weighting.mode == WeightingMode.IDENTITY:
        # the one-sided PSD of a unit variance per sample
        sigma = jnp.full(lead, jnp.sqrt(2 / fs), dtype=tod_format.dtype)
        return WhiteNoiseModel(sigma=sigma, sample_rate=fs), fs
    if config.weighting.mode == WeightingMode.LOW_RANK:
        raise ValueError('the low-rank noise model is fitted over whole observations, not a tile')
    if config.weighting.source == NoiseSource.FIT:
        fit_config = config.weighting.fitting
        noise_model_class = WhiteNoiseModel if config.binned else OneOverFNoiseModel
        fhwp = _hwp_frequency(data[ReaderField.TIMESTAMPS], data[ReaderField.HWP_ANGLES])

        shape = (*lead, tod_format.n_samples)
        noise_model = noise_model_class.fit(
            tod_format.as_array(data[ReaderField.SAMPLE_DATA]).reshape(shape),
            sample_rate=fs,
            hwp_frequency=fhwp,
            n_real=n_samples,
            # the flags of each detector apply to all its streams
            valid=jnp.broadcast_to(_sample_mask(data, config), shape),
            config=fit_config,
        )
    else:
        fits = data[ReaderField.NOISE_MODEL_FITS]  # (*lead, 4), already a plain array
        noise_model = OneOverFNoiseModel(*jnp.moveaxis(fits, -1, 0), sample_rate=fs)
        if config.binned:
            noise_model = noise_model.white()
    if isinstance(noise_model, OneOverFNoiseModel):
        correlation_length = config.weighting.correlation_length
        noise_model = replace(noise_model, correlation_length=correlation_length)
    return noise_model, fs


def _hwp_frequency(
    timestamps: Float[Array, ' ...'], hwp_angles: Float[Array, ' ...']
) -> Float[Array, '']:
    """The rotation frequency of the HWP, from its angles at `timestamps`."""
    # Note that the reader extrapolates hwp_angles in the padded region, keeping hwp freq constant
    return (jnp.unwrap(hwp_angles)[-1] - hwp_angles[0]) / jnp.ptp(timestamps) / (2 * jnp.pi)


def low_rank_tile_spectra(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    padding: Any,
    sample_rate: float,
    temperature: Float[Array, 'det samp'] | None,
    shared_amplitudes: PyTree[Array] | None = None,
) -> tuple[Array, Array, Array]:
    """What the low-rank noise fit takes of the channels of a tile, and the samples of its observation.

    Returns the spectrum and the white level of each channel (see
    [`_low_rank_spectra`][furax.mapmaking.noise.lowrank._low_rank_spectra]), streams then detectors, and the
    number of samples before the padding. With `deproject`, the TOD is first rid of its fit by the
    templates: those the detectors share with `shared_amplitudes`, fitted over the whole
    observation (see [`shared_deprojection`][furax.mapmaking._tile.shared_deprojection]), then
    each detector's own, on the tile.

    Args:
        data: The fields read for the tile's detectors.
        config: The mapmaking configuration.
        tod_format: The format of the tile's TOD.
        padding: The padding widths of the fields.
        sample_rate: The sample rate at which the bins of the model are placed, in Hz.
        temperature: The I timestream, from which the T2P template is built; by default, the I
            stream of the sample data.
        shared_amplitudes: The amplitudes of the templates the detectors share, fitted over the
            observation; `None` without such templates.
    """
    fit_config = config.weighting.low_rank
    _, n_real = real_extent(padding, (1, tod_format.n_samples))
    tod = data[ReaderField.SAMPLE_DATA]
    if fit_config.deproject and config.templates is not None:
        tod = _deprojected_tod(data, config, tod_format, n_real, temperature, shared_amplitudes)
    samples = tod_format.as_array(tod).reshape(-1, tod_format.n_samples)
    spectra, level = _low_rank_spectra(samples, sample_rate, n_real, fit_config)
    return spectra, level, n_real


def shared_deprojection(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    padding: Any,
    temperature: Float[Array, 'det samp'] | None,
) -> dict[str, Any]:
    r"""A tile's share of the fit of the templates its detectors share, before the low-rank fit.

    The amplitudes $a_s$ of the templates $T_s$ the detectors share are fitted over the whole
    observation, so that the TOD the noise is estimated from does not depend on how the
    detectors are tiled. Under the white weight $W$ of the deprojection, with the templates of
    each detector $T_l$ marginalized, $W' = W - W T_l G_l^{-1} T_l^\top W$, a tile contributes
    $T_s^\top W' T_s$ and $T_s^\top W' d$: summed over the tiles of the observation, they give
    $a_s$. Those of the templates every observation shares, $a_g$, are fitted per observation
    here, from the diagonal of $T_g^\top W T_g$ and $T_g^\top W d$ alone, apart from the other
    templates: the fit only shapes the data the noise is estimated from.

    Returns:
        The tile's share, under `'gram'` and `'rhs'`, of the Gram and right-hand side of the
        templates the detectors of the observation share, and under `'global_diagonal'` and
        `'global_rhs'` of the diagonal of the Gram and the right-hand side of those every
        observation shares; `None` for those without templates.
    """
    _, n_real = real_extent(padding, (1, tod_format.n_samples))
    parts = _deprojection_parts(data, config, tod_format, n_real, temperature)
    assert config.templates is not None  # deprojection comes with templates
    templates = config.templates
    tod = data[ReaderField.SAMPLE_DATA]
    terms: dict[str, Any] = dict.fromkeys(('gram', 'rhs', 'global_diagonal', 'global_rhs'))
    if parts.global_ is not None:
        terms['global_diagonal'] = template_gram_diagonal(parts.global_, parts.weight)
        terms['global_rhs'] = (parts.global_.T @ parts.weight)(tod)
    if parts.shared is not None:
        weight = parts.weight
        if parts.local is not None:
            bundle = TemplateBundle.create(parts.local, weight, templates)
            assert bundle.gram_inverse is not None  # set by `create`
            weight = MarginalizedWeightOperator.create(weight, parts.local, bundle.gram_inverse)
        terms['gram'] = template_gram(parts.shared, weight, batch_size=templates.gram_batch_size)
        terms['rhs'] = (parts.shared.T @ weight)(tod)
    return terms


def _deprojected_tod(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    n_real: Int[Array, ''],
    temperature: Float[Array, 'det samp'] | None,
    shared_amplitudes: PyTree[Array] | None,
) -> PyTree[Array]:
    r"""The TOD less its fit by every configured template, $d - T_s a_s - T_l \hat a_l$.

    The amplitudes $a_s$ of the templates the detectors share, and of those every observation
    shares, are given, fitted over the whole observation; those of each detector's own templates are fitted by least squares under a white
    weight: each channel's inverse variance, from its sample-to-sample differences, on its valid
    samples. The fit only shapes the data the noise is estimated from, so a loose tolerance serves.
    """
    assert config.templates is not None
    parts = _deprojection_parts(data, config, tod_format, n_real, temperature)
    tod = data[ReaderField.SAMPLE_DATA]
    for templates in (parts.shared, parts.global_):
        if templates is not None:
            assert shared_amplitudes is not None  # fitted over the observation beforehand
            amplitudes = {name: shared_amplitudes[name] for name in templates.templates}
            tod = tree.sub(tod, templates(amplitudes))
    if parts.local is None:
        return tod
    fit = TemplateFit(
        parts.local,
        parts.weight,
        regularization=config.templates.regularization,
        batch_size=config.templates.gram_batch_size,
        solver=CGSolver(rtol=1e-4, max_steps=_DEPROJECTION_STEPS),
    )
    return tree.sub(tod, fit.model(fit(tod).solution))


class _DeprojectionParts(NamedTuple):
    """The templates of the deprojection before the low-rank fit, by kind, and its weight.

    A template operator is `None` without templates of its kind.
    """

    local: TemplateOperator | None
    """The templates of each detector."""
    shared: TemplateOperator | None
    """The templates the detectors of the observation share."""
    global_: TemplateOperator | None
    """The templates every observation shares."""
    weight: AbstractLinearOperator
    """Each channel's inverse variance, from its sample-to-sample differences, on its valid
    samples."""


def _deprojection_parts(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    n_real: Int[Array, ''],
    temperature: Float[Array, 'det samp'] | None,
) -> _DeprojectionParts:
    """The templates of the deprojection before the low-rank fit, and its weight."""
    assert config.templates is not None
    explicit, implicit = build_templates(
        config.templates, data, tod_format, temperature=temperature
    )
    of_kind = _by_kind({**explicit, **implicit})
    samples = tod_format.as_array(data[ReaderField.SAMPLE_DATA])  # (stream, det, sample)
    valid = _sample_mask(data, config) & (jnp.arange(tod_format.n_samples) < n_real)
    pairs = valid[..., 1:] & valid[..., :-1]
    steps = jnp.where(pairs, jnp.diff(samples, axis=-1), 0)
    variance = jnp.sum(steps**2, axis=-1) / jnp.maximum(2 * pairs.sum(axis=-1), 1)
    inverse = jnp.where(variance > 0, 1 / jnp.where(variance > 0, variance, 1), 0)
    channel_weight = tod_format.diagonal(inverse[..., None].astype(samples.dtype))
    sample_weight = tod_format.diagonal(valid[None].astype(samples.dtype))
    weight = (channel_weight @ sample_weight).reduce()
    return _DeprojectionParts(
        of_kind.get(TEMPLATES), of_kind.get(SHARED_TEMPLATES), of_kind.get(GLOBAL_TEMPLATES), weight
    )


def _by_kind(templates: dict[str, Template]) -> dict[TemplateKind, TemplateOperator]:
    """The operator of the templates of each kind of amplitudes among `templates`, if any.

    The amplitudes are each detector's own, those the detectors of the observation share, or those
    every observation shares (see [`GLOBAL_TEMPLATE_NAMES`][furax.mapmaking.templates]).
    """
    grouped: dict[TemplateKind, dict[str, Template]] = {}
    for name, template in templates.items():
        if name in GLOBAL_TEMPLATE_NAMES:
            kind: TemplateKind = GLOBAL_TEMPLATES
        else:
            kind = TEMPLATES if template.per_detector else SHARED_TEMPLATES
        grouped.setdefault(kind, {})[name] = template
    return {kind: TemplateOperator(chosen) for kind, chosen in grouped.items()}


_DEPROJECTION_STEPS = 50
"""The most conjugate gradient steps of the template fit that precedes the low-rank noise fit."""
