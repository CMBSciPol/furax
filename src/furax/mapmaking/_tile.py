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
from furax.core import BlockColumnOperator, BlockRowOperator
from furax.linalg import CGSolver
from furax.obs import TODFormat
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._gaps import GapBasisOperator, GapSplines
from ._observation import ReaderField, _sample_rate, real_extent
from .acquisition import build_acquisition_operator
from .config import GapTreatment, MapMakingConfig, NoiseSource, TemplatesConfig, WeightingMode
from .lowrank import LowRankNoiseModel, ModesCore, low_rank_spectra
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel, padding_aware_welch
from .templates import TemplateOperator, build_templates
from .templates.fit import TemplateFit
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

AMPLITUDE_STRIDES: dict[AmplitudeKind, Literal[1, 'segment']] = {
    TEMPLATES: 1,
    SHARED_TEMPLATES: 'segment',
    GAPS: 1,
}
"""How the tiles hold each kind of amplitudes: per tile (stride 1), or per observation
(`'segment'`)."""

DROP_REASONS = ('over_flagged', 'over_gap_budget')
"""Why a detector is dropped: the fields of [`Tile`][] marking the detectors dropped for each."""


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """A detector tile of an observation: the model of its TOD, and its templates.

    The templates are partitioned by their `explicit` config flag: the amplitudes of the implicit
    ones are marginalized in the weight, those of the explicit ones solved for with the map.
    """

    H: AbstractLinearOperator
    """The acquisition operator $H$, from the sky to the TOD."""

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
        # a per-sample mask (det, samp) broadcasts over the leading stream axis of a demodulated TOD
        M = MaskOperator.from_boolean_mask(
            mask & valid_detectors, in_structure=tod_format.structure
        )
        if noise_model is None:
            noise_model, sample_rate = _noise_model(data, config, tod_format, n_samples)
        else:
            sample_rate = _sample_rate(data[ReaderField.TIMESTAMPS])
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
            H, W, F, noise_model, sample_rate, over_flagged, over_gap_budget, n_samples, gaps
        )
        if gaps is None:
            return tile
        # computed once: it is amplitude-sized, while the basis it comes from is TOD-sized
        return replace(tile, gap_preconditioner=tile._gap_preconditioner())

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """The reader fields [`create`][furax.mapmaking._tile.Tile.create] needs."""
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

    def normal(self, selection: AbstractLinearOperator | None = None) -> AbstractLinearOperator:
        r"""$J_t^\top W'_t J_t$, its block of the normal system.

        The selection is that of [`system`][furax.mapmaking._tile.Tile.system].
        """
        J, W = self.system(selection)
        return J.T @ W @ J

    def rhs(self, selection: AbstractLinearOperator | None = None) -> AbstractLinearOperator:
        r"""$J_t^\top W'_t M_t$: from the TOD to its right-hand side, keyed as the unknowns.

        The flagged samples of the TOD are masked: with explicit gaps, $W'_t$ does not mask them.
        The selection is that of [`system`][furax.mapmaking._tile.Tile.system].
        """
        J, W = self.system(selection)
        return J.T @ W @ self.M

    def response(self, selection: AbstractLinearOperator | None = None) -> AbstractLinearOperator:
        r"""$J_t^\top W'_t M_t H_t$: from a sky to the right-hand side of the TOD it gives."""
        return self.rhs(selection) @ self.H

    def modes(self, selection: AbstractLinearOperator | None = None) -> AbstractLinearOperator:
        r"""$T_t^\top D_t^{-1} P_t J_t$: from the unknowns to the amplitudes of the correlated modes.

        Under a low-rank noise model $N = D + V G V^\dagger$, the weight of a tile is that of the
        diagonal $D$, which couples no detectors, and the correlated modes $T$ (see
        [`LowRankModesOperator`][furax.mapmaking.lowrank.LowRankModesOperator]) are marginalized
        over the tiles of the observation: $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$, $K$ the
        [`modes_core`][furax.mapmaking._tile.Tile.modes_core]. $P_t$ is the mask $M_t$, the
        identity with explicit gaps, which leave the weight unmasked. The selection is that of
        [`system`][furax.mapmaking._tile.Tile.system].
        """
        J, _ = self.system(selection)
        T, inverse_diagonal, P = self.low_rank_weight()
        return T.T @ inverse_diagonal @ P @ J

    def normal_and_modes(
        self, selection: AbstractLinearOperator | None = None
    ) -> AbstractLinearOperator:
        r"""[`normal`][furax.mapmaking._tile.Tile.normal] and [`modes`][furax.mapmaking._tile.Tile.modes] at once.

        The output is a dictionary of $J_t^\top W'_t J_t x$, under `'unknowns'`, and of
        $T_t^\top D_t^{-1} P_t J_t x$, under `'modes'`, both from one $D_t^{-1} P_t J_t x$. The
        selection is that of [`system`][furax.mapmaking._tile.Tile.system].
        """
        J, _ = self.system(selection)
        return self._weights_and_modes(J) @ J

    def rhs_and_modes(
        self, selection: AbstractLinearOperator | None = None
    ) -> AbstractLinearOperator:
        r"""[`rhs`][furax.mapmaking._tile.Tile.rhs] and $T_t^\top D_t^{-1} M_t$ at once.

        From the TOD, its right-hand side, under `'unknowns'`, and the amplitudes of its
        correlated modes, under `'modes'`, both from one $D_t^{-1} M_t$ of it.
        """
        J, _ = self.system(selection)
        return self._weights_and_modes(J) @ self.M

    def response_and_modes(
        self, selection: AbstractLinearOperator | None = None
    ) -> AbstractLinearOperator:
        r"""[`rhs_and_modes`][furax.mapmaking._tile.Tile.rhs_and_modes] of the TOD of a sky."""
        return self.rhs_and_modes(selection) @ self.H

    def low_rank_weight(
        self,
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator, AbstractLinearOperator]:
        r"""The modes $T_t$, $D_t^{-1}$ and $P_t$ of the low-rank weight $W'_t = P_t D_t^{-1} P_t$.

        See [`modes`][furax.mapmaking._tile.Tile.modes].
        """
        assert isinstance(self.noise_model, LowRankNoiseModel)
        # the low-rank weight is never nested, and has neither implicit templates nor Pomme
        assert isinstance(self.W, WeightOperator) and self.implicit is None
        assert isinstance(self.F, IdentityOperator)
        T = self.noise_model.modes_operator(self.tod_format.structure)
        P = self.M if self.gaps is None else IdentityOperator(in_structure=self.W.in_structure)
        return T, self.W.weight, P

    def binned_normal(self) -> AbstractLinearOperator:
        r"""$H_t^\top W_t F_t H_t$: the map block of the system, without the implicit templates."""
        return self._map_normal(self.W)

    def white_normal(self) -> AbstractLinearOperator:
        r"""$H_t^\top \operatorname{diag}(W_t) F_t H_t$: the same under the white noise only."""
        return self._map_normal(self.diag_W())

    def system(
        self, selection: AbstractLinearOperator | None = None
    ) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
        r"""The joint operator $J_t$, from the unknowns to the TOD, and the weight $W'_t$.

        The unknowns are a dictionary of the `'map'` and of the amplitudes of each kind the tile
        has (see [`kinds`][furax.mapmaking._tile.Tile.kinds]): $J_t = [H_t S^\top, T_t, T_{s,t}]$.
        With explicit gaps, whose amplitudes stand for the flagged samples, the other blocks are
        masked: $J_t = [M_t H_t S^\top, M_t T_t, M_t T_{s,t}, C_t]$ (see
        [`system_weight`][furax.mapmaking._tile.Tile.system_weight]).

        Args:
            selection: The selection $S$ of the estimated pixels; the whole sky without it.
        """
        H = self.H if selection is None else self.H @ selection.T
        blocks = {'map': H} | {kind: bundle.operator for kind, bundle in self.explicit.items()}
        if self.gaps is not None:
            blocks = {kind: self.M @ block for kind, block in blocks.items()}
            gaps = self.amplitudes(GAPS)
            assert gaps is not None
            blocks[GAPS] = gaps.operator
        return BlockRowOperator(blocks), self.system_weight()

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
            assert isinstance(self.W, WeightOperator)
            return self.W.weight
        weight = (self.W @ self.F).reduce()
        if self.implicit is None:
            return weight
        implicit = self.implicit
        assert implicit.gram_inverse is not None  # implicit templates are factored per tile
        return MarginalizedWeightOperator.create(weight, implicit.operator, implicit.gram_inverse)

    @property
    def kinds(self) -> tuple[AmplitudeKind, ...]:
        """The kinds of amplitudes the tile has, each a key of the unknowns besides the `'map'`."""
        has_gaps = self.gaps is not None
        return tuple(k for k in AMPLITUDE_STRIDES if k in self.explicit or (k == GAPS and has_gaps))

    def amplitudes(self, kind: AmplitudeKind) -> TemplateBundle | None:
        r"""The amplitudes of `kind`; `None` if the tile has none.

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

    def diag_W(self) -> WeightOperator:
        """The weight of the tile under the white part of its noise, with its mask."""
        white = self.noise_model.to_white_noise_model()
        inv = _noise_operator(white, self.tod_format, self.sample_rate)
        return WeightOperator.create(inv, self.M)

    def noise_operator(
        self, correlation_length: int, *, inverse: bool = True
    ) -> AbstractLinearOperator:
        """The noise covariance $N$ of the TOD, or its inverse.

        Args:
            correlation_length: The band of the Toeplitz operator of a correlated noise model.
            inverse: Whether to return $N^{-1}$.
        """
        return _noise_operator(
            self.noise_model,
            self.tod_format,
            self.sample_rate,
            correlation_length,
            inverse=inverse,
        )

    @property
    def tod_format(self) -> TODFormat:
        """The format of the tile's TOD."""
        return _tod_format(self.H)

    @property
    def M(self) -> MaskOperator:
        """The mask $M$ of the good samples."""
        return self.W.mask

    def _weights_and_modes(self, J: AbstractLinearOperator) -> AbstractLinearOperator:
        r"""From a TOD $y$ to $J^\top W'_t y$ and $T_t^\top D_t^{-1} P_t y$, from one $D_t^{-1} P_t y$.

        They are under `'unknowns'` and `'modes'` (see
        [`low_rank_weight`][furax.mapmaking._tile.Tile.low_rank_weight]).
        """
        T, inverse_diagonal, P = self.low_rank_weight()
        blocks = {'unknowns': J.T @ P, 'modes': T.T}
        return BlockColumnOperator(blocks) @ inverse_diagonal @ P

    def _gap_preconditioner(self) -> DiagonalOperator:
        assert self.gaps is not None
        C = self.gaps.basis(self.M, self.n_samples)
        diagonal = C.squared().T(self.diag_W().weight(self.tod_format.ones()))
        usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
        inverse = jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)
        return DiagonalOperator(inverse, in_structure=C.in_structure)

    def _map_normal(self, W: AbstractLinearOperator) -> AbstractLinearOperator:
        r"""$H_t^\top W F_t H_t$ under the weight `W`."""
        H = self.H
        return H.T @ (W @ self.F).reduce() @ H


@register_dataclass
@dataclass
class TemplateBundle:
    r"""A kind of amplitudes of a tile: their operator, the inverse of their Gram, and its ridge.

    The gap amplitudes have a bundle too: the gap basis $C$, the inverse of the diagonal of their
    white-noise Gram, and no ridge (see [`Tile.amplitudes`][furax.mapmaking._tile.Tile.amplitudes]).
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
    # `tile.W` and `tau`, when it is diagonal. A correlated weight couples samples or detectors,
    # which the Grams cannot take. The templates are then all explicit (see `MapMakingConfig`), so their
    # Grams only precondition the solve and set its ridge: those under the zero-lag white
    # weight serve.
    pomme_tau = config.pomme_tau
    gram_weight = tile.W if config.binned else tile.diag_W()

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


def _tod_format(H: AbstractLinearOperator) -> TODFormat:
    """The format of the TOD the acquisition operator `H` maps the sky to."""
    structure = H.out_structure
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
    if config.weighting.mode == WeightingMode.LOW_RANK:
        raise ValueError('the low-rank noise model is fitted over whole observations, not a tile')
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
    model ignores it. The weight of a low-rank model is that of its diagonal alone: its correlated
    modes are marginalized over the tiles of the observation (see [`Tile.modes`][furax.mapmaking._tile.Tile.modes]).
    """
    if inverse and isinstance(noise_model, LowRankNoiseModel):
        return noise_model.diagonal_inverse_operator(tod_format.structure)
    build = noise_model.inverse_operator if inverse else noise_model.operator
    return build(
        tod_format.structure, sample_rate=sample_rate, correlation_length=correlation_length
    )


def low_rank_tile_spectra(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    padding: Any,
    sample_rate: float,
    temperature: Float[Array, 'det samp'] | None,
) -> tuple[Array, Array, Array]:
    """What the low-rank noise fit takes of the channels of a tile, and the samples of its observation.

    Returns the spectrum and the white level of each channel (see
    [`low_rank_spectra`][furax.mapmaking.lowrank.low_rank_spectra]), streams then detectors, and the
    number of samples before the padding. With `deproject`, the TOD is first rid of its fit by the
    templates, on the tile alone.

    Args:
        data: The fields read for the tile's detectors.
        config: The mapmaking configuration.
        tod_format: The format of the tile's TOD.
        padding: The padding widths of the fields.
        sample_rate: The sample rate at which the bins of the model are placed, in Hz.
        temperature: The I timestream, from which the T2P template is built; by default, the I
            stream of the sample data.
    """
    fit_config = config.weighting.low_rank
    _, n_real = real_extent(padding, (1, tod_format.n_samples))
    tod = data[ReaderField.SAMPLE_DATA]
    if fit_config.deproject and config.templates is not None:
        tod = _deprojected_tod(data, config, tod_format, n_real, temperature)
    samples = tod_format.as_array(tod).reshape(-1, tod_format.n_samples)
    spectra, level = low_rank_spectra(samples, sample_rate, n_real, fit_config)
    return spectra, level, n_real


def _deprojected_tod(
    data: Any,
    config: MapMakingConfig,
    tod_format: TODFormat,
    n_real: Int[Array, ''],
    temperature: Float[Array, 'det samp'] | None,
) -> PyTree[Array]:
    r"""The TOD less its fit by every configured template, $d - T \hat a$.

    The amplitudes are fitted by least squares under a white weight: each channel's inverse
    variance, from its sample-to-sample differences, on its valid samples. The fit only shapes the
    data the noise is estimated from, so a loose tolerance serves.
    """
    assert config.templates is not None
    explicit, implicit = build_templates(
        config.templates, data, tod_format, temperature=temperature
    )
    operator = TemplateOperator({**explicit, **implicit})
    tod = data[ReaderField.SAMPLE_DATA]
    samples = tod_format.as_array(tod)  # (stream, det, sample)
    valid = _sample_mask(data, config) & (jnp.arange(tod_format.n_samples) < n_real)
    pairs = valid[..., 1:] & valid[..., :-1]
    steps = jnp.where(pairs, jnp.diff(samples, axis=-1), 0)
    variance = jnp.sum(steps**2, axis=-1) / jnp.maximum(2 * pairs.sum(axis=-1), 1)
    inverse = jnp.where(variance > 0, 1 / jnp.where(variance > 0, variance, 1), 0)
    channel_weight = tod_format.diagonal(inverse[..., None].astype(samples.dtype))
    sample_weight = tod_format.diagonal(valid[None].astype(samples.dtype))
    weight = channel_weight @ sample_weight
    fit = TemplateFit(
        operator,
        weight,
        regularization=config.templates.regularization,
        batch_size=config.templates.gram_batch_size,
        solver=CGSolver(rtol=1e-4, max_steps=_DEPROJECTION_STEPS),
    )
    return tree.sub(tod, fit.model(fit(tod).solution))


_DEPROJECTION_STEPS = 50
"""The most conjugate gradient steps of the template fit that precedes the low-rank noise fit."""
