r"""The normal system of the multi-observation mapmaker, evaluated one detector tile at a time.

The mapmaker solves $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$ over the detector
tiles $t$ of every observation, with $J_t = [H_t S^\top, T_{e,t}, T_{g,t}]$ mapping the selected
sky pixels, the tile's own template amplitudes and those its observation's detectors share to the
tile's TOD, and $W'_t$ its weight (see
[`ObservationModel.deprojected_weight`][furax.mapmaking._model.ObservationModel.deprojected_weight]).

With explicit gaps, the flagged samples are modelled by the spline amplitudes of the tile's gaps,
$C_t$: $J_t = [M_t H_t S^\top, M_t T_{e,t}, M_t T_{g,t}, C_t]$, with $M_t$ the mask of the good
samples, under the inverse noise $N_t^{-1}$ unmasked. The map block is that of the masked weight,
$H^\top M N^{-1} M H$; marginalizing the gap amplitudes would give the nested weight on the span of
the splines.

Under a low-rank noise model, $N = D + V G V^\dagger$, the weight of each tile is that of the
diagonal $D$, which couples no detectors, and the correlated modes are marginalized across the
tiles of each observation, $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$ (see
[`LowRankNoiseModel.modes_core`][furax.mapmaking.lowrank.LowRankNoiseModel.modes_core]): a first
sweep over the tiles sums $T^\top D^{-1}$ of their TODs over each observation, a second one
subtracts the correction. The tiles of an observation are on one device, so the sums do not
communicate.

A bucket keeps what its tiles hold as a `BucketModel`: a stacked ensemble of its tiles, split
over the devices, grouped in segments by observation. Each leaf is stacked per tile, per segment
when the tiles of an observation share it (the boresight pointing, the bases and Gram inverse of
the templates with shared amplitudes, ...), or held once for the bucket. The operators of this
module evaluate the sums over it with the ensemble engine.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax
from jax.tree_util import register_dataclass
from jaxtyping import Array, PyTree

import furax.tree
from furax import AbstractLinearOperator, IdentityOperator, symmetric
from furax.core import BlockRowOperator, Policy, StackedEnsemble, map_reduce
from furax.obs.stokes import StokesType

from ._model import ObservationModel, ObservationTemplates
from .lowrank import LowRankNoiseModel, ModesCore
from .weight import WeightOperator


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """What the normal system needs of one detector tile of an observation."""

    model: ObservationModel
    """The tile's acquisition operator, weight and noise model."""
    templates: ObservationTemplates | None
    """The tile's templates, with amplitudes per detector or shared by the observation."""
    amplitude_rhs: PyTree[Array] | None
    """The right-hand side of the tile's own explicit amplitudes (None without them)."""
    gap_rhs: Array | None = None
    """The right-hand side of the amplitudes of the tile's explicit gaps (None without them)."""
    gap_preconditioner: Array | None = None
    """The inverse of the diagonal of the white-noise Gram of those amplitudes."""


@register_dataclass
@dataclass(frozen=True)
class BucketModel:
    """The detector tiles of every observation of a bucket."""

    tiles: StackedEnsemble
    """The [`Tile`][]s, one member per tile, grouped in segments by observation and laid out
    over the devices; the right-hand sides of their own amplitudes are stacked per tile."""
    common_rhs: PyTree[Array] | None
    """Right-hand side of each observation's shared amplitudes, `(segment, ...)` (None without)."""
    modes_data: Array | None = None
    """$T^\top D^{-1}$ of the data of each observation, `(segment, frequency, mode, 2)`, under a
    low-rank noise model (None otherwise)."""
    modes_core: ModesCore | None = None
    """$K$ of each observation, its leaves `(segment, ...)`."""

    @property
    def templates(self) -> ObservationTemplates | None:
        """The kinds of templates of the bucket, from the structure of its tiles."""
        return self.tiles.template.templates


def bucket_apply(
    model: BucketModel,
    apply: Callable[
        [Tile, PyTree[Array], PyTree[Array] | None, PyTree[Array] | None], tuple[Any, ...]
    ],
    whole: PyTree[Array],
    own: PyTree[Array] | None,
    common: PyTree[Array] | None,
    policy: Policy | None = None,
    extra: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    """`apply` on every detector tile of a bucket.

    Args:
        model: The bucket.
        apply: From a tile, `whole`, its own amplitudes and those of its observation, to `(map,
            own, common)`: a map, the tile's own amplitudes and its observation's, any of them
            `None`.
        whole: What every tile takes whole, such as a map, held by every device. Its leaves must
            be arrays; anything else (an operator, whose index may hold an `Ellipsis`) is closed
            over `apply` instead.
        own: The amplitudes of each tile alone, `(tile, ...)`, if any.
        common: The amplitudes of each observation, `(segment, ...)`, if any.
        policy: How the tiles are walked; by default that of `model.tiles`. A round of `batch`
            tiles holds the temporaries of that many tiles.
        extra: How further outputs of `apply`, after the first three, combine (see
            [`map_reduce`][furax.core.map_reduce]).

    Returns:
        The maps summed over the bucket, the tiles' own amplitudes `(tile, ...)`, and those of
        each observation summed over its tiles, `(segment, ...)`; then the `extra` outputs.
    """
    tiles = model.tiles
    strides = (
        *tiles.strides,
        *(1 for _ in jax.tree.leaves(own)),
        *('segment' for _ in jax.tree.leaves(common)),
    )
    ensemble = StackedEnsemble(
        tiles.size,
        (tiles.stacked, own, common),
        strides,
        owner=tiles.owner,
        segments=tiles.segments,
        policy=policy if policy is not None else tiles.policy,
    )

    def tile_fn(member: tuple[Any, ...], whole: PyTree[Array]) -> tuple[Any, ...]:
        tile, own_t, common_o = member
        return apply(tile, whole, own_t, common_o)

    out, _ = map_reduce(tile_fn, ensemble, ('sum', 1, 'segment_sum', *extra), whole)
    return out


def joint_operator(
    tile: Tile,
    selection: AbstractLinearOperator | None,
    own: bool,
    common: bool,
    gaps: bool = False,
) -> AbstractLinearOperator:
    r"""$J_t = [H_t S^\top, T_{e,t}, T_{g,t}, C_t]$ of a tile, with the amplitudes asked for.

    With the gaps, the other blocks are masked: the gap amplitudes stand for the flagged samples.

    Args:
        tile: The tile.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        own: Whether to include the tile's own explicit amplitudes.
        common: Whether to include those its observation's detectors have in common.
        gaps: Whether to include the amplitudes of its explicit gaps.
    """
    H = tile.model.H if selection is None else tile.model.H @ selection.T
    blocks = [H]
    if own:
        assert tile.templates is not None and tile.templates.explicit is not None
        blocks.append(tile.templates.explicit.operator)
    if common:
        assert tile.templates is not None and tile.templates.common is not None
        blocks.append(tile.templates.common.operator)
    if gaps:
        blocks = [tile.model.M @ block for block in blocks]
        blocks.append(tile.model.gap_basis())
    return BlockRowOperator(blocks)


def system_weight(tile: Tile, gaps: bool = False) -> AbstractLinearOperator:
    """$W_t$ of a tile, its implicit templates marginalized; $N_t^{-1}$ with its explicit gaps."""
    implicit = None if tile.templates is None else tile.templates.implicit
    if gaps:
        assert implicit is None and isinstance(tile.model.W, WeightOperator)
        return tile.model.W.weight
    return tile.model.deprojected_weight(implicit)


def _modes_operator(tile: Tile) -> AbstractLinearOperator:
    """The correlated modes $T_t$ of a tile."""
    model = tile.model
    assert isinstance(model.noise_model, LowRankNoiseModel)
    return model.noise_model.modes_operator(model.tod_format.structure)


def segment_ensemble(ensemble: StackedEnsemble) -> StackedEnsemble:
    """An ensemble of tiles holding only segment and shared leaves, as one of its segments.

    Raises:
        ValueError: If a leaf varies with the tile.
    """
    if any(isinstance(s, int) for s in ensemble.strides):
        raise ValueError('a leaf varies with the tile, not with its segment')
    assert ensemble.segments is not None
    strides = tuple(None if s is None else 1 for s in ensemble.strides)
    return StackedEnsemble(ensemble.segments, ensemble.stacked, strides, policy=ensemble.policy)


@symmetric
class BucketNormalOperator(AbstractLinearOperator):
    r"""$\sum_t J_t^\top W'_t J_t$ over the detector tiles of a bucket.

    The unknowns are a list: the map, then the tiles' own amplitudes, their observations', and
    those of the tiles' explicit gaps, if the bucket has them and `amplitudes` is set. Otherwise
    they are the map alone. Without `selection`, the map is the whole sky.
    """

    model: BucketModel
    selection: AbstractLinearOperator | None
    """The selection of the estimated pixels, if any."""
    weight: Literal['system', 'plain', 'white'] = field(default='system', metadata={'static': True})
    """$W'_t$: the weight of the system, or $W F$ without the implicit templates, or the same with
    the white part of the noise only."""
    amplitudes: bool = field(default=True, metadata={'static': True})
    """Whether the unknowns include the amplitudes of the explicit templates."""
    policy: Policy | None = field(default=None, metadata={'static': True})
    """How the tiles are walked (see `bucket_apply`); by default that of `model.tiles`."""

    def mv(self, x: Any) -> Any:
        m, own, common = self._split(x)
        present = self._amplitudes_present

        def tod(tile, m, own_t, common_o):
            explicit, gaps = own_t
            inputs = [v for v in (explicit, common_o, gaps) if v is not None]
            return joint_operator(tile, self.selection, *present)([m, *inputs])

        return self._join(*self._weigh(tod, m, own, common))

    def response(self, sky: StokesType) -> Any:
        r"""$\sum_t J_t^\top W'_t H_t \, s$: the right-hand side of the TOD of a sky `sky`."""
        gaps = self._amplitudes_present[2]

        def tod(tile, sky, _own, _common):
            tod = tile.model.H(sky)
            return tile.model.M(tod) if gaps else tod  # the weight is unmasked

        return self._join(*self._weigh(tod, sky, None, None))

    def correction(self, data: Array) -> Any:
        r"""What the correlated modes take from a right-hand side.

        That is $\sum_t J_t^\top P^\top D_t^{-1} T_t K u$. `data` $u$ is the $T^\top D^{-1}$ of the TOD, summed over the tiles of each observation.
        """
        return self._join(*self._correct(data))

    def _weigh(
        self,
        tod: Callable[[Tile, Any, Any, Any], PyTree[Array]],
        whole: Any,
        own: Any,
        common: Any,
    ) -> tuple[Any, Any, Any]:
        r"""$\sum_t J_t^\top W'_t$ of the TOD each tile makes of `whole`, `own` and `common`.

        Under a low-rank noise model, the correlated modes are marginalized: a first sweep weights
        each TOD by the diagonal and sums its $T^\top D^{-1}$ over each observation, a second one
        subtracts the correction.
        """
        selection = self.selection
        if not self._modes:

            def apply(tile, whole, own_t, common_o):
                weighted = self._weight(tile)(tod(tile, whole, own_t, common_o))
                return self._project(tile, weighted, selection)

            return bucket_apply(self.model, apply, whole, own, common, self.policy)

        def apply_modes(tile, whole, own_t, common_o):
            # D⁻¹ P of the TOD, P the mask unless the gaps are explicit (the weight is P D⁻¹ P)
            inverse_diagonal, mask = self._diagonal(tile)
            weighted = inverse_diagonal(mask(tod(tile, whole, own_t, common_o)))
            data = _modes_operator(tile).T(weighted)
            return (*self._project(tile, mask(weighted), selection), data)

        m, own_out, common_out, data = bucket_apply(
            self.model, apply_modes, whole, own, common, self.policy, extra=('segment_sum',)
        )
        m2, own2, common2 = self._correct(data)
        return (
            furax.tree.sub(m, m2),
            furax.tree.sub(own_out, own2),
            furax.tree.sub(common_out, common2),
        )

    def _correct(self, data: Array) -> tuple[Any, Any, Any]:
        """The correction of the correlated modes, from the sums `data` of each observation."""
        assert self.model.modes_core is not None
        template = self.model.tiles.template.model
        noise = template.noise_model
        assert isinstance(noise, LowRankNoiseModel)
        n_samples = template.tod_format.n_samples
        z = jax.vmap(lambda core, u: core.apply(u, noise.edges, noise.sample_rate, n_samples))(
            self.model.modes_core, data
        )
        selection = self.selection

        def apply(tile, _whole, _own, z_o):
            inverse_diagonal, mask = self._diagonal(tile)
            modes = inverse_diagonal(_modes_operator(tile)(z_o))
            return self._project(tile, mask(modes), selection)

        return bucket_apply(self.model, apply, None, None, z, self.policy)

    def _diagonal(self, tile: Tile) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
        """$D^{-1}$ of a tile, unmasked, and the mask $P$ of its weight $P D^{-1} P$."""
        W = tile.model.W
        assert isinstance(W, WeightOperator)
        if self._amplitudes_present[2]:  # explicit gaps: the weight is unmasked
            return W.weight, IdentityOperator(in_structure=W.in_structure)
        return W.weight, tile.model.M

    @property
    def _modes(self) -> bool:
        """Whether the weight marginalizes the correlated modes of a low-rank noise model."""
        return self.weight == 'system' and self.model.modes_core is not None

    def _project(
        self, tile: Tile, u: PyTree[Array], selection: AbstractLinearOperator | None
    ) -> tuple[Any, Any, Any]:
        r"""$J_t^\top u$, in parts."""
        present = self._amplitudes_present
        return _unpack(joint_operator(tile, selection, *present).T(u), present)

    def _weight(self, tile: Tile) -> AbstractLinearOperator:
        gaps = self._amplitudes_present[2]
        if self.weight == 'system':
            return system_weight(tile, gaps)
        model = tile.model
        W = model.diag_W() if self.weight == 'white' else model.W
        if gaps:
            assert isinstance(W, WeightOperator)
            return W.weight
        return (W @ model.F).reduce()

    @property
    def _amplitudes_present(self) -> tuple[bool, bool, bool]:
        """Whether the unknowns have the tiles' own amplitudes, their observations', their gaps'."""
        if not self.amplitudes:
            return False, False, False
        templates = self.model.templates
        gaps = self.model.tiles.template.gap_rhs is not None
        if templates is None:
            return False, False, gaps
        return templates.explicit is not None, templates.common is not None, gaps

    def _split(self, x: Any) -> tuple[Any, Any, Any]:
        """The map, the amplitudes of each tile, `(explicit, gaps)`, and of each observation.

        Those absent are `None`.
        """
        present = self._amplitudes_present
        if not any(present):
            return x, (None, None), None
        return _unpack(x, present)

    def _join(self, m: Any, own: Any, common: Any) -> Any:
        present = self._amplitudes_present
        if not any(present):
            return m
        explicit, gaps = own
        parts = (explicit, common, gaps)
        return [m, *(part for part, p in zip(parts, present, strict=True) if p)]


@symmetric
class MemberOperator(AbstractLinearOperator):
    """The operator of each member of an ensemble, on its own amplitudes: `x[i] -> operators[i](x[i])`.

    The members are the tiles of a bucket, such as the inverses of the Grams of their own explicit
    amplitudes, which precondition them, or its segments, for the amplitudes the detectors of an
    observation share (see [`segment_ensemble`][]).
    """

    operators: StackedEnsemble
    """The operators, one member per entry of the amplitudes."""

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        # small computations: every member of a device is evaluated at once, vectorized
        ensemble = self.operators.zip(x)
        policy = replace(ensemble.policy, batch=ensemble.size_per_device)

        def apply(member: tuple[Any, ...]) -> PyTree[Array]:
            operator, v = member
            return operator(v)

        out, _ = map_reduce(apply, replace(ensemble, policy=policy), 1)
        return out


def _unpack(
    parts: list[PyTree[Array]], present: tuple[bool, bool, bool]
) -> tuple[Any, tuple[Any, Any], Any]:
    """The map, the amplitudes of each tile, `(explicit, gaps)`, and of each observation.

    `parts` lists the map, then the explicit, common and gap amplitudes `present`; those absent
    are `None`.
    """
    m, *rest = parts
    explicit, common, gaps = (rest.pop(0) if p else None for p in present)
    return m, (explicit, gaps), common
