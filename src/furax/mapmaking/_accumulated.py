"""What the mapmaker accumulates over the observations, and what it makes of it.

`AccumulatedModel` holds the per-bucket models and the map-domain sums of the pass over the data.
Its methods assemble from it the normal system and its block-Jacobi preconditioner, and unpack the
noise parameters and template amplitudes into observation order.
"""

from dataclasses import dataclass
from functools import reduce
from typing import Any

import jax
import numpy as np
from jax.tree_util import register_dataclass
from jaxtyping import Array, Int64, PyTree

import furax.tree
from furax import AbstractLinearOperator, IdentityOperator
from furax.core import (
    AdditionOperator,
    BlockDiagonalOperator,
    BlockSelectOperator,
    DiagonalOperator,
    StackedEnsemble,
    lift,
)
from furax.obs.stokes import StokesType

from ._bucket import GAPS, BucketModel, BucketNormalOperator, MemberOperator, Tile, segment_ensemble
from ._model import DROP_REASONS, FACTORED_KINDS, TemplateBundle
from ._solve import MapMakingSystem
from .layout import TilePlacement
from .preconditioner import BJPreconditioner


@register_dataclass
@dataclass(frozen=True)
class AccumulatedModel:
    """Per-bucket observation models and the map-domain sums."""

    buckets: tuple[BucketModel, ...]
    """One `BucketModel` per bucket of the layout, in bucket order."""
    hit_map: Int64[Array, '...']
    """Hit map over every observation, replicated on every device."""
    map_rhs: StokesType
    """Map RHS over every observation, replicated on every device."""

    def block_jacobi(self, binned: bool, in_structure: PyTree) -> BJPreconditioner:
        """The block-Jacobi preconditioner, from the diagonal pixel system summed over the buckets.

        The system is under the weight without the noise correlations, nor the implicit templates.
        """
        weight = 'plain' if binned else 'white'
        A_diag = AdditionOperator(
            [
                BucketNormalOperator(
                    bm,
                    None,
                    weight,
                    amplitudes=False,
                    in_structure=in_structure,
                )
                for bm in self.buckets
            ],
            sequential=True,
        )
        return BJPreconditioner.create(A_diag)

    def build_system(
        self,
        S: AbstractLinearOperator,
        BJ: BJPreconditioner,
        sky: StokesType | None = None,
    ) -> MapMakingSystem:
        r"""Assembles the GLS normal system $J^T W' J x = J^T W' d$ over the selected pixels.

        Args:
            S: Selection of the estimated pixels out of the full sky grid.
            BJ: Block-Jacobi preconditioner.
            sky: A simulated sky on the full grid, whose right-hand side `injected_b` is added.

        Returns:
            The system to solve. Its unknowns are the selected sky pixels; with amplitudes, a
            dictionary of them under `'map'` and of the amplitudes of each kind, one block per
            bucket.
        """
        # Restrict BJ preconditioner to selected pixels
        M = (S @ BJ.I @ S.T).reduce()

        # Explicit templates and gaps are configured for the run, so every bucket has them or none.
        kinds = self.buckets[0].amplitude_kinds

        if not kinds:
            terms = [
                BucketNormalOperator(bm, S, in_structure=S.out_structure) for bm in self.buckets
            ]
            A = AdditionOperator(terms, sequential=True)
            response = None
            if sky is not None:
                response = reduce(furax.tree.add, [op.response(sky) for op in terms])
            return MapMakingSystem(A, S(self.map_rhs), M, has_amplitudes=False, injected_b=response)

        # The unknowns are the selected sky pixels, then, per kind of amplitudes present, one block
        # per bucket: those of each detector tile alone, `(tile, ...)`, and those the detectors of
        # an observation share, `(segment, ...)`.
        structures = {
            kind: [_amplitude_structure(bm, kind) for bm in self.buckets] for kind in kinds
        }

        # Joint system. Each bucket's system sees the selected pixels and its own amplitudes,
        # picked out of the lists and embedded back by `E`.
        terms, responses = [], []
        for b, bm in enumerate(self.buckets):
            in_structure = {'map': S.out_structure} | {kind: structures[kind][b] for kind in kinds}
            normal = BucketNormalOperator(bm, S, in_structure=in_structure)
            picks = {kind: BlockSelectOperator(b, in_structure=structures[kind]) for kind in kinds}
            E = BlockDiagonalOperator(
                {'map': IdentityOperator(in_structure=S.out_structure)} | picks
            )
            terms.append((E.T @ normal @ E).reduce())
            if sky is not None:
                responses.append(E.T(normal.response(sky)))

        # The Gram inverses of the amplitudes of each detector tile alone are per tile, those of
        # the common amplitudes per observation.
        inverses: dict[str, list[AbstractLinearOperator]] = {kind: [] for kind in kinds}
        ridges: dict[str, list[AbstractLinearOperator | None]] = {kind: [] for kind in kinds}
        for b, bm in enumerate(self.buckets):
            for kind in kinds:
                if kind == GAPS:
                    # the inverse of the diagonal of the white-noise Gram of the gap amplitudes
                    jacobi = bm.tiles.stacked.gap_preconditioner
                    inverses[kind].append(
                        DiagonalOperator(jacobi, in_structure=structures[kind][b])
                    )
                    ridges[kind].append(None)
                    continue
                inverse, ridge = _gram_inverse(bm, kind, structures[kind][b])
                inverses[kind].append(inverse)
                ridges[kind].append(ridge)

        # The ridge the Gram inverses were factored with acts on the amplitudes alone, so it is
        # one term over them all, outside the passes over the TOD.
        for kind in kinds:
            ridged = ridges[kind]
            assert all(r is None for r in ridged) or all(r is not None for r in ridged)
            if ridged[0] is not None:
                amplitudes = BlockSelectOperator(kind, in_structure=terms[0].in_structure)
                terms.append((amplitudes.T @ BlockDiagonalOperator(ridged) @ amplitudes).reduce())

        rhs = {'map': S(self.map_rhs)} | {
            # declared stacked over the tiles, the right-hand sides of each tile's own amplitudes
            # are wholly per tile
            kind: [
                bm.tiles.stacked.gap_rhs
                if kind == GAPS
                else bm.tiles.stacked.amplitude_rhs
                if FACTORED_KINDS[kind] == 'stack'
                else bm.common_rhs
                for bm in self.buckets
            ]
            for kind in kinds
        }
        M = BlockDiagonalOperator({'map': M} | inverses)
        A = AdditionOperator(terms, sequential=True)
        response = None
        if sky is not None:
            response = reduce(furax.tree.add, responses)
        return MapMakingSystem(A, rhs, M, has_amplitudes=True, injected_b=response)

    def template_amplitudes(
        self, x: dict[str, PyTree[Array]], placement: TilePlacement
    ) -> dict[str, np.ndarray]:
        """The explicit-template amplitudes solved for, per observation, from the unknowns `x`.

        Every device holds the same replicated copy after the gather, so the host array is
        complete on every process; then back to observation order.
        """
        amplitudes = {}
        if (own := x.get('explicit')) is not None:
            templates = self.buckets[0].templates
            assert templates is not None and templates.explicit is not None
            explicit = templates.explicit
            per_bucket = [placement.gather(a) for a in own]
            for name in per_bucket[0]:
                # the detector axis follows the stream axis, if any (see `Template`), after the
                # tile axis
                axis = 1 + explicit.operator.templates[name].per.index('det')
                amplitudes[name] = placement.layout.to_observation_order(
                    [a[name] for a in per_bucket], detector_axis=axis
                )
        if (common := x.get('common')) is not None:  # one set per segment, no detector axis
            amplitudes |= jax.tree.map(
                lambda *leaves: placement.layout.to_observation_order(leaves),
                *[placement.gather(a) for a in common],
            )
        return amplitudes

    def detector_drops(self, placement: TilePlacement) -> dict[str, np.ndarray]:
        """Per detector, whether it was dropped, by reason (see [`DROP_REASONS`][]).

        Each array is `(n_observations, detector)`, in observation order, its detector axis
        padded with `False`.
        """
        drops = {}
        for reason in DROP_REASONS:
            per_bucket = []
            for bm in self.buckets:
                flags = lift(lambda tile, reason=reason: getattr(tile.model, reason), bm.tiles)
                assert isinstance(flags, StackedEnsemble)
                values = placement.gather(flags.stacked)
                if flags.strides[0] is None:  # the same in every tile, as when none is dropped
                    values = np.broadcast_to(values, (bm.tiles.size, *values.shape))
                per_bucket.append(values)
            drops[reason] = placement.layout.to_observation_order(per_bucket, detector_axis=-1)
        return drops

    def noise_parameters(self, placement: TilePlacement) -> dict[str, np.ndarray]:
        """The noise model parameters of every detector, in observation order.

        Each parameter is `(n_observations, [stream,] detector)`, its detector axis padded.
        """
        per_bucket = []
        for bm in self.buckets:
            # the parameters of the detectors are held per tile: `(tile, ..., detector)`
            noise = lift(lambda tile: tile.model.noise_model, bm.tiles)
            assert isinstance(noise, StackedEnsemble)
            leaves, _ = jax.tree_util.tree_flatten_with_path(placement.gather(noise.stacked))
            per_bucket.append(
                {
                    _parameter_name(path): leaf
                    for (path, leaf), stride in zip(leaves, noise.strides, strict=True)
                    if stride == 1
                }
            )
        return {
            name: placement.layout.to_observation_order(
                [p[name] for p in per_bucket], detector_axis=-1
            )
            for name in per_bucket[0]
        }


def _amplitude_structure(bm: BucketModel, kind: str) -> PyTree[jax.ShapeDtypeStruct]:
    """The amplitudes of a kind of a bucket.

    Those of each tile alone are `(tile, ...)`, those of each observation `(segment, ...)`.
    """
    if kind == GAPS:
        return furax.tree.as_structure(bm.tiles.stacked.gap_rhs)
    assert bm.templates is not None
    bundle = getattr(bm.templates, kind)
    assert bundle is not None
    tiles = bm.tiles
    assert tiles.segments is not None
    leading = tiles.size if FACTORED_KINDS[kind] == 'stack' else tiles.segments
    return {
        name: jax.ShapeDtypeStruct((leading, *s.shape), s.dtype)
        for name, s in bundle.operator.in_structure.items()
    }


def _gram_inverse(
    bm: BucketModel, kind: str, structure: dict[str, jax.ShapeDtypeStruct]
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """The inverse of the Gram of a kind of amplitudes of a bucket, and the ridge added."""

    def bundle(tile: Tile) -> TemplateBundle:
        assert tile.templates is not None
        bundle = getattr(tile.templates, kind)
        assert bundle is not None
        return bundle

    def member(part: str) -> AbstractLinearOperator | None:
        if getattr(bundle(bm.tiles.template), part) is None:
            return None
        # structural: the operators keep the leaves of the tiles, and their strides
        operators = lift(lambda tile: getattr(bundle(tile), part), bm.tiles)
        assert isinstance(operators, StackedEnsemble)
        if FACTORED_KINDS[kind] != 'stack':
            operators = segment_ensemble(operators)
        return MemberOperator(operators, in_structure=structure)

    inverse = member('gram_inverse')
    assert inverse is not None
    return inverse, member('ridge')


def _parameter_name(path: tuple[Any, ...]) -> str:
    """The name of a leaf of a noise model, from its path: `sigma`, or `<key>/sigma` if nested."""
    return '/'.join(str(getattr(k, 'name', getattr(k, 'key', k))) for k in path)
