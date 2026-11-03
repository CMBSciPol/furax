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
from furax.core import AdditionOperator, BlockDiagonalOperator, BlockSelectOperator
from furax.obs.stokes import StokesType

from ._bucket import Batch, BucketModel, BucketNormalOperator, MemberOperator
from ._model import FACTORED_KINDS, TemplateBundle
from ._solve import MapMakingSystem
from .layout import Bucket, SlotLayout, SlotPlacement
from .noise import NoiseModel
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

    def block_jacobi(
        self, placement: SlotPlacement, binned: bool, in_structure: PyTree
    ) -> BJPreconditioner:
        """The block-Jacobi preconditioner, from the diagonal pixel system summed over the buckets.

        The system is under the weight without the noise correlations, nor the implicit templates.
        """
        weight = 'plain' if binned else 'white'
        A_diag = AdditionOperator(
            [
                BucketNormalOperator(
                    bm,
                    None,
                    placement.axis,
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
        layout: SlotLayout,
        placement: SlotPlacement,
        sky: StokesType | None = None,
    ) -> MapMakingSystem:
        r"""Assembles the GLS normal system $J^T W' J x = J^T W' d$ over the selected pixels.

        Args:
            S: Selection of the estimated pixels out of the full sky grid.
            BJ: Block-Jacobi preconditioner.
            layout: The layout of the observations in buckets.
            placement: The placement of the data on the devices.
            sky: A simulated sky on the full grid, whose right-hand side `injected_b` is added.

        Returns:
            The system to solve. Its unknowns are the selected sky pixels, then the amplitudes
            of each kind of explicit template present, if any, one block per bucket.
        """
        axis = placement.axis
        # Restrict BJ preconditioner to selected pixels
        M = (S @ BJ.I @ S.T).reduce()

        # Explicit templates are configured for the run, so every bucket carries them or none.
        templates = self.buckets[0].templates
        kinds = []
        if templates is not None:
            kinds = [kind for kind in FACTORED_KINDS if getattr(templates, kind) is not None]

        if not kinds:
            terms = [
                BucketNormalOperator(bm, S, axis, in_structure=S.out_structure)
                for bm in self.buckets
            ]
            A = AdditionOperator(terms, sequential=True)
            response = None
            if sky is not None:
                response = reduce(furax.tree.add, [op.response(sky) for op in terms])
            return MapMakingSystem(A, S(self.map_rhs), M, has_amplitudes=False, injected_b=response)

        # The unknowns are the selected sky pixels, then, per kind of amplitudes present, one block
        # per bucket: those of each detector batch alone, `(slot, batch, ...)`, and those the
        # detectors of an observation share, `(slot, ...)`.
        structures = {
            kind: [
                _amplitude_structure(bucket, bm, kind)
                for bucket, bm in zip(layout.buckets, self.buckets, strict=True)
            ]
            for kind in kinds
        }
        at = {kind: 1 + k for k, kind in enumerate(kinds)}  # their block among the unknowns

        # Joint system. Each bucket's system sees the selected pixels and its own amplitudes,
        # picked out of the lists and embedded back by `E`.
        terms, responses = [], []
        for b, bm in enumerate(self.buckets):
            in_structure = [S.out_structure, *(structures[kind][b] for kind in kinds)]
            normal = BucketNormalOperator(bm, S, axis, in_structure=in_structure)
            picks = [BlockSelectOperator(b, in_structure=structures[kind]) for kind in kinds]
            E = BlockDiagonalOperator([IdentityOperator(in_structure=S.out_structure), *picks])
            terms.append((E.T @ normal @ E).reduce())
            if sky is not None:
                responses.append(E.T(normal.response(sky)))

        # The Gram inverses of the amplitudes of each detector batch alone are per batch, those
        # of the common amplitudes per observation.
        inverses: dict[str, list[AbstractLinearOperator]] = {kind: [] for kind in kinds}
        ridges: dict[str, list[AbstractLinearOperator | None]] = {kind: [] for kind in kinds}
        for b, bm in enumerate(self.buckets):
            for kind in kinds:
                inverse, ridge = _gram_inverse(bm, kind, structures[kind][b])
                inverses[kind].append(inverse)
                ridges[kind].append(ridge)

        # The ridge the Gram inverses were factored with acts on the amplitudes alone, so it is
        # one term over them all, outside the passes over the TOD.
        for kind in kinds:
            ridged = ridges[kind]
            assert all(r is None for r in ridged) or all(r is not None for r in ridged)
            if ridged[0] is not None:
                amplitudes = BlockSelectOperator(at[kind], in_structure=terms[0].in_structure)
                terms.append((amplitudes.T @ BlockDiagonalOperator(ridged) @ amplitudes).reduce())

        rhs = [
            S(self.map_rhs),
            *(
                # declared stacked over the batches, the right-hand sides of each batch's own
                # amplitudes are wholly per batch
                [
                    bm.batches.amplitude_rhs if FACTORED_KINDS[kind] == 'stack' else bm.common_rhs
                    for bm in self.buckets
                ]
                for kind in kinds
            ),
        ]
        M = BlockDiagonalOperator([M, *(inverses[kind] for kind in kinds)])
        A = AdditionOperator(terms, sequential=True)
        response = None
        if sky is not None:
            response = reduce(furax.tree.add, responses)
        return MapMakingSystem(A, rhs, M, has_amplitudes=True, injected_b=response)

    def template_amplitudes(
        self,
        blocks: list[list[PyTree[Array]]],
        layout: SlotLayout,
        placement: SlotPlacement,
    ) -> dict[str, np.ndarray]:
        """The explicit-template amplitudes solved for, per observation, from their CG blocks.

        Every device holds the same replicated copy after the gather, so the host array is
        complete on every process; then back to observation order.
        """
        # Explicit templates are configured for the run, so every bucket carries them or none.
        templates = self.buckets[0].templates
        explicit = None if templates is None else templates.explicit
        amplitudes = {}
        if explicit is not None:
            own, *blocks = blocks
            per_bucket = [placement.gather(a) for a in own]
            for name in per_bucket[0]:
                # the detector axis follows the stream axis, if any (see `Template`), after the
                # slot and batch axes
                axis = 2 + explicit.operator.templates[name].per.index('det')
                amplitudes[name] = layout.to_observation_order(
                    [a[name] for a in per_bucket], detector_axis=axis
                )
        for common in blocks:  # one set per slot, no detector axis
            amplitudes |= jax.tree.map(
                lambda *leaves: layout.to_observation_order(leaves),
                *[placement.gather(a) for a in common],
            )
        return amplitudes

    def noise_parameters(
        self, layout: SlotLayout, placement: SlotPlacement
    ) -> dict[str, np.ndarray]:
        """The noise model parameters of every detector slot, in observation order.

        Each parameter is `(n_observations, [stream,] detector)`, its detector axis padded.
        """
        per_bucket = []
        for bm in self.buckets:
            # every parameter is a detector's, so held per batch: `(slot, batch, ..., detector)`
            def per_detector(models):
                return jax.tree.map(
                    lambda m: m.detector_parameters(),
                    models,
                    is_leaf=lambda m: isinstance(m, NoiseModel),
                )

            parameters = jax.vmap(jax.vmap(per_detector))(bm.batches.model.noise_model)
            leaves, _ = jax.tree_util.tree_flatten_with_path(placement.gather(parameters))
            per_bucket.append({_parameter_name(path): leaf for path, leaf in leaves})
        return {
            name: layout.to_observation_order([p[name] for p in per_bucket], detector_axis=-1)
            for name in per_bucket[0]
        }


def _amplitude_structure(
    bucket: Bucket, bm: BucketModel, kind: str
) -> dict[str, jax.ShapeDtypeStruct]:
    """The amplitudes of a kind of a bucket.

    Those of each batch alone are `(slot, batch, ...)`, those of each observation `(slot, ...)`.
    """
    assert bm.templates is not None
    bundle = getattr(bm.templates, kind)
    assert bundle is not None
    leading = (bucket.n_slots,)
    if FACTORED_KINDS[kind] == 'stack':
        leading += (bucket.n_batches,)
    return {
        name: jax.ShapeDtypeStruct((*leading, *s.shape), s.dtype)
        for name, s in bundle.operator.in_structure.items()
    }


def _gram_inverse(
    bm: BucketModel, kind: str, structure: dict[str, jax.ShapeDtypeStruct]
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """The inverse of the Gram of a kind of amplitudes of a bucket, and the ridge added."""

    def bundle(batch: Batch) -> TemplateBundle:
        assert batch.templates is not None
        bundle = getattr(batch.templates, kind)
        assert bundle is not None
        return bundle

    # those of each batch alone hold per batch what depends on the batch, the others whole
    own, kept = bundle(bm.batches), bundle(bm.shared)

    def member(part: str) -> AbstractLinearOperator | None:
        operators, shared = getattr(own, part), getattr(kept, part)
        if operators is None and shared is None:
            return None
        if FACTORED_KINDS[kind] == 'stack':
            return MemberOperator(operators, shared, batches=True, in_structure=structure)
        return MemberOperator(shared, in_structure=structure)

    inverse = member('gram_inverse')
    assert inverse is not None
    return inverse, member('ridge')


def _parameter_name(path: tuple[Any, ...]) -> str:
    """The name of a leaf of a noise model, from its path: `sigma`, or `<key>/sigma` if nested."""
    return '/'.join(str(getattr(k, 'name', getattr(k, 'key', k))) for k in path)
