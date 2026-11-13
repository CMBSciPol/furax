"""What the mapmaker accumulates over the observations, and the normal system it makes of it.

`AccumulatedModel` holds the tiles of each bucket, the hit map, and the right-hand side of the normal
system from the pass over the data (see `_pass`). Its methods assemble from them the normal system
and its block-Jacobi preconditioner.
"""

from dataclasses import dataclass
from functools import reduce
from typing import Any

from jax.tree_util import register_dataclass
from jaxtyping import Array, Int64, PyTree

import furax.tree
from furax import AbstractLinearOperator, IdentityOperator
from furax.core import AdditionOperator, BlockDiagonalOperator, BlockSelectOperator, StackedEnsemble
from furax.obs.stokes import StokesType

from ._bucket import BucketNormalOperator, amplitude_kinds, preconditioner, ridge
from ._solve import MapMakingSystem
from .preconditioner import BJPreconditioner


@register_dataclass
@dataclass(frozen=True)
class AccumulatedModel:
    """The result of the read pass: its buckets, the hit map and the right-hand side."""

    buckets: list[StackedEnsemble]
    """The [`Tile`][furax.mapmaking._bucket.Tile]s of each bucket of the layout, in bucket order."""
    hit_map: Int64[Array, '...']
    """Hit map over every observation, replicated on every device."""
    rhs: dict[str, Any]
    """The right-hand side of the normal system, keyed as its unknowns: the map RHS over the full
    sky under `'map'`, replicated on every device, and for each kind of amplitudes present (see
    [`AMPLITUDE_KINDS`][furax.mapmaking._bucket.AMPLITUDE_KINDS]) a list with the right-hand side
    of each bucket, `(tile, ...)`, or `(segment, ...)` for the amplitudes the detectors of an
    observation share."""

    def block_jacobi(self, binned: bool, in_structure: PyTree) -> BJPreconditioner:
        """The block-Jacobi preconditioner, from the diagonal pixel system summed over the buckets.

        The system is under the weight without the noise correlations, nor the implicit templates.
        """
        weight = 'plain' if binned else 'white'
        unknowns = {'map': in_structure}
        E = BlockSelectOperator('map', in_structure=unknowns)
        terms = [
            BucketNormalOperator(tiles, None, weight, amplitudes=False, in_structure=unknowns)
            for tiles in self.buckets
        ]
        return BJPreconditioner.create(
            (E @ AdditionOperator(terms, sequential=True) @ E.T).reduce()
        )

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
            The system to solve. Its unknowns are a dictionary of the selected sky pixels, under
            `'map'`, and of the amplitudes of each kind, one block per bucket.
        """
        # Restrict BJ preconditioner to selected pixels
        M = (S @ BJ.I @ S.T).reduce()

        # Explicit templates and gaps are configured for the run, so every bucket has them or none.
        kinds = amplitude_kinds(self.buckets[0])

        # The unknowns are the selected sky pixels, then, per kind of amplitudes present, one block
        # per bucket
        rhs = self.rhs | {'map': S(self.rhs['map'])}
        structures = furax.tree.as_structure(rhs)

        # Joint system. Each bucket's system sees the selected pixels and its own amplitudes,
        # picked out of the lists and embedded back by `E`.
        terms, responses = [], []
        for b, tiles in enumerate(self.buckets):
            in_structure = {'map': S.out_structure} | {kind: structures[kind][b] for kind in kinds}
            normal = BucketNormalOperator(tiles, S, in_structure=in_structure)
            picks = {kind: BlockSelectOperator(b, in_structure=structures[kind]) for kind in kinds}
            E = BlockDiagonalOperator(
                {'map': IdentityOperator(in_structure=S.out_structure)} | picks
            )
            terms.append((E.T @ normal @ E).reduce())
            if sky is not None:
                responses.append(E.T(normal.response(sky)))

        # The ridge the Gram inverses were factored with acts on the amplitudes alone, so it is
        # one term over them all, outside the passes over the TOD.
        for kind in kinds:
            ridged = [ridge(tiles, kind) for tiles in self.buckets]
            assert all(r is None for r in ridged) or all(r is not None for r in ridged)
            if ridged[0] is not None:
                amplitudes = BlockSelectOperator(kind, in_structure=terms[0].in_structure)
                terms.append((amplitudes.T @ BlockDiagonalOperator(ridged) @ amplitudes).reduce())

        # The preconditioner of the amplitudes of each kind, block-diagonal over the buckets
        inverses = {kind: [preconditioner(tiles, kind) for tiles in self.buckets] for kind in kinds}
        M = BlockDiagonalOperator({'map': M} | inverses)
        A = AdditionOperator(terms, sequential=True)
        response = None
        if sky is not None:
            response = reduce(furax.tree.add, responses)
        return MapMakingSystem(A, rhs, M, injected_b=response)
