"""Implicit marginalisation of template amplitudes into the weighting operator.

A template family with operator ``T_m`` (amplitudes → TOD) can be removed from the GLS
problem *without* solving for its amplitudes by replacing the noise weight ``W`` with the
template-marginalised weight

    W_m = W − W T_m (T_mᵀ W T_m)⁻¹ T_mᵀ W ,

the ``W``-metric projector off ``range(T_m)`` (``W_m T_m = 0``). Marginalising over the
amplitudes is *exactly* equivalent to jointly solving for them and discarding the result:
the map and any retained (explicit) amplitudes are identical, but the marginalised degrees
of freedom never enter the conjugate-gradient solve.

The cost is the Gram inverse ``G⁻¹ = (T_mᵀ W T_m)⁻¹``. This is tractable only because the
Gram is block-diagonal: for a :class:`~furax.mapmaking.basis_templates.PerDetectorTemplate`
each detector fits its own amplitudes (block-diagonal over detectors), and for a
:class:`~furax.mapmaking.basis_templates.SegmentedBasis` (per-interval Legendre) distinct
scan intervals share no sample (block-diagonal over intervals too). Only the trailing
*coupled* sub-basis index (the polynomial order, harmonic, ...) gives a dense block, which
is small (``k`` ≲ 10) and inverted in closed form.

The block structure is discovered generically by *probing*: applying ``G`` to a unit
amplitude on each coupled sub-index recovers one column of every block at once (the
block-diagonal structure guarantees no cross-block leakage), so ``sum_k`` applies of ``G``
build all blocks. This keeps the builder agnostic to the specific basis.
"""

from dataclasses import field
from math import prod

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree

from furax import AbstractLinearOperator

from ._scan_blocks import ScanBlockDiagonalOperator
from .basis_templates import PerDetectorTemplate

__all__ = [
    'BlockDiagInverseOperator',
    'build_marginal_weight',
    'marginal_weight',
]


class BlockDiagInverseOperator(AbstractLinearOperator):
    """Apply a block-diagonal inverse Gram ``G⁻¹`` in template-amplitude space.

    Each amplitude leaf has shape ``(*lead, *coupled)`` where ``lead`` are the independent
    (block-diagonal) axes — detectors, and for segmented bases the scan-interval axis — and
    ``coupled`` is the dense sub-basis index. ``blocks`` holds, per leaf, the inverse Gram
    block ``(*lead, c, c)`` with ``c = prod(coupled)``; ``mv`` contracts it against the
    flattened coupled index of the amplitude. Symmetric (inverse of a symmetric Gram).
    """

    blocks: PyTree[Array]
    lead_ndims: tuple[int, ...] = field(metadata={'static': True})

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        treedef = jax.tree.structure(x)
        x_leaves = jax.tree.leaves(x)
        b_leaves = jax.tree.leaves(self.blocks)
        out = []
        for b, xi, lead in zip(b_leaves, x_leaves, self.lead_ndims):
            lead_shape = xi.shape[:lead]
            c = prod(xi.shape[lead:])
            xf = xi.reshape(*lead_shape, c)
            yf = jnp.einsum('...ij,...j->...i', b, xf)
            out.append(yf.reshape(xi.shape))
        return jax.tree.unflatten(treedef, out)

    def transpose(self) -> AbstractLinearOperator:
        # G is symmetric ⇒ G⁻¹ symmetric; transpose the dense block axes to be exact.
        blocks_T = jax.tree.map(lambda b: jnp.swapaxes(b, -1, -2), self.blocks)
        return BlockDiagInverseOperator(blocks_T, self.lead_ndims, in_structure=self.in_structure)


def _lead_ndims(T_m: AbstractLinearOperator) -> tuple[int, ...]:
    """Per-amplitude-leaf count of leading block-diagonal axes (detector + basis-independent),
    aligned with ``jax.tree.leaves(T_m.in_structure)``."""
    perdets = [
        o
        for o in jax.tree.leaves(T_m, is_leaf=lambda o: isinstance(o, PerDetectorTemplate))
        if isinstance(o, PerDetectorTemplate)
    ]
    return tuple(1 + o.operator.independent_ndim for o in perdets)  # type: ignore[attr-defined]


def _build_gram_inverse(
    G: AbstractLinearOperator,
    probe_structure: PyTree[jax.ShapeDtypeStruct],
    probe_lead: tuple[int, ...],
    out_structure: PyTree[jax.ShapeDtypeStruct],
    out_lead: tuple[int, ...],
    regularization: float,
    mesh: jax.sharding.Mesh | None = None,
) -> BlockDiagInverseOperator:
    """Probe ``G = T_mᵀ W T_m`` to assemble and invert its block-diagonal Gram.

    For leaf ``i`` and coupled sub-index ``j``, applying ``G`` to the amplitude that is one
    on ``j`` (broadcast over all block-diagonal axes) and zero elsewhere returns column ``j``
    of every block of leaf ``i`` simultaneously — the block-diagonal structure guarantees the
    response stays within leaf ``i`` and within each ``(detector, interval)`` block.

    ``probe_structure``/``probe_lead`` describe the amplitude space ``G`` acts on (stacked,
    with an obs axis, when ``G`` is a :class:`ScanBlockDiagonalOperator`). The returned
    operator is declared over ``out_structure``/``out_lead`` (the per-observation amplitude
    space) so it can be wrapped by ``ScanBlockDiagonalOperator.create``; its ``blocks`` keep
    the obs axis as the leaf's leading axis, which the scan peels off per observation.
    """
    treedef = jax.tree.structure(probe_structure)
    leaf_structs = jax.tree.leaves(probe_structure)

    def _zeros(s: jax.ShapeDtypeStruct) -> Array:
        z = jnp.zeros(s.shape, s.dtype)
        if mesh is None:
            return z
        # shard the probe over the obs axis (leaf axis 0) to match the scan-block kernel,
        # using the concrete mesh (the structure's own sharding is over an abstract mesh).
        spec = P(mesh.axis_names[0], *([None] * (z.ndim - 1)))
        return jax.device_put(z, jax.sharding.NamedSharding(mesh, spec))  # type: ignore[no-any-return]

    def _invert(block: Array) -> Array:
        # block: (*lead, c, c). Regularise then invert the trailing c×c blocks. Run inside a
        # shard_map (below) so each device inverts its own local blocks — jnp.linalg.inv batches
        # over the leading axes via vmap, which rejects an obs-sharded batch axis directly.
        if regularization:
            # relative ridge: scale by the mean block diagonal so it is dimensionally sound.
            diag = jnp.diagonal(block, axis1=-2, axis2=-1)
            scale = jnp.mean(diag, axis=-1)[..., None, None]
            block = block + regularization * scale * jnp.eye(block.shape[-1], dtype=block.dtype)
        return jnp.linalg.inv(block)  # type: ignore[no-any-return]

    inv_blocks = []
    for i, (s, lead) in enumerate(zip(leaf_structs, probe_lead)):
        lead_shape = s.shape[:lead]
        c = prod(s.shape[lead:])
        cols = []
        for j in range(c):
            unit = _zeros(s).reshape(*lead_shape, c).at[..., j].set(1.0).reshape(s.shape)
            amp = [unit if k == i else _zeros(t) for k, t in enumerate(leaf_structs)]
            Ga = G(jax.tree.unflatten(treedef, amp))
            col = jax.tree.leaves(Ga)[i].reshape(*lead_shape, c)  # (*lead, out)
            cols.append(col)
        block = jnp.stack(cols, axis=-1)  # (*lead, out, in)
        if mesh is None:
            inv_blocks.append(_invert(block))
        else:
            spec = P(mesh.axis_names[0], *([None] * (block.ndim - 1)))
            inv_blocks.append(jax.shard_map(_invert, out_specs=spec, check_vma=False)(block))
    return BlockDiagInverseOperator(
        jax.tree.unflatten(treedef, inv_blocks), out_lead, in_structure=out_structure
    )


def marginal_weight(
    W: AbstractLinearOperator,
    T_m: AbstractLinearOperator,
    regularization: float = 0.0,
) -> AbstractLinearOperator:
    """Build the template-marginalised weight ``W_m = W − W T_m (T_mᵀ W T_m)⁻¹ T_mᵀ W``.

    ``W`` is the noise weight and ``T_m`` the marginalised template operator (amplitudes →
    TOD); both act on a single observation's TOD (the obs axis is added later by
    :class:`~furax.mapmaking._scan_blocks.ScanBlockDiagonalOperator`). The returned operator
    is the same shape as ``W`` and satisfies ``W_m T_m = 0``.
    """
    lead = _lead_ndims(T_m)
    G = (T_m.T @ W @ T_m).reduce()
    Ginv = _build_gram_inverse(G, T_m.in_structure, lead, T_m.in_structure, lead, regularization)
    return (W - W @ T_m @ Ginv @ T_m.T @ W).reduce()


def build_marginal_weight(
    W: AbstractLinearOperator,
    T_m: AbstractLinearOperator,
    regularization: float = 0.0,
    mesh: jax.sharding.Mesh | None = None,
) -> AbstractLinearOperator:
    """Per-observation template-marginalised weight ``W_m`` for the multi-observation path.

    ``W`` and ``T_m`` are the *per-observation* noise weight and marginalised template
    operators with an obs-stacked leading leaf axis (``model.W`` and ``templates.marginal``).
    They are wrapped in :class:`ScanBlockDiagonalOperator` only to probe the Gram per
    observation under the active mesh; the returned ``W_m`` is itself a *per-observation*
    operator (obs-stacked leaves, including the obs-stacked Gram-inverse blocks), so it slots
    in exactly where ``model.W`` did — wrap it with ``ScanBlockDiagonalOperator.create`` for
    the operator algebra, or apply it per observation inside the streaming accumulators.
    Requires an active mesh context (``jax.set_mesh``).
    """
    Ws = ScanBlockDiagonalOperator.create(W)
    Ts = ScanBlockDiagonalOperator.create(T_m)
    out_lead = _lead_ndims(T_m)
    # the stacked amplitude space gains a leading obs axis, so the probe's block-diagonal lead
    # grows by one; the resulting blocks keep that obs axis for the per-obs operator below.
    probe_lead = tuple(n + 1 for n in out_lead)
    Gs = (Ts.T @ Ws @ Ts).reduce()
    Ginv = _build_gram_inverse(
        Gs, Gs.in_structure, probe_lead, T_m.in_structure, out_lead, regularization, mesh
    )
    return (W - W @ T_m @ Ginv @ T_m.T @ W).reduce()
