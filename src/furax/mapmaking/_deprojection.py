"""Deprojection of implicit template families into the noise weight.

A template family ``T`` (amplitudes → TOD) can be removed from the GLS problem
*without* solving for its amplitudes by replacing the noise weight ``W`` with the
template-marginalised weight

    W' = W − W T (Tᵀ W T)⁻¹ Tᵀ W ,

the ``W``-metric projector off ``range(T)`` (so ``W' T = 0``). Marginalising over
the amplitudes is *exactly* equivalent to jointly solving for them and discarding the
result: the map and any retained (explicit) amplitudes are identical, but the implicit
degrees of freedom never enter the joint conjugate-gradient solve.

This is the "implicit" path; the "explicit" path keeps the amplitudes as unknowns and
returns them. Both can be active at once (some families deprojected, others solved).

Design note: Gaussian amplitude priors (not implemented)
---------------------------------------------------------
A Gaussian prior ``a ~ N(0, Σ_a)`` on the amplitudes would generalise the weight to

    W' = W − W T (Tᵀ W T + Σ_a⁻¹)⁻¹ Tᵀ W = (N + T Σ_a Tᵀ)⁻¹   (Woodbury, W = N⁻¹),

i.e. Wiener-filter the amplitudes instead of fitting them exactly; the current
deprojection is the improper (flat-prior) limit. Since ``W' T = W T M⁻¹ Σ_a⁻¹`` with
``M = Tᵀ W T + Σ_a⁻¹``, modes in ``ker(Σ_a⁻¹)`` stay *exactly* annihilated even when
coupled to proper ones. Any prior that is block-diagonal per detector (diagonal or dense
``K×K``) slots straight into the block inversion here. Cross-detector priors (e.g. HWPSS
is strongly correlated across detectors) break the per-detector blocks; tractable routes:

- low-rank detector correlation (common mode): Woodbury correction on top of the
  per-detector inverses — one small ``rK×rK`` dense solve;
- reparametrise the correlated part as a *shared* template family (one amplitude set
  expanded to all detectors) plus per-detector residuals: Gram becomes arrow-shaped,
  solved by Schur complement on the small shared block;
- otherwise keep the family explicit and add ``Σ_a⁻¹`` to the amplitude block of the
  joint CG system (one prior apply per iteration, no dense inverse).

The ``regularization`` ridge is itself the MAP limit of an isotropic prior with
data-adaptive precision ``λ·mean(diag G)``; it is kept as a numerical safeguard, not a
statistical statement.

Restrictions (kept deliberately simple)
---------------------------------------
- ``W`` must be *diagonal* (white / ATOP weighting). The Gram ``G = Tᵀ W T`` is then
  block-diagonal over detectors — each detector fits its own amplitudes — and is built by
  probing the (small) per-detector coupled amplitude index. Correlated (Toeplitz) ``W`` is
  not handled here.
- Tuned for :class:`~furax.mapmaking.templates.PerDetectorTemplate` families with
  ``shared_basis=True``. Several families are combined into one ``T`` (e.g. a
  ``BlockRowOperator``); cross-family coupling within a detector is captured by probing the
  combined index, so the families need not be mutually orthogonal.
"""

from math import prod
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, symmetric

__all__ = [
    'PerDetGramInverse',
    'deprojector',
    'marginal_weight',
    'stacked_gram_inverse',
]


@symmetric
class PerDetGramInverse(AbstractLinearOperator):
    """Apply the per-detector inverse Gram ``G_d⁻¹`` in template-amplitude space.

    Amplitudes are a PyTree whose leaves have shape ``(*batch, *coupled_f)`` (one leaf per
    template family). The ``batch`` axes are block-diagonal (detector, and an outer
    observation axis in the multi-observation path); ``coupled_f`` are the within-block
    indices that couple. For each batch element the coupled indices of every family are
    flattened and concatenated into a single block of size ``K = sum_f prod(coupled_f)``;
    ``mv`` contracts the stored ``(*batch, K, K)`` inverse against that block and unflattens.
    Symmetric (inverse of a symmetric Gram).
    """

    inv: Float[Array, '*batch k k']

    @classmethod
    def from_gram(
        cls,
        gram: AbstractLinearOperator,
        batch_ndim: int = 1,
        regularization: float = 0.0,
    ) -> Self:
        """Probe ``G = Tᵀ W T`` to assemble and invert its block-diagonal Gram.

        For diagonal ``W`` and per-detector templates, ``G`` is block-diagonal over the ``batch``
        axes (detectors, plus an outer observation axis in the multi-observation path). Applying
        ``G`` to the amplitude that is one on a single coupled index (broadcast over all batch
        elements) and zero elsewhere returns that column of every block at once. The coupled
        index is the concatenation of all families' coupled axes, so cross-family blocks are
        captured. The probes run as a single traced ``lax.map`` iteration, so trace/compile
        time stays constant in ``K`` and TOD-sized intermediates are never batched.

        Args:
            gram: The Gram operator ``Tᵀ W T`` acting on the amplitude PyTree
                (leaves ``(*batch, *coupled_f)``).
            batch_ndim: Number of leading block-diagonal axes (1 = detector, 2 = obs + detector).
            regularization: Relative ridge added to each block diagonal before inversion.

        Returns:
            The block-diagonal inverse Gram operator.
        """
        amp_structure = gram.in_structure
        leaves = jax.tree.leaves(amp_structure)
        treedef = jax.tree.structure(amp_structure)
        batch_shape = leaves[0].shape[:batch_ndim]
        dtype = leaves[0].dtype
        sizes = [int(prod(s.shape[batch_ndim:])) for s in leaves]
        offsets = np.cumsum([0, *sizes])
        n_basis = int(offsets[-1])

        def probe(col: Array) -> Array:
            # one-hot on the concatenated coupled index, split at the (static) family offsets
            # and broadcast over the batch axes.
            flat = jnp.zeros((n_basis,), dtype).at[col].set(1.0)
            parts = [
                jnp.broadcast_to(
                    flat[offsets[k] : offsets[k + 1]].reshape(s.shape[batch_ndim:]), s.shape
                )
                for k, s in enumerate(leaves)
            ]
            response = gram(jax.tree.unflatten(treedef, parts))
            flats = [leaf.reshape(*batch_shape, -1) for leaf in jax.tree.leaves(response)]
            return jnp.concatenate(flats, axis=-1)  # (*batch, n_basis) = column `col`

        cols = jax.lax.map(probe, jnp.arange(n_basis))  # (col, *batch, row)
        block = jnp.moveaxis(cols, 0, -1)  # (*batch, row, col)

        if regularization:
            # relative ridge: scale by the mean block diagonal so it is dimensionally sound.
            diag = jnp.diagonal(block, axis1=-2, axis2=-1)
            scale = jnp.mean(diag, axis=-1)[..., None, None]
            block = block + regularization * scale * jnp.eye(n_basis, dtype=dtype)

        return cls(jnp.linalg.inv(block), in_structure=amp_structure)

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        batch_ndim = self.inv.ndim - 2
        treedef = jax.tree.structure(x)
        leaves = jax.tree.leaves(x)
        flats = [leaf.reshape(*leaf.shape[:batch_ndim], -1) for leaf in leaves]
        xf = jnp.concatenate(flats, axis=-1)  # (*batch, K)
        yf = jnp.einsum('...ij,...j->...i', self.inv, xf)
        out = []
        offset = 0
        for leaf in leaves:
            c = prod(leaf.shape[batch_ndim:])
            out.append(yf[..., offset : offset + c].reshape(leaf.shape))
            offset += c
        return jax.tree.unflatten(treedef, out)


def deprojector(
    weight: AbstractLinearOperator,
    template: AbstractLinearOperator,
    batch_ndim: int = 1,
    regularization: float = 0.0,
) -> AbstractLinearOperator:
    """``W``-metric projector ``P = T (Tᵀ W T)⁻¹ Tᵀ W`` onto ``range(T)`` (diagonal ``W``).

    Precomputes the per-detector inverse Gram ``G⁻¹`` — so the forward apply is cheap (project
    the TOD onto amplitudes ``Tᵀ W``, normalise by ``G⁻¹``, expand back ``T``) — and returns it
    as the reduced composition ``T @ G⁻¹ @ Tᵀ @ W``. ``P`` is idempotent but *not* symmetric
    (``Pᵀ = W T G⁻¹ Tᵀ``); the symmetric object is the complementary weight ``W' = W − W P``
    (see :func:`marginal_weight`). ``batch_ndim`` is the number of leading block-diagonal
    amplitude axes (1 = detector, 2 = observation + detector).
    """
    gram = (template.T @ weight @ template).reduce()
    gram_inverse = PerDetGramInverse.from_gram(gram, batch_ndim, regularization)
    return (template @ gram_inverse @ template.T @ weight).reduce()


def marginal_weight(
    weight: AbstractLinearOperator,
    template: AbstractLinearOperator,
    batch_ndim: int = 1,
    regularization: float = 0.0,
) -> AbstractLinearOperator:
    """Deprojected weight ``W' = W − W T (Tᵀ W T)⁻¹ Tᵀ W`` for diagonal ``W``.

    Returned as the reduced composition ``W @ (I − P)``, which applies the diagonal ``W`` one
    time fewer per solve iteration than the expanded ``W − W @ P``. Symmetric (``W P`` is
    symmetric since ``Pᵀ W = W P``) and annihilates the template (``W' T = 0``).
    For the multi-observation path build ``W'`` from scan-block pieces —
    ``W − W @ T @ G⁻¹ @ Tᵀ @ W`` with each factor a
    :class:`~furax.mapmaking._scan_blocks.ScanBlockDiagonalOperator` (see
    :func:`stacked_gram_inverse`) — so the scan-block algebra fuses it into one block and keeps
    the obs-axis sharding and the closed-over ``−1`` correct.

    Args:
        weight: The diagonal noise weight ``W`` (acts on a single observation's TOD).
        template: The implicit template operator ``T`` (amplitudes → TOD).
        batch_ndim: Number of leading block-diagonal amplitude axes (1 = detector,
            2 = observation + detector).
        regularization: Relative ridge on the Gram blocks before inversion.
    """
    projector = deprojector(weight, template, batch_ndim, regularization)
    identity = IdentityOperator(in_structure=weight.in_structure)
    return (weight @ (identity - projector)).reduce()


def stacked_gram_inverse(
    weight: AbstractLinearOperator,
    template: AbstractLinearOperator,
    regularization: float = 0.0,
) -> AbstractLinearOperator:
    """Per-observation inverse Gram ``G⁻¹ = (Tᵀ W T)⁻¹`` for obs-stacked operators.

    ``weight`` and ``template`` are *per-observation* operators whose array leaves carry a
    leading observation axis (as produced by ``jax.lax.scan`` over observations). The returned
    operator holds the inverse blocks of shape ``(n_obs, n_dets, K, K)`` and acts on the
    obs-stacked amplitude PyTree.

    Compose it with scan-block pieces to form the deprojected weight as a single fused
    ``ScanBlockDiagonalOperator`` — ``W' = W − W T G⁻¹ Tᵀ W`` (see
    ``tests/mapmaking/test_scan_blocks.py::test_marginal_weight_fusion``); the same blocks build
    the per-observation projector ``P = T G⁻¹ Tᵀ W`` used to apply ``W'`` inline while streaming
    the TOD.

    Requires an active mesh context (``jax.set_mesh``). The per-observation Gram inverses are
    built inside a ``shard_map`` + ``scan`` over each device's local observations, so the
    ``K×K`` block inversions stay device-local (no collective over the obs-sharded batch).
    """
    mesh = jax.sharding.get_abstract_mesh()
    axis = mesh.axis_names[0]

    @jax.shard_map(out_specs=P(axis), check_vma=False)
    def build_inv(
        template: AbstractLinearOperator, weight: AbstractLinearOperator
    ) -> Float[Array, 'obs det k k']:
        def step(_: None, args: Any) -> tuple[None, Array]:
            t, w = args
            gram = (t.T @ w @ t).reduce()
            return None, PerDetGramInverse.from_gram(gram, 1, regularization).inv

        _, inv = jax.lax.scan(step, None, (template, weight))
        return inv  # (n_local_obs, *det_batch, K, K)

    inv = build_inv(template, weight)
    return PerDetGramInverse(inv, in_structure=template.in_structure)
