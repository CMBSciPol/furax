r"""The normal system of the multi-observation mapmaker, evaluated one detector batch at a time.

The mapmaker solves $\sum_b J_b^\top W'_b J_b \, x = \sum_b J_b^\top W'_b d_b$ over the detector
batches $b$ of every observation, with $J_b = [H_b S^\top, T_{e,b}, T_{g,b}]$ mapping the selected
sky pixels, the batch's own template amplitudes and those its observation's detectors share to the
batch's TOD, and $W'_b$ its weight (see
[`ObservationModel.deprojected_weight`][furax.mapmaking._model.ObservationModel.deprojected_weight]).

A bucket keeps what its observations hold as a `BucketModel`, and the operators of this module
evaluate the sums over it with the engine: a family of the bucket's observations, split over the
devices, each observation a family of its detector batches.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from jax.tree_util import register_dataclass
from jaxtyping import Array, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.core import BlockRowOperator
from furax.obs.stokes import StokesType

from ._engine import (
    Policy,
    ReplicatedFamily,
    StackedFamily,
    ZippedFamily,
    family_apply,
)
from ._model import ObservationModel, ObservationTemplates


@register_dataclass
@dataclass(frozen=True)
class Batch:
    """What the system needs of one detector batch of an observation."""

    model: ObservationModel
    templates: ObservationTemplates | None
    """The batch's templates; those its observation's detectors share, factored over all its
    batches (None for a run without templates)."""
    amplitude_rhs: PyTree[Array] | None
    """The right-hand side of the batch's own explicit amplitudes (None without them)."""


@register_dataclass
@dataclass(frozen=True)
class BucketModel:
    """The detector batches of every observation of a bucket.

    What a batch holds is split in two: `batches`, stacked over the slots then over the batches of
    each, and `shared`, what the batches of an observation all hold (its boresight
    pointing, the bases and Gram inverse of the templates its detectors share, ...), stacked over
    the slots only. Each is `None` where the other has a leaf; a batch is their combination.
    """

    batches: Batch
    """Leaves `(slot, batch, ...)`."""
    shared: Batch
    """Leaves `(slot, ...)`."""
    common_rhs: PyTree[Array] | None
    """The right-hand side of the amplitudes the detectors of each observation have in common, `(slot,
    ...)` (None without them)."""

    @property
    def templates(self) -> ObservationTemplates | None:
        """The kinds of templates of the bucket, from the structure of its batches."""
        return self.batches.templates


def bucket_apply(
    model: BucketModel,
    apply: Callable[
        [Batch, PyTree[Any], PyTree[Array] | None, PyTree[Array] | None], tuple[Any, ...]
    ],
    whole: PyTree[Any],
    own: PyTree[Array] | None,
    common: PyTree[Array] | None,
    axis: str,
) -> tuple[Any, Any, Any]:
    """`apply` on every detector batch of every observation of a bucket.

    Args:
        model: The bucket.
        apply: From a batch, `whole`, its own amplitudes and those of its observation, to `(map,
            own, common)`: a map, the batch's own amplitudes and its observation's, any of them
            `None`.
        whole: What every batch takes whole, such as a map, held by every device.
        own: The amplitudes of each batch alone, `(slot, batch, ...)`, if any.
        common: The amplitudes of each observation, `(slot, ...)`, if any.
        axis: The mesh axis the slots are sharded over.

    Returns:
        The maps summed over the bucket, the batches' own amplitudes `(slot, batch, ...)`, and
        those of each observation summed over its batches, `(slot, ...)`.
    """

    def observation(members):
        (batches, kept, own_o, common_o), whole_o = members
        families = [StackedFamily(batches, shared=kept)]
        if own_o is not None:
            families.append(StackedFamily(own_o))

        def batch(members):
            member, *rest = members
            return apply(member, whole_o, rest[0] if rest else None, common_o)

        out, _ = family_apply(ZippedFamily(tuple(families)), batch, ('sum', 'stack', 'sum'))
        return out

    slots = StackedFamily((model.batches, model.shared, own, common))
    family = ZippedFamily((slots, ReplicatedFamily(whole, slots.size)))
    policy = Policy(devices=axis)
    out, _ = family_apply(family, observation, ('sum', 'stack', 'stack'), policy)
    return out


def joint_operator(
    batch: Batch, selection: AbstractLinearOperator | None, own: bool, common: bool
) -> AbstractLinearOperator:
    r"""$J_b = [H_b S^\top, T_{e,b}, T_{g,b}]$ of a batch, with the amplitudes asked for.

    Args:
        batch: The batch.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        own: Whether to include the batch's own explicit amplitudes.
        common: Whether to include those its observation's detectors have in common.
    """
    H = batch.model.H if selection is None else batch.model.H @ selection.T
    blocks = [H]
    if own:
        assert batch.templates is not None and batch.templates.explicit is not None
        blocks.append(batch.templates.explicit.operator)
    if common:
        assert batch.templates is not None and batch.templates.common is not None
        blocks.append(batch.templates.common.operator)
    return BlockRowOperator(blocks)


def system_weight(batch: Batch) -> AbstractLinearOperator:
    """$W_b$ of a batch, its implicit templates marginalized."""
    implicit = None if batch.templates is None else batch.templates.implicit
    return batch.model.deprojected_weight(implicit)


@symmetric
class BucketNormalOperator(AbstractLinearOperator):
    r"""$\sum_b J_b^\top W'_b J_b$ over the detector batches of a bucket.

    The unknowns are a list: the map, then the batches' own amplitudes and their observations', if
    the bucket has such templates and `amplitudes` is set. Otherwise they are the map alone. Without
    `selection`, the map is the whole sky.
    """

    model: BucketModel
    selection: AbstractLinearOperator | None
    """The selection of the estimated pixels, if any."""
    axis: str = field(metadata={'static': True})
    """The mesh axis the slots are sharded over."""
    weight: Literal['system', 'plain', 'white'] = field(default='system', metadata={'static': True})
    """$W'_b$: the weight of the system, or $W F$ without the implicit templates, or the same with
    the white part of the noise only."""
    amplitudes: bool = field(default=True, metadata={'static': True})
    """Whether the unknowns include the amplitudes of the explicit templates."""

    def mv(self, x: Any) -> Any:
        m, own, common = self._split(x)

        def apply(batch, whole, own_b, common_o):
            m, selection = whole
            J = joint_operator(batch, selection, *self._amplitudes_present)
            inputs = [x for x in (own_b, common_o) if x is not None]
            return self._project(batch, self._weight(batch)(J([m, *inputs])), selection)

        return self._join(
            *bucket_apply(self.model, apply, (m, self.selection), own, common, self.axis)
        )

    def response(self, sky: StokesType) -> Any:
        r"""$\sum_b J_b^\top W'_b H_b \, s$: the right-hand side of the TOD of a sky `sky`."""

        def apply(batch, whole, _own, _common):
            sky, selection = whole
            return self._project(batch, self._weight(batch)(batch.model.H(sky)), selection)

        whole = (sky, self.selection)
        return self._join(*bucket_apply(self.model, apply, whole, None, None, self.axis))

    def _project(
        self, batch: Batch, u: PyTree[Array], selection: AbstractLinearOperator | None
    ) -> tuple[Any, Any, Any]:
        r"""$J_b^\top u$, in parts."""
        own, common = self._amplitudes_present
        return _unpack(joint_operator(batch, selection, own, common).T(u), own, common)

    def _weight(self, batch: Batch) -> AbstractLinearOperator:
        if self.weight == 'system':
            return system_weight(batch)
        model = batch.model
        W = model.diag_W() if self.weight == 'white' else model.W
        return (W @ model.F).reduce()

    @property
    def _amplitudes_present(self) -> tuple[bool, bool]:
        """Whether the unknowns have the batches' own amplitudes, and their observations'."""
        templates = self.model.templates
        if not self.amplitudes or templates is None:
            return False, False
        return templates.explicit is not None, templates.common is not None

    def _split(self, x: Any) -> tuple[Any, Any, Any]:
        """The map and the amplitudes of each kind, `None` for those absent."""
        own, common = self._amplitudes_present
        if not own and not common:
            return x, None, None
        return _unpack(x, own, common)

    def _join(self, m: Any, own: Any, common: Any) -> Any:
        present_own, present_common = self._amplitudes_present
        if not present_own and not present_common:
            return m
        return [m, *([own] if present_own else []), *([common] if present_common else [])]


@symmetric
class MemberOperator(AbstractLinearOperator):
    """The operator of each observation, or of each detector batch, on its own amplitudes.

    `x[s] -> operator[s](x[s])` over the slots of a bucket, or `x[s, b] -> operator[s, b](x[s, b])`
    over their batches too, such as the inverses of the Grams of the explicit amplitudes, which
    precondition them. Per batch, the leaves the batches share are those of `shared`, held once
    per observation.
    """

    operators: AbstractLinearOperator
    """The operators, stacked `(slot, ...)`, or `(slot, batch, ...)` with `batches`."""
    shared: AbstractLinearOperator | None = None
    """With `batches`, the leaves the batches of an observation share, stacked `(slot, ...)`."""
    batches: bool = field(default=False, metadata={'static': True})
    """Whether there is an operator per detector batch rather than per observation."""

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        def apply(member):
            operator, v = member
            return operator(v)

        if not self.batches:
            slots = ZippedFamily((StackedFamily(self.operators), StackedFamily(x)))
            return _at_once(slots, apply)

        def observation(member):
            (operators, kept), x_o = member
            batches = (StackedFamily(operators, shared=kept), StackedFamily(x_o))
            return _at_once(ZippedFamily(batches), apply)

        slots = (StackedFamily((self.operators, self.shared)), StackedFamily(x))
        return _at_once(ZippedFamily(slots), observation)


def _at_once(family: ZippedFamily, apply: Callable[[Any], PyTree[Array]]) -> PyTree[Array]:
    """`apply` on every member of a family, a small computation, in a single round."""
    out, _ = family_apply(family, apply, 'stack', Policy(batch=family.size))
    return out


def _unpack(parts: list[PyTree[Array]], own: bool, common: bool) -> tuple[Any, Any, Any]:
    """The map and the amplitudes of each kind of a list of parts, `None` for those absent."""
    m, *rest = parts
    return m, rest.pop(0) if own else None, rest.pop(0) if common else None
