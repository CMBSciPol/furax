from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

import furax.tree
from furax import AbstractLinearOperator

__all__ = [
    'BucketSumOperator',
    'apply_bucket_sum',
]


def apply_bucket_sum(x: PyTree[Array], terms: Sequence[AbstractLinearOperator]) -> PyTree[Array]:
    r"""Apply a sum of per-bucket operators, $\sum_b A_b x$, one bucket at a time.

    Buckets hold different envelopes, so their contributions cannot be batched into a single
    scan. Summing them as independent expressions lets XLA reserve every bucket's working set
    at once: the memory a run reserves then grows with the bucket count instead of shrinking
    with it, undoing what bucketing is for. Dispatching the buckets through a `lax.switch`
    inside a `fori_loop` keeps exactly one of them live per iteration, so their temporaries are
    overlaid and the reservation is set by the largest bucket alone.

    Args:
        x: The vector to apply the sum to.
        terms: One operator per bucket, all with the same input and output structures.

    Returns:
        The summed contributions of every bucket.
    """
    if len(terms) == 1:
        return terms[0](x)
    branches = [lambda x, term=term: term(x) for term in terms]
    zero = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), terms[0].out_structure)
    return jax.lax.fori_loop(
        0,
        len(branches),
        lambda b, acc: furax.tree.add(acc, jax.lax.switch(b, branches, x)),
        zero,
    )


class BucketSumOperator(AbstractLinearOperator):
    """The sum of one operator per bucket, applied a bucket at a time.

    Same as `AdditionOperator` but applies operands one at a time (see [`apply_bucket_sum`][])
    to avoid excessive memory usage.
    """

    operands: list[AbstractLinearOperator]

    def __init__(self, operands: Sequence[AbstractLinearOperator]) -> None:
        operands = list(operands)
        if not operands:
            raise ValueError('BucketSumOperator needs at least one operand')
        object.__setattr__(self, 'operands', operands)
        super().__init__(in_structure=operands[0].in_structure)

    @property
    def is_square(self) -> bool:
        return super().is_square or self.operands[0].is_square

    @property
    def is_symmetric(self) -> bool:
        return super().is_symmetric or all(op.is_symmetric for op in self.operands)

    @property
    def is_positive_semidefinite(self) -> bool:
        return super().is_positive_semidefinite or all(
            op.is_positive_semidefinite for op in self.operands
        )

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return apply_bucket_sum(x, self.operands)

    def transpose(self) -> AbstractLinearOperator:
        return BucketSumOperator([op.T for op in self.operands])
