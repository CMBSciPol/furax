r"""Pomme: implicit deprojection of the mean of every short interval of the TOD.

Pomme fits and removes one offset per detector and per interval of `tau` consecutive samples. It
is a template in the sense of [`furax.mapmaking.templates`][], with the indicator template $Z$
(one column per detector and interval), but the template matrix is never formed: with a diagonal
weight $W$ that is constant on each interval, the marginalised weight $W' = W D_Z$ reduces to
$W F$ with $F = I - Z (Z^\top Z)^{-1} Z^\top$ the plain interval-mean remover,
[`PommeProjectionOperator`][].

The constant-weight requirement is what the mapmaker's mask widening provides: every interval with
a flagged sample is flagged whole (and so is the partial tail interval), so per-detector white
noise gives a weight that commutes with $F$. Then $W F$ is symmetric positive semidefinite and
the normal equations $P^\top W F P\, m = P^\top W F\, d$ are a valid GLS system.
"""

from dataclasses import field

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, square

__all__ = [
    'PommeProjectionOperator',
]


@square
class PommeProjectionOperator(AbstractLinearOperator):
    r"""Remove the mean of every `tau`-sample interval of a `(det, samp)` TOD.

    This is the Pomme deprojector $F = I - Z (Z^\top Z)^{-1} Z^\top$ for the template $Z$ whose
    columns are the indicators of the consecutive `tau`-sample intervals (one amplitude per
    detector and interval): an unweighted projector, the same for every detector. The samples
    past the last whole interval (the tail) are left untouched.
    """

    tau: int = field(metadata={'static': True})

    def __init__(
        self,
        tau: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'tau', tau)
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det samp']:
        n_det, n_samp = self.in_structure.shape
        n_int, n_rem = divmod(n_samp, self.tau)
        y = x[:, : n_int * self.tau].reshape(n_det, n_int, self.tau)
        y = y - jnp.mean(y, axis=-1, keepdims=True)
        y = y.reshape(n_det, n_int * self.tau)
        if n_rem == 0:
            return y
        return jnp.concatenate([y, x[:, -n_rem:]], axis=1)
