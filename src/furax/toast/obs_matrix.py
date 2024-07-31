from pathlib import Path

import jax
import numpy as np
from jax import Array
from jax.experimental.sparse.csr import CSR
from jaxtyping import Inexact, PyTree

from furax.operators import AbstractLazyTransposeOperator, AbstractLinearOperator, square


@square
class ToastObservationMatrixOperator(AbstractLinearOperator):
    """Class for applying a Toast observation matrix.

    Usage:
        >>> path = '/path/to/nside064/toast_telescope_all_time_all_obs_matrix.npz'
        >>> O = ToastObservationMatrixOperator(path)
        >>> x = jnp.ones(O.in_structure().shape)
        >>> y = O(x)
        >>> y = O.T(x)
    """

    matrix: CSR

    def __init__(self, path: Path | str):
        with np.load(path) as data:
            fmt = data['format']
            if isinstance(fmt, np.ndarray):
                fmt = fmt[()]
            if isinstance(fmt, bytes):
                fmt = fmt.decode('utf-8')
            if fmt != 'csr':
                raise NotImplementedError
            matrix = CSR((data['data'], data['indices'], data['indptr']), shape=data['shape'])
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('The observation matrix is not square.')
        self.matrix = matrix

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _a']]:
        return self.matrix @ x

    def transpose(self) -> AbstractLinearOperator:
        return ToastObservationMatrixTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct((self.matrix.shape[0],), self.matrix.dtype)


class ToastObservationMatrixTransposeOperator(AbstractLazyTransposeOperator):
    """Class for applying the transpose of a Toast observation matrix."""

    operator: ToastObservationMatrixOperator

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _a']]:
        return self.operator.matrix.T @ x
