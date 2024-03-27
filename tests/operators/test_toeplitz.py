import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit
from numpy.testing import assert_allclose

from astrosim.operators.toeplitz import SymmetricBandToeplitzOperator


@pytest.mark.parametrize('method', SymmetricBandToeplitzOperator.METHODS)
@pytest.mark.parametrize('do_jit', [False, True])
def test(method: str, do_jit: bool) -> None:
    x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    expected_y = x - 0.5
    kernel = jnp.array([0.5, 0.5])
    func = SymmetricBandToeplitzOperator(kernel, x.shape, method=method).mv
    if do_jit:
        func = jit(func)
    y = func(x)
    assert_allclose(y, expected_y, rtol=1e-7, atol=1e-7)
