import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy

from furax import Config
from furax.core._linear import PackOperator
from furax.interfaces.toast.obs_matrix import ToastObservationMatrixOperator


@pytest.mark.xfail(reason='We need to add an observation matrix of smaller nside.')
def test() -> None:
    matobs_path = (
        '/home/chanial/work/scipol/data/nside064/toast_telescope_all_time_all_obs_matrix.npz'
    )
    obs = ToastObservationMatrixOperator(matobs_path)
    ones = jnp.ones(obs.in_structure().shape, np.float32)
    y = obs(ones)
    mask = y != 0
    pack = PackOperator(mask, obs.in_structure())
    unpack = pack.T

    with Config(solver=lx.CG(rtol=1e-6, atol=1e-6, max_steps=500)):
        A = (pack @ obs.T @ obs @ unpack).I @ pack @ obs.T
    ones2 = A(y)
    print(ones2)
    rms = jnp.sum((pack(ones) - ones2) ** 2)
    print(scipy.stats.describe(jnp.abs(pack(ones) - ones2)))
    print(rms)
    # Diverges...
    # 200:    21049.54724850748
    # 1000: 179169463.15897495
