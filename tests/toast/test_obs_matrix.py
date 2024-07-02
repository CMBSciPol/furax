import jax.numpy as jnp
import pytest

from astrosim.toast.obs_matrix import ToastObservationMatrixOperator


@pytest.mark.xfail(reason='We need to add an observation matrix of smaller nside.')
def test() -> None:
    matobs_path = (
        '/home/chanial/work/scipol/data/nside064/toast_telescope_all_time_all_obs_matrix.npz'
    )
    O = ToastObservationMatrixOperator(matobs_path)
    x = jnp.ones(O.in_structure().shape)
    y = O(x)
    oTo = O.T @ O
    z = oTo(x)
