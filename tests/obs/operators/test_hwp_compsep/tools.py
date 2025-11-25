from typing import get_args

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest
from fgbuster import Dust, MixingMatrix
from fgbuster.observation_helpers import _jysr2rj, _rj2cmb
from jaxtyping import Array, Float, Integer
from numpy.random import PCG64, Generator

from furax.obs.landscapes import StokesLandscape
from furax.obs.operators import (
    CMBOperator,
    DustOperator,
    MixingMatrixOperator,
)
from furax.obs.stokes import (
    ValidStokesType,
)


@pytest.fixture(params=get_args(ValidStokesType))
def stokes(request: pytest.FixtureRequest) -> ValidStokesType:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param


def get_random_generator(seed: int) -> np.random.Generator:
    return Generator(PCG64(seed))


@jax.tree_util.register_pytree_node_class
class MAG_Landscape(StokesLandscape):
    def __init__(
        self,
        shape=None,
        stokes: ValidStokesType = 'IQU',
        dtype=np.float64,
        pixel_shape=None,
    ):
        if shape is None and pixel_shape is None:
            raise TypeError('The shape is not specified.')
        if shape is not None and pixel_shape is not None:
            raise TypeError(
                'Either the shape or pixel_shape should be specified.'
            )
        shape = shape if pixel_shape is None else pixel_shape[::-1]
        assert shape is not None  # mypy assert
        super().__init__(shape, dtype)
        self.stokes = stokes
        self.pixel_shape = shape[::-1]

    # --- PyTree interface ---
    def tree_flatten(self):
        children = ()
        aux_data = {
            "shape": self.shape,
            "stokes": self.stokes,
            "dtype": self.dtype,
            "pixel_shape": self.pixel_shape,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            shape=aux_data["shape"],
            stokes=aux_data["stokes"],
            dtype=aux_data["dtype"],
            pixel_shape=aux_data["pixel_shape"],
        )

    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], ...]:
        nside = int(jnp.sqrt(self.shape[0] // 12))
        return (jhp.ang2pix(nside, theta, phi),)


def eval_A(params, f_c, nu_r, in_structure):
    cmb_template = CMBOperator(
        nu_r, in_structure=in_structure, units="K_CMB"
    )
    dust_template = DustOperator(
        nu_r,
        frequency0=f_c,
        temperature=20.,
        beta=params["beta_dust"],
        in_structure=in_structure,
        units="K_CMB"
    )
    # synchrotron_template = SynchrotronOperator(
    #     nu_r,
    #     frequency0=f_c,
    #     beta_pl=best_params["beta_pl"],
    #     in_structure=in_structure,units="K_CMB"
    # )
    A = MixingMatrixOperator(
        cmb=cmb_template, dust=dust_template
    )  # , synchrotron=synchrotron_template)
    return A


def get_bp(freq_c, NFREQ):
    nu_r = np.linspace(0.8, 1.2, NFREQ) * freq_c
    bp = np.zeros_like(nu_r)
    bp[nu_r > 0.925 * freq_c] = 1.
    bp[nu_r > 1.075 * freq_c] = 0.
    return nu_r, np.nan_to_num(bp, nan=0)


def get_bp_uKCMB(freq_c, NFREQ):
    nu_r = np.linspace(0.8, 1.2, NFREQ) * freq_c
    bp = np.zeros_like(nu_r)
    bp[nu_r > 0.925 * freq_c] = 1.
    bp[nu_r > 1.075 * freq_c] = 0.
    weights = bp / _jysr2rj(nu_r)
    weights /= _rj2cmb(nu_r)
    weights /= np.trapz(np.nan_to_num(weights, nan=0), nu_r * 1E9)
    bp_norm = [nu_r, np.nan_to_num(weights, nan=0)]
    return bp_norm


def get_Adust_fgbuster(nu_v):
    mixing_matrix = MixingMatrix(Dust(353.))
    mm_val = mixing_matrix.eval(nu_v, 1.6, 20.)
    return mm_val[:, 0]