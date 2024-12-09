from typing import get_args

import os
import healpy as hp
import jax
import numpy as np
import pytest
from jaxtyping import Array, Float

from furax.landscapes import StokesPyTree, StokesIQUPyTree, ValidStokesType, HealpixLandscape
from tests.helpers import TEST_DATA_PLANCK, TEST_DATA_SAT, TEST_DATA_FGBUSTER


def load_planck(nside: int) -> np.array:
    PLANCK_URL = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits'
    map_2048 = hp.read_map(PLANCK_URL, field=['I_STOKES', 'Q_STOKES', 'U_STOKES'])
    return hp.ud_grade(map_2048, nside)


@pytest.fixture(scope='session')
def planck_iqu_256() -> StokesIQUPyTree:
    nside = 256
    path = TEST_DATA_PLANCK / f'HFI_SkyMap_143_{nside}_R3.01_full_IQU.fits'
    if path.exists():
        maps = hp.read_map(path, field=[0, 1, 2])
    else:
        maps = load_planck(nside)
        TEST_DATA_PLANCK.mkdir(parents=True, exist_ok=True)
        hp.write_map(path, maps)
    i, q, u = maps.astype(float)
    return StokesIQUPyTree(
        i=jax.device_put(i),
        q=jax.device_put(q),
        u=jax.device_put(u),
    )


@pytest.fixture(scope='session')
def sat_nhits() -> Float[Array, '...']:
    nhits = hp.read_map(TEST_DATA_SAT / 'norm_nHits_SA_35FOV_G_nside512.fits').astype('<f8')
    npixel = nhits.size
    nhits[: npixel // 2] = 0
    nhits /= np.sum(nhits)
    return jax.device_put(nhits)


@pytest.fixture(params=get_args(ValidStokesType))
def stokes(request: pytest.FixtureRequest) -> ValidStokesType:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param


@pytest.fixture(scope='session')
def get_fgbuster_data():
    TEST_DATA_FGBUSTER.mkdir(exist_ok=True)
    # Check if file already exists
    data_filename = f'{TEST_DATA_FGBUSTER}/fgbuster_data.npz'
    nside = 32
    stokes_type = 'IQU'
    in_structure = HealpixLandscape(nside, stokes_type).structure
    try:
        # If the file already exists, we can skip data generation
        fg_data = np.load(data_filename)
        print(f"Data file '{data_filename}' already exists, skipping generation.")
        freq_maps: Array = fg_data['freq_maps']
        d = StokesPyTree.from_stokes(
            I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :]
        )
        return fg_data, d, in_structure
    except FileNotFoundError:
        try:
            from fgbuster import CMB, Dust, Synchrotron, get_observation, get_instrument
        except ImportError:
            raise ImportError(
                'fgbuster is not installed. Please install it using `pip install fgbuster`'
            )
        instrument = get_instrument('LiteBIRD')
        freq_maps = get_observation(instrument, 'c1d0s0', nside=nside)
        nu = instrument['frequency'].values

        # Generate FGBuster components
        cmb_fgbuster_K_CMB = CMB().eval(nu)
        dust_fgbuster_K_CMB = Dust(150.0).eval(nu, 1.54, 20.0)
        synchrotron_fgbuster_K_CMB = Synchrotron(20.0).eval(nu, -3.0)

        cmb_fgbuster_K_RJ = CMB(units='K_RJ').eval(nu)
        dust_fgbuster_K_RJ = Dust(150.0, units='K_RJ').eval(nu, 1.54, 20.0)
        synchrotron_fgbuster_K_RJ = Synchrotron(20.0, units='K_RJ').eval(nu, -3.0)

        fg_data = {
            'frequencies': nu,
            'freq_maps': freq_maps,
            'CMB_K_CMB': cmb_fgbuster_K_CMB,
            'DUST_K_CMB': dust_fgbuster_K_CMB,
            'SYNC_K_CMB': synchrotron_fgbuster_K_CMB,
            'CMB_K_RJ': cmb_fgbuster_K_RJ,
            'DUST_K_RJ': dust_fgbuster_K_RJ,
            'SYNC_K_RJ': synchrotron_fgbuster_K_RJ,
        }
        # Save all required arrays to an .npz file
        np.savez(data_filename, **fg_data)
        print(f"Data saved to '{data_filename}'")

        d = StokesPyTree.from_stokes(
            I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :]
        )
        return fg_data, d, in_structure
