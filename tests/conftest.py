import healpy as hp
import numpy as np
import pytest

from tests.helpers import TEST_DATA_PLANCK, TEST_DATA_SAT


def load_planck(nside: int) -> np.array:
    PLANCK_URL = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits'
    map_2048 = hp.read_map(PLANCK_URL, field=['I_STOKES', 'Q_STOKES', 'U_STOKES'])
    return hp.ud_grade(map_2048, nside)


@pytest.fixture(scope='session')
def planck_iqu_256() -> np.array:
    nside = 256
    path = TEST_DATA_PLANCK / f'HFI_SkyMap_143_{nside}_R3.01_full_IQU.fits'
    if path.exists():
        maps = hp.read_map(path, field=[0, 1, 2])
    else:
        maps = load_planck(nside)
        TEST_DATA_PLANCK.mkdir(parents=True, exist_ok=True)
        hp.write_map(path, maps)
    return maps.astype(float)


@pytest.fixture(scope='session')
def sat_nhits():
    nhits = hp.read_map(TEST_DATA_SAT / 'norm_nHits_SA_35FOV_G_nside512.fits')
    npixel = nhits.size
    nhits[: npixel // 2] = 0
    nhits /= np.sum(nhits)
    return nhits
