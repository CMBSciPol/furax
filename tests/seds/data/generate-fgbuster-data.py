import numpy as np

from furax.landscapes import StokesPyTree, HealpixLandscape
from pathlib import Path

try:
    from fgbuster import CMB, Dust, Synchrotron, get_observation, get_instrument
except ImportError:
    raise ImportError(
        'fgbuster is not installed. Please install it using `pip install git+https://github.com/fgbuster/fgbuster`'
    )


def write_fgbuster_data():
    data_filename = Path(__file__).parent / 'fgbuster_data.npz'
    nside = 32
    stokes_type = 'IQU'
    in_structure = HealpixLandscape(nside, stokes_type).structure

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

    d = StokesPyTree.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    return fg_data, d, in_structure


write_fgbuster_data()
