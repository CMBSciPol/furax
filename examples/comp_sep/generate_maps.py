# FGBUSTER IMPORTS

from fgbuster import (
    get_observation,
    get_instrument,
)


import os
import pickle

def save_to_cache(nside, noise=False , instrument_name = 'LiteBIRD'):

    instrument = get_instrument(instrument_name)
    # Define cache file path
    cache_dir = 'freq_maps_cache'
    os.makedirs(cache_dir, exist_ok=True)
    noise_str = 'noise' if noise else 'no_noise'
    cache_file = os.path.join(cache_dir, f'freq_maps_nside_{nside}_{noise_str}.pkl')

    # Check if file exists, load if it does, otherwise create and save it
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            freq_maps = pickle.load(f)
        print(f'Loaded freq_maps for nside {nside} from cache.')
    else:
        # Generate freq_maps if not already cached
        freq_maps = get_observation(instrument, 'c1d0s0', nside=nside, noise=noise)

        # Save freq_maps to the cache
        with open(cache_file, 'wb') as f:
            pickle.dump(freq_maps, f)
        print(f'Generated and saved freq_maps for nside {nside}.')


def load_from_cache(nside , noise=False , instrument_name = 'LiteBIRD'):
    # Define cache file path
    instrument = get_instrument(instrument_name)
    noise_str = 'noise' if noise else 'no_noise'
    cache_dir = 'freq_maps_cache'
    cache_file = os.path.join(cache_dir, f'freq_maps_nside_{nside}_{noise_str}.pkl')

    # Check if file exists and load if it does; otherwise raise an error with guidance
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            freq_maps = pickle.load(f)
        print(f'Loaded freq_maps for nside {nside} from cache.')
    else:
        raise FileNotFoundError(
            f'Cache file for freq_maps with nside {nside} not found.\n'
            f'Please generate it first by calling `generate_maps({nside})`.'
        )


    return instrument['frequency'].values , freq_maps


