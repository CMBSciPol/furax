import argparse
import os
import pickle
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from sotodlib.core import AxisManager
from sotodlib.preprocess.preprocess_util import init_logger, load_and_preprocess
from sotodlib.site_pipeline.util import main_launcher

from furax.interfaces.sotodlib.observation import SOTODLibObservation
from furax.mapmaking.config import MapMakingConfig
from furax.mapmaking.mapmaker import MapMaker

logger = init_logger('preprocess')


def main(
    preprocess_config: str | dict[str, Any] | None,
    mapmaking_config: str | MapMakingConfig,
    obs_id: str | None = None,
    det_select: dict[str, str] | None = None,
    verbosity: int = 3,
    binary_filepath: str | None = None,
    obs: AxisManager | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """
    Mapmaking script for SO observation data.
    Args
    ----
    preprocess_config : str or dict
        Preprocessing configuration dictionary or path to a yaml file.
    mapmaking_config : str or dict
        Mapmaking configuration dictionary or path to a yaml file.
    obs_id : str
        Observation id.
        e.g., obs_1714550584_satp3_1111111.
    det_select : dict or None
        If specified, select a subset of detectors
        e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}.
    verbosity : int
        verboisity level of logging.Logger.
    binary_filepath : str or None
        If specified, load the observation from a binary file instead.
        Overrides preprocess_config, obs_id, det_select.
    obs : AxisManager or None
        If specified, use the observation object passed.
        Overrides preprocess_config, obs_id, det_select, binary_filepath.
    output_path : str or None
        If specified, create directories leading to the path and
        save the output map and configuration files.

    Returns
    -------
    map_results : dict
        Product of the mapmaking pipeline.
        The contents vary depending on the mapmaker used.
        See individual mapmakers for details on the outputs.
    """

    logger = init_logger('preprocess', verbosity=verbosity)
    logger.info('Initialised logger')

    if isinstance(mapmaking_config, str):
        mapmaking_config = MapMakingConfig.load_yaml(path=mapmaking_config)
        logger.info('Mapmaking config loaded from file')
    assert isinstance(mapmaking_config, MapMakingConfig), 'Bad mapmaking config file'

    maker = MapMaker.from_config(config=mapmaking_config, logger=logger)

    if obs is None:
        if binary_filepath is not None:
            # If provided, load the observation from a binary file
            obs = AxisManager.load(binary_filepath)
            logger.info(f'Observation data loaded from {binary_filepath}')

        else:
            if isinstance(preprocess_config, str):
                preprocess_config = yaml.safe_load(open(preprocess_config))
                logger.info('Preprocessing config loaded from file')

            obs = load_and_preprocess(obs_id, preprocess_config, dets=det_select)
            logger.info('Observationa data loaded')
    observation = SOTODLibObservation(data=obs)

    # Make maps
    results = maker.run(observation=observation, out_dir=output_path)
    logger.info('Mapmaking finished')

    # Save results
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        with open(f'{output_path}/preprocess_config.yaml', 'w') as f:
            yaml.dump(preprocess_config, f, default_flow_style=False)
            logger.info('Preproces config saved to file')

    return results


def load_result(result_path: str) -> dict[str, Any]:
    # Load results from a directory into a dictionary

    if not os.path.isdir(result_path):
        raise ValueError(f"Provided path '{result_path}' is not a directory.")

    results = {}
    for filename in os.listdir(result_path):
        file_path = os.path.join(result_path, filename)

        if os.path.isfile(file_path):
            fn, extension = os.path.splitext(filename)
            if extension == '.yaml':
                results[fn] = yaml.safe_load(open(file_path))
            elif extension == '.npy':
                results[fn] = np.load(file_path)
            elif extension == '.pkl':
                with open(file_path, 'rb') as f:
                    results[fn] = pickle.load(f)
            elif filename == 'wcs.fits':
                with fits.open(file_path) as hdul:
                    results['wcs'] = WCS(hdul[0].header)

    return results


def get_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('preprocess_config', help='Preprocessing configuration file')
    parser.add_argument('mapmaking_config', help='Mapmaking configuration file')
    parser.add_argument('--obs-id', help='obs-id of the observation')
    # TODO: add support for det_select

    parser.add_argument(
        '--verbosity',
        help='Output verbosity level. 0:Error, 1:Warning, 2:Info(default), 3:Debug',
        default=3,
        type=int,
    )
    parser.add_argument(
        '--binary-filepath',
        help='Optional path to binary file containing observation',
        default=None,
    )
    parser.add_argument('--output-path', help='File output path', default=None)

    return parser


def main_cli() -> None:
    main_launcher(main, get_parser)


if __name__ == '__main__':
    main_cli()
