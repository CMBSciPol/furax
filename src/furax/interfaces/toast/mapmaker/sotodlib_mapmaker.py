import argparse
import logging
import os

import jax
import numpy as np
import pixell.enmap
import yaml
from sotodlib.core import AxisManager
from sotodlib.preprocess.preprocess_util import init_logger, load_and_preprocess
from sotodlib.site_pipeline.util import main_launcher

from .sotodlib_utils import binned_demod_mapmaker, binned_mapmaker, ml_mapmaker, two_step_mapmaker

logger = init_logger('preprocess')


def main(
    preprocess_config: str | dict | None,
    mapmaking_config: str | dict | None,
    obs_id: str | None = None,
    det_select: dict[str, str] | None = None,
    verbosity: int = 3,
    binary_filepath: str | None = None,
    obs: AxisManager | None = None,
    output_path: str | None = None,
):
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
        mapmaking_config = yaml.safe_load(open(mapmaking_config))
        logger.info('Mapmaking config loaded from file')

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

    # Make maps
    map_results = make_maps(obs, mapmaking_config, logger)
    logger.info('Mapmaking finished')

    # Save results
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

        for key, m in map_results.items():
            if isinstance(m, pixell.enmap.ndmap):
                pixell.enmap.write_map(f'{output_path}/{key}.hdf', m, allow_modify=True)
            elif isinstance(m, jax.Array) or isinstance(m, np.ndarray):
                np.save(f'{output_path}/{key}.npy', np.array(m))
            else:
                continue
            logger.info(f'Mapmaking result [{key}] saved to file')

        with open(f'{output_path}/preprocess_config.yaml', 'w') as f:
            yaml.dump(preprocess_config, f, default_flow_style=False)
            logger.info('Preproces config saved to file')

        with open(f'{output_path}/mapmaking_config.yaml', 'w') as f:
            yaml.dump(mapmaking_config, f, default_flow_style=False)
            logger.info('Mapmaking config saved to file')

    return map_results


def make_maps(
    obs: AxisManager,
    config: dict,
    logger: logging.Logger,
):
    demodulated = config.get('demodulated', True)
    binned = config.get('binned', True)
    has_templates = 'template' in config.keys()

    if binned and demodulated:
        return binned_demod_mapmaker(obs, config, logger=logger)

    if binned and not demodulated:
        if has_templates:
            return two_step_mapmaker(obs, config, logger=logger)
        else:
            return binned_mapmaker(obs, config, logger=logger)

    if not binned and not demodulated:
        return ml_mapmaker(obs, config, logger=logger)

    raise NotImplementedError('Specified mapmaker is currently not implemented')


def get_parser(parser=None):
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


if __name__ == '__main__':
    main_launcher(main, get_parser)
