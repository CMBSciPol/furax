import argparse
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from sotodlib.preprocess.preprocess_util import init_logger, load_and_preprocess
from sotodlib.site_pipeline.util import main_launcher

from furax.interfaces.sotodlib.observation import SotodlibObservationData
from furax.mapmaking.config import MapMakingConfig
from furax.mapmaking.mapmaker import MapMaker

logger = init_logger('preprocess')


def main(
    preprocess_config: str | dict[str, Any] | None,
    mapmaking_configs: list[str],
    obs_ids: list[str],
    wafer_slot: str | None = None,
    wafer_bandpass: str | None = None,
    verbosity: int = 3,
    output_path: str | None = None,
    log_path: str | None = None,
) -> None:
    """
    Mapmaking script for SO observation data.
    Args
    ----
    preprocess_config : str or dict
        Preprocessing configuration dictionary or path to a yaml file.
    mapmaking_configs : list
        Paths to mapmaking configuration yaml files.
    obs_ids : list
        Observation ids.
        e.g., ['obs_1714550584_satp3_1111111']
    wafer_slot : str or None
        If specified, select a subset of wafers
        e.g. 'ws0'
    wafer_bandpass : str or None
        If specified, select a subset of bandpasses
        e.g. 'f150'
    verbosity : int
        verboisity level of logging.Logger.
    output_path : str or None
        If specified, create directories leading to the path and
        save the output map and configuration files.
    log_path : str or None
        If specified, save output logs in a file of the given path.

    Returns
    -------
    map_results : dict
        Product of the mapmaking pipeline.
        The contents vary depending on the mapmaker used.
        See individual mapmakers for details on the outputs.
    """

    logger = init_logger('preprocess', verbosity=verbosity)
    logger.info('Initialised logger')

    logger.info(f'{len(obs_ids)} observations in the list')
    logger.info(f'{len(mapmaking_configs)} mapmakers')
    logger.info(mapmaking_configs)

    if wafer_slot is None and wafer_bandpass is None:
        det_select = None
    else:
        det_select = {}
        if wafer_slot:
            det_select['wafer_slot'] = wafer_slot
        if wafer_bandpass:
            det_select['wafer.bandpass'] = wafer_bandpass

    if log_path is not None:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    preprocess_config = yaml.safe_load(open(preprocess_config))  # type: ignore[arg-type]
    logger.info('Preprocessing config loaded from file')

    for obs_id in obs_ids:
        # First check if we need to load this obs or not
        completed = True
        for mapmaking_config in mapmaking_configs:
            output_dir = get_output_dir(
                root_dir=output_path,  # type: ignore[arg-type]
                mapmaking_config=mapmaking_config,
                obs_id=obs_id,
                det_select=det_select,
            )
            if not os.path.exists(output_dir):
                completed = False
                break
        if completed:
            logger.info(f'Skipping {obs_id}...')
            continue

        # Load obs
        try:
            obs = load_and_preprocess(obs_id, preprocess_config, dets=det_select)
            logger.info('Observationa data loaded')
            observation = SotodlibObservationData(observation=obs)
        except Exception as exception:
            logger.info(f'Loading failed for {obs_id}')
            logger.info(exception)
            continue

        for mapmaking_config in mapmaking_configs:
            # First check if we need to do mapmaking or not
            mm_name = os.path.basename(mapmaking_config)
            output_dir = get_output_dir(
                root_dir=output_path,  # type: ignore[arg-type]
                mapmaking_config=mapmaking_config,
                obs_id=obs_id,
                det_select=det_select,
            )
            if os.path.exists(output_dir):
                # Skip if the data exist already
                logger.info(f'Skipping [{mm_name}] on {obs_id}...')
                continue

            # Load mapmaking config
            config = MapMakingConfig.load_yaml(path=mapmaking_config)
            logger.info(f'Mapmaking config [{mm_name}] loaded from file')

            # Set up mapmaker and output path
            maker = MapMaker.from_config(config=config, logger=logger)

            # Make maps
            try:
                maker.make_maps(observation=observation, out_dir=output_dir)
                logger.info('Mapmaking finished')
                logger.info(f'Output directory: {output_dir}')
            except Exception as exception:
                logger.info(f'Mapmaking failed for [{mm_name}] on {obs_id}')
                logger.info(exception)
                continue

        logger.info(f'Finished mapmaking for {obs_id}')
    logger.info('Finished mapmaking for all observations provided.')

    return


def get_output_dir(
    root_dir: str, mapmaking_config: str, obs_id: str, det_select: dict[str, Any] | None = None
) -> str:
    name = Path(mapmaking_config).stem
    output_dir = f'{root_dir}/{name}/{obs_id}'
    if det_select is not None:
        output_dir = f'{output_dir}/{det_select["wafer.bandpass"]}/{det_select["wafer_slot"]}'
    return output_dir


def get_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--preprocess-config', help='Preprocessing configuration file')
    parser.add_argument('--mapmaking-configs', nargs='+', help='Mapmaking configuration file(s)')
    parser.add_argument('--obs-ids', nargs='+', help='Observation id(s)')
    parser.add_argument('--wafer-slot', help='wafer slot selection', default=None)
    parser.add_argument('--wafer-bandpass', help='wafer bandpass selection', default=None)

    parser.add_argument(
        '--verbosity',
        help='Output verbosity level. 0:Error, 1:Warning, 2:Info(default), 3:Debug',
        default=3,
        type=int,
    )
    parser.add_argument('--output-path', help='File output path', default=None)
    parser.add_argument('--log-path', help='Log output path', default=None)

    return parser


def main_cli() -> None:
    main_launcher(main, get_parser)


if __name__ == '__main__':
    main_cli()
