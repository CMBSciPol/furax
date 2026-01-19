import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from sotodlib.core import AxisManager
from sotodlib.preprocess.preprocess_util import init_logger

from furax.interfaces.sotodlib.observation import SOTODLibObservation
from furax.mapmaking.config import MapMakingConfig
from furax.mapmaking.mapmaker import MapMaker

logger = init_logger('preprocess')


def main(
    preprocess_config: str | Path | dict[str, Any] | None,
    mapmaking_config: str | Path | MapMakingConfig,
    obs_id: str | None = None,
    det_select: dict[str, str] | None = None,
    verbosity: int = 3,
    binary_filepath: str | Path | None = None,
    obs: AxisManager | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Mapmaking script for SO observation data.

    Args
    ----
    preprocess_config : path or dict, or None
        Preprocessing configuration dictionary or path to a yaml file
    mapmaking_config : path or dict
        Mapmaking configuration dictionary or path to a yaml file
    obs_id : str
        Observation id.
        e.g., obs_1714550584_satp3_1111111.
    det_select : dict, optional
        If specified, select a subset of detectors
        e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}.
    verbosity : int
        verbosity level of logging.Logger
    binary_filepath : path, optional
        If specified, load the observation from a binary file instead.
        Overrides preprocess_config, obs_id, det_select.
    obs : AxisManager, optional
        If specified, use the observation object passed.
        Overrides preprocess_config, obs_id, det_select, binary_filepath.
    output_path : path, optional
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

    # Load map-making config
    if not isinstance(mapmaking_config, MapMakingConfig):
        mapmaking_config = MapMakingConfig.load_yaml(path=mapmaking_config)
        assert isinstance(mapmaking_config, MapMakingConfig), 'Bad mapmaking config file'
        logger.info('Mapmaking config loaded from file')

    # Create map-maker
    maker = MapMaker.from_config(mapmaking_config, logger=logger)

    # Load observation
    if obs is None:
        if binary_filepath is not None:
            observation = SOTODLibObservation.from_file(binary_filepath)
            logger.info(f'Observation data loaded from {binary_filepath}')
        else:
            if obs_id is None or preprocess_config is None:
                msg = 'obs_id and preprocess_config must be specified if obs is not given'
                raise ValueError(msg)
            observation = SOTODLibObservation.from_preprocess(preprocess_config, obs_id, det_select)
            logger.info('Observation data loaded from preprocessing config')
    else:
        observation = SOTODLibObservation(obs)

    # Make maps
    results = maker.run(observation=observation, out_dir=output_path)
    logger.info('Mapmaking finished')

    # Save preprocessing config
    if output_path is not None and preprocess_config is not None:
        dest = Path(output_path) / 'preprocess_config.yaml'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.dump(preprocess_config, default_flow_style=False))
        logger.info('Preprocess config saved to file')

    return results


def load_result(result_path: str | Path) -> dict[str, Any]:
    """Load results from a directory into a dictionary."""

    if not isinstance(result_path, Path):
        result_path = Path(result_path)

    if not result_path.is_dir():
        raise ValueError(f"Provided path '{result_path.as_posix()}' is not a directory.")

    results = {}
    for path in result_path.iterdir():
        if path.is_file():
            filename, extension = path.stem, path.suffix
            if extension == '.yaml':
                results[filename] = yaml.safe_load(path.read_text())
            elif extension == '.npy':
                results[filename] = np.load(path)
            elif extension == '.pkl':
                with path.open('rb') as f:
                    results[filename] = pickle.load(f)
            elif path.name == 'wcs.fits':
                with fits.open(path) as hdul:
                    results[filename] = WCS(hdul[0].header)
    return results


def get_parser(parser: ArgumentParser | None = None) -> ArgumentParser:
    if parser is None:
        parser = ArgumentParser()

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
        '--binary-filepath', help='Optional path to binary file containing observation'
    )
    parser.add_argument('--output-path', help='Path where to save results')
    return parser


def main_cli() -> None:
    #main_launcher(main, get_parser)
    # Script currently not supported due to an error in packaged sotodlib
    pass


if __name__ == '__main__':
    main_cli()
