r"""Plan a multi-observation run's bucket layout before submitting it.

The mapmaker groups observations into buckets padded to a common buffer shape, and rounds each
bucket up to a whole number of slots per device (see [`furax.mapmaking.layout`][]). How much that
padding costs depends on the observations' shapes, the device count and `max_buckets`. This command
reports the layout a run would choose for each device count and bucket budget, so the job can be
sized from real shapes instead of guessed.

Probing opens every observation once, as the mapmaker does, but runs no map-making, so it is best
done once and cached; the sweep is instant:

    furax-so-layout probe --init-config INIT --proc-config PROC --obsids-file OBS \
        --mapmaking-config MM --wafer ws0 --band f090 --cache shapes.json
    furax-so-layout sweep --cache shapes.json --devices 4,5,10,20 --max-buckets 1,2,4

`--devices` is the device count of the planned run (one per rank with the default CPU backend),
not the one the command itself runs on: it is a single process.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cyclopts import App

# furax is imported where used, once x64 is set: importing it with x64 off makes jax_healpy warn
if TYPE_CHECKING:
    from furax.mapmaking import ObservationBufferShape

app = App(help='Plan the bucket layout of a multi-observation run from probed observation shapes.')


@dataclass(frozen=True)
class LayoutRow:
    """The layout a run would choose for one device count and bucket budget."""

    n_devices: int
    max_buckets: int
    n_buckets: int
    n_slots: int
    slot_overhead: float
    """Empty slots as a fraction of the observation count."""
    padded_volume: int
    """Buffer volume summed over every slot, in detector-samples."""
    volume_overhead: float
    """Padded volume over the unpadded one, minus one: the cost of shape and slot padding."""
    peak_device_volume: int
    """The largest single bucket's buffer on one device, which is what has to fit in memory."""
    bucket_summaries: tuple[str, ...]


def sweep_layouts(
    shapes: Sequence['ObservationBufferShape'],
    devices: Sequence[int],
    max_buckets: Sequence[int],
) -> list[LayoutRow]:
    """The layout for every pair of device count and bucket budget, device count first."""
    from furax.mapmaking.layout import SlotLayout, real_volume

    unpadded = real_volume(shapes)
    rows = []
    for n_devices in devices:
        for budget in max_buckets:
            layout = SlotLayout.create(shapes, n_devices=n_devices, max_buckets=budget)
            # Buckets stream one after another, so a device's peak buffer is set by the largest
            # single bucket it holds, not by the run's total padded volume.
            peak = max(b.envelope.volume * (b.n_slots // n_devices) for b in layout.buckets)
            rows.append(
                LayoutRow(
                    n_devices=n_devices,
                    max_buckets=budget,
                    n_buckets=len(layout.buckets),
                    n_slots=layout.n_slots,
                    slot_overhead=layout.slot_overhead,
                    padded_volume=layout.padded_volume,
                    volume_overhead=layout.padded_volume / unpadded - 1,
                    peak_device_volume=peak,
                    bucket_summaries=tuple(b.summary(n_devices) for b in layout.buckets),
                )
            )
    return rows


def probe_shapes(
    observations: Sequence[Any], config: Any, logger: logging.Logger
) -> tuple[list['ObservationBufferShape'], list[int]]:
    """Every observation's buffer shape, as the mapmaker would bucket it, and those that failed.

    The mapmaker's own probe decides which fields are needed (hence whether scanning intervals are
    probed) and raises the sample count to the minimum its operators accept, so reusing it keeps
    the shapes identical to the run's. Observations that fail to probe are given the largest shape,
    as in the run.
    """
    from furax.mapmaking import MultiObservationMapMaker

    maker = MultiObservationMapMaker(observations, config=config, logger=logger)
    shapes, failed = maker._probe_shapes
    return list(shapes), [int(i) for i in failed.nonzero()[0]]


@app.command
def probe(
    *,
    init_config: Path,
    obsids_file: Path,
    cache: Path,
    proc_config: Path | None = None,
    mapmaking_config: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    downsample: int = 1,
    loglevel: str = 'info',
) -> None:
    """Probe the observations and cache their buffer shapes.

    Args:
        init_config: Base-layer preprocessing config file.
        obsids_file: Text file with one obsid per line.
        cache: Where to write the shapes (JSON).
        proc_config: Optional second-layer preprocessing config file.
        mapmaking_config: Mapmaking config file; decides which fields are probed.
        wafer: Wafer slot selection.
        band: Wafer bandpass selection.
        downsample: Downsampling factor applied after preprocessing.
        loglevel: Logging level (debug, info, warning, error).
    """
    import shutil
    import tempfile

    import jax

    # as in `furax-so-map`: x64 before the furax imports, the config's precision once loaded
    jax.config.update('jax_enable_x64', True)

    from furax.interfaces.sotodlib import LazyPreprocSOTODLibObservation
    from furax.mapmaking import MapMakingConfig

    from . import _preproc as pp
    from .util import detector_selection, resolve_obsids, setup_logger

    config = MapMakingConfig.load_yaml(mapmaking_config) if mapmaking_config else MapMakingConfig()
    jax.config.update('jax_enable_x64', config.double_precision)
    logger = setup_logger(loglevel, None, process_index=0)

    obsids = resolve_obsids(None, obsids_file)
    if not obsids:
        raise SystemExit('no observations to probe')

    # absolute copies of the cwd-sensitive configs, as `furax-so-map` makes
    stage_dir = Path(tempfile.mkdtemp(prefix='furax-so-layout-'))
    try:
        init = pp.normalize_config(init_config, stage_dir)
        proc = pp.normalize_config(proc_config, stage_dir) if proc_config else None
        det_select = detector_selection(wafer, band)
        observations = [
            LazyPreprocSOTODLibObservation(
                obs_id, init, proc, det_select, downsample, sotodlib_config=config.sotodlib
            )
            for obs_id in obsids
        ]
        logger.info(f'probing {len(observations)} observations')
        shapes, failed = probe_shapes(observations, config, logger)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    if failed:
        logger.warning(f'{len(failed)} observation(s) failed to probe')
    payload = {
        'obsids_file': str(obsids_file),
        'wafer': wafer,
        'band': band,
        'downsample': downsample,
        'mapmaking_config': str(mapmaking_config) if mapmaking_config else None,
        'failed': [obsids[i] for i in failed],
        'shapes': [list(shape) for shape in shapes],
    }
    cache.write_text(json.dumps(payload, indent=2))
    print(f'wrote {len(shapes)} shapes to {cache}')


def _format_volume(volume: int, bytes_per_element: float | None) -> str:
    if bytes_per_element is None:
        return f'{volume / 1e9:.2f}G'
    return f'{volume * bytes_per_element / 2**30:.1f}GiB'


@app.command
def sweep(
    *,
    cache: Path,
    devices: str = '1,2,4,5,8,10,16,20,25',
    max_buckets: str = '1,2,3,4',
    bytes_per_element: float | None = None,
    detail: bool = False,
) -> None:
    """Report the layout each (device count, bucket budget) pair would produce.

    Args:
        cache: Shapes written by `probe`.
        devices: Comma-separated device counts to try.
        max_buckets: Comma-separated bucket budgets to try.
        bytes_per_element: Bytes per detector-sample, to print volumes as memory. A previous
            run's log gives it: a bucket's logged `slot_size` over its envelope volume.
        detail: Also print each layout's per-bucket breakdown.
    """
    import jax

    jax.config.update('jax_enable_x64', True)

    from furax.mapmaking import ObservationBufferShape
    from furax.mapmaking.layout import real_volume

    payload = json.loads(cache.read_text())
    shapes = [ObservationBufferShape(*row) for row in payload['shapes']]
    rows = sweep_layouts(
        shapes,
        [int(d) for d in devices.split(',')],
        [int(b) for b in max_buckets.split(',')],
    )

    print(f'{len(shapes)} observations, unpadded volume ', end='')
    print(_format_volume(real_volume(shapes), bytes_per_element))
    detectors = [s.detector_count for s in shapes]
    samples = [s.sample_count for s in shapes]
    print(f'detectors {min(detectors)}-{max(detectors)}, samples {min(samples)}-{max(samples)}')
    print()
    header = (
        f'{"dev":>4} {"budget":>6} {"buckets":>7} {"slots":>5} {"slotpad":>8} '
        f'{"volume":>10} {"volpad":>7} {"peak/dev":>9}'
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        print(
            f'{row.n_devices:>4} {row.max_buckets:>6} {row.n_buckets:>7} {row.n_slots:>5} '
            f'{row.slot_overhead:>+7.1%} '
            f'{_format_volume(row.padded_volume, bytes_per_element):>10} '
            f'{row.volume_overhead:>+6.1%} '
            f'{_format_volume(row.peak_device_volume, bytes_per_element):>9}'
        )
        if detail:
            for b, summary in enumerate(row.bucket_summaries):
                print(f'        bucket {b}: {summary}')
    print()
    print('slotpad:  empty slots as a fraction of the observation count')
    print('volpad:   padded volume over unpadded, the cost of shape and slot padding')
    print('peak/dev: largest per-device bucket buffer, which is what has to fit in memory')


if __name__ == '__main__':
    app()
