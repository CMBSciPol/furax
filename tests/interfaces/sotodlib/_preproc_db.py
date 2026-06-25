"""Build a minimal sotodlib preprocessing database on the fly from saved AxisManagers.

This lets the preproc-db loading path (``load_and_preprocess`` and the Context machinery
behind it) run for real in tests, without checking in a multi-GB Context + raw data tree.
Each saved observation is split into a raw part (served by a test obs loader) and a
preprocess archive (ManifestDb + hdf5), wired together by a Context.
"""

from pathlib import Path

import sotodlib.io.metadata  # noqa: F401  registers the hdf5 metadata loaders (AxisManagerHdf)
import yaml
from sotodlib.core import OBSLOADER_REGISTRY, AxisManager, metadata

# obs_id -> raw observation file, consulted by the registered obs loader below.
_RAW_FILES: dict[str, str] = {}


def _furax_test_loader(_obsfiledb, obs_id, dets=None, samples=None, no_signal=None, **_kwargs):  # type: ignore[no-untyped-def]
    """Serve the raw observation from a saved AxisManager file (test obs loader).

    Mimics a real obs loader: returns signal + geometry only, leaving obs_info and the
    preprocess products to the metadata system. Honours dets/samples/no_signal selection.
    """
    am = AxisManager.load(_RAW_FILES[obs_id])
    for field in ('preprocess', 'obs_info'):
        if field in am._fields:
            am.move(field, None)
    if no_signal and 'signal' in am._fields:
        am.move('signal', None)
    if dets is not None:
        am.restrict('dets', list(dets))
    if samples is not None:
        am.restrict('samps', slice(samples[0], samples[1]))
    return am


OBSLOADER_REGISTRY.setdefault('furax_test', _furax_test_loader)


def build_preproc_db(root: Path, files: list[Path]) -> Path:
    """Materialise a minimal preproc db for ``files`` under ``root``; return the config path.

    The returned config is exactly what ``from_preproc_group`` / ``furax-so-map
    --init-config`` consume.
    """
    obsfiledb = metadata.ObsFileDb(map_file=(root / 'obsfiledb.sqlite').as_posix())
    obsdb = metadata.ObsDb()
    obsdb.add_obs_columns(['timestamp float', 'telescope string'])

    archive_h5 = root / 'preprocess_archive.h5'
    scheme = metadata.ManifestScheme()
    scheme.add_exact_match('obs:obs_id')
    scheme.add_data_field('loader')
    scheme.add_data_field('dataset')
    mdb = metadata.ManifestDb((root / 'preprocess.sqlite').as_posix(), scheme=scheme)

    for i, file in enumerate(files):
        am = AxisManager.load(Path(file).as_posix())
        obs_id = str(am.obs_info.obs_id)
        _RAW_FILES[obs_id] = Path(file).resolve().as_posix()

        obsfiledb.add_detset(f'detset{i}', [str(v) for v in am.dets.vals])
        obsfiledb.add_obsfile(_RAW_FILES[obs_id], obs_id, f'detset{i}', 0, am.samps.count)
        obsdb.update_obs(
            obs_id,
            {'timestamp': float(am.timestamps[0]), 'telescope': str(am.obs_info.telescope)},
        )

        preprocess = am.preprocess
        # pcore.Pipeline.run reads frequency_cutoffs; provide an empty one for the no-op pipe
        if 'frequency_cutoffs' not in preprocess._fields:
            preprocess.wrap('frequency_cutoffs', AxisManager())
        preprocess.save(archive_h5.as_posix(), obs_id, overwrite=True)
        mdb.add_entry(
            {'obs:obs_id': obs_id, 'loader': 'AxisManagerHdf', 'dataset': obs_id},
            archive_h5.as_posix(),
        )

    obsfiledb.to_file((root / 'obsfiledb.sqlite').as_posix())
    obsdb.to_file((root / 'obsdb.sqlite').as_posix())
    mdb.to_file((root / 'preprocess.sqlite').as_posix())

    context = {
        'tags': {},
        'obsfiledb': (root / 'obsfiledb.sqlite').as_posix(),
        'obsdb': (root / 'obsdb.sqlite').as_posix(),
        'obs_loader_type': 'furax_test',
        'metadata': [],
    }
    (root / 'context.yaml').write_text(yaml.safe_dump(context))

    config = {
        'context_file': (root / 'context.yaml').as_posix(),
        'archive': {'index': (root / 'preprocess.sqlite').as_posix()},
        'process_pipe': [],
    }
    config_path = root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return config_path
