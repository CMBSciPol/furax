"""Build a minimal sotodlib preprocessing database on the fly from saved AxisManagers.

This lets the preproc-db loading path (``load_and_preprocess`` and the Context machinery
behind it) run for real in tests, without checking in a multi-GB Context + raw data tree.
Each saved observation is split into a raw part (served by a test obs loader) and a
preprocess archive (ManifestDb + hdf5), wired together by a Context.
"""

from pathlib import Path

import numpy as np
import pytest
import so3g.proj
import sotodlib.io.metadata  # noqa: F401  registers the hdf5 metadata loaders (AxisManagerHdf)
import yaml
from sotodlib.core import OBSLOADER_REGISTRY, AxisManager, Context, metadata
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.preprocess import preprocess_util as pu

# obs_id -> raw observation file, consulted by the registered obs loader below.
_RAW_FILES: dict[str, str] = {}


def _furax_test_loader(_obsfiledb, obs_id, dets=None, samples=None, no_signal=None, **_kwargs):
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


def build_preproc_db(
    root: Path,
    files: list[Path],
    n_detsets: int = 1,
    n_uncut: dict[str, int] | None = None,
) -> Path:
    """Materialise a minimal preproc db for ``files`` under ``root``; return the config path.

    The returned config is exactly what ``from_preproc_group`` / ``furax-so-map
    --init-config`` consume. The detectors of each observation are split into ``n_detsets``
    detsets, its preprocessing groups.

    `n_uncut` maps an observation id to the number of its leading detectors that survive the
    cuts, stored as `valid_data`; the others have no valid sample. Without an entry, none is cut.
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

        for j, dets in enumerate(np.array_split([str(v) for v in am.dets.vals], n_detsets)):
            obsfiledb.add_detset(f'detset{i}_{j}', list(dets))
            # one file per detset, by name only: the test loader reads _RAW_FILES
            name = f'{_RAW_FILES[obs_id]}#{j}'
            obsfiledb.add_obsfile(name, obs_id, f'detset{i}_{j}', 0, am.samps.count)
        obsdb.update_obs(
            obs_id,
            {'timestamp': float(am.timestamps[0]), 'telescope': str(am.obs_info.telescope)},
        )

        preprocess = am.preprocess
        # pcore.Pipeline.run reads frequency_cutoffs; provide an empty one for the no-op pipe
        if 'frequency_cutoffs' not in preprocess._fields:
            preprocess.wrap('frequency_cutoffs', AxisManager())
        if n_uncut and obs_id in n_uncut:
            _wrap_valid_data(preprocess, n_uncut[obs_id])
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
        'subobs': {'use': 'detset'},
        'process_pipe': [],
    }
    config_path = root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _wrap_valid_data(preprocess: AxisManager, n_uncut: int) -> None:
    """Store the cuts as sotodlib does: a `valid_data` flag of the samples each detector keeps."""
    n_samples = preprocess.samps.count
    ranges = []
    for i in range(preprocess.dets.count):
        valid = so3g.proj.Ranges(n_samples)
        if i < n_uncut:
            valid.add_interval(0, n_samples)
        ranges.append(valid)
    flags = AxisManager(preprocess.dets, preprocess.samps)
    flags.wrap('valid_data', so3g.proj.RangesMatrix(ranges), [(0, 'dets'), (1, 'samps')])
    preprocess.wrap('valid_data', flags)


def drop_group_record(monkeypatch: pytest.MonkeyPatch, detset: str) -> None:
    """Make `Context.get_meta` find no preprocessing record for the group of a detset."""
    get_meta = Context.get_meta

    def get_meta_without_group(self, *args, dets=None, **kwargs):
        if dets is not None and dets.get('detset') == detset:
            raise LoaderError('There were no metadata records found for this observation.')
        return get_meta(self, *args, dets=dets, **kwargs)

    monkeypatch.setattr(Context, 'get_meta', get_meta_without_group)


def cut_group(monkeypatch: pytest.MonkeyPatch, detset: str) -> list[str]:
    """Make preprocessing cut every detector of the group of a detset.

    Returns the list that records the detsets each load was asked for.
    """
    load_and_preprocess = pu.load_and_preprocess
    loaded: list[str] = []

    def load_without_group(*args, dets=None, **kwargs):
        loaded.append((dets or {}).get('detset'))
        if dets is not None and dets.get('detset') == detset:
            return None
        return load_and_preprocess(*args, dets=dets, **kwargs)

    monkeypatch.setattr(pu, 'load_and_preprocess', load_without_group)
    return loaded
