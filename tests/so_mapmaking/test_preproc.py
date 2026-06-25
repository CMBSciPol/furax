"""Path-resolution behaviour for so_mapmaking preproc configs (no JAX/sotodlib needed)."""

import sqlite3
from pathlib import Path

import pytest
import yaml

from so_mapmaking import _preproc as pp
from so_mapmaking.stage import Stager, stage_config


def _make_db(path: Path, *, manifest: bool) -> None:
    """Write a tiny sqlite db: a ManifestDb-style files(id, name) table, or a plain stub."""
    con = sqlite3.connect(path)
    if manifest:
        con.execute('create table files (id integer primary key, name text)')
        con.execute("insert into files (name) values ('data.h5')")  # relative companion
    else:
        con.execute('create table files (a text)')
    con.commit()
    con.close()


def _make_config(
    root: Path, *, context_dbs_relative: bool = True, extra: dict | None = None
) -> Path:
    """Build a `<root>/preprocessing/satpy/config.yaml` tree with relative internal paths."""
    cfg_dir = root / 'preprocessing' / 'satpy'
    cfg_dir.mkdir(parents=True)
    _make_db(root / 'preprocess.sqlite', manifest=True)
    _make_db(root / 'obsfiledb.sqlite', manifest=False)
    _make_db(root / 'obsdb.sqlite', manifest=False)

    obsfiledb = (
        'obsfiledb.sqlite' if context_dbs_relative else (root / 'obsfiledb.sqlite').as_posix()
    )
    context = {
        'tags': {'basedir': root.as_posix()},
        'obsfiledb': obsfiledb,
        'obsdb': '{basedir}/obsdb.sqlite',  # exercise tag expansion
        'metadata': [],
    }
    (root / 'context.yaml').write_text(yaml.safe_dump(context))

    config: dict = {
        'context_file': 'context.yaml',
        'archive': {'index': 'preprocess.sqlite'},
        'process_pipe': [],
    }
    if extra:
        config.update(extra)
    cfg = cfg_dir / 'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    return cfg


def test_iter_config_paths_resolves_index_and_context_dbs(tmp_path: Path) -> None:
    root = (tmp_path / 'vx').resolve()
    cfg = _make_config(root)
    by_token = {c.token: c for c in pp.iter_config_paths(cfg)}

    # archive index lives in the config and resolves against the root
    assert by_token['preprocess.sqlite'].where == 'config'
    assert by_token['preprocess.sqlite'].path == root / 'preprocess.sqlite'
    # context dbs live in the context; tags expand and relative names resolve against the context dir
    assert by_token['obsfiledb.sqlite'].where == 'context'
    assert by_token['obsfiledb.sqlite'].path == root / 'obsfiledb.sqlite'
    assert by_token['{basedir}/obsdb.sqlite'].path == root / 'obsdb.sqlite'
    # the context_file itself is not yielded (it is rewritten, not copied)
    assert all('context.yaml' not in c.token for c in pp.iter_config_paths(cfg))


def test_shared_root_accepts_common_root_and_rejects_mismatch(tmp_path: Path) -> None:
    init = _make_config(tmp_path / 'vx')
    proc_same = tmp_path / 'vx' / 'preprocessing' / 'satpy' / 'proc.yaml'
    proc_same.write_text((tmp_path / 'vx' / 'preprocessing' / 'satpy' / 'config.yaml').read_text())
    assert pp.shared_root([init, proc_same]) == (tmp_path / 'vx').resolve()

    proc_other = _make_config(tmp_path / 'vy')
    with pytest.raises(ValueError, match='same root directory'):
        pp.shared_root([init, proc_other])


def test_normalize_config_absolutizes_cwd_sensitive_paths(tmp_path: Path, monkeypatch) -> None:
    root = (tmp_path / 'vx').resolve()
    cfg = _make_config(root)
    out_dir = tmp_path / 'stage'
    out_dir.mkdir()

    # run from an unrelated cwd: normalisation must not depend on it
    monkeypatch.chdir(tmp_path)
    normalized = pp.normalize_config(cfg, out_dir)
    doc = yaml.safe_load(normalized.read_text())

    assert doc['archive']['index'] == (root / 'preprocess.sqlite').as_posix()
    assert doc['context_file'] == (root / 'context.yaml').as_posix()
    # context dbs are sotodlib's job (resolved relative to the context file) -> left untouched
    assert yaml.safe_load((root / 'context.yaml').read_text())['obsfiledb'] == 'obsfiledb.sqlite'
    assert doc['process_pipe'] == []  # unrelated keys preserved


def test_normalize_config_is_noop_for_absolute_config(tmp_path: Path) -> None:
    # an already-absolute config (e.g. emitted by furax-so-stage) is handed back unchanged
    cfg = _make_config(tmp_path / 'vx', context_dbs_relative=False)
    root = (tmp_path / 'vx').resolve()
    doc = yaml.safe_load(cfg.read_text())
    doc['archive']['index'] = (root / 'preprocess.sqlite').as_posix()
    doc['context_file'] = (root / 'context.yaml').as_posix()
    cfg.write_text(yaml.safe_dump(doc))

    out_dir = tmp_path / 'stage'
    out_dir.mkdir()
    assert pp.normalize_config(cfg, out_dir) == cfg.resolve()
    assert list(out_dir.iterdir()) == []  # no copy written


def test_normalize_config_distinct_names_for_same_filename(tmp_path: Path) -> None:
    init = _make_config(tmp_path / 'vx')
    proc = _make_config(tmp_path / 'vy')  # both named config.yaml, different roots
    out_dir = tmp_path / 'stage'
    out_dir.mkdir()
    assert pp.normalize_config(init, out_dir).name != pp.normalize_config(proc, out_dir).name


def test_stage_config_copies_and_repoints_to_dest(tmp_path: Path) -> None:
    root = tmp_path / 'vx'
    cfg = _make_config(root)
    dest = tmp_path / 'shm'
    dest.mkdir()

    out = stage_config(Stager(dest), cfg)
    doc = yaml.safe_load(out.read_text())
    ctx = yaml.safe_load((dest / f'context_{cfg.stem}.yaml').read_text())

    # config index + context_file now point inside dest
    assert Path(doc['archive']['index']).parent == dest.resolve()
    assert Path(doc['context_file']).parent == dest.resolve()
    # context dbs (incl. the tagged one) repointed into dest
    assert Path(ctx['obsfiledb']).parent == dest.resolve()
    assert Path(ctx['obsdb']).parent == dest.resolve()
    # ManifestDb companion .h5 anchored back to its original dir (data stays on shared FS)
    staged_index = next(dest.glob('*preprocess.sqlite'))
    con = sqlite3.connect(staged_index)
    name = con.execute('select name from files').fetchone()[0]
    con.close()
    assert name == (root.resolve() / 'data.h5').as_posix()


def test_stage_config_warns_on_leftover_relative_path(tmp_path: Path, capsys) -> None:
    cfg = _make_config(tmp_path / 'vx', extra={'stray': 'unhandled.h5'})
    dest = tmp_path / 'shm'
    dest.mkdir()
    stage_config(Stager(dest), cfg)
    assert 'unhandled.h5' in capsys.readouterr().err


def test_find_relative_paths_flags_only_relative_tagfree_paths() -> None:
    doc = {
        'abs': '/data/x.h5',  # absolute -> ok
        'rel': 'sub/y.sqlite',  # relative path -> flagged
        'tagged': '{base}/z.db',  # unexpanded tag -> skipped
        'plain': 'not_a_path',  # no path suffix -> skipped
        'nested': [{'deep': 'deep.fits'}],
    }
    assert sorted(pp.find_relative_paths(doc)) == ['deep.fits', 'sub/y.sqlite']
