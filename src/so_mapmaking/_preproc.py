"""Shared knowledge about preproc-config file layout and path resolution.

A sotodlib preproc config carries paths that are *relative to a root directory* and that
sotodlib resolves against the process cwd at load time. This single fact is needed in several
places (``furax-so-map``, ``furax-so-prepare``, ``furax-so-stage``); centralising it here keeps
them consistent. The convention:

- The config lives at ``<root>/preprocessing/satpy/cfg.yaml`` -- i.e. ``root`` is
  ``config.parents[PREPROC_ROOT_DEPTH]``.
- ``archive.index`` and ``context_file`` are relative to ``root``.
- The context's own db paths (``obsfiledb`` / ``obsdb`` / ``detdb`` and any ``metadata[].db``)
  are relative to the *context file's* directory, and may use ``{tag}`` placeholders.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PREPROC_ROOT_DEPTH = 2
"""Number of levels from a config file up to its root (``<root>/preprocessing/satpy/cfg.yaml``)."""

# Suffixes that mark a yaml string value as a filesystem path, used by find_relative_paths.
_PATH_SUFFIXES = ('.sqlite', '.db', '.h5', '.hdf5', '.g3', '.npy', '.fits', '.yaml', '.txt')


@dataclass(frozen=True)
class ConfigPath:
    """A path a preproc run will read, located both in the source text and on disk.

    Attributes:
        token: The path string exactly as written in the source file (keeps any ``{tag}``), so it
            can be found and replaced with a string operation.
        path: The resolved absolute path on disk.
        where: Which file's text carries ``token`` -- ``'config'`` or ``'context'``.
        kind: ``'index'`` (the archive index) or ``'db'`` (a sotodlib sqlite db).
    """

    token: str
    path: Path
    where: str
    kind: str


def config_root(config: str | Path) -> Path:
    """Return the root directory a preproc config's relative paths resolve against."""
    return Path(config).resolve().parents[PREPROC_ROOT_DEPTH]


def shared_root(layers: Iterable[str | Path]) -> Path:
    """Return the common root of all ``layers``, raising if they disagree.

    Only callers that ``chdir`` to a single root (e.g. ``furax-so-prepare``) need this; the
    map path normalises each layer independently and has no single-root constraint.
    """
    roots = {config_root(layer) for layer in layers}
    if len(roots) > 1:
        raise ValueError('all preproc configs must share the same root directory')
    return roots.pop()


def resolve(path: str, base: Path) -> Path:
    """Resolve ``path`` (absolute, or relative to ``base``) to an absolute Path."""
    p = Path(path)
    return p if p.is_absolute() else (base / p)


def expand_tags(token: str, tags: dict[str, str]) -> str:
    """Expand ``{tag}`` placeholders the way a sotodlib context does.

    The context's ``tags:`` block defines substitutions (e.g. ``{basedir}``, ``{manifestdir}``)
    used in db paths. Tags may reference other tags, so format repeatedly until stable.
    """
    prev = None
    while prev != token and '{' in token:
        prev = token
        try:
            token = token.format_map(tags)
        except (KeyError, IndexError, ValueError):
            break  # an unknown placeholder we can't expand; leave as-is
    return token


def collect_db_tokens(doc: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    """Return the literal sqlite path strings written under ``keys`` or in ``metadata[].db``."""
    tokens: list[str] = []
    for key in keys:
        val = doc.get(key)
        if isinstance(val, str):
            tokens.append(val)
    for meta in doc.get('metadata', []) or []:
        db = meta.get('db') if isinstance(meta, dict) else None
        if isinstance(db, str) and db.endswith(('.sqlite', '.db')):
            tokens.append(db)
    return tokens


def context_file(config_path: Path) -> tuple[str, Path]:
    """Return the config's ``context_file`` (literal token, resolved absolute path)."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    token = config['context_file']
    return token, resolve(token, config_path.parents[PREPROC_ROOT_DEPTH])


def iter_config_paths(config_path: Path) -> Iterator[ConfigPath]:
    """Yield every sqlite index a preproc run will read for ``config_path``.

    Covers the archive index and any config-level ``metadata`` db (resolved against the config
    root), plus the context's ``obsfiledb`` / ``obsdb`` / ``detdb`` and ``metadata`` dbs (resolved
    against the context dir, with tags expanded). The ``context_file`` itself is not yielded -- it
    is a yaml that is rewritten rather than copied; use :func:`context_file` for it.
    """
    config_path = config_path.resolve()
    root = config_path.parents[PREPROC_ROOT_DEPTH]
    config = yaml.safe_load(config_path.read_text())

    index_token = (config.get('archive') or {}).get('index')
    if isinstance(index_token, str):
        yield ConfigPath(index_token, resolve(index_token, root), 'config', 'index')

    # config-level metadata dbs (rare, but resolved against the config root like the index)
    for token in collect_db_tokens(config, ()):
        yield ConfigPath(token, resolve(token, root), 'config', 'db')

    _, ctx_path = context_file(config_path)
    ctx_doc = yaml.safe_load(ctx_path.read_text())
    ctx_dir = ctx_path.parent
    tags = ctx_doc.get('tags') or {}
    for token in collect_db_tokens(ctx_doc, ('obsfiledb', 'obsdb', 'detdb')):
        yield ConfigPath(token, resolve(expand_tags(token, tags), ctx_dir), 'context', 'db')


def normalize_config(config_path: Path, out_dir: Path) -> Path:
    """Write a copy of the preproc config with its cwd-sensitive paths made absolute.

    sotodlib abspath's the config's ``context_file`` and ``archive.index`` against the process
    cwd, so a relative config only loads from the right directory. Anchoring those two to the
    config root frees the run from any ``chdir`` (and from the single-root constraint a chdir
    imposes). The context's own db paths are left untouched: sotodlib resolves them relative to
    the context file, which this still points at, so they stay correct regardless of cwd.

    Returns the path of the normalised config written under ``out_dir`` (named with a hash of the
    source directory so init/proc layers never collide).
    """
    config_path = config_path.resolve()
    root = config_path.parents[PREPROC_ROOT_DEPTH]
    doc = yaml.safe_load(config_path.read_text())

    archive = doc.get('archive')
    index = archive.get('index') if isinstance(archive, dict) else None
    ctx = doc.get('context_file')

    # Already-absolute configs (e.g. those emitted by furax-so-stage) need no rewriting; skip the
    # copy and hand the file back as-is so cwd stays irrelevant either way.
    cwd_sensitive = [p for p in (index, ctx) if isinstance(p, str)]
    if all(Path(p).is_absolute() for p in cwd_sensitive):
        return config_path

    if isinstance(index, str):
        archive['index'] = resolve(index, root).as_posix()
    if isinstance(ctx, str):
        doc['context_file'] = resolve(ctx, root).as_posix()

    digest = hashlib.sha1(str(config_path.parent).encode()).hexdigest()[:8]
    out = out_dir / f'{digest}_{config_path.name}'
    out.write_text(yaml.safe_dump(doc))
    return out


def find_relative_paths(doc: Any) -> list[str]:
    """Return path-like string values in ``doc`` that are still relative (and tag-free).

    A diagnostic for normalised/patched configs: any leftover relative path will be resolved by
    sotodlib against the process cwd, which is rarely what the caller intends.
    """
    out: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if (
                '{' not in value
                and value.endswith(_PATH_SUFFIXES)
                and not Path(value).is_absolute()
            ):
                out.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(doc)
    return out
