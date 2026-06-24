#!/usr/bin/env python
"""Stage a preproc run's sqlite indices to node-local storage and emit patched configs.

The furax lazy-preproc reader opens the obsdb / obsfiledb / archive sqlite indices on every
observation load. This helper copies the (small) indices to a node-local directory (e.g. /dev/shm)
and writes patched copies of the init/proc configs + their context that point at the local copies,
so the run reads indices locally. Only the .sqlite indices move; the large .h5/.g3 data stay on the
shared filesystem.

Run it once per node in the slurm prologue (for multi-node, use --ntasks-per-node=1):

    mapfile -t CFGS < <(furax-so-stage --init-config A --proc-config B --dest "$STAGE")
    srun ... --init-config "${CFGS[0]}" --proc-config "${CFGS[1]}"

Prints the patched init and proc config paths (one per line) to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import sys
from pathlib import Path

import yaml
from cyclopts import App

from . import _preproc as pp

app = App(help='Stage a preproc run sqlite indices to node-local storage and emit patched configs.')


def _log(msg: str) -> None:
    print(f'stage: {msg}', file=sys.stderr)


def _absolutize_manifest_paths(db_path: Path, src_dir: Path) -> None:
    """Rewrite a staged ManifestDb's relative ``files.name`` entries to absolute paths.

    ManifestDb-style indices (the preprocess process_archive + the metadata ``*_local.sqlite``)
    store their companion ``.h5`` data file by a path relative to the db's own directory; sotodlib
    resolves it via ``os.path.join(prefix, name)`` where ``prefix`` is that directory. Copying the
    sqlite to node-local storage moves the directory out from under those relative names, so the
    ``.h5`` (which stays on the shared FS) can no longer be found. Anchoring each relative name to the
    original source dir makes the join idempotent. obsfiledb/obsdb use a different ``files`` schema
    (and already store absolute g3 paths), so they are detected by signature and left untouched.
    """
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute('PRAGMA table_info(files)')]
        if cols != ['id', 'name']:  # not a ManifestDb files table
            return
        for fid, name in con.execute('select id, name from files').fetchall():
            if name and not Path(name).is_absolute():
                con.execute(
                    'update files set name=? where id=?', ((src_dir / name).as_posix(), fid)
                )
        con.commit()
    finally:
        con.close()


class Stager:
    """Copies sqlite indices to ``dest`` (once each) and rewrites the configs that name them."""

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self._copied: dict[Path, str] = {}  # source abs -> local basename

    def stage(self, src: Path) -> str | None:
        """Copy ``src`` into dest (deduped) and return its local basename, or None if missing."""
        src = src.resolve()
        if src in self._copied:
            return self._copied[src]
        if not src.is_file():
            _log(f'missing, not staged: {src}')
            return None
        # Prefix with a short hash of the source dir so same-named indices (e.g. init/proc
        # process_archive.sqlite) do not collide in the flat dest.
        digest = hashlib.sha1(str(src.parent).encode()).hexdigest()[:8]
        local_name = f'{digest}_{src.name}'
        shutil.copy2(src, self.dest / local_name)
        _absolutize_manifest_paths(self.dest / local_name, src.parent)
        self._copied[src] = local_name
        _log(f'staged {src} -> {local_name}')
        return local_name

    def patch_text(self, text: str, replacements: dict[str, str]) -> str:
        """Replace each literal token (as written in the file) with its local path."""
        for token, local in replacements.items():
            text = text.replace(token, local)
        return text


def stage_config(stager: Stager, config_path: Path) -> Path:
    """Stage one preproc config's indices + its context; write patched copies; return new config."""
    config_path = config_path.resolve()
    config_text = config_path.read_text()
    ctx_token, ctx_path = pp.context_file(config_path)
    ctx_text = ctx_path.read_text()

    # Stage every sqlite index the run will read, splitting the path->local rewrites by which file's
    # text carries the token (the index/metadata live in the config, the dbs in the context).
    config_repl: dict[str, str] = {}
    ctx_repl: dict[str, str] = {}
    for cp in pp.iter_config_paths(config_path):
        local = stager.stage(cp.path)
        if local is None:
            continue
        repl = config_repl if cp.where == 'config' else ctx_repl
        repl[cp.token] = (stager.dest / local).as_posix()

    # write patched context, then point the config at it (absolute, so chdir is irrelevant)
    patched_ctx = stager.dest / f'context_{config_path.stem}.yaml'
    patched_ctx_text = stager.patch_text(ctx_text, ctx_repl)
    patched_ctx.write_text(patched_ctx_text)
    config_repl[ctx_token] = patched_ctx.as_posix()

    patched_config = stager.dest / f'{config_path.stem}.yaml'
    patched_config_text = stager.patch_text(config_text, config_repl)
    patched_config.write_text(patched_config_text)

    # Warn on any path the run will still resolve against its cwd: with the patched config living in
    # dest, furax-so-map chdirs to dest's root, so a leftover relative path resolves to the wrong
    # place. iter_config_paths handles the indices; anything else flagged here needs manual review.
    for text, label in ((patched_config_text, config_path.name), (patched_ctx_text, ctx_path.name)):
        for leftover in pp.find_relative_paths(yaml.safe_load(text)):
            _log(f'WARNING: {label} still has a relative path (resolved against cwd): {leftover}')

    return patched_config


@app.default  # type: ignore[untyped-decorator]
def stage(init_config: Path, proc_config: Path, dest: Path) -> None:
    """Stage the init/proc preproc indices to ``dest`` and print the patched config paths.

    Args:
        init_config: Base-layer preprocessing config file.
        proc_config: Second-layer preprocessing config file.
        dest: Node-local staging directory (e.g. /dev/shm/...).
    """
    dest.mkdir(parents=True, exist_ok=True)
    stager = Stager(dest)
    init_out = stage_config(stager, init_config)
    proc_out = stage_config(stager, proc_config)

    # stdout: the two patched config paths the launcher passes to furax-so-map
    print(init_out.as_posix())
    print(proc_out.as_posix())


if __name__ == '__main__':
    app()
