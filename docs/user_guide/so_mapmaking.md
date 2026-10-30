# Mapping SO Observations

Furax installs command-line tools that run the multi-observation mapmaker on Simons Observatory
data, loading the observations from a sotodlib preprocessing database.

| Command | Purpose |
| --- | --- |
| `furax-so-map` | Make maps from the preprocessing database. |
| `furax-so-cache` | Manage the cache of preprocessed observations. |
| `furax-so-stage` | Copy the sqlite indices of a preprocessing run to node-local storage. |

Each command lists its options with `--help`.

## Making maps

```bash
furax-so-map --init-config init.yaml --proc-config proc.yaml \
    --obsids-file obsids.txt --wafer ws0 --band f090 \
    --mapmaking-config mapmaking.yaml --outdir maps/
```

- `--init-config` and `--proc-config` are the preprocessing configs of the database, the second
  one optional. Each observation is loaded and preprocessed through them when the mapmaker reads it.
- The observations are given with `--obsid` or `--obsids-file`, one id per line. Obsids are
  deduplicated.
- `--wafer` and `--band` select the detectors. With `--wafer all`, the wafers of the band are
  joined into one observation, so that the templates spanning the focal plane, such as the common
  mode and the T2P field, are fitted over all of them at once.
- `--mapmaking-config` is the YAML file of a `MapMakingConfig`. Its `sotodlib` section sets how
  the observations are loaded, e.g. `downsample` and `demodulated`.

The output directory records the state of the run in `status.json`: `running` from the start,
then `done` once every product is written, or `failed`, with the error. A run killed by a signal,
e.g. at its time limit, stays `running`. The mapmaking config, the provenance and the logs, one
`furax-so-map.rank<N>.log` per process, are written at the start, so that a run that does not
finish leaves them behind. `furax-so-map` refuses a directory that holds the output of an earlier
run, unless given `--overwrite`, which deletes the files of that run first. Other files, such as
the output of a batch scheduler, are kept.

`furax-so-map` initializes `jax.distributed` itself, so that a run over several processes, e.g.
launched with `srun`, maps the observations together.

With `--obsdir DIR` instead of the preprocessing configs, `furax-so-map` maps the `<obsid>.h5`
files of a directory.

## Caching preprocessed observations

Preprocessing takes minutes per observation, and is repeated by every run that maps the same
observations. With `--cache-dir DIR`, the first run saves each observation once loaded, i.e.
preprocessed, joined over the wafers and downsampled, to `DIR/<key>/<obsid>.h5`. Later runs load
that file instead.

The key is a hash of every input that changes the loaded data:

- the detector selection, and whether the wafers are joined (`--wafer all`);
- the content of the preprocessing configs and of their sotodlib contexts;
- `sotodlib.downsample` from the mapmaking config;
- the version of sotodlib, and a version of the cache format.

A `key.json` next to the files records these inputs. Runs that differ only in the rest of the
mapmaking config, the other `sotodlib` settings included, share a cache. Runs that differ in one
of the inputs above use a key of their own: in particular, an observation cached with
`--wafer all` holds every wafer joined, and serves only runs with `--wafer all`, while a run
with `--wafer ws0` caches that wafer alone.

The content of the databases the configs point to, such as the preprocessing archive, is not
part of the key. After preprocessing again under the same configs, delete the key (see
[`prune`](#listing-and-pruning)), or the runs keep loading the earlier data. Likewise, a wafer
skipped when an observation was cached, because none of its detectors survived the cuts or it
had no preprocessing record, stays missing from the cached observation.

Concurrent runs can share a cache: each file is written under a temporary name and renamed into
place once complete, so a run never reads a partial file. A cached file that cannot be read is
loaded from the preprocessing database again and replaced.

### Filling the cache ahead of time

`furax-so-cache fill` preprocesses observations into the cache without making maps, in several
processes, so that the mapmaking runs start with every observation cached. It takes the
arguments of the `furax-so-map` run to prepare:

```bash
furax-so-cache fill --cache-dir $SCRATCH/cache --init-config init.yaml --proc-config proc.yaml \
    --obsids-file obsids.txt --wafer all --band f090 --mapmaking-config mapmaking.yaml --nproc 16
```

Observations already cached are skipped, so `fill` can be run again after an interruption. With
`--dry-run`, it only prints the key and the observations missing from it. `fill` runs on one
node; to spread a long list over several jobs, give each job part of the ids.

If some observations cannot be cached, `fill` lists them and exits with status 1. A worker
process that dies, e.g. killed out of memory, stops the others: the observations it leaves
uncached are listed the same way.

### Listing and pruning

`furax-so-cache list DIR` prints each key with the inputs recorded in its `key.json`, its number
of observations, its size and its last write.

`furax-so-cache prune DIR` deletes the temporary files of interrupted writes once they are older
than `--older-than` hours (24 by default): younger ones may still be written by a running job.
With `--key KEY`, it deletes the given keys, as `list` prints them. A run reading a key while it
is deleted fails.

## Staging the indices

A run opens the sqlite indices of the preprocessing database (archive, obsdb, obsfiledb and the
metadata databases) on every observation load. `furax-so-stage` copies them to node-local
storage once per node and prints patched configs that point to the copies:

```bash
mapfile -t CFGS < <(furax-so-stage --init-config init.yaml --proc-config proc.yaml --dest "$STAGE")
srun ... furax-so-map --init-config "${CFGS[0]}" --proc-config "${CFGS[1]}" ...
```

A patched config names the config it was staged from in a `furax_staged_from` entry, and the
cache keys it as that config: runs with staged configs, wherever they are staged, and runs with
the original configs share a cache.
