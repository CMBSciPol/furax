"""Cross-process collectives for the per-process ("Model B") execution.

A single [`jax.shard_map`][] over a global observation axis forces one uniform buffer shape on
every process, so the whole run pads to the global maximum. Owning a *segment* per process lifts
that: each process pads only to its own segment's maximum and runs its time-ordered work on its
own devices, at its own shapes, with no global array involved.

What still has to be global is the sky map: every process accumulates a partial over its own
observations, and the map is the sum of those partials. That is the only quantity crossing process
boundaries, and it is uniform by construction (``npix`` does not depend on the segment), so it can
travel as a regular global array even though the time-ordered data cannot.

[`cross_process_sum`][] performs that reduction. It stacks each process's partial into a global
array sharded over a ``proc`` axis and sums the leading axis, which lowers to an all-reduce; the
partials never leave the device.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree

__all__ = ['cross_process_mesh', 'cross_process_sum', 'split_stream', 'stream_is_split']

_state = threading.local()


@contextmanager
def split_stream() -> Iterator[None]:
    """Declare that the active mesh covers only this process's share of the stream.

    Operators that reduce over the stream axis (see
    [`AbstractStreamOperator`][furax.mapmaking.streaming.AbstractStreamOperator]) ``psum`` over the
    active mesh, which under this context reaches only the local devices. Inside it they finish the
    job with [`cross_process_sum`][], so their reduced outputs mean the same thing they do on a
    mesh spanning the whole job.

    Declared rather than detected: a process-local mesh is not by itself evidence that the stream
    continues elsewhere, and silently adding a collective to one built for other reasons would be
    both wrong and hard to see.
    """
    previous = stream_is_split()
    _state.split = True
    try:
        yield
    finally:
        _state.split = previous


def stream_is_split() -> bool:
    """Whether the stream axis stops at the process boundary; see [`split_stream`][]."""
    return getattr(_state, 'split', False)


def cross_process_mesh() -> Mesh:
    """A ``(proc, dev)`` mesh over every device in the job, grouped by owning process.

    ``jax.devices()`` is not guaranteed to be ordered by process, so the devices are sorted by
    ``(process_index, id)`` before being folded into the process-major grid that ``P('proc')``
    indexes.
    """
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    n_proc = jax.process_count()
    return Mesh(np.array(devices).reshape(n_proc, -1), ('proc', 'dev'))


def cross_process_sum(tree: PyTree[Array]) -> PyTree[Array]:
    """Sum a process-local pytree across all processes, returning the total on every process.

    Each leaf must be process-local (not a global sharded array) and identically shaped on every
    process -- true of sky-sized quantities, which is what this is for. Single-process runs return
    the input untouched, so callers need no branch of their own.

    Args:
        tree: Pytree of process-local arrays holding this process's partial sums.

    Returns:
        A pytree of the same structure where every leaf is the sum over all processes.
    """
    if jax.process_count() == 1:
        return tree

    mesh = cross_process_mesh()
    n_proc = jax.process_count()
    # One shard per *local* device is required, but the partial is replicated within the process,
    # so every local device contributes the same slice of this process's row.
    local_devices = sorted(jax.local_devices(), key=lambda d: d.id)

    def reduce_leaf(x: Array) -> Array:
        sharding = NamedSharding(mesh, P('proc'))
        stacked = jax.make_array_from_single_device_arrays(
            (n_proc, *x.shape),
            sharding,
            [jax.device_put(x.reshape(1, *x.shape), d) for d in local_devices],
        )
        total: Array = jax.jit(lambda s: s.sum(axis=0))(stacked)
        return total

    return jax.tree.map(reduce_leaf, tree)
