"""Cross-process collectives for the mapmaking pipeline."""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree


def _cross_process_mesh() -> Mesh:
    """A ``(proc, dev)`` mesh over every device in the job, grouped by owning process.

    Assumes every process contributes the same number of devices, so that row ``p`` holds exactly
    the devices of process ``p``.
    """
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    n_proc = jax.process_count()
    return Mesh(np.array(devices).reshape(n_proc, -1), ('proc', 'dev'))


def cross_process_sum(tree: PyTree[Array]) -> PyTree[Array]:
    """Sum a process-local pytree across all processes, returning the total on every process.

    Each leaf must be process-local (not a global sharded array) and identically shaped on every
    process. Leaves are replicated across a process's local devices rather than split over them,
    so those copies carry the same row and do not contribute to the sum. Single-process runs
    return the input untouched.

    Args:
        tree: Pytree of process-local arrays holding this process's partial sums.

    Returns:
        A pytree of the same structure where every leaf is the sum over all processes, replicated
        on every device.
    """
    if jax.process_count() == 1:
        return tree

    mesh = _cross_process_mesh()
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
