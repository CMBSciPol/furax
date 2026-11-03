import logging
import os
import threading
from pathlib import Path

import jax
import numpy as np
from jaxtyping import Array, PyTree

CHECKPOINT_FILES = ('checkpoint.npz', 'checkpoint_injected.npz')
TEMPORARY_SUFFIX = '.tmp.'

logger = logging.getLogger(__name__)


def remove_checkpoint_files(directory: Path) -> None:
    """Delete checkpoints from `directory`."""
    for name in CHECKPOINT_FILES:
        for path in [directory / name, *directory.glob(f'{name}{TEMPORARY_SUFFIX}*')]:
            path.unlink(missing_ok=True)


class SolveCheckpoint:
    """Manages checkpoints of a CG solve.

    The checkpoint file holds the leaves of the iterate, in the order of `jax.tree.leaves`, and the
    number of steps taken by the CG.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.resumed_after = 0
        self._saved = 0
        self._lock = threading.Lock()

    def save(self, n_steps: Array, x: PyTree[Array]) -> None:
        """Save the iterate after `n_steps` steps, unless a later checkpoint exists."""
        total = self.resumed_after + int(n_steps)
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(x)]
        # The callbacks of the solver are unordered and run on concurrent host threads: the lock
        # keeps a slow write of an early iterate from replacing a later one.
        with self._lock:
            if total <= self._saved:
                return
            tmp = self.path.with_name(f'{self.path.name}{TEMPORARY_SUFFIX}{os.getpid()}')
            with tmp.open('wb') as f:
                np.savez(f, *leaves, n_steps=total)
            os.replace(tmp, self.path)
            self._saved = total
        logger.debug(f'Saved the checkpoint after {total} steps to {self.path}')

    def load(self, like: PyTree[Array]) -> tuple[PyTree[Array], int] | None:
        """Load a saved iterate sharded like `like`, and its number of steps.

        Returns:
            `None` if no checkpoint is found, or if shapes and dtype don't match.
        """
        leaves, treedef = jax.tree.flatten(like)
        try:
            with np.load(self.path) as data:
                saved = [data.get(f'arr_{i}') for i in range(len(leaves))]
                n_steps = int(data['n_steps'])
        except FileNotFoundError:
            return None
        if any(
            a is None or a.shape != leaf.shape or a.dtype != leaf.dtype
            for a, leaf in zip(saved, leaves, strict=True)
        ):
            return None
        x0 = [
            jax.make_array_from_callback(leaf.shape, leaf.sharding, lambda index, a=a: a[index])
            for a, leaf in zip(saved, leaves, strict=True)
        ]
        return treedef.unflatten(x0), n_steps

    def resume(self, n_steps: int) -> None:
        self.resumed_after = self._saved = n_steps

    def remove(self) -> None:
        """Delete the checkpoint, if any."""
        self.path.unlink(missing_ok=True)
        for tmp in self.path.parent.glob(f'{self.path.name}{TEMPORARY_SUFFIX}*'):
            tmp.unlink(missing_ok=True)
