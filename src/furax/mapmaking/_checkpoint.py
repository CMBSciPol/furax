"""Checkpoints of the CG solves of the mapmaker, from which an interrupted run resumes."""

import os
from pathlib import Path

import jax
import numpy as np
from jaxtyping import Array, PyTree

CHECKPOINT_FILES = ('checkpoint.npz', 'checkpoint_injected.npz')
"""The checkpoints of the GLS solve and of the solve of the injected sky, in that order."""


class SolveCheckpoint:
    """The iterate of a CG solve, saved to a file every few steps, and read back to resume it.

    The file holds the leaves of the iterate, in the order of `jax.tree.leaves`, and the number of
    steps the iterate took, over every run that resumed it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.resumed_after = 0
        """The number of steps of the iterate loaded, which the steps of this run add to."""
        self._saved = 0

    def save(self, n_steps: Array, x: PyTree[Array]) -> None:
        """Save the iterate after `n_steps` steps of this run, unless a later one is saved."""
        total = self.resumed_after + int(n_steps)
        # the callbacks of the solver are unordered
        if total <= self._saved:
            return
        tmp = self.path.with_name(f'{self.path.name}.tmp.{os.getpid()}')
        with tmp.open('wb') as f:
            np.savez(f, *(np.asarray(leaf) for leaf in jax.tree.leaves(x)), n_steps=total)
        os.replace(tmp, self.path)
        self._saved = total

    def load(self, like: PyTree[Array]) -> tuple[PyTree[Array], int] | None:
        """The saved iterate, sharded like `like`, and its number of steps.

        `None` if there is no checkpoint, or one of a system of other shapes or precision.
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
        self.resumed_after = self._saved = n_steps
        return treedef.unflatten(x0), n_steps

    def remove(self) -> None:
        """Delete the checkpoint, if any."""
        self.path.unlink(missing_ok=True)
