import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax import Array
from jax.experimental import io_callback
from jax.tree_util import register_static
from jaxtyping import PyTree

from furax.tree import nbytes, zeros_like

logger = logging.getLogger(__name__)


@register_static
class AbstractReader(ABC):
    """Abstract class for reading data from disk, avoiding recompilation and large static datasets.

    Attributes:
        count (int): The number of data to read.
        args (list[tuple[Any, ...]]): For each data, the positional arguments to be passed to
            the read function.
        keywords (list[dict[str, Any]]): For each data, the keyword arguments to be passed to
            the read function.
        common_keywords (dict[str, Any]): For all data, the keyword arguments to be passed to
            the read function.
        out_structure (PyTree[jax.ShapeDtypeStruct]): The structure of the data that is returned by
            the read function. The structure is the same for all data.
    """

    def __init__(
        self,
        *args: Sequence[Any],
        common_keywords: Mapping[str, Any] | None = None,
        structures: list[PyTree[jax.ShapeDtypeStruct]] | None = None,
        known_failures: Sequence[int] | None = None,
        **keywords: Sequence[Any],
    ) -> None:
        """Initialize the reader.

        Args:
            *args: One list per positional argument to the read function, one element per item.
            common_keywords: Keyword arguments shared by all data items.
            structures: Pre-computed per-item output structures.
                When provided, constructor skips all I/O and uses them directly.
            known_failures: Item indices known to be unreadable up front (e.g. their shape probe
                failed); [`read`][furax.io.readers.AbstractReader.read] returns filler for them
                without attempting a load.
            **keywords: One list per keyword argument to the read function, one element per item.
        """
        self.args, self.keywords = self._normalize_args_keywords(args, keywords)
        self.count = len(self.args)
        self.common_keywords = common_keywords or {}
        self.known_failures = set(known_failures or ())
        self.reset_failures()
        if structures is None:
            structures = self._read_structures()
        elif len(structures) != self.count:
            raise ValueError(
                f'structures length {len(structures)} does not match data count {self.count}'
            )
        self.total_nbytes = sum(nbytes(s) for s in structures)
        self.out_structure = self._get_common_structure(structures)

    def reset_failures(self) -> None:
        """Reset runtime read failures to the known-failure baseline.

        [`read`][furax.io.readers.AbstractReader.read] records caught failures in ``failed_indices``
        as a host side effect, so a reused reader would otherwise carry failures across read passes.
        Call this before a fresh pass to start from just the up-front ``known_failures``.

        ``failed_indices`` is a set: ``read`` runs inside ``io_callback``, which JAX may dispatch
        on concurrent host threads, so two threads can record the same failing index at once.
        ``set.add`` is idempotent, sidestepping the check-then-append race a list would need.
        """
        self.failed_indices = set(self.known_failures)

    @staticmethod
    def _normalize_args_keywords(
        args: Sequence[Any], keywords: dict[str, Sequence[Any]]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Normalize the arguments and keywords to ensure they are lists of the same length."""
        invalid_args = [i for i, v in enumerate(args) if not isinstance(v, Sequence)]
        if invalid_args:
            raise ValueError(
                f'Values in positional arguments must be lists. Invalid args: {invalid_args}'
            )

        invalid_keys = [k for k, v in keywords.items() if not isinstance(v, Sequence)]
        if invalid_keys:
            raise ValueError(f'Values in keywords must be lists. Invalid keys: {invalid_keys}')

        # transform a tuple of lists to a list of tuples
        try:
            list_of_args = list(zip(*args, strict=True))
        except ValueError:
            raise ValueError(
                'Values in positional arguments must be lists of the same length.'
            ) from None

        # transform a dict of lists to a list of dicts
        try:
            list_of_keywords = [
                dict((k, v) for k, v in zip(keywords, values))
                for values in zip(*keywords.values(), strict=True)
            ]
        except ValueError:
            raise ValueError('Values in keywords must be lists of the same length.') from None

        count_args = len(list_of_args)
        count_keywords = len(list_of_keywords)
        if count_args == 0 and count_keywords == 0:
            raise ValueError(
                'The specification of the data to be loaded through args and keywords cannot be '
                'empty'
            )

        if count_args == 0:
            return [()] * count_keywords, list_of_keywords

        if count_keywords == 0:
            return list_of_args, [{}] * count_args

        if count_args != count_keywords:
            raise ValueError('Positional arguments and keywords must have the same length')

        return list_of_args, list_of_keywords

    def _read_structures(self) -> list[PyTree[jax.ShapeDtypeStruct]]:
        structures = [
            self._read_structure_impure(*self.args[i], **self.keywords[i], **self.common_keywords)
            for i in range(self.count)
        ]
        if not jax.tree.all(
            jax.tree.map(
                lambda leaf, *leaves: all(l.dtype == leaf.dtype for l in leaves),
                structures[0],
                *structures[1:],
            )
        ):
            raise ValueError('All structures must have the same dtype')
        if not jax.tree.all(
            jax.tree.map(
                lambda leaf, *leaves: all(l.ndim == leaf.ndim for l in leaves),
                structures[0],
                *structures[1:],
            )
        ):
            raise ValueError('All structures must have the same number of dimensions')
        return structures

    def _get_common_structure(
        self, structures: list[PyTree[jax.ShapeDtypeStruct]]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        structure = jax.tree.map(
            lambda *leaves: jax.ShapeDtypeStruct(
                tuple(max(leaf.shape[i] for leaf in leaves) for i in range(leaves[0].ndim)),
                leaves[0].dtype,
            ),
            *structures,
        )
        return structure

    def _actual_padding(self, data: PyTree[Any]) -> PyTree[tuple[int, ...]]:
        """Per-leaf padding from the actual loaded shape to ``out_structure``.

        Computed at read time inside the io_callback, since the loaded shape may be smaller than
        the (upper-bound) probe shape used to size ``out_structure``. Raises if any loaded leaf
        exceeds ``out_structure``: that means a ``probe_shape`` under-estimated, which would force a
        negative pad and silently corrupt the buffers.
        """

        def leaf_padding(common_leaf: jax.ShapeDtypeStruct, leaf: Any) -> tuple[int, ...]:
            padding = tuple(common_leaf.shape[i] - leaf.shape[i] for i in range(leaf.ndim))
            if any(p < 0 for p in padding):
                raise ValueError(
                    f'loaded shape {leaf.shape} exceeds buffer shape {common_leaf.shape}: '
                    'probe_shape under-estimated the observation size'
                )
            return padding

        return jax.tree.map(leaf_padding, self.out_structure, data)

    @abstractmethod
    def _failure_filler(self) -> PyTree[np.ndarray]:
        """Finite, ``out_structure``-shaped data to substitute when a read fails."""

    def _padding_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """Per-leaf shape of the padding pytree: one ``int32`` entry per axis of each leaf."""
        return jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct((len(leaf.shape),), np.int32), self.out_structure
        )

    @jax.jit
    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array], Array]:
        """Read the data at the given index.

        Returns:
            A triple ``(data, padding, valid)``: the padded data pytree (matching
            ``out_structure``), the padding pytree, and a scalar boolean that is ``False`` when the
            read failed.
        """
        padding_structure = self._padding_structure()

        def callback(
            i_arr: Array,
        ) -> tuple[PyTree[np.ndarray], PyTree[np.ndarray], np.ndarray]:
            i = int(i_arr)  # io_callback passes the index as a (host) array
            if i in self.known_failures:
                # skip load entirely
                logger.info('read item %d: skipped (known failure)', i)
                return self._failure_filler(), zeros_like(padding_structure), np.array(False)
            logger.info('read item %d: start', i)
            start = time.perf_counter()
            try:
                data = self._read_data_impure(
                    *self.args[i], **self.keywords[i], **self.common_keywords
                )
            except Exception:
                self.failed_indices.add(i)
                logger.exception(
                    'read item %d: failed after %.1fs; substituting filler data',
                    i,
                    time.perf_counter() - start,
                )
                return self._failure_filler(), zeros_like(padding_structure), np.array(False)
            # Pad from the actual loaded shape (not the precomputed, probe-based padding): the
            # io_callback contract requires the result to match ``out_structure`` exactly, and
            # ``probe_shape`` is only an upper bound, so the load may be smaller than probed.
            actual_padding = self._actual_padding(data)
            result = (
                self._pad(data, actual_padding),
                self._padding_to_arrays(actual_padding),
                np.array(True),
            )
            logger.info('read item %d: ok (%.1fs)', i, time.perf_counter() - start)
            return result

        result_shape = (
            self.out_structure,
            padding_structure,
            jax.ShapeDtypeStruct((), bool),
        )
        data, padding, valid = jax.lax.switch(
            data_index,
            [lambda i=i: io_callback(callback, result_shape, i) for i in range(self.count)],
        )
        return data, padding, valid

    @jax.jit
    def read_filler(self) -> tuple[PyTree[Array], PyTree[Array], Array]:
        """Return finite filler data without touching any backing store."""
        padding_structure = self._padding_structure()

        def callback():  # type: ignore[no-untyped-def]
            return self._failure_filler(), zeros_like(padding_structure), np.array(False)

        result_shape = (self.out_structure, padding_structure, jax.ShapeDtypeStruct((), bool))
        return io_callback(callback, result_shape)  # type: ignore[no-any-return]

    @staticmethod
    def _padding_to_arrays(padding: PyTree[tuple[int, ...]]) -> PyTree[np.ndarray]:
        """Turn a pytree of per-axis padding tuples into one ``int32`` array per leaf."""
        return jax.tree.map(
            lambda pad: np.asarray(pad, dtype=np.int32),
            padding,
            is_leaf=lambda x: isinstance(x, tuple),
        )

    def _pad(self, data: PyTree[Any], padding: PyTree[tuple[int, ...]]) -> PyTree[np.ndarray]:
        """Pad the host (numpy) data to the common structure.

        Runs on the host inside the read callback, so it must use numpy. Pads with 0 by default;
        subclasses may override to apply field-specific padding.
        """
        return jax.tree.map(
            lambda leaf, pad: (
                np.pad(leaf, [(0, p) for p in pad]) if len(pad) > 0 else np.asarray(leaf)
            ),
            data,
            padding,
        )

    @abstractmethod
    def _read_data_impure(self, *args: Any, **keywords: Any) -> PyTree[Any]: ...

    @abstractmethod
    def _read_structure_impure(
        self, *args: Any, **keywords: Any
    ) -> PyTree[jax.ShapeDtypeStruct]: ...
