import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np
from jax import Array
from jax.experimental import io_callback
from jax.tree_util import register_static
from jaxtyping import PyTree

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
        paddings (list[PyTree[tuple[int, ...]]): For each data, the padding that is applied to
            the data that is returned by the read function.
    """

    def __init__(
        self,
        *args: Sequence[Any],
        common_keywords: Mapping[str, Any] | None = None,
        structures: list[PyTree[jax.ShapeDtypeStruct]] | None = None,
        known_failures: Sequence[int] | None = None,
        **keywords: Sequence[Any],
    ) -> None:
        """
        Args:
            *args: One list per positional argument to the read function, one element per data item.
            common_keywords: Keyword arguments shared by all data items.
            structures: Pre-computed per-item output structures.
                When provided, constructor skips all I/O and uses them directly.
            known_failures: Item indices known to be unreadable up front (e.g. their shape probe
                failed); :meth:`read` returns filler for them without attempting a load.
            **keywords: One list per keyword argument to the read function, one element per data item.
        """
        self.args, self.keywords = self._normalize_args_keywords(args, keywords)
        self.count = len(self.args)
        self.common_keywords = common_keywords or {}
        self.known_failures = set(known_failures or ())
        self.failed_indices = sorted(self.known_failures)
        if structures is None:
            structures = self._read_structures()
        elif len(structures) != self.count:
            raise ValueError(
                f'structures length {len(structures)} does not match data count {self.count}'
            )
        self._infer_structure_and_paddings(structures)

    def _infer_structure_and_paddings(self, structures: list[PyTree[jax.ShapeDtypeStruct]]) -> None:
        self.out_structure = self._get_common_structure(structures)
        self.paddings: list[PyTree[tuple[int, ...]]] = [
            self._get_padding(structures[i]) for i in range(self.count)
        ]

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

    def _get_padding(
        self, structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[tuple[int, ...] | None]:
        padding = jax.tree.map(
            lambda leaf, common_leaf: tuple(
                common_leaf.shape[i] - leaf.shape[i] for i in range(leaf.ndim)
            ),
            structure,
            self.out_structure,
        )
        return padding

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

    def _failure_filler(self) -> PyTree[np.ndarray] | None:
        """Finite, ``out_structure``-shaped data to substitute when a read fails.

        Subclasses override to return finite filler so a failing item degrades to a valid-shaped
        result flagged invalid, instead of crashing.
        """
        return None

    @jax.jit
    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array], Array]:
        """Read the data at the given index.

        Returns:
            A triple ``(data, padding, valid)``: the padded data pytree (matching
            ``out_structure``), the padding pytree, and a scalar boolean that is ``False`` when the
            read failed.
        """

        def callback(i_arr: Array) -> tuple[PyTree[np.ndarray], np.ndarray]:
            i = int(i_arr)  # io_callback passes the index as a (host) array
            if i in self.known_failures:
                # skip load entirely
                return self._failure_filler(), np.array(False)
            try:
                data = self._read_data_impure(
                    *self.args[i], **self.keywords[i], **self.common_keywords
                )
            except Exception:
                filler = self._failure_filler()
                if filler is None:
                    # no filler override -> crash
                    raise
                if i not in self.failed_indices:
                    self.failed_indices.append(i)
                logger.exception('read of item %d failed; substituting filler data', i)
                return filler, np.array(False)
            # Pad from the actual loaded shape (not the precomputed, probe-based padding): the
            # io_callback contract requires the result to match ``out_structure`` exactly, and
            # ``probe_shape`` is only an upper bound, so the load may be smaller than probed.
            return self._pad(data, self._actual_padding(data)), np.array(True)

        result_shape = (self.out_structure, jax.ShapeDtypeStruct((), bool))
        data, valid = jax.lax.switch(
            data_index,
            [lambda i=i: io_callback(callback, result_shape, i) for i in range(self.count)],
        )
        padding = jax.lax.switch(
            data_index, [lambda i=i: self.paddings[i] for i in range(self.count)]
        )
        return data, padding, valid

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
