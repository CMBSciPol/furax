from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import io_callback
from jax.tree_util import register_static
from jaxtyping import PyTree


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
        **keywords: Sequence[Any],
    ) -> None:
        self.args, self.keywords = self._normalize_args_keywords(args, keywords)
        self.count = len(self.args)
        if common_keywords is None:
            common_keywords = {}
        self.common_keywords = common_keywords
        structures = self._read_structures()
        self.out_structure = self._get_common_structure(structures)
        self.paddings: list[PyTree[tuple[int, ...]]] = [
            self._get_padding(structures[i]) for i in range(self.count)
        ]

    def _normalize_args_keywords(
        self, args: Sequence[Any], keywords: dict[str, Sequence[Any]]
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
            raise ValueError('Values in positional arguments must be lists of the same length.')

        # transform a dict of lists to a list of dicts
        try:
            list_of_keywords = [
                dict((k, v) for k, v in zip(keywords, values))
                for values in zip(*keywords.values(), strict=True)
            ]
        except ValueError:
            raise ValueError('Values in keywords must be lists of the same length.')

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

    @jax.jit
    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array]]:
        """Read the data at the given index.

        Args:
            data_index: The index of the data to read.

        Returns:
            A pair of PyTrees, the first one containing the data and the second one containing the
            padding. The structure of the data is the same as the structure of the padding.
        """

        def callback(i: int) -> PyTree[Array]:
            args = self.args[i]
            keywords = self.keywords[i]
            padding = self.paddings[i]
            data = self._read_data_impure(*args, **keywords, **self.common_keywords)
            device_data = jax.tree.map(
                lambda leaf, pad: jnp.pad(leaf, [(0, p) for p in pad]) if len(pad) > 0 else leaf,
                data,
                padding,
            )
            return device_data

        data = jax.lax.switch(
            data_index,
            [lambda i=i: io_callback(callback, self.out_structure, i) for i in range(self.count)],
        )
        padding = jax.lax.switch(
            data_index, [lambda i=i: self.paddings[i] for i in range(self.count)]
        )
        return data, padding

    @abstractmethod
    def _read_data_impure(self, *args: Any, **keywords: Any) -> PyTree[Any]: ...

    @abstractmethod
    def _read_structure_impure(
        self, *args: Any, **keywords: Any
    ) -> PyTree[jax.ShapeDtypeStruct]: ...
