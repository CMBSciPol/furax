"""Pixel indexing of time-ordered samples, with an index for "no pixel".

A sample, or a neighbour of an interpolation stencil, that reads no pixel of the map, because it
falls outside the map or has no pointing, carries the index `NO_PIXEL`. Gathering reads zero
there and scatter-adding drops it, so such a sample costs no memory traffic. On GPU, this also
keeps the padding of an observation from serializing the atomic additions of a scatter on a
single pixel.
"""

from jaxtyping import Array, Float, Integer

NO_PIXEL = -1
"""The index of a sample or neighbour that reads no pixel."""


def gather_pixels(
    data: Float[Array, '... pixels'], indices: Integer[Array, '...']
) -> Float[Array, '...']:
    """Read the pixels of `indices` along the last axis of `data`, zero for `NO_PIXEL`."""
    return data.at[..., indices].get(mode='fill', fill_value=0, wrap_negative_indices=False)


def scatter_add_pixels(
    data: Float[Array, '... pixels'], indices: Integer[Array, ' n'], values: Float[Array, '... n']
) -> Float[Array, '... pixels']:
    """Add `values` to the pixels of `indices` along the last axis, dropping `NO_PIXEL`."""
    return data.at[..., indices].add(values, mode='drop', wrap_negative_indices=False)
