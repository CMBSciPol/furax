import numpy as np
import numpy.typing as npt


def minimum_enclosing_arc(arcs_rad: npt.ArrayLike) -> tuple[float, float]:
    """Finds the minimum arc enclosing all input intervals.

    Merges all intervals on the circle, finds the largest gap in the union, and cuts
    there. Return [-π, π] if intervals cover the full circle.

    Args:
        arcs_rad: array-like of shape (n, 2) with [lo, hi] per interval, in radians.
    """
    two_pi = 2 * np.pi
    arcs = np.asarray(arcs_rad)

    # Normalize lo to [0, 2π) and shift hi by the same amount to preserve spans
    lo = arcs[:, 0] % two_pi
    hi = lo + (arcs[:, 1] - arcs[:, 0])

    # Sort by lo, then find group boundaries: a new group starts where lo[i] exceeds
    # the running max of all previous hi values (i.e. no overlap with prior intervals)
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    new_group = np.concatenate([[True], lo[1:] > np.maximum.accumulate(hi[:-1])])
    group_starts = np.where(new_group)[0]
    merged_lo = lo[group_starts]
    merged_hi = np.maximum.reduceat(hi, group_starts)

    # Find the largest gap (including wrap-around) and cut there
    gap_starts = merged_hi
    gap_ends = np.append(merged_lo[1:], merged_lo[0] + two_pi)
    gap_sizes = gap_ends - gap_starts
    i = np.argmax(gap_sizes)
    if gap_sizes[i] <= 0:
        return -np.pi, np.pi  # full-sky coverage

    cut = (gap_starts[i] + gap_ends[i]) / 2
    remapped = cut + ((arcs - cut) % two_pi)
    lo, hi = remapped.min(), remapped.max()
    shift = np.floor(hi / two_pi) * two_pi
    return lo - shift, hi - shift
