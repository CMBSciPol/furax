import numpy as np
import pytest

from furax.mapmaking._geometry import minimum_enclosing_arc


@pytest.mark.parametrize(
    ('intervals', 'span'),
    [
        ([[10, 30]], 20),
        ([[350, 370]], 20),  # crossing zero, as pixell returns it after unwinding
        ([[345, 365], [355, 375]], 30),  # two observations crossing zero
        ([[5, 355]], 350),
        ([[100, 120], [350, 370]], 130),  # two disjoint patches: 350° -> 120°
        ([[i, i + 20] for i in range(0, 360, 20)], 360),  # contiguous full sky
        ([[0, 360]], 360),
    ],
)
def test_minimum_enclosing_arc(intervals, span):
    lo, hi = minimum_enclosing_arc(np.radians(intervals))
    assert np.degrees(hi - lo) == pytest.approx(span)
    if span == 360:
        assert (lo, hi) == (-np.pi, np.pi)
    else:
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi
