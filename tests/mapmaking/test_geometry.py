import numpy as np
import pytest

from furax.mapmaking._geometry import minimum_enclosing_arc


class TestMinimumEnclosingArc:
    @staticmethod
    def arcs_deg(*intervals):
        return np.radians(intervals)

    @staticmethod
    def span_deg(lo, hi):
        return np.degrees(hi - lo)

    def test_simple_patch(self):
        lo, hi = minimum_enclosing_arc(self.arcs_deg([10, 30]))
        assert self.span_deg(lo, hi) == pytest.approx(20.0)
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi

    def test_crosses_zero(self):
        # Observation from 350° to 10°, as pixell would return after unwinding
        lo, hi = minimum_enclosing_arc(self.arcs_deg([350, 370]))
        assert self.span_deg(lo, hi) == pytest.approx(20.0)
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi

    def test_two_observations_crossing_zero(self):
        lo, hi = minimum_enclosing_arc(self.arcs_deg([345, 365], [355, 375]))
        assert self.span_deg(lo, hi) == pytest.approx(30.0)
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi

    def test_nearly_full_sky(self):
        lo, hi = minimum_enclosing_arc(self.arcs_deg([5, 355]))
        assert self.span_deg(lo, hi) == pytest.approx(350.0)
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi

    def test_full_sky_contiguous(self):
        intervals = [[i, i + 20] for i in range(0, 360, 20)]
        lo, hi = minimum_enclosing_arc(self.arcs_deg(*intervals))
        assert self.span_deg(lo, hi) == pytest.approx(360.0)
        assert lo == -np.pi
        assert hi == np.pi

    def test_full_sky_explicit(self):
        lo, hi = minimum_enclosing_arc(self.arcs_deg([0, 360]))
        assert self.span_deg(lo, hi) == pytest.approx(360.0)
        assert lo == -np.pi
        assert hi == np.pi

    def test_two_disjoint_patches(self):
        # 100°–120° and 350°–10°: enclosing arc goes 350°→120° = 130°
        lo, hi = minimum_enclosing_arc(self.arcs_deg([100, 120], [350, 370]))
        assert self.span_deg(lo, hi) == pytest.approx(130.0)
        assert -2 * np.pi <= lo <= hi <= 2 * np.pi
