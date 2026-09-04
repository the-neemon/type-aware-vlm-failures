"""Boundary tests for relaxed accuracy (TASKS P1.1).

The TASKS file is explicit: "unit-test it exactly at the tolerance boundary in
both directions."  Getting this wrong marks correct answers as errors, and those
mislabelled items land in the structural class and poison every probe.

Run:
    python -m pytest src/eval/test_relaxed_accuracy.py -v
"""

import pytest

from src.eval.relaxed_accuracy import _try_parse_float, is_correct, relaxed_accuracy


# -----------------------------------------------------------------------
# Numeric parsing
# -----------------------------------------------------------------------

class TestTryParseFloat:
    """Ensure the parser handles common ChartQA formatting."""

    def test_plain_integer(self):
        assert _try_parse_float("42") == 42.0

    def test_plain_float(self):
        assert _try_parse_float("3.14") == 3.14

    def test_negative(self):
        assert _try_parse_float("-7.5") == -7.5

    def test_comma_thousands(self):
        assert _try_parse_float("1,234,567") == 1234567.0

    def test_dollar_sign(self):
        assert _try_parse_float("$100") == 100.0

    def test_dollar_with_comma(self):
        assert _try_parse_float("$1,200.50") == 1200.50

    def test_percent(self):
        assert _try_parse_float("85%") == 85.0

    def test_whitespace(self):
        assert _try_parse_float("  42  ") == 42.0

    def test_non_numeric_returns_none(self):
        assert _try_parse_float("yes") is None

    def test_empty_returns_none(self):
        assert _try_parse_float("") is None

    def test_mixed_text_returns_none(self):
        assert _try_parse_float("about 50") is None

    def test_euro_sign(self):
        assert _try_parse_float("€99.99") == 99.99

    def test_positive_sign(self):
        assert _try_parse_float("+3.5") == 3.5


# -----------------------------------------------------------------------
# Boundary tests — the critical ones
# -----------------------------------------------------------------------

class TestBoundaryNumeric:
    """Test exactly AT the 5 % tolerance boundary, and one tick either side."""

    # gold = 100, tolerance = 5 %  →  accept range [95, 105]

    def test_exactly_at_upper_boundary(self):
        """gold=100, pred=105  →  |105 - 100| = 5 = 0.05 * 100  →  correct."""
        assert is_correct("100", "105") is True

    def test_just_above_upper_boundary(self):
        """gold=100, pred=105.01  →  |105.01 - 100| = 5.01 > 5  →  wrong."""
        assert is_correct("100", "105.01") is False

    def test_just_below_upper_boundary(self):
        """gold=100, pred=104.99  →  |104.99 - 100| = 4.99 < 5  →  correct."""
        assert is_correct("100", "104.99") is True

    def test_exactly_at_lower_boundary(self):
        """gold=100, pred=95  →  |95 - 100| = 5 = 0.05 * 100  →  correct."""
        assert is_correct("100", "95") is True

    def test_just_below_lower_boundary(self):
        """gold=100, pred=94.99  →  |94.99 - 100| = 5.01 > 5  →  wrong."""
        assert is_correct("100", "94.99") is False

    def test_just_above_lower_boundary(self):
        """gold=100, pred=95.01  →  |95.01 - 100| = 4.99 < 5  →  correct."""
        assert is_correct("100", "95.01") is True


class TestBoundaryNegative:
    """Same boundary logic but with negative gold."""

    def test_negative_gold_exact_boundary(self):
        """gold=-200, pred=-190  →  |-190 - (-200)| = 10 = 0.05 * 200  →  correct."""
        assert is_correct("-200", "-190") is True

    def test_negative_gold_just_outside(self):
        """gold=-200, pred=-189.99  →  |10.01| > 10  →  wrong."""
        assert is_correct("-200", "-189.99") is False

    def test_negative_gold_just_inside(self):
        """gold=-200, pred=-190.01  →  |9.99| < 10  →  correct."""
        assert is_correct("-200", "-190.01") is True


class TestZeroGold:
    """gold == 0 is a singularity: tolerance * |gold| = 0, so only 0 passes."""

    def test_zero_zero(self):
        assert is_correct("0", "0") is True

    def test_zero_tiny(self):
        assert is_correct("0", "0.001") is False

    def test_zero_negative_tiny(self):
        assert is_correct("0", "-0.001") is False

    def test_zero_zero_float(self):
        assert is_correct("0", "0.0") is True


class TestSmallGold:
    """Small but non-zero gold: tolerance window is small but non-degenerate."""

    def test_gold_one_pred_one_point_zero_five(self):
        """gold=1, pred=1.05  →  |0.05| = 0.05 * 1  →  correct."""
        assert is_correct("1", "1.05") is True

    def test_gold_one_pred_one_point_zero_six(self):
        """gold=1, pred=1.06  →  |0.06| > 0.05  →  wrong."""
        assert is_correct("1", "1.06") is False


# -----------------------------------------------------------------------
# String comparison
# -----------------------------------------------------------------------

class TestStringComparison:
    """Non-numeric answers: case-insensitive, stripped exact match."""

    def test_exact_match(self):
        assert is_correct("yes", "yes") is True

    def test_case_insensitive(self):
        assert is_correct("Yes", "yes") is True
        assert is_correct("YES", "yes") is True

    def test_whitespace_stripped(self):
        assert is_correct("  yes  ", "yes") is True

    def test_mismatch(self):
        assert is_correct("yes", "no") is False

    def test_multiword(self):
        assert is_correct("United States", "united states") is True

    def test_multiword_mismatch(self):
        assert is_correct("United States", "United Kingdom") is False


# -----------------------------------------------------------------------
# Mixed types
# -----------------------------------------------------------------------

class TestMixedTypes:
    """One side numeric, the other not: fall to string comparison → fail."""

    def test_numeric_gold_string_pred(self):
        assert is_correct("100", "one hundred") is False

    def test_string_gold_numeric_pred(self):
        assert is_correct("yes", "1") is False


# -----------------------------------------------------------------------
# Formatting robustness
# -----------------------------------------------------------------------

class TestFormatting:
    """ChartQA answers sometimes have currency, commas, percent signs."""

    def test_comma_in_gold(self):
        """gold='1,000', pred='1000' → both parse as 1000 → correct."""
        assert is_correct("1,000", "1000") is True

    def test_comma_in_pred(self):
        assert is_correct("1000", "1,000") is True

    def test_dollar_gold(self):
        assert is_correct("$500", "500") is True

    def test_dollar_pred(self):
        assert is_correct("500", "$500") is True

    def test_percent_gold(self):
        assert is_correct("85%", "85") is True

    def test_percent_pred(self):
        assert is_correct("85", "85%") is True

    def test_dollar_with_tolerance(self):
        """$100 vs 104.99 → numeric, within 5% → correct."""
        assert is_correct("$100", "104.99") is True


# -----------------------------------------------------------------------
# Batch metric
# -----------------------------------------------------------------------

class TestRelaxedAccuracy:
    """The batch function over parallel sequences."""

    def test_all_correct(self):
        assert relaxed_accuracy(["100", "yes"], ["100", "yes"]) == 1.0

    def test_all_wrong(self):
        assert relaxed_accuracy(["100", "yes"], ["200", "no"]) == 0.0

    def test_half_correct(self):
        assert relaxed_accuracy(["100", "yes"], ["105", "no"]) == 0.5

    def test_empty(self):
        assert relaxed_accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            relaxed_accuracy(["100"], ["100", "200"])


# -----------------------------------------------------------------------
# Custom tolerance
# -----------------------------------------------------------------------

class TestCustomTolerance:
    """Verify the tolerance parameter threads through correctly."""

    def test_tighter_tolerance(self):
        """1% tolerance: gold=100, pred=101 → correct; pred=102 → wrong."""
        assert is_correct("100", "101", tolerance=0.01) is True
        assert is_correct("100", "102", tolerance=0.01) is False

    def test_looser_tolerance(self):
        """10% tolerance: gold=100, pred=110 → correct; pred=111 → wrong."""
        assert is_correct("100", "110", tolerance=0.10) is True
        assert is_correct("100", "111", tolerance=0.10) is False
