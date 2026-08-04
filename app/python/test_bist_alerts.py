"""Tests for the single-date screen in tools/bist_alerts.py.

`screen()` is thin — it truncates a panel, trains, and scores, all of which is
covered by test_algotrade.py. `build_table` is the part with rules of its own:
the percentile cut, the ordering and the NaN handling are easy to get subtly
wrong, and they have to match `AlgoTradeStrategy._rank` or the screen stops
describing what the strategy would enter.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import bist_alerts  # noqa: E402
from algo_trade import EXPECTED_PCT, SCORE  # noqa: E402


def sig(rows):
    """{symbol: score} -> a signal frame shaped like `signal` returns."""
    frame = pd.DataFrame({SCORE: pd.Series(rows, dtype=float)})
    frame[EXPECTED_PCT] = np.expm1(frame[SCORE]) * 100.0
    return frame


class FakeModel:
    liquid = {"GOOD"}
    min_turnover_percentile = 0.40
    train_start = "2024-01-01"
    exit_ma = 20
    max_hold = 60


def test_the_table_is_the_top_slice_best_first():
    table = bist_alerts.build_table(
        sig({"LOW": 0.1, "HIGH": 0.9, "MID": 0.5, "LOWEST": 0.0}), top_pct=50.0)
    assert list(table.index) == ["HIGH", "MID"]


def test_the_slice_is_a_share_of_the_scored_universe():
    rows = {f"S{i:02d}": 1.0 - i / 100.0 for i in range(40)}
    assert len(bist_alerts.build_table(sig(rows), top_pct=50.0)) == 20
    assert len(bist_alerts.build_table(sig(rows), top_pct=25.0)) == 10
    assert len(bist_alerts.build_table(sig(rows), top_pct=5.0)) == 2


def test_a_slice_narrower_than_one_name_still_takes_one():
    """`max(1, ...)`, matching `_rank` — a screen that lists nothing is useless."""
    table = bist_alerts.build_table(sig({"A": 0.5, "B": 0.4}), top_pct=0.001)
    assert list(table.index) == ["A"]


def test_ties_break_on_symbol():
    """Two runs of the same fit must print the same order."""
    table = bist_alerts.build_table(sig({"ZZZ": 0.5, "AAA": 0.5, "MMM": 0.5}),
                                    top_pct=100.0)
    assert list(table.index) == ["AAA", "MMM", "ZZZ"]


def test_unscored_symbols_are_dropped():
    """A NaN score means the model refused the row — it is not a zero.

    This is how a name trading below its exit average disappears: `signal` NaNs
    it out rather than ranking it last.
    """
    table = bist_alerts.build_table(sig({"GOOD": 0.5, "BELOW_MA": np.nan}),
                                    top_pct=100.0)
    assert list(table.index) == ["GOOD"]


def test_the_slice_is_measured_after_the_unscored_are_removed():
    """Ten names with five unscored is a five-name universe, not a ten."""
    rows = {f"S{i}": (0.5 - i / 100.0 if i < 5 else np.nan) for i in range(10)}
    assert len(bist_alerts.build_table(sig(rows), top_pct=40.0)) == 2


def test_an_empty_screen_is_not_an_error():
    table = bist_alerts.build_table(sig({"A": np.nan}))
    assert table.empty
    assert list(table.columns) == [SCORE, EXPECTED_PCT]


def test_expected_pct_rides_along():
    """The percent is carried so a pick reads without re-joining against `sig`."""
    table = bist_alerts.build_table(sig({"A": 0.5}), top_pct=100.0)
    assert table.at["A", EXPECTED_PCT] == pytest.approx(np.expm1(0.5) * 100.0)


def test_render_labels_itself_as_a_screen():
    """The header has to say so, or the output reads like a backtest result."""
    table = bist_alerts.build_table(sig({"GOOD": 0.5}), top_pct=100.0)
    text = bist_alerts.render(table, FakeModel(), pd.Timestamp("2026-07-24"), 632)

    assert "NOT A BACKTEST" in text
    assert "2026-07-24" in text
    assert "GOOD" in text
    assert "score" in text and "exp%" in text


def test_render_states_the_trade_the_scores_are_about():
    """A score is meaningless without the exit rule it was fitted against."""
    table = bist_alerts.build_table(sig({"GOOD": 0.5}), top_pct=100.0)
    text = bist_alerts.render(table, FakeModel(), pd.Timestamp("2026-07-24"), 632)
    assert "MA20" in text and "60-bar cap" in text


def test_render_survives_an_empty_table():
    text = bist_alerts.render(bist_alerts.build_table(sig({"A": np.nan})),
                              FakeModel(), pd.Timestamp("2026-07-24"), 632)
    assert "nothing scored" in text
