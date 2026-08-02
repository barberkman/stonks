"""Tests for the single-date screen in tools/bist_alerts.py.

`screen()` is thin — it truncates a panel, trains, and scores, all of which is
covered by test_algotrade.py. `build_table` is the part with rules of its own: the
threshold, the `n` count, the composite and the ordering are bist's and are easy
to get subtly wrong.
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
from algo_trade import HEADS  # noqa: E402


def sig(rows):
    """{symbol: (up_h5, up_h10, dn_h5)} -> a signal frame like `signal` returns."""
    frame = pd.DataFrame.from_dict(
        rows, orient="index", columns=[h.name for h in HEADS])
    for head in HEADS:
        frame[f"{head.name}_pct"] = frame[head.name] * 10.0
    return frame


def test_threshold_admits_a_symbol_on_either_up_head():
    """bist's rule is "above it in ANY up config", not both."""
    table = bist_alerts.build_table(sig({
        "BOTH": (1.5, 1.2, 0.0),
        "H5":   (1.5, 0.4, 0.0),
        "H10":  (0.4, 1.5, 0.0),
        "NONE": (0.9, 0.9, 0.0),
    }), HEADS)
    assert set(table.index) == {"BOTH", "H5", "H10"}


def test_n_counts_the_configs_above_the_threshold():
    table = bist_alerts.build_table(sig({
        "BOTH": (1.5, 1.2, 0.0),
        "H5":   (1.5, 0.4, 0.0),
    }), HEADS)
    assert table.at["BOTH", "n"] == 2
    assert table.at["H5", "n"] == 1


def test_composite_is_the_mean_of_the_up_heads_and_sets_the_order():
    table = bist_alerts.build_table(sig({
        "LOW":  (1.1, 1.0, 0.0),   # 1.05
        "HIGH": (4.0, 2.0, 0.0),   # 3.00
        "MID":  (2.0, 1.4, 0.0),   # 1.70
    }), HEADS)
    assert list(table.index) == ["HIGH", "MID", "LOW"]
    assert table.at["MID", "composite"] == pytest.approx(1.7)


def test_the_down_head_is_carried_but_never_filters():
    """bist displays dn as an overlay; a screaming downside must still show."""
    table = bist_alerts.build_table(sig({
        "UGLY": (2.0, 1.5, 9.9),
    }), HEADS)
    assert list(table.index) == ["UGLY"]
    assert table.at["UGLY", "dn_h5"] == pytest.approx(9.9)


def test_unscored_symbols_are_dropped():
    """A NaN up head means the model refused the row — it is not a zero."""
    table = bist_alerts.build_table(sig({
        "GOOD": (2.0, 1.5, 0.1),
        "NAN":  (np.nan, np.nan, np.nan),
        "HALF": (2.0, np.nan, 0.1),
    }), HEADS)
    assert list(table.index) == ["GOOD"]


def test_an_empty_screen_is_not_an_error():
    table = bist_alerts.build_table(sig({"A": (0.1, 0.1, 0.0)}), HEADS)
    assert table.empty


def test_render_labels_itself_as_a_screen():
    """The header has to say so, or the output reads like a backtest result."""
    class FakeModel:
        heads = HEADS
        liquid = {"GOOD"}
        min_turnover_percentile = 0.40
        train_start = "2024-01-01"

    table = bist_alerts.build_table(sig({"GOOD": (2.0, 1.5, 0.1)}), HEADS)
    text = bist_alerts.render(table, FakeModel(), pd.Timestamp("2026-07-24"), 632)

    assert "NOT A BACKTEST" in text
    assert "2026-07-24" in text
    assert "GOOD" in text
    assert "h5 sigma" in text and "h10 exp%" in text and "h5 dn" in text


def test_render_survives_an_empty_table():
    class FakeModel:
        heads = HEADS
        liquid = set()
        min_turnover_percentile = 0.40
        train_start = "2024-01-01"

    text = bist_alerts.render(bist_alerts.build_table(sig({"A": (0.1, 0.1, 0.0)}),
                                                      HEADS),
                              FakeModel(), pd.Timestamp("2026-07-24"), 632)
    assert "nothing cleared" in text
