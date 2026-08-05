"""Tests for tools/pattern_compare.py — the benchmark math and the sweep join.

The comparison's whole load-bearing claim is that the market number covers the
same bars the strategy saw, so the fixtures here are tiny panels with a
hand-computable answer and one listing quirk each: a late listing, a delisting,
and a day a listed symbol did not print.

Run from the project root with the app-local venv (pandas/pyarrow needed):

    app/python/.venv/bin/pytest tools/test_pattern_compare.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import pattern_compare as pc  # noqa: E402

DAY = 86_400_000


def panel(tmp_path, series: dict) -> Path:
    """`{symbol: [close, ...]}` -> a parquet the benchmark can read. None is a
    bar the symbol did not print."""
    rows = []
    for symbol, closes in series.items():
        for i, close in enumerate(closes):
            if close is None:
                continue
            rows.append({"timestamp": i * DAY, "symbol": symbol,
                         "open": close, "high": close, "low": close,
                         "close": close, "volume": 1000.0})
    path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_equal_weight_index_compounds_the_cross_sectional_mean(tmp_path):
    # A: +10%, +10%, 0    B: -10%, 0, +100%
    # daily mean: 0, +5%, +50%  ->  1.05 * 1.5 = 1.575
    b = pc.benchmark(panel(tmp_path, {"A": [100, 110, 121, 121],
                                      "B": [100, 90, 90, 180]}))
    assert b["return_pct"] == pytest.approx(57.5)
    assert b["symbols"] == 2


def test_passive_and_median_are_per_symbol_buy_and_hold(tmp_path):
    # A holds +21%, B holds +80%: mean 50.5, median 50.5.
    b = pc.benchmark(panel(tmp_path, {"A": [100, 110, 121, 121],
                                      "B": [100, 90, 90, 180]}))
    assert b["passive_pct"] == pytest.approx(50.5)
    assert b["median_symbol_pct"] == pytest.approx(50.5)


def test_a_late_listing_is_not_credited_with_the_move_it_missed(tmp_path):
    # C lists on day 2 and doubles on day 3. Days 1-2 are A's alone; day 3
    # averages A's 0 with C's +100%. Were C carried back at its first price it
    # would have added a phantom flat leg, and its own +100% would still land.
    b = pc.benchmark(panel(tmp_path, {"A": [100, 110, 121, 121],
                                      "C": [None, None, 100, 200]}))
    assert b["return_pct"] == pytest.approx((1.10 * 1.10 * 1.50 - 1) * 100)
    # C only ever quoted 100 -> 200, so its buy-and-hold is its own +100%.
    assert b["median_symbol_pct"] == pytest.approx((21.0 + 100.0) / 2)
    # ...but the do-nothing benchmark only buys what was listed on day one.
    assert b["passive_pct"] == pytest.approx(21.0)


def test_a_delisting_stops_contributing_instead_of_carrying_flat(tmp_path):
    # D halves and stops quoting. If its last price were forward-filled it
    # would keep voting 0% and halve the index's daily mean forever.
    b = pc.benchmark(panel(tmp_path, {"A": [100, 110, 121, 121],
                                      "D": [100, 50, None, None]}))
    day1 = (0.10 + -0.50) / 2
    assert b["return_pct"] == pytest.approx(((1 + day1) * 1.10 * 1.00 - 1) * 100)


def test_a_missed_day_inside_a_listing_is_carried_flat(tmp_path):
    # E does not print on day 2, then prints unchanged on day 3: that is one
    # 0% day and one 0% day, not a gap the index skips.
    b = pc.benchmark(panel(tmp_path, {"E": [100, 110, None, 110]}))
    assert b["return_pct"] == pytest.approx(10.0)
    assert b["max_dd_pct"] == pytest.approx(0.0)


def test_max_drawdown_is_the_worst_peak_to_trough(tmp_path):
    # One symbol, so the index is that symbol: 100 -> 120 -> 60 -> 90.
    b = pc.benchmark(panel(tmp_path, {"A": [100, 120, 60, 90]}))
    assert b["max_dd_pct"] == pytest.approx(50.0)
    assert b["return_pct"] == pytest.approx(-10.0)


def sweep(tmp_path, first_return=None):
    """Every registered pattern as a MISSING row, optionally with index 0
    turned into a run that traded."""
    rows = {}
    for index, name, side, tradeable in pc.load_patterns():
        rows[index] = pc.row_for(index, name, side, tradeable,
                                 tmp_path / "nope.json")
    if first_return is not None:
        r = rows[0]
        r.status, r.closed, r.return_pct = "ok", 5, first_return
    return rows


def test_excess_is_the_pattern_return_less_its_own_market(tmp_path):
    """build_frame subtracts each market's benchmark, not a shared one."""
    us = pc.Market("us", Path("us.parquet"), Path("us"), return_pct=10.0)
    tr = pc.Market("tr", Path("tr.parquet"), Path("tr"), return_pct=400.0)
    sweeps = {"us": sweep(tmp_path, 25.0), "tr": sweep(tmp_path, 25.0)}

    frame = pc.build_frame([us, tr], sweeps)
    first = frame.iloc[0]
    assert first["us_excess_pct"] == pytest.approx(15.0)
    assert first["tr_excess_pct"] == pytest.approx(-375.0)


def test_a_missing_report_is_flagged_not_scored(tmp_path):
    m = pc.Market("us", Path("us.parquet"), Path("us"), return_pct=10.0)
    rows = sweep(tmp_path)
    assert rows[0].status == "MISSING"

    frame = pc.build_frame([m], {"us": rows})
    assert pd.isna(frame.iloc[0]["us_return_pct"])
    assert pd.isna(frame.iloc[0]["us_excess_pct"])
    assert pc.traded(frame, m).empty
