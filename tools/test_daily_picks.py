"""Tests for tools/daily_picks.py's gate.

Everything else in that script is either subprocess orchestration or a call into
`algo_trade`, which has its own suite — the sigma conversion moved there when
`AlgoTradeStrategy` started gating on it, and is covered by
app/python/test_algotrade.py. `top_slice` and the markdown renderer are the
logic this script still owns, and `top_slice` is the piece that silently
degenerates: the one-name floor means a smaller
`--top-pct` stops being a tighter gate below 100/n, which is exactly the failure
that looks like a working table.

Run from the project root with the app-local venv:

    app/python/.venv/bin/pytest tools/test_daily_picks.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from daily_picks import markdown_table, naive_rank, top_slice


def cross_section(n):
    """`n` names whose scores descend, so the expected pick is unambiguous."""
    return pd.DataFrame({"symbol": [f"S{i:03d}" for i in range(n)],
                         "score": [float(n - i) for i in range(n)]})


@pytest.mark.parametrize("n, pct, expected", [
    (200, 2.0, 4),
    (200, 1.0, 2),
    (150, 2.0, 3),
    (134, 1.0, 1),      # the live cross-section this script actually sees
    (1000, 0.5, 5),
])
def test_take_is_truncated_not_rounded(n, pct, expected):
    """`int()`, not `round()` — 150 * 2% is 3.0 and 134 * 1% is 1, not 2."""
    picks, take = top_slice(cross_section(n), pct)
    assert take == expected
    assert len(picks) == expected


def test_returns_the_highest_scores():
    picks, _ = top_slice(cross_section(200), 2.0)
    assert list(picks["symbol"]) == ["S000", "S001", "S002", "S003"]


def test_floor_keeps_one_name_when_the_percentage_rounds_to_zero():
    """A gate that selected nothing would print an empty table, not a warning."""
    picks, take = top_slice(cross_section(134), 0.1)
    assert take == 1
    assert list(picks["symbol"]) == ["S000"]


def test_percentages_below_the_floor_are_the_same_gate():
    """0.1% and 0.5% of 134 names are not two gates — they are one.

    The script's docstring makes this claim; if the floor ever changes, the
    claim has to change with it.
    """
    a, _ = top_slice(cross_section(134), 0.1)
    b, _ = top_slice(cross_section(134), 0.5)
    assert list(a["symbol"]) == list(b["symbol"])


def blotter_shaped():
    """One resolved row and one open row — the two shapes a pick table holds."""
    frame = pd.DataFrame({
        "symbol": ["IEYHO", "TEHOL"],
        "signal_date": [pd.Timestamp("2026-08-05")] * 2,
        "entry_date": [pd.NaT, pd.NaT],
        "bars_held": [np.nan, 3.0],
        "score": [1.1116, 0.6144],
        "limit_up": [False, True],
    }, index=pd.Index([1, 2], name="rank"))
    return frame


def test_markdown_table_is_well_formed():
    lines = markdown_table(blotter_shaped()).splitlines()
    assert len(lines) == 4                      # header, rule, two rows
    pipes = {line.count("|") for line in lines}
    assert len(pipes) == 1                      # every row has the same columns
    assert set(lines[1]) <= {"|", "-"}          # the separator is only a rule


def test_markdown_table_names_the_index_and_columns():
    header = markdown_table(blotter_shaped()).splitlines()[0]
    for name in ("rank", "symbol", "signal_date", "score", "limit_up"):
        assert name in header


def test_markdown_table_distinguishes_nat_from_nan():
    """A missing *date* and a missing *number* are different facts about a
    trade — no entry yet vs. no result yet — and the table has to keep them
    apart the way the notebook's blotter does."""
    body = markdown_table(blotter_shaped()).splitlines()[2:]
    assert "NaT" in body[0] and "NaN" in body[0]


def test_markdown_table_renders_timestamps_as_dates():
    assert "2026-08-05" in markdown_table(blotter_shaped())
    assert "00:00:00" not in markdown_table(blotter_shaped())


def naive_frame(vol, extended):
    return pd.DataFrame({"realized_vol_60": vol, "distance_from_ma_20": extended})


def test_naive_rewards_the_calmest_and_most_extended_name():
    """The two ranks pull in opposite directions on `realized_vol_60`.

    Low vol is good and high distance is good, so the name that is both best is
    the only one that can reach 1.0 — that orientation is the whole column, and
    an inverted `ascending` would flip the gate without changing its shape.
    """
    naive = naive_rank(naive_frame([0.01, 0.02, 0.03], [3.0, 2.0, 1.0]))
    assert naive.iloc[0] == pytest.approx(1.0)
    assert naive.iloc[1] == pytest.approx(2.0 / 3.0)
    assert naive.iloc[2] == pytest.approx(1.0 / 3.0)


def test_naive_is_a_rank_not_a_level():
    """Only the ordering matters, so a monotone rescale must not move it."""
    a = naive_rank(naive_frame([0.01, 0.02, 0.03], [3.0, 2.0, 1.0]))
    b = naive_rank(naive_frame([0.10, 0.90, 5.00], [900.0, 2.5, -4.0]))
    assert list(a) == list(b)


def test_naive_halves_disagree_independently():
    """Calm and extended are averaged, not intersected.

    The loudest name in the field still outranks a quieter one when it is far
    enough above its average — ranking on vol alone would order these two the
    other way round. If the average ever collapses to a single feature, this is
    the case that catches it.

    (Two names cannot show this: with n=2 the two ranks are exact mirrors and
    every row ties at 0.75.)
    """
    vol = [0.04, 0.01, 0.02, 0.03]
    naive = naive_rank(naive_frame(vol, [10.0, 5.0, 1.0, 0.0]))
    assert vol[0] > vol[3]                      # row 0 is the louder of the two
    assert naive.iloc[0] > naive.iloc[3]        # ...and still ranks above it


def test_naive_stays_inside_the_unit_interval():
    rng = np.random.default_rng(0)
    naive = naive_rank(naive_frame(rng.random(50), rng.random(50)))
    assert naive.min() > 0.0 and naive.max() <= 1.0


def test_naive_is_nan_where_a_feature_is_missing():
    """A name with no 60-bar vol has no naive rank, and says so.

    It must not be imputed to mid-field, and it must not drag the names around
    it — the other two keep ranking against each other alone.
    """
    naive = naive_rank(naive_frame([0.01, np.nan, 0.03], [3.0, 2.0, 1.0]))
    assert np.isnan(naive.iloc[1])
    assert naive.iloc[0] > naive.iloc[2]
    assert naive.notna().sum() == 2


def test_ties_do_not_select_more_than_the_quota():
    """`nlargest` breaks ties by position, so the count is exact regardless."""
    flat = pd.DataFrame({"symbol": [f"S{i}" for i in range(200)],
                         "score": [1.0] * 200})
    picks, take = top_slice(flat, 2.0)
    assert take == 4 and len(picks) == 4
