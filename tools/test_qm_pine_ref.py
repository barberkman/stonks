"""Unit tests for tools/qm_pine_ref.py — the pine v12 replayer.

Hand-built bar fixtures with hand-computable pine outcomes. Small parameter
overrides (short lookbacks, require_mas off) keep fixtures compact, but every
fixture still carries >= 20 warmup bars: liqOK reads avgVol20, and pine's
na-comparison-is-false semantics keep the universe shut until it exists. The
volume-gate test runs at real defaults because volBreakOK hardcodes the
51-bar avgVol50[1] window, exactly like the pine.

Run from the project root:

    app/python/.venv/bin/pytest tools/test_qm_pine_ref.py -q
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from qm_pine_ref import PineParams, replay

DAY = 86_400_000
HOUR = 3_600_000


def frame(rows, spacing=DAY, start=DAY):
    """rows = [(open, high, low, close, volume)] -> one-symbol DataFrame."""
    return pd.DataFrame({
        "timestamp": [start + i * spacing for i in range(len(rows))],
        "symbol": "TT",
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
        "volume": [float(r[4]) for r in rows],
    })


def flat_rows(n, px=100.0, vol=1000.0):
    return [(px, px * 1.003, px * 0.997, px, vol)] * n


def ramp_rows(n, start=100.0, step=1.01, vol=1000.0):
    rows = []
    c = start
    for _ in range(n):
        o = c
        c = o * step
        rows.append((o, max(o, c) * 1.003, min(o, c) * 0.997, c, vol))
    return rows


def flag_rows(prev_close, n, vol=1000.0):
    rows = []
    c = prev_close
    for j in range(n):
        o = c
        c = o * (0.99 if j == 0 else 1.0005)
        rows.append((o, max(o, c) * 1.001, min(o, c) * 0.999, c, vol))
    return rows


def events_of(events, kind):
    return [e for e in events if e["event"] == kind]


def small_params(**overrides):
    """Compact-fixture params: short lookbacks, no MA gate, volume gate off."""
    base = dict(mom_len=5, adr_len=3, base_max_len=8, min_base_bars=2,
                bo_order_bars=3, require_mas=False, use_bo_vol=False,
                trail_len=3, mintick=0.01)
    base.update(overrides)
    return PineParams(**base)


def bo_fixture():
    """flat warmup + ramp + 2 flag bars: with small_params the BO arms on the
    second flag bar (index 23) at level = peak high + mintick."""
    rows = flat_rows(10) + ramp_rows(12, start=100.0)
    rows += flag_rows(rows[-1][3], 2)
    level = max(r[1] for r in rows) + 0.01
    return rows, level

ARM = 23   # bo_fixture's arm bar


# ─── Volume-gated fill at real defaults ───────────────────────────────────────

def test_arm_skip_volume_then_fill_at_defaults():
    rows = ramp_rows(50)
    rows += flag_rows(rows[-1][3], 3)
    level = max(r[1] for r in rows) + 0.01
    o1 = rows[-1][3]
    # poke through the level at flat volume, close back UNDER it
    rows.append((o1, level * 1.002, o1 * 0.999, o1 * 1.0005, 1000.0))
    o2 = rows[-1][3]
    rows.append((o2, level * 1.003, o2 * 0.999, level * 1.002, 5000.0))  # + volume
    df = frame(rows)
    events, intervals = replay(df, PineParams())

    arms = events_of(events, "arm")
    assert arms and arms[0]["bar_index"] == 52   # 3rd flag bar: sincePk == 3
    assert arms[0]["level"] == pytest.approx(level, rel=1e-12)

    skips = events_of(events, "skip_volume")
    assert len(skips) == 1 and skips[0]["bar_index"] == 53
    assert skips[0]["setup"] == "BO"

    fills = events_of(events, "fill")
    assert len(fills) == 1
    f = fills[0]
    assert f["bar_index"] == 54 and f["setup"] == "BO"
    # entryPx = max(open, level); the open sits below the level here
    assert f["entry"] == pytest.approx(level, rel=1e-12)
    # stop = min(max(entry x (1 - adr/100), fill-bar low), entry x 0.999),
    # with adr = SMA(100 x (high/low - 1), 20) on the fill bar
    fill_bar = rows[54]
    window = rows[54 - 19: 54 + 1]
    adr = np.mean([100.0 * (hi / lo - 1.0) for (_, hi, lo, _, _) in window])
    expected_stop = min(max(f["entry"] * (1.0 - adr / 100.0), fill_bar[2]),
                        f["entry"] * 0.999)
    assert f["stop"] == pytest.approx(expected_stop, rel=1e-9)
    assert f["target"] == pytest.approx(f["entry"] + 2.0 * (f["entry"] - f["stop"]),
                                        rel=1e-9)
    assert intervals and intervals[0]["entry_bar"] == 54


# ─── Resting-order timer: strict > (last fillable bar = arm + N) ──────────────

def timer_rows(cross_at):
    """Arm at ARM, then declining non-setup bars below the level, then one
    crossing bar `cross_at` bars after the arm."""
    rows, level = bo_fixture()
    c = rows[-1][3]
    for _ in range(cross_at - 1):              # -1.5%/bar: gain gate dies fast
        o = c
        c = o * 0.985
        rows.append((o, o * 1.0005, c * 0.999, c, 1000.0))
    o = c
    rows.append((o, level * 1.001, o * 0.999, level * 1.0005, 1000.0))
    return rows, level


def test_timer_fills_on_last_valid_bar():
    rows, level = timer_rows(cross_at=3)       # bo_order_bars = 3
    events, _ = replay(frame(rows), small_params())
    arms = events_of(events, "arm")
    assert arms and arms[-1]["bar_index"] == ARM
    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["bar_index"] == ARM + 3
    assert fills[0]["entry"] == pytest.approx(level, rel=1e-12)


def test_timer_expires_one_bar_later():
    rows, _ = timer_rows(cross_at=4)
    events, intervals = replay(frame(rows), small_params())
    assert events_of(events, "fill") == []
    expires = events_of(events, "expire")
    assert len(expires) == 1 and expires[-1]["bar_index"] == ARM + 4
    assert intervals == []


# ─── Trade management: partial -> breakeven -> trail ──────────────────────────

def managed_fill_rows():
    """Fill on the bar after the arm with a DEEP low: the ADR stop wins over
    the LOD stop, R is ~1% of entry, so drift bars can't tag 2R by accident.
    The fill bar gaps through the level -> entryPx = its open."""
    rows, level = bo_fixture()
    o = level * 1.0005
    c = o * 1.001
    rows.append((o, c * 1.001, o * 0.98, c, 1000.0))   # bar ARM+1 = fill
    return rows, level, o                              # entryPx = o


def drift_rows(rows, n, step=1.002):
    c = rows[-1][3]
    for _ in range(n):
        o = c
        c = o * step
        rows.append((o, c * 1.001, o * 0.998, c, 1000.0))


def test_partial_be_then_trail_exit():
    rows, level, entry = managed_fill_rows()
    drift_rows(rows, 2)                        # ARM+2, ARM+3: quiet hold
    c = rows[-1][3]
    # partial bar: tags 2R, closes strong (above the 3-EMA, above the stop)
    rows.append((c, entry * 1.03, c * 0.999, c * 1.004, 1000.0))
    c = rows[-1][3]
    # trail bar: closes below the 3-EMA but above the breakeven stop
    rows.append((c, c * 1.0005, entry * 1.0005, entry * 1.001, 1000.0))
    events, intervals = replay(frame(rows), small_params(partial_days=99))

    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["bar_index"] == ARM + 1
    assert fills[0]["entry"] == pytest.approx(entry, rel=1e-12)

    partials = events_of(events, "partial")
    assert len(partials) == 1 and partials[0]["bar_index"] == ARM + 4
    be = events_of(events, "be_move")
    assert len(be) == 1 and be[0]["stop"] == pytest.approx(entry, rel=1e-12)

    exits = events_of(events, "exit")
    assert len(exits) == 1 and exits[0]["reason"] == "trail"
    assert exits[0]["bar_index"] == ARM + 5
    assert intervals[-1]["exit_reason"] == "trail"


def test_time_partial_after_partial_days():
    rows, level, entry = managed_fill_rows()
    drift_rows(rows, 3, step=1.001)            # no 2R touch, no stop touch
    events, _ = replay(frame(rows), small_params(partial_days=2))
    partials = events_of(events, "partial")
    assert len(partials) == 1
    assert partials[0]["bar_index"] == ARM + 3   # fill bar + partial_days
    assert events_of(events, "be_move")          # BE follows the time partial


def test_same_bar_partial_and_trail_exit():
    rows, level, entry = managed_fill_rows()
    drift_rows(rows, 4)                        # EMA-3 catches up under the closes
    c = rows[-1][3]
    # one bar tags 2R AND closes below the 3-EMA but above breakeven:
    # pine takes the partial, moves to BE, then trail-exits the same bar
    rows.append((c, entry * 1.03, entry * 1.002, entry * 1.004, 1000.0))
    events, intervals = replay(frame(rows), small_params(partial_days=99))
    partials = events_of(events, "partial")
    exits = events_of(events, "exit")
    assert len(partials) == 1 and len(exits) == 1
    assert partials[0]["bar_index"] == exits[0]["bar_index"] == ARM + 6
    assert exits[0]["reason"] == "trail"
    assert intervals[-1]["exit_reason"] == "trail"


def test_same_bar_collapse_bail():
    rows, level = bo_fixture()
    o = rows[-1][3]
    # spikes through the level, closes in the basement (below the ADR stop)
    rows.append((o, level * 1.001, o * 0.90, o * 0.905, 1000.0))
    events, intervals = replay(frame(rows), small_params())
    fills = events_of(events, "fill")
    exits = events_of(events, "exit")
    assert len(fills) == 1 and len(exits) == 1
    assert exits[0]["reason"] == "same_bar_collapse"
    assert exits[0]["bar_index"] == fills[0]["bar_index"] == ARM + 1
    assert intervals[-1]["exit_reason"] == "same_bar_collapse"


# ─── PARA: first red bar + covers ─────────────────────────────────────────────

def para_base_rows():
    rows = flat_rows(25)
    c = 100.0
    for _ in range(5):                         # 5 consecutive up-closes
        o = c
        c = o * 1.025
        rows.append((o, c * 1.002, o * 0.998, c, 1000.0))
    o = c
    c = o * 0.99                               # first red bar -> PARA fires
    rows.append((o, o * 1.001, c * 0.998, c, 1000.0))
    return rows


def test_para_fill_and_bounce_cover():
    rows = para_base_rows()
    stop_ref = max(r[1] for r in rows[-3:]) + 0.01
    o = rows[-1][3]
    c = o * 1.004                              # green bar, high stays below stop
    rows.append((o, c * 1.001, o * 0.999, c, 1000.0))
    events, intervals = replay(frame(rows), PineParams(enable_para=True))

    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["setup"] == "PARA"
    assert fills[0]["bar_index"] == 30
    assert fills[0]["entry"] == pytest.approx(rows[30][3], rel=1e-12)
    assert fills[0]["stop"] == pytest.approx(stop_ref, rel=1e-12)

    exits = events_of(events, "exit")
    assert len(exits) == 1 and exits[0]["reason"] == "para_bounce"
    assert intervals[-1]["exit_reason"] == "para_bounce"


def test_para_time_cover():
    rows = para_base_rows()
    c = rows[-1][3]
    for _ in range(2):                         # non-increasing closes, no stop touch
        o = c
        rows.append((o, o * 1.001, o * 0.997, c, 1000.0))
    events, _ = replay(frame(rows), PineParams(enable_para=True, ps_max_hold=2))
    exits = events_of(events, "exit")
    assert len(exits) == 1 and exits[0]["reason"] == "para_time"
    assert exits[0]["bar_index"] == 32         # entry 30 + ps_max_hold 2


# ─── canEnter: armed break with volume while holding is only a skip ───────────

def ep_while_armed_rows(ep_crosses_level):
    """45 flat bars (avgVol50[1] warmup for the EP volume gate) + ramp + flag:
    BO arms; then an EP gap bar whose high stays under / crosses the level."""
    rows = flat_rows(45) + ramp_rows(8, start=100.0)
    rows += flag_rows(rows[-1][3], 2)          # BO arms on the 2nd flag bar (54)
    level = max(r[1] for r in rows) + 0.01
    o = rows[-1][3] * 1.006                    # +0.6% gap, strong close, volume
    c = level * 1.002 if ep_crosses_level else o * 1.003
    rows.append((o, c * 1.0005, o * 0.999, c, 5000.0))
    return rows, level


def test_skip_in_pos_while_holding():
    rows, level = ep_while_armed_rows(ep_crosses_level=False)
    assert rows[55][1] < level                 # the EP bar itself never crossed
    o2 = rows[-1][3]
    rows.append((o2, level * 1.002, o2 * 0.999, level * 1.001, 1000.0))
    events, intervals = replay(frame(rows), small_params(enable_ep=True))

    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["setup"] == "EP"
    assert fills[0]["bar_index"] == 55

    skips = events_of(events, "skip_in_pos")
    assert len(skips) == 1 and skips[0]["bar_index"] == 56
    assert skips[0]["setup"] == "BO"


def test_bo_beats_ep_on_the_same_bar():
    rows, level = ep_while_armed_rows(ep_crosses_level=True)
    events, _ = replay(frame(rows), small_params(enable_ep=True))
    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["setup"] == "BO"
    o = rows[55][0]
    assert fills[0]["entry"] == pytest.approx(max(o, level), rel=1e-12)


# ─── ta.highestbars tie-break ─────────────────────────────────────────────────

def test_tie_break_recent_vs_oldest():
    """Two equal window highs, 2 and 5 bars back of the evaluation bar:
    tie=recent -> sincePk=2 < min_base_bars(3) -> no arm;
    tie=oldest -> sincePk=5 -> arms at level = peak + mintick."""
    rows = flat_rows(10) + ramp_rows(10, start=100.0, step=1.005)
    peak = 120.0

    def bar(hi_val):
        o = bar.c
        bar.c = o * 1.002                      # +0.2%/bar keeps the gain gate open
        return (o, hi_val, min(o, bar.c) * 0.999, bar.c, 1000.0)
    bar.c = rows[-1][3]
    rows.append(bar(peak))                     # bar 20: first equal high
    rows.append(bar(bar.c * 1.001))            # 21
    rows.append(bar(bar.c * 1.001))            # 22
    rows.append(bar(peak))                     # bar 23: second equal high
    rows.append(bar(bar.c * 1.001))            # 24
    rows.append(bar(bar.c * 1.001))            # bar 25: evaluation bar
    df = frame(rows)

    ev_recent, _ = replay(df, small_params(min_base_bars=3))
    ev_oldest, _ = replay(df, small_params(min_base_bars=3, tie="oldest"))

    def armlike(events):
        return {e["bar_index"]: e for e in events
                if e["event"] in ("arm", "rearm")}
    recent_arms = armlike(ev_recent)
    oldest_arms = armlike(ev_oldest)
    assert recent_arms == {}                   # sincePk <= 2 everywhere -> no arm
    assert 25 in oldest_arms                   # sincePk = 5 -> (re)arms
    assert oldest_arms[25]["level"] == pytest.approx(peak + 0.01, rel=1e-12)


# ─── ORB on intraday spacing ──────────────────────────────────────────────────

def test_orb_fill_on_hourly_bars():
    rows = flat_rows(24)                       # day 1: warmup, universe shut
    c = rows[-1][3]
    rows.append((c, c * 1.007, c * 0.999, c * 1.004, 1000.0))   # bar 24: day 2 opens
    or_high = rows[24][1]
    c = rows[-1][3]
    rows.append((c, or_high - 0.5, c * 0.999, c * 1.001, 1000.0))  # 25: under the OR
    o = rows[-1][3]
    rows.append((o, or_high * 1.01, o * 0.999, or_high * 1.005, 1000.0))  # 26: break
    df = frame(rows, spacing=HOUR, start=0)    # bar 24 starts a new UTC day
    events, _ = replay(df, small_params(enable_orb=True, enable_bo=False))
    fills = events_of(events, "fill")
    assert len(fills) == 1 and fills[0]["setup"] == "ORB"
    assert fills[0]["bar_index"] == 26
    assert fills[0]["entry"] == pytest.approx(max(o, or_high + 0.01), rel=1e-12)


# ─── Warmup safety ────────────────────────────────────────────────────────────

def test_nan_warmup_produces_no_events():
    rows = ramp_rows(10)
    events, intervals = replay(frame(rows), PineParams())
    assert events == [] and intervals == []
