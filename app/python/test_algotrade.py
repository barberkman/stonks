"""Unit tests for AlgoTrade — the bist manipulation model as a strategy.

Two halves. The model half checks the properties the port rests on: features are
causal, they are window-bounded except for the two `ScoringState` carries, and
labels simulate the trade the strategy holds without ever reading past the
training cutoff. The strategy half injects a stub model so the entry rule, the
bracket shape and the leak guard can be tested without training anything.

What is *not* here is parity with bist itself — that lives in test_bist_parity.py,
which diffs this module against bist's own code on the same input. These tests
pin behaviour; that one pins fidelity.

FakeContext never fills orders on its own; a fill is simulated by setting
ctx.positions, mirroring what the broker would report on the next tick. Engine
fill mechanics (next-open market fills, bracket arming, reduce-only) are pinned
by the C++ suite under tests/core/.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import stonks
from stonks import OrderSide
from stonks.testing import FakeContext, FakeKLine, FakePosition

import algo_trade
from algo_trade import (
    DAYS_SINCE_COLUMN,
    EXPECTED_PCT,
    FEATURE_NAMES,
    LOOKBACK,
    MAX_HOLD,
    OBV_COLUMN,
    SCORE,
    AlgoTradeStrategy,
    ManipulationModel,
)

# Every column except the two that accumulate from a symbol's first bar ever:
# `obv` is a running signed-volume sum and `days_since_past_extreme` is a counter
# with a "never" sentinel. A bounded window cannot reproduce either, which is what
# `ScoringState` exists for. See test_two_features_need_the_carried_state.
CARRIED = (OBV_COLUMN, DAYS_SINCE_COLUMN)
WINDOWED = [j for j in range(len(FEATURE_NAMES)) if j not in CARRIED]


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_from_arrays(close, volume=None, high=None, low=None, open_=None):
    """(T, N) close (and optional friends) -> the panel dict `_features` wants."""
    close = np.asarray(close, dtype=float)
    T, N = close.shape
    if volume is None:
        volume = np.full((T, N), 1_000_000.0)
    if high is None:
        high = close * 1.01
    if low is None:
        low = close * 0.99
    if open_ is None:
        open_ = close * 1.002
    panel = {
        "open": pd.DataFrame(open_), "high": pd.DataFrame(high),
        "low": pd.DataFrame(low), "close": pd.DataFrame(close),
        "volume": pd.DataFrame(volume),
    }
    panel["ret"] = algo_trade._returns(panel["close"])
    return panel


def random_panel(T=400, N=14, seed=0, *, quirks=True):
    """A synthetic BIST-ish panel: random walks plus the pathologies that matter.

    The quirks are the point. Flat windows, dead volume, limit locks and late
    listings are exactly the cases where an incrementally-computed rolling
    moment stops being reproducible, which is what `_std` / `_wsum` / `_moments`
    exist to avoid.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(T, N)), axis=0))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.01, size=(T, N))))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.01, size=(T, N))))
    open_ = low + (high - low) * rng.random((T, N))
    volume = rng.lognormal(10.0, 1.5, size=(T, N))

    if quirks:
        close[: T // 8, N - 1] = np.nan                  # late listing
        high[: T // 8, N - 1] = low[: T // 8, N - 1] = np.nan
        open_[: T // 8, N - 1] = np.nan
        close[T // 2: T // 2 + 5, 3] = np.nan            # mid-window halt
        volume[T // 3: T // 3 + 20, 4] = 0.0             # dead name
        flat = slice(T - 60, T - 45)                     # limit-locked, no range
        high[flat, 2] = low[flat, 2] = close[flat, 2] = close[T - 61, 2]
        open_[flat, 2] = close[T - 61, 2]

    return panel_from_arrays(close, volume, high, low, open_)


def window_of(panel, i, lookback=LOOKBACK):
    """The trailing `lookback` rows ending at row i, as a panel."""
    return {k: v.iloc[i - lookback + 1: i + 1] for k, v in panel.items()}


# ---------------------------------------------------------------------------
# The windowing contract
# ---------------------------------------------------------------------------

def test_features_are_window_bounded():
    """features(panel)[i] must equal features(panel[i-299:i+1])[-1] bit for bit,
    at the precision that reaches the trees.

    This is the invariant that lets one model be fit on a whole panel and scored
    one bar at a time. Not "close enough": a tree split sitting on zero turns a
    tiny disagreement into a different leaf, so the NaN pattern and the values
    both have to match exactly.

    Compared through `_design_matrix`, which is float32 — bist's own narrowing,
    applied at the last step before the model. In raw float64 the two paths agree
    only to about 1e-14 relative, because pandas' rolling mean accumulates
    incrementally and a window starts at a different row. Asserting float64
    equality here would be asserting something the port does not need and cannot
    deliver; asserting float32 equality is exactly the guarantee the fit relies on.

    `obv` is excluded because it provably cannot hold — see
    test_obv_is_the_only_feature_a_window_cannot_reproduce.
    """
    panel = random_panel(T=420, N=12, seed=7)
    full = algo_trade._design_matrix(algo_trade._features(panel))

    for i in (LOOKBACK - 1, 340, 400, 419):
        windowed = algo_trade._design_matrix(
            algo_trade._features(window_of(panel, i)))[-1][:, WINDOWED]
        row = full[i][:, WINDOWED]
        assert np.array_equal(row, windowed), f"feature values differ at row {i}"


def test_features_are_window_bounded_in_float64_to_a_tight_tolerance():
    """The float64 residual is bounded, so the float32 test above is not a dodge.

    If a future change made a feature genuinely path-dependent — a cumulative sum,
    a counter, an expanding window — the error would be O(1), not O(1e-12), and
    this catches it well before the float32 comparison stops noticing.
    """
    panel = random_panel(T=420, N=12, seed=7)
    full = algo_trade._features(panel)

    for i in (LOOKBACK - 1, 419):
        windowed = algo_trade._features(window_of(panel, i))[-1][:, WINDOWED]
        row = full[i][:, WINDOWED]
        assert np.array_equal(np.isnan(row), np.isnan(windowed)), (
            f"NaN pattern differs at row {i}")
        both = np.isfinite(row) & np.isfinite(windowed)
        scale = np.maximum(np.abs(row[both]), 1e-12)
        assert (np.abs(row[both] - windowed[both]) / scale).max() < 1e-9


def test_two_features_need_the_carried_state():
    """Exactly two features are not window-bounded, and both are carried.

    Pinning the count both ways. If a third feature became unbounded this fails and
    says so, which matters because the failure mode is silent: the strategy would
    keep scoring, just against values the fit never saw.
    """
    panel = random_panel(T=420, N=12, seed=7)
    i = 419
    full = algo_trade._design_matrix(algo_trade._features(panel))[i]
    windowed = algo_trade._design_matrix(
        algo_trade._features(window_of(panel, i)))[-1]

    unbounded = {j for j in range(len(FEATURE_NAMES))
                 if not np.array_equal(full[:, j], windowed[:, j])}
    # A subset, not equality: whether `days_since_past_extreme` actually diverges
    # depends on when the panel's last trigger fired, and on this synthetic one no
    # trigger fires at all so both paths report the 9999 sentinel. Its divergence
    # is pinned separately below.
    assert unbounded <= set(CARRIED), (
        "a feature outside the carried set is not window-bounded: "
        + str(sorted(FEATURE_NAMES[j] for j in unbounded - set(CARRIED))))
    assert OBV_COLUMN in unbounded, "obv must rebase in a window"

    # obv_slope_20 is NOT among them, even though obv is: the slope kernel demeans
    # its window, so the additive history constant cancels.
    slope = FEATURE_NAMES.index("obv_slope_20")
    assert np.array_equal(full[:, slope], windowed[:, slope])


def test_days_since_past_extreme_rebases_in_a_window():
    """A trigger older than the window reads as "never", which is a real number.

    Constructed rather than sampled: one symbol gets an idiosyncratic run early
    enough that the trigger falls off the front of a 300-bar window. The full panel
    then reports a real count where the window reports the 9999 sentinel.
    """
    T, N = 700, 12
    panel = random_panel(T=T, N=N, seed=2, quirks=False)
    close = panel["close"].to_numpy().copy()
    # A sustained idiosyncratic ramp around row 150, well before row 699 - 299.
    close[150:170, 0] *= np.linspace(1.0, 3.0, 20)
    close[170:, 0] *= 3.0
    panel = panel_from_arrays(close, panel["volume"].to_numpy(),
                             panel["high"].to_numpy() * 3.0,
                             panel["low"].to_numpy() / 3.0,
                             panel["open"].to_numpy())

    i = T - 1
    full = algo_trade._features(panel)[i][:, DAYS_SINCE_COLUMN]
    windowed = algo_trade._features(window_of(panel, i))[-1][:, DAYS_SINCE_COLUMN]
    assert not np.array_equal(full, windowed, equal_nan=True), (
        "expected the window to lose an out-of-range trigger")


def test_features_ignore_the_future():
    """Scrambling every row after i must not move the features at row i.

    The post-cut scramble from trade_algo/check_lookahead.py. Rolling windows
    only reach backwards, so this should hold by construction — the test is here
    to catch a future feature that quietly reaches forward.
    """
    panel = random_panel(T=380, N=10, seed=11)
    i = 330
    baseline = algo_trade._features(panel)[i]

    rng = np.random.default_rng(99)
    scrambled = {}
    for key, frame in panel.items():
        values = frame.to_numpy(dtype=float).copy()
        tail = values[i + 1:]
        rng.shuffle(tail, axis=0)
        values[i + 1:] = tail * 3.0
        scrambled[key] = pd.DataFrame(values, index=frame.index, columns=frame.columns)
    scrambled["ret"] = algo_trade._returns(scrambled["close"])

    # Includes obv: it is a *backward* running sum, so the future cannot reach it.
    assert np.array_equal(baseline, algo_trade._features(scrambled)[i],
                          equal_nan=True)


def test_feature_layout_matches_the_contract():
    panel = random_panel(T=320, N=6, seed=3)
    assert algo_trade._features(panel).shape == (320, 6, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 72


# ---------------------------------------------------------------------------
# Returns and row drops
# ---------------------------------------------------------------------------

def test_returns_are_raw_like_bists():
    """No corporate-action handling, because bist has none.

    bist's `close_adj` is a verbatim copy of its raw close, so a bonus issue
    prints as a 2800% "return" there and reaches its label. An earlier version of
    this port booked anything outside the price band as 0%, which is a defensible
    data fix and a different model.
    """
    close = np.array([[100.0], [110.0], [100.0], [2_800.0], [2_800.0], [np.nan],
                      [2_900.0]])
    ret = algo_trade._returns(pd.DataFrame(close)).iloc[:, 0].to_numpy()

    assert np.isnan(ret[0])                       # no previous bar
    assert ret[1] == pytest.approx(0.10)
    assert ret[2] == pytest.approx(-1.0 / 11.0)
    assert ret[3] == pytest.approx(27.0)          # kept, not clamped
    assert ret[4] == pytest.approx(0.0)
    assert np.isnan(ret[5])                       # halted: unknown, not flat
    assert np.isnan(ret[6])                       # no usable previous close


def test_returns_keep_nan_unknown():
    """A halt is unknown, not flat — NaN must not become 0.0."""
    close = pd.DataFrame([[100.0], [np.nan], [np.nan], [101.0]])
    ret = algo_trade._returns(close).iloc[:, 0]
    assert ret.isna().tolist() == [True, True, True, True]


def test_limit_hit_fires_at_the_ten_percent_band():
    """bist's LIMIT_MOVE_THRESHOLD is 0.095, not 0.195.

    Load-bearing: the flag masks every feature that divides by the intraday
    range, and at 0.195 it never fired on this feed's real band at all.
    """
    close = pd.DataFrame([[100.0], [109.6], [104.0], [104.0]])
    high = pd.DataFrame([[101.0], [110.0], [105.0], [104.0]])
    low = pd.DataFrame([[99.0], [109.0], [103.0], [104.0]])
    volume = pd.DataFrame([[1.0], [1.0], [1.0], [1.0]])
    limit = algo_trade._limit_hit(algo_trade._returns(close), high, low, volume)

    assert not limit.iloc[0, 0], "no previous bar, so no return to test"
    assert limit.iloc[1, 0], "+9.6% printed at the band"
    assert not limit.iloc[2, 0], "-5.1% is an ordinary move"
    assert limit.iloc[3, 0], "H == L with volume is a lock, whatever the return"

    # And the old 0.195 threshold would have caught none of the band days.
    assert 0.095 <= algo_trade.LIMIT_HIT < 0.1


def test_drop_unusable_removes_the_rows_bist_removes():
    """Halt rows, zero-volume-with-price rows and disordered OHLC all go."""
    frame = pd.DataFrame([
        # timestamp, symbol, o, h, l, c, v
        (1, "AAA", 10.0, 11.0, 9.0, 10.5, 100.0),    # fine
        (2, "AAA", 0.0, 0.0, 0.0, 10.5, 0.0),        # vendor halt stamp
        (3, "AAA", 10.0, 11.0, 9.0, 10.5, 0.0),      # zero volume, real price
        (4, "AAA", 10.0, 8.0, 9.0, 10.5, 100.0),     # high below low
        (5, "AAA", 10.0, 11.0, 9.0, 10.5, 100.0),    # fine
    ], columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    kept = algo_trade._drop_unusable(frame)
    assert kept["timestamp"].tolist() == [1, 5]


def test_drop_unusable_tolerates_float_noise_in_ohlc():
    """bist's 1e-4 tolerance: a low a hair above the close is not a bad row."""
    frame = pd.DataFrame([
        (1, "AAA", 10.5, 10.5, 10.50001, 10.5, 100.0),
    ], columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    assert len(algo_trade._drop_unusable(frame)) == 1

    # Past the tolerance it is a bad row again.
    frame.loc[0, "low"] = 10.6
    assert len(algo_trade._drop_unusable(frame)) == 0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def trade_panel(series, T=200):
    """{symbol: {dated row: close}} -> a panel. Absent rows are halted sessions.

    `open` is set equal to `close` so the expected fills read straight off the
    closes the test wrote, which is the whole point of hand-building these.
    """
    syms = sorted(series)
    close = pd.DataFrame(np.nan, index=range(T), columns=range(len(syms)))
    for sym, rows in series.items():
        for row, value in rows.items():
            close.iloc[row, syms.index(sym)] = value
    high, low = close * 1.01, close * 0.99
    panel = {"open": close.copy(), "high": high, "low": low, "close": close,
             "volume": pd.DataFrame(1_000_000.0, index=close.index,
                                    columns=close.columns)}
    panel["ret"] = algo_trade._returns(panel["close"])
    return panel, {s: i for i, s in enumerate(syms)}


def flat_then_ramp(T=200, break_row=44, level=100.0, step=2.0):
    """Flat for 30 bars, a ramp, then one close far under the average."""
    rows = {r: level for r in range(30)}
    rows.update({r: level + step * (r - 29) for r in range(30, break_row)})
    rows[break_row] = level * 0.9
    rows.update({r: level * 0.9 for r in range(break_row + 1, T)})
    return rows


def halted_variant(T=200, break_row=44, halt_at=35, gap=2):
    """`flat_then_ramp` with `gap` halted sessions spliced in at `halt_at`.

    The same traded bars as `flat_then_ramp(T, break_row)` over the range that
    matters, just `gap` dates further along — so every traded-bar answer has to
    come out identical from a later dated row.
    """
    rows = flat_then_ramp(T - gap, break_row)
    out = {r: v for r, v in rows.items() if r < halt_at}
    out.update({r + gap: v for r, v in rows.items() if r >= halt_at})
    return out


def test_label_runs_from_the_next_open_to_the_open_after_the_break():
    """The trade: buy open[T+1], sell open[K+1] where K is the first close < MA.

    Three places to be off by one and none of them announce themselves, so the
    fills are asserted against the exact rows rather than against a shape.
    """
    T, brk = 200, 44
    panel, ci = trade_panel({"A": flat_then_ramp(T, brk)}, T)
    got = algo_trade._labels(panel)
    o = panel["open"].to_numpy()
    a = ci["A"]

    # Signalled at bar 40; the break lands at 44, so the pair is open[41] -> open[45].
    assert got["entry_ok"][40, a] == 1.0
    assert got["bars_held"][40, a] == 4.0
    assert got["raw"][40, a] == pytest.approx(o[45, a] / o[41, a] - 1.0)
    assert got["exit_row"][40, a] == 45.0


def test_y_is_the_log_of_the_realised_return():
    """The training target is a log return; `raw` is the same trade as a percent."""
    T = 200
    panel, ci = trade_panel({"A": flat_then_ramp(T)}, T)
    got = algo_trade._labels(panel)
    live = np.isfinite(got["y"])
    assert live.any()
    assert np.allclose(got["y"][live], np.log1p(got["raw"][live]))


def test_the_break_is_strict():
    """A close exactly on the average holds, matching `_exit_below_ma`'s `>=`.

    A flat series sits exactly on its own mean forever, so if the comparison
    were `<=` every bar would break and nothing would ever be held.
    """
    T = 200
    panel, ci = trade_panel({"FLAT": {r: 100.0 for r in range(T)}}, T)
    got = algo_trade._labels(panel)
    a = ci["FLAT"]
    assert got["entry_ok"][40, a] == 1.0
    # Never breaks, so the cap is what closes it.
    assert got["bars_held"][40, a] == float(MAX_HOLD)


def test_a_trade_that_never_breaks_is_capped():
    T = 200
    panel, ci = trade_panel({"UP": {r: 100.0 + r for r in range(T)}}, T)
    got = algo_trade._labels(panel)
    o, a = panel["open"].to_numpy(), ci["UP"]

    assert got["bars_held"][40, a] == float(MAX_HOLD)
    assert got["raw"][40, a] == pytest.approx(
        o[40 + MAX_HOLD + 1, a] / o[41, a] - 1.0)
    assert got["exit_row"][40, a] == float(40 + MAX_HOLD + 1)


def test_a_close_below_the_average_is_never_entered():
    """No trade, so no label — not a zero, and not a poor score.

    This is the population the fit is restricted to, which is why
    `ManipulationModel.signal` refuses the same rows at scoring time.
    """
    T = 200
    panel, ci = trade_panel({"DOWN": {r: 500.0 - 2.0 * r for r in range(T)}}, T)
    got = algo_trade._labels(panel)
    a = ci["DOWN"]
    assert np.nansum(got["entry_ok"][:, a]) == 0.0
    assert not np.isfinite(got["raw"][:, a]).any()


def test_the_breaking_bar_is_not_itself_enterable():
    T, brk = 200, 44
    panel, ci = trade_panel({"A": flat_then_ramp(T, brk)}, T)
    got = algo_trade._labels(panel)
    a = ci["A"]
    assert got["entry_ok"][brk, a] == 0.0
    assert not np.isfinite(got["raw"][brk, a])


def test_the_average_counts_traded_bars_not_calendar_rows():
    """Halted sessions must not consume a window slot.

    `D` is `A` with two halted sessions inserted mid-ramp: the same traded bars,
    two dates further along. It has to resolve to the identical trade from a
    different dated row, or the label is measuring a different average than
    `AlgoTradeStrategy._tail_closes` does on the live exit.
    """
    T, brk = 200, 44
    panel, ci = trade_panel(
        {"A": flat_then_ramp(T, brk), "D": halted_variant(T, brk)}, T)

    got = algo_trade._labels(panel)
    a, d = ci["A"], ci["D"]
    o = panel["open"].to_numpy()

    assert got["bars_held"][42, d] == got["bars_held"][40, a] == 4.0
    assert got["raw"][42, d] == pytest.approx(got["raw"][40, a])
    assert got["raw"][42, d] == pytest.approx(o[47, d] / o[43, d] - 1.0)
    assert got["exit_row"][42, d] == 47.0
    # A halted date carries nothing at all.
    assert not np.isfinite(got["raw"][35, d])
    assert not np.isfinite(got["entry_ok"][35, d])


def test_a_trade_whose_exit_would_fall_off_the_end_is_unresolved():
    """Still open at the last bar means unknown, not zero."""
    T = 200
    panel, ci = trade_panel({"UP": {r: 100.0 + r for r in range(T)}}, T)
    got = algo_trade._labels(panel)
    a = ci["UP"]
    # The capped exit fills at T + MAX_HOLD + 1, so this is the last row that fits.
    assert np.isfinite(got["raw"][T - 2 - MAX_HOLD, a])
    assert not np.isfinite(got["raw"][T - 1 - MAX_HOLD, a])


def test_a_price_scale_glitch_voids_the_trades_that_straddle_it():
    """A 100x one-bar spike is a unit error, not a move. See GLITCH_RATIO.

    BIST's daily band is +/-10%, so nothing legitimate travels 5x in a session.
    Left in, the spike prices an entry or an exit off a tick that never traded
    and books a -99% "loss" in one bar.
    """
    T, spike = 200, 60
    rows = {r: 100.0 for r in range(T)}
    rows[spike] = 10_000.0                    # kurus recorded as lira
    panel, ci = trade_panel({"G": rows}, T)
    got = algo_trade._labels(panel)
    a = ci["G"]

    # Every trade whose window contains the spike is dropped...
    for row in range(40, spike):
        assert not np.isfinite(got["raw"][row, a]), row
    # ...and the -99% revert is never booked as a return.
    finite = got["raw"][np.isfinite(got["raw"][:, a]), a]
    assert (finite > -0.5).all()


def test_the_exit_row_is_the_dated_row_of_the_exit_fill():
    """What makes the training embargo exact rather than horizon arithmetic.

    `train` admits a row only when `exit_row < end`, so this value has to be a
    dated row index into the panel — not a packed slot, and not an offset.
    """
    T, brk = 200, 44
    panel, ci = trade_panel(
        {"A": flat_then_ramp(T, brk), "D": halted_variant(T, brk)}, T)
    got = algo_trade._labels(panel)

    for symbol, signal_row, expected in (("A", 40, 45.0), ("D", 42, 47.0)):
        col = ci[symbol]
        assert got["exit_row"][signal_row, col] == expected
        # The dated row it names has to be one this symbol actually traded.
        assert np.isfinite(panel["close"].to_numpy()[int(expected), col])


def test_label_never_resolves_inside_a_symbols_dead_tail():
    """A symbol that stops printing must not borrow its own delisting.

    Packing lifts each symbol's traded bars to the top of the column, leaving dead
    slots below. A forward read would happily resolve inside that dead region and
    pull the value back onto real rows, where the symbol has genuinely ended.
    """
    T, N = 200, 6
    panel = random_panel(T=T, N=N, seed=4, quirks=False)
    # Symbol 1 delists 30 rows early.
    for key in ("open", "high", "low", "close", "volume"):
        panel[key].iloc[T - 30:, 1] = np.nan
    panel["ret"] = algo_trade._returns(panel["close"])

    got = algo_trade._labels(panel)
    live = np.flatnonzero(np.isfinite(got["raw"][:, 1]))
    assert live.max() <= T - 30 - 1


def test_a_symbol_shorter_than_the_average_has_no_trade():
    """No full window means no average to be above, so nothing is enterable."""
    T = 200
    panel, ci = trade_panel({"SHORT": {r: 100.0 + r for r in range(15)}}, T)
    got = algo_trade._labels(panel)
    a = ci["SHORT"]
    assert np.nansum(got["entry_ok"][:, a]) == 0.0


# ---------------------------------------------------------------------------
# Panel plumbing
# ---------------------------------------------------------------------------

def test_panel_from_window_pivots_ragged_history():
    """A ragged long window becomes a (T, N) panel with holes where bars are."""
    bars = [
        FakeKLine(1_000, "AAA", 1.0, 1.5, 0.5, 1.2, 10.0),
        FakeKLine(2_000, "AAA", 1.2, 1.6, 0.6, 1.3, 11.0),
        FakeKLine(3_000, "AAA", 1.3, 1.7, 0.7, 1.4, 12.0),
        FakeKLine(2_000, "BBB", 5.0, 5.5, 4.5, 5.2, 20.0),   # late listing
        FakeKLine(3_000, "BBB", 5.2, 5.6, 4.6, 5.3, 21.0),
    ]
    ctx = FakeContext(bars)
    for _ in range(3):
        ctx.advance()
    panel = algo_trade._panel_from_window(ctx.history(5))

    close = panel["close"]
    assert list(close.columns) == ["AAA", "BBB"]
    assert list(close.index) == [1_000, 2_000, 3_000]
    assert np.isnan(close.at[1_000, "BBB"])          # not listed yet
    assert close.at[3_000, "BBB"] == pytest.approx(5.3)
    assert close.at[1_000, "AAA"] == pytest.approx(1.2)


def test_panel_from_window_drops_symbols_that_did_not_print():
    """ctx.history only returns printing symbols, so the panel has no others."""
    bars = [
        FakeKLine(1_000, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0),
        FakeKLine(2_000, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0),
        FakeKLine(1_000, "GONE", 9.0, 9.0, 9.0, 9.0, 1.0),   # delisted after bar 1
    ]
    ctx = FakeContext(bars)
    ctx.advance()
    ctx.advance()
    panel = algo_trade._panel_from_window(ctx.history(5))
    assert list(panel["close"].columns) == ["AAA"]


# ---------------------------------------------------------------------------
# Training: embargo, persistence, calibration
# ---------------------------------------------------------------------------

def long_frame(panel, start="2020-01-02"):
    """A panel back to the long format `train` takes."""
    close = panel["close"]
    stamps = pd.bdate_range(start=start, periods=len(close))
    rows = []
    for j, symbol in enumerate(close.columns):
        for i, ts in enumerate(stamps):
            rows.append({
                "timestamp": ts,
                "symbol": f"S{symbol:02d}" if isinstance(symbol, int) else str(symbol),
                "open": panel["open"].iat[i, j], "high": panel["high"].iat[i, j],
                "low": panel["low"].iat[i, j], "close": close.iat[i, j],
                "volume": panel["volume"].iat[i, j],
            })
    return pd.DataFrame(rows).dropna(subset=["close"])


@pytest.fixture(scope="module")
def trained():
    """A small real fit. Slow enough to share, fast enough to keep honest.

    `train_start=None` because the synthetic frame starts in 2020 and bist's real
    2024-01-01 cutoff would leave it nothing to fit.
    """
    panel = random_panel(T=420, N=22, seed=13)
    frame = long_frame(panel)
    cutoff = sorted(frame["timestamp"].unique())[-40]
    model = ManipulationModel(params={**algo_trade.XGB_PARAMS, "n_estimators": 20},
                              train_start=None).train(frame, train_end=cutoff)
    return model, frame, cutoff


def test_train_stops_at_the_cutoff(trained):
    """train_end is a hard boundary: the fit records the last bar it could see."""
    model, frame, cutoff = trained
    assert model.train_end == int(pd.Timestamp(cutoff).value // 1_000_000)
    assert model.booster is not None


def test_train_labels_never_read_past_the_cutoff(trained):
    """The embargo is structural, so assert it on the labels themselves.

    Holding periods vary, so there is no `horizon` to subtract — each label
    carries the dated row of its own exit fill instead, and every trade the fit
    could keep has to have closed inside the training window.
    """
    _, frame, cutoff = trained
    panel = algo_trade._panel(frame)
    index = panel["close"].index.to_numpy()
    end = int(np.searchsorted(index, np.datetime64(pd.Timestamp(cutoff)), side="right"))
    truncated = algo_trade._truncate(panel, end)

    label = algo_trade._labels(truncated)
    exit_row = label["exit_row"]
    resolved = np.isfinite(exit_row)
    assert resolved.any(), "nothing resolved at all"
    assert exit_row[resolved].max() <= end - 1


def test_a_trade_still_open_at_the_cutoff_is_dropped_not_truncated(trained):
    """The embargo's actual mechanism, asserted against the trades it removes.

    `train` truncates before labelling, so a trade that would still be open at
    the cutoff cannot resolve and goes NaN. The strong form of that: every row
    whose *full-panel* trade closes at or after `end` must be unlabelled in the
    truncated panel — otherwise the fit would be learning from a bar it was
    never allowed to see.
    """
    _, frame, cutoff = trained
    panel = algo_trade._panel(frame)
    index = panel["close"].index.to_numpy()
    end = int(np.searchsorted(index, np.datetime64(pd.Timestamp(cutoff)), side="right"))

    full = algo_trade._labels(panel)
    truncated = algo_trade._labels(algo_trade._truncate(panel, end))

    exit_row = full["exit_row"][:end]
    reaches_past = np.isfinite(exit_row) & (exit_row >= end)
    assert reaches_past.any(), "no trade straddles the cutoff — weak fixture"
    assert not np.isfinite(truncated["y"][:end][reaches_past]).any()

    # ...and a trade that closed well inside the window is untouched by the cut.
    inside = np.isfinite(exit_row) & (exit_row < end - MAX_HOLD)
    assert np.allclose(full["y"][:end][inside], truncated["y"][:end][inside],
                       equal_nan=True)


def test_train_records_the_liquid_universe(trained):
    """The turnover cut is fit at train time, so scoring carries no future info."""
    model, frame, _ = trained
    assert model.liquid is not None
    symbols = set(frame["symbol"].unique())
    assert model.liquid < symbols, "the cut must exclude somebody"
    # 40th percentile of 22 names keeps roughly the top 60%.
    assert 0.5 * len(symbols) <= len(model.liquid) <= 0.7 * len(symbols)


def test_train_fills_the_design_matrix(trained):
    """bist fills NaN with 0.0 before every fit, so the trees never see missing.

    Asserted through the booster rather than the array: a tree trained on filled
    data has no `missing` branch to take, so scoring a row full of NaN must give
    the same answer as scoring the same row zero-filled.
    """
    model, frame, cutoff = trained
    window = _window_from_frame(frame, cutoff)
    F = algo_trade._features(window)
    lo, hi = model._bounds
    rows = np.clip(F[-1], lo, hi)

    filled = model._predict(np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0))
    assert np.isfinite(filled).all()


def _window_from_frame(frame, cutoff):
    """The trailing LOOKBACK bars ending at `cutoff`, as a window-shaped panel."""
    dates = np.sort(frame["timestamp"].unique())
    end = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff)))) + 1
    keep = dates[max(0, end - LOOKBACK):end]
    sub = frame[frame["timestamp"].isin(keep)]
    return algo_trade._panel(sub)


def test_signal_reports_a_score_and_its_expected_pct(trained):
    """Two columns, on the model's raw log-return scale. No percentile layer."""
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff))

    assert set(sig.columns) == {SCORE, EXPECTED_PCT}
    assert sig[SCORE].notna().any()


def test_expected_pct_is_the_score_read_as_a_return(trained):
    """A log return converts directly — no vol and no horizon in the way."""
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff)).dropna()
    assert not sig.empty
    assert np.allclose(sig[EXPECTED_PCT], np.expm1(sig[SCORE]) * 100.0)


def test_signal_refuses_names_below_their_exit_average(trained):
    """The entry half of the trade, enforced where the fit's population is set.

    The model never saw a row trading under its average, so a name below it is
    not ranked poorly — it is not scored at all.
    """
    model, frame, cutoff = trained
    window = _window_from_frame(frame, cutoff)
    sig = model.signal(window)

    closes = window["close"].to_numpy(dtype=float)
    printed = window["close"].notna()
    order = algo_trade._pack_order(printed)
    packed = algo_trade._pack(window["close"], order)
    sma = algo_trade._sma(packed, model.exit_ma,
                          algo_trade._real_slots(printed).to_numpy())
    dated = algo_trade._unpack(pd.DataFrame(sma, columns=packed.columns), order,
                               window["close"].index, printed).to_numpy()

    below = closes[-1] < dated[-1]
    assert below.any(), "no name is below its average — weak fixture"
    assert sig[SCORE].to_numpy()[below].size
    assert np.isnan(sig[SCORE].to_numpy()[below]).all()


def test_signal_refuses_symbols_without_enough_history(trained):
    """A name with 5 bars gets NaN, not a number derived from missing inputs."""
    model, frame, cutoff = trained
    dates = np.sort(frame["timestamp"].unique())
    end = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff)))) + 1
    keep = dates[max(0, end - LOOKBACK):end]
    sub = frame[frame["timestamp"].isin(keep)].copy()

    one = sub["symbol"].iloc[0]
    newborn = sub[(sub["symbol"] == one) & sub["timestamp"].isin(keep[-5:])].copy()
    newborn["symbol"] = "NEW"
    # `liquid` gates on the artifact's symbol set, which "NEW" is not in; clear it
    # so the assertion is about history alone.
    liquid, model.liquid = model.liquid, None
    try:
        sig = model.signal(algo_trade._panel(pd.concat([sub, newborn])))
    finally:
        model.liquid = liquid
    assert np.isnan(sig.at["NEW", SCORE])


def test_signal_only_scores_the_liquid_universe(trained):
    """Symbols under bist's turnover percentile are not scored at all."""
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff))
    scored = set(sig.index[sig[SCORE].notna()])
    assert scored <= model.liquid
    assert scored, "the liquid universe cannot be empty"


def test_signal_before_training_is_an_error():
    with pytest.raises(RuntimeError, match="train"):
        ManipulationModel().signal({})


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------

TRAIN_END = int(pd.Timestamp("2024-12-31", tz="UTC").value // 1_000_000)
SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


class StubModel:
    """Stands in for ManipulationModel with a scripted signal frame.

    The strategy is a decision rule over `signal`'s output; what produced that
    output is the model's business and is tested above.
    """

    LOOKBACK = LOOKBACK

    def __init__(self, frame=None, train_end=TRAIN_END, exit_ma=algo_trade.EXIT_MA,
                 **kwargs):
        self.frame = frame
        self.train_end = train_end
        self.trained = None
        self.kwargs = kwargs
        # The strategy compares this against its own `exit_ma`: the entry gate is
        # the artifact's average and the exit is the run's, so a mismatch means
        # the two ends of the trade disagree.
        self.exit_ma = exit_ma
        self.calls = 0
        self.states = []

    def signal(self, window, state=None):
        self.calls += 1
        self.states.append(state)
        return self.frame

    def train(self, panel, *, train_end=None):
        """The strategy fits in-run, so the harness intercepts that too.

        What was trained on is the model's business and is tested above; the
        panel is kept only so the accumulation tests can assert what the
        strategy actually handed over.
        """
        self.trained = panel
        return self


def yyyymmdd(timestamp_ms):
    """The `train_on` param's units — see algo_trade.TRAIN_ON."""
    return int(pd.Timestamp(timestamp_ms, unit="ms").strftime("%Y%m%d"))


def stub_model(stub):
    """Patch the class the strategy constructs, not the loader it no longer calls.

    The factory records what the strategy asked for, so a test can assert the fit
    was configured for the trade the run is actually holding.
    """
    def factory(**kwargs):
        stub.kwargs = kwargs
        return stub
    return patch.object(algo_trade, "ManipulationModel", factory)


def signal_frame(rows):
    """{symbol: score} -> a signal frame shaped like `signal` returns."""
    frame = pd.DataFrame({SCORE: pd.Series(rows, dtype=float)})
    frame[EXPECTED_PCT] = np.expm1(frame[SCORE]) * 100.0
    return frame


# Every name scored and enterable. The strategy gates on the ranking, not on a
# level, so the value only has to be finite — `signal` has already refused
# anything trading below its exit average by the time `_rank` sees it.
ALL_PASS = {s: 0.4 for s in SYMBOLS}


def sessions(count, start="2024-01-02"):
    return [int(d.value // 1_000_000)
            for d in pd.bdate_range(start=start, periods=count, tz="UTC")]


def feed(timestamps, symbols=SYMBOLS, price=100.0, paths=None):
    """Flat bars at `price`, except where `paths` gives a symbol its own closes.

    A path is aligned to the *end* of the feed, so {"AAA": [90.0]} means AAA
    closed at 90 on the final bar and at `price` on every bar before it.
    """
    bars = []
    for i, ts in enumerate(timestamps):
        for symbol in symbols:
            value = price
            path = (paths or {}).get(symbol)
            if path is not None:
                offset = i - (len(timestamps) - len(path))
                if offset >= 0:
                    value = path[offset]
            bars.append(FakeKLine(ts, symbol, value, value, value, value, 1_000.0))
    return bars


def _universe_bars(rows, ticks_after_cutoff=1, train_end=TRAIN_END):
    """A flat feed covering exactly the symbols in `rows`.

    The default feed carries SYMBOLS; the slice-width tests need a universe big
    enough for a percentage of it to be more than one name.
    """
    before = sessions(LOOKBACK, start="2023-10-02")
    before = [t for t in before if t <= train_end][-LOOKBACK:]
    after = [train_end + (i + 1) * 86_400_000 for i in range(ticks_after_cutoff)]
    return feed(before + after, symbols=list(rows))


def start_run(rows=ALL_PASS, *, bars=None, ticks_after_cutoff=1, cash=100_000.0,
              train_end=TRAIN_END, paths=None, **overrides):
    """A strategy driven to just past the cutoff, with a stub model attached.

    Always goes through the real `on_start` with the model class patched, rather
    than setting up the strategy's state by hand — hand-rolled setup silently
    rots the moment `on_start` grows another attribute.

    Warmup is really ticked, not skipped. It used to be skipped as pure cost,
    but the strategy now fits from the bars it has accumulated, so a warmup that
    never called `on_tick` would hand `_fit` an empty accumulator. `train_on`
    defaults to the last warmup bar, so the fit happens on the final warmup tick
    — inside the patch — and every driven tick afterwards takes the ordinary
    already-fitted path.
    """
    if bars is None:
        # LOOKBACK bars up to the cutoff, then a few past it.
        before = sessions(LOOKBACK, start="2023-10-02")
        before = [t for t in before if t <= train_end][-LOOKBACK:]
        after = [train_end + (i + 1) * 86_400_000 for i in range(ticks_after_cutoff)]
        bars = feed(before + after, paths=paths)

    ctx = FakeContext(bars, cash=cash)
    strategy = AlgoTradeStrategy()
    stub = StubModel(signal_frame(rows), train_end=train_end)
    # Most tests here are about ordering, sizing or exits, not about how wide the
    # entry slice is — and with a five-symbol universe the production 0.5% would
    # admit exactly one name and mask all of it. Widening to the whole universe
    # keeps those tests measuring what they claim to; the slice width has its own
    # tests below.
    overrides.setdefault("top_pct", 100.0)
    stamps = sorted({b.timestamp for b in bars})
    warmup = len(stamps) - ticks_after_cutoff
    overrides.setdefault("train_on", yyyymmdd(stamps[warmup - 1]))
    # Applied BEFORE on_start, which is where the engine applies them — see
    # pythonstrategy.h, "set on the instance ... before on_start, so strategies
    # see final values". Setting them afterwards would hide anything on_start
    # decides from a param.
    for name, value in overrides.items():
        setattr(strategy, name, value)
    with stub_model(stub):
        strategy.on_start(ctx)
        for _ in range(warmup):
            ctx.advance()
            strategy.on_tick(ctx)
    return ctx, strategy, stub


def drive(strategy, ctx, ticks=1):
    for _ in range(ticks):
        ctx.advance()
        strategy.on_tick(ctx)


def buys(ctx):
    return [o for o in ctx.orders
            if o.side == OrderSide.Buy and o.order_type == "market"]


def legs(ctx, entry, order_type):
    return [o for o in ctx.orders
            if o.parent == entry.id and o.order_type == order_type]


def sells(ctx):
    """The strategy's own exits: unparented market sells, not the resting stop."""
    return [o for o in ctx.orders
            if o.side == OrderSide.Sell and o.parent is None
            and o.order_type == "market"]


def fill_entries(ctx):
    """Report every pending entry as filled, like the broker would next bar."""
    for order in buys(ctx):
        ctx.positions.setdefault(
            order.symbol, FakePosition(quantity=order.quantity, price=100.0))


def test_on_start_clears_state():
    ctx, strategy, stub = start_run()
    assert strategy.model is stub
    assert strategy.book == set()
    assert strategy.exiting == set()
    assert strategy.state.obv == {} and strategy.state.bars == 0


def test_warmup_takes_no_positions():
    """Fewer than LOOKBACK distinct timestamps means nothing is fit or scored.

    The date alone does not license a fit: `train_on` here is long past, and the
    strategy still refuses, because `train` handed a panel this short raises
    rather than returning a bad model.
    """
    stamps = sessions(20, start="2025-01-02")
    ctx = FakeContext(feed(stamps), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    strategy.train_on = yyyymmdd(stamps[0])
    stub = StubModel(signal_frame(ALL_PASS), train_end=0)
    with stub_model(stub):
        strategy.on_start(ctx)
        drive(strategy, ctx, ticks=20)

    assert ctx.orders == []
    assert stub.calls == 0
    assert strategy.model is None


def test_no_trading_on_bars_the_model_was_trained_on():
    """The fit saw those bars; trading them is not a backtest result.

    The fit stamps `train_end` with its own bar, so this is what keeps the
    strategy off the bar it just trained on as well as everything before it.
    """
    before = sessions(LOOKBACK + 5, start="2023-10-02")
    ctx = FakeContext(feed(before), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    # Fit on the last bar, so every bar of the run is in-sample.
    strategy.train_on = yyyymmdd(before[-1])
    stub = StubModel(signal_frame(ALL_PASS), train_end=before[-1])
    with stub_model(stub):
        strategy.on_start(ctx)
        drive(strategy, ctx, ticks=len(before))

    assert ctx.orders == [], "in-sample bars must not be traded"
    assert stub.calls == 1, "but the fit still scores, to seed the state"


def test_enters_the_qualifying_names():
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)
    assert sorted(o.symbol for o in buys(ctx)) == sorted(SYMBOLS)
    assert strategy.book == set(SYMBOLS)


def test_the_slice_is_a_share_of_the_scored_universe():
    """`top_pct` selects the day's best N, not everything above a fixed sigma.

    bist gates on absolute sigmas; this takes a cross-sectional slice instead,
    because the heads are a ranker and a fixed threshold both discards the
    ranking and drifts in meaning with every retrain. See ENTRY_TOP_PCT.
    """
    rows = {f"S{i:02d}": 1.0 - i / 100.0 for i in range(20)}
    for pct, expected in ((100.0, 20), (50.0, 10), (25.0, 5), (5.0, 1)):
        ctx, strategy, _ = start_run(rows, bars=_universe_bars(rows), top_pct=pct,
                                     max_positions=99)
        drive(strategy, ctx)
        assert len(buys(ctx)) == expected, f"top_pct={pct}"
        # and it is the TOP of the ranking, not an arbitrary subset
        assert [o.symbol for o in buys(ctx)] == [f"S{i:02d}" for i in range(expected)]


def test_a_slice_narrower_than_one_name_still_takes_one():
    """int() truncation must not silently disable the strategy on a small universe."""
    rows = dict(ALL_PASS)
    ctx, strategy, _ = start_run(rows, top_pct=0.5)
    drive(strategy, ctx)
    assert len(buys(ctx)) == 1


def test_the_slice_is_measured_before_held_names_are_removed():
    """Otherwise the gate quietly widens as the book fills.

    Ranking the *unheld* remainder would promote the next-best candidate into the
    slice every time a name is held, so a nearly-full book would end up buying
    names the gate was built to exclude. The cut is taken against the whole
    scored universe and the book applied afterwards.
    """
    rows = {f"S{i:02d}": 1.0 - i / 100.0 for i in range(20)}
    ctx, strategy, _ = start_run(rows, bars=_universe_bars(rows), top_pct=25.0)
    held = {f"S{i:02d}" for i in range(5)}            # the whole top slice
    strategy.book = set(held)
    for symbol in held:                              # the broker must agree, or
        ctx.positions[symbol] = FakePosition(quantity=1.0, price=100.0)
    drive(strategy, ctx)
    assert buys(ctx) == [], "the slice is exhausted by held names, not refilled"


def test_the_percentile_gate_never_sits_out():
    """A behaviour change worth pinning: there is no longer a "nothing qualifies".

    bist's absolute gate could return an empty day — every name below 3.0 sigma
    meant no trade. A cross-sectional slice always has a top, so the strategy
    takes its best available name however weak the whole cohort is, and cannot
    step aside in a bad tape. Whatever regime protection the strategy has now
    comes from the exit, not the entry.
    """
    ctx, strategy, _ = start_run({s: 0.0001 for s in SYMBOLS}, top_pct=25.0)
    drive(strategy, ctx)
    assert len(buys(ctx)) == 1, "weak scores still produce the day's best pick"


def test_nothing_is_entered_when_the_model_scores_nobody():
    """NaN sigmas are the model declining to score, not a weak score."""
    ctx, strategy, _ = start_run({s: np.nan for s in SYMBOLS})
    drive(strategy, ctx)
    assert ctx.orders == []


def test_ranks_on_the_score():
    """Order is by score descending, not frame order."""
    rows = {
        "AAA": 0.35,
        "BBB": 0.85,    # best
        "CCC": 0.30,
        "DDD": 0.55,    # second
        "EEE": 0.21,
    }
    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    assert [o.symbol for o in buys(ctx)][:2] == ["BBB", "DDD"]


def test_ties_break_on_symbol():
    """Equal scores must order deterministically, not by frame order."""
    rows = {s: 0.4 for s in ["EEE", "CCC", "AAA", "DDD", "BBB"]}
    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    assert [o.symbol for o in buys(ctx)] == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_the_book_is_capped_at_max_positions():
    """bist caps nothing, which only survives because its gate rarely fires.

    A percentile gate fires every session, so an uncapped book would exhaust
    cash within a few entries and everything after would be a broker rejection.
    """
    rows = {f"S{i:02d}": 1.0 - i / 100.0 for i in range(30)}
    ctx, strategy, _ = start_run(rows, bars=_universe_bars(rows),
                                 max_positions=4)
    drive(strategy, ctx)
    assert len(buys(ctx)) == 4, "only the free slots are filled"
    assert [o.symbol for o in buys(ctx)] == [f"S{i:02d}" for i in range(4)]


def test_a_full_book_takes_nothing():
    rows = dict(ALL_PASS)
    ctx, strategy, _ = start_run(rows, ticks_after_cutoff=2, max_positions=2)
    drive(strategy, ctx)
    fill_entries(ctx)
    assert len(buys(ctx)) == 2

    drive(strategy, ctx)
    assert len(buys(ctx)) == 2, "no slot free, so no new entry"


def test_every_entry_carries_a_stop_and_no_target():
    """bist's bracket is SL -10% / TP +30%; the stop is kept, the target is not.

    The +30% leg is the half with nothing behind it — the label encodes a
    drawdown floor and no ceiling — and it capped every winner. `exit_ma`
    replaces it.
    """
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)

    entry = buys(ctx)[0]
    stops = legs(ctx, entry, "stop")
    assert len(stops) == 1
    assert legs(ctx, entry, "limit") == [], "no fixed target"
    assert stops[0].price == pytest.approx(100.0 * 0.90)
    assert stops[0].side == OrderSide.Sell
    assert stops[0].reduce_only is True
    assert stops[0].quantity == pytest.approx(entry.quantity)


def test_ma_exit_fires_when_the_close_breaks_below():
    """A held name closing under its 20-bar average is sold at market.

    The average moves every bar, so this cannot be a resting leg: it is
    re-decided each bar and fills at the next open.
    """
    ctx, strategy, _ = start_run(ticks_after_cutoff=2, paths={"AAA": [100.0, 90.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    drive(strategy, ctx)

    exits = sells(ctx)
    assert [o.symbol for o in exits] == ["AAA"]
    assert exits[0].parent is None, "not a bracket leg"
    assert exits[0].reduce_only is True
    assert exits[0].quantity == pytest.approx(ctx.positions["AAA"].quantity)
    assert strategy.exiting == {"AAA"}


def test_no_ma_exit_while_the_close_holds_above():
    """The break is strict: a close exactly AT the average holds the position."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2)   # every close flat at 100
    drive(strategy, ctx)
    fill_entries(ctx)
    drive(strategy, ctx)

    assert sells(ctx) == []
    assert strategy.exiting == set()


def test_ma_exit_is_not_resent_while_it_is_pending():
    """The sale fills at the next open, so the rule sees the position again."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=3,
                                 paths={"AAA": [100.0, 90.0, 80.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    drive(strategy, ctx, ticks=2)      # under the average on both bars

    assert len(sells(ctx)) == 1


def test_ma_exit_runs_when_nothing_qualifies_to_enter():
    """Exits are decided before the entry block, which returns early on most bars.

    Ordering is load-bearing: with the exit after the `picks.empty` return, a
    held name would only be evaluated on bars that happened to produce a signal.
    """
    ctx, strategy, stub = start_run(ticks_after_cutoff=2,
                                    paths={"AAA": [100.0, 90.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    stub.frame = signal_frame({s: 0.2 for s in SYMBOLS})

    drive(strategy, ctx)
    assert [o.symbol for o in sells(ctx)] == ["AAA"]


def test_an_exiting_name_is_not_re_entered_until_it_is_flat():
    """Its sale settles at the next open; the broker rejects what it cannot fund."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=4,
                                 paths={"AAA": [100.0, 90.0, 100.0, 100.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    drive(strategy, ctx)                        # AAA breaks its average
    assert strategy.exiting == {"AAA"} and "AAA" in strategy.book
    entered = sum(1 for o in buys(ctx) if o.symbol == "AAA")

    drive(strategy, ctx)                        # back above it, sale still in flight
    assert sum(1 for o in buys(ctx) if o.symbol == "AAA") == entered

    ctx.positions.pop("AAA")                    # the sale filled
    drive(strategy, ctx)
    assert strategy.exiting == set()
    assert sum(1 for o in buys(ctx) if o.symbol == "AAA") == entered + 1


def test_ma_exit_needs_a_full_window():
    """Fewer than `exit_ma` traded bars means there is no average to break."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2,
                                 paths={"AAA": [100.0, 1.0]}, exit_ma=LOOKBACK + 1)
    drive(strategy, ctx)
    fill_entries(ctx)
    drive(strategy, ctx)

    assert sells(ctx) == [], "a collapsed close still cannot break a short window"


def test_ma_exit_only_sells_a_position_the_broker_reports():
    """Defensive: a name we believe we hold but the broker does not is left alone."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2, paths={"AAA": [100.0, 90.0]})
    drive(strategy, ctx)
    ctx.advance()

    strategy.book = {"AAA"}          # believed held; no position was ever filled
    strategy.exiting = set()
    strategy._exit_below_ma(ctx, ctx.history(LOOKBACK))
    assert sells(ctx) == []


def test_entry_size_is_equity_over_the_position_cap():
    """Full deployment lands exactly at the cap, never before it."""
    ctx, strategy, _ = start_run({"AAA": 0.4}, cash=1_000.0,
                                 max_positions=4, cash_buffer=0.02)
    drive(strategy, ctx)

    orders = buys(ctx)
    assert len(orders) == 1
    assert orders[0].quantity == pytest.approx(1_000.0 / 4 / 100.0)


def test_a_bar_wanting_several_names_cannot_order_more_cash_than_it_has():
    """The cash leg binds when one bar fills many slots at once.

    equity/max_positions alone would commit 5 x 25% of a 4-slot book on a bar
    that picks five names, and the broker rejects — never queues — an order it
    cannot fund at fill time.
    """
    ctx, strategy, _ = start_run(cash=1_000.0, max_positions=4, cash_buffer=0.02)
    drive(strategy, ctx)

    orders = buys(ctx)
    assert len(orders) == 4, "five names qualify but only four slots exist"
    committed = sum(o.quantity * 100.0 for o in orders)
    assert committed <= 1_000.0 * 0.98 + 1e-9, f"committed {committed}"
    # the cash leg is the binding one here: equity/max_positions alone would be
    # 250 a name, and four of those overdraw the 980 that is actually spendable.
    assert orders[0].quantity == pytest.approx(1_000.0 * 0.98 / 4 / 100.0)


def test_held_names_are_not_re_entered():
    ctx, strategy, _ = start_run(ticks_after_cutoff=2)
    drive(strategy, ctx)
    fill_entries(ctx)
    before = len(ctx.orders)

    drive(strategy, ctx)
    assert len(ctx.orders) == before, "every name is already held"


def test_a_position_closed_behind_our_back_can_be_re_entered():
    """A filled stop drops the name from the book; nothing embargoes it."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2)
    drive(strategy, ctx)
    fill_entries(ctx)
    held = buys(ctx)[0].symbol
    assert held in strategy.book

    ctx.positions.pop(held)          # the stop filled
    drive(strategy, ctx)
    assert sum(1 for o in buys(ctx) if o.symbol == held) == 2


def test_limit_locked_reads_this_ticks_return():
    """The gate must not compare a close to itself.

    An earlier version remembered the previous close in strategy state and
    advanced it before the gate ran, so the gate compared a bar to itself and
    never fired. It is read off the window now; pinned directly because the
    integration test below drives 300 ticks.
    """
    strategy = AlgoTradeStrategy()
    strategy.skip_limit_locked = 1
    strategy._ret = {"AAA": 0.0995, "BBB": 0.05, "CCC": -0.10}
    assert strategy._limit_locked("AAA")
    assert not strategy._limit_locked("BBB")
    assert not strategy._limit_locked("CCC"), "a limit-DOWN close is still buyable"
    assert not strategy._limit_locked("MISSING")

    # Off by default, because bist's own backtest disables it.
    strategy.skip_limit_locked = 0
    assert not strategy._limit_locked("AAA")


@pytest.mark.parametrize("skip,expected", [
    (1, {"BBB", "CCC", "DDD", "EEE"}),
    (0, {"AAA", "BBB", "CCC", "DDD", "EEE"}),
])
def test_the_limit_locked_gate_skips_the_name_without_backfilling(skip, expected):
    """A close at the +10% band cannot be bought at tomorrow's open.

    On by default now. bist's backtest disables it (LU_OVERRIDE), and while the
    gate was bist's absolute sigma that mattered enormously — 77 of its 92
    out-of-sample signals closed at the band, and refusing them left 15 trades.
    The percentile gate makes it affordable, and measurably better.

    Note the skipped name is *not* replaced by the next-best candidate: it
    consumed its place in the slice. Backfilling would let a locked day quietly
    reach further down the ranking than the gate allows.
    """
    # One out-of-sample bar, and it is the locked one, so AAA never gets an
    # unlocked tick on which it could have been bought first. The previous close
    # comes off the window, so no prior tick is needed.
    ctx, strategy, _ = start_run(paths={"AAA": [111.0]},
                                 skip_limit_locked=skip)
    drive(strategy, ctx)
    assert {o.symbol for o in buys(ctx)} == expected


def test_a_token_sized_entry_is_skipped_rather_than_placed():
    """Cash still tied up in an unsettled sale must not buy a dust position.

    The moving-average exit settles at the next open, so on the bar it is placed
    the proceeds do not exist yet. Sizing off the remaining cash would buy a
    token quantity that holds a slot for the whole trade and contributes nothing
    — the first engine run produced four, the smallest 0.0009 lira.
    """
    # FakeContext reports equity == cash, but the situation being modelled is
    # equity held in positions with the cash leg still in flight, so equity is
    # pinned high while cash is drained.
    def run_with(cash):
        ctx, strategy, _ = start_run({"AAA": 0.4}, max_positions=4)
        ctx._cash = cash
        ctx.equity = lambda: 1_000.0        # target slot = 250
        drive(strategy, ctx)
        return buys(ctx)

    assert run_with(1.0) == [], "1 lira against a 250 slot is not a position"
    # ...and a budget that is merely reduced, not vestigial, still trades.
    assert len(run_with(200.0)) == 1


def test_the_limit_locked_gate_is_on_by_default():
    """The one bist default this port overrides on measured grounds."""
    assert AlgoTradeStrategy.skip_limit_locked == 1


def test_tail_closes_slices_each_symbol_independently():
    """A ragged window: symbols have different bar counts and share no index.

    `_tail_closes`, `_closes` and `_returns_this_tick` all read the window
    through `_segments`, so getting the boundaries wrong would silently hand one
    symbol another's prices.
    """
    stamps = sessions(6)
    bars = [FakeKLine(ts, "AAA", v, v, v, v, 1_000.0)
            for ts, v in zip(stamps, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
    # BBB lists late, so its slice is shorter than AAA's.
    bars += [FakeKLine(ts, "BBB", v, v, v, v, 1_000.0)
             for ts, v in zip(stamps[3:], [40.0, 50.0, 60.0])]
    ctx = FakeContext(sorted(bars, key=lambda b: b.timestamp))
    for _ in range(len(stamps)):
        ctx.advance()
    w = ctx.history(6)

    tails = AlgoTradeStrategy._tail_closes(w, {"AAA", "BBB"}, 3)
    assert list(tails["AAA"]) == [4.0, 5.0, 6.0]
    assert list(tails["BBB"]) == [40.0, 50.0, 60.0]

    # Asking for more than a symbol has clamps to what it has, rather than
    # reaching back into the previous symbol's rows.
    assert list(AlgoTradeStrategy._tail_closes(w, {"BBB"}, 10)["BBB"]) == [
        40.0, 50.0, 60.0]
    # Only the requested names come back.
    assert set(AlgoTradeStrategy._tail_closes(w, {"AAA"}, 3)) == {"AAA"}

    assert AlgoTradeStrategy._closes(w) == {"AAA": 6.0, "BBB": 60.0}


def test_entries_are_logged_with_the_signal_that_produced_them(capsys):
    """One line per entry, carrying the order and the score the gate ranked on.

    The gate is a percentile, so the raw score is the only record of how strong
    a pick actually was — the line has to carry it, or a pick cannot be read
    back without re-running the model.
    """
    rows = {"AAA": 0.4}
    ctx, strategy, _ = start_run(rows, cash=1_000.0, max_positions=4,
                                 cash_buffer=0.02, stop_pct=10.0, exit_ma=20)
    drive(strategy, ctx)

    line = next(l for l in capsys.readouterr().out.splitlines() if " AAA " in l)
    assert line.startswith("[2025-01-01 00:00 UTC] AAA ")
    assert "enter market @ 100.0000" in line
    assert "SL 90.0000 (-10.00%)" in line
    assert "exit < MA20" in line
    # the score, then that score read as a return
    assert "score 0.4000" in line
    assert f"({100.0 * np.expm1(0.4):+.2f}%)" in line
    assert f"qty {1_000.0 / 4 / 100.0:.6g}" in line


def test_nothing_is_logged_when_no_order_is_placed(capsys):
    """Weak scores are no longer silent — only an unscored universe is."""
    ctx, strategy, _ = start_run({s: np.nan for s in SYMBOLS})
    drive(strategy, ctx)
    assert capsys.readouterr().out == ""


def test_entries_are_plotted_with_their_score():
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)
    plotted = {(p.name, p.symbol) for p in ctx.plots}
    for symbol in SYMBOLS:
        assert (SCORE, symbol) in plotted


# ---------------------------------------------------------------------------
# The in-run fit — no artifact, trained from the bars the run has seen
# ---------------------------------------------------------------------------

def fit_run(train_on_index, *, extra=0, symbols=SYMBOLS, bars=None, **overrides):
    """Tick a feed to its end, fitting on `train_on_index` (negative = from end)."""
    stamps = sessions(LOOKBACK + extra, start="2023-10-02")
    if bars is None:
        bars = feed(stamps, symbols=symbols)
    ctx = FakeContext(bars, cash=100_000.0)
    strategy = AlgoTradeStrategy()
    strategy.train_on = yyyymmdd(stamps[train_on_index])
    strategy.top_pct = 100.0
    for name, value in overrides.items():
        setattr(strategy, name, value)
    stub = StubModel(signal_frame({s: 0.4 for s in symbols}),
                     train_end=stamps[train_on_index])
    with stub_model(stub):
        strategy.on_start(ctx)
        drive(strategy, ctx, ticks=len(stamps))
    return ctx, strategy, stub


def test_nothing_is_fit_or_scored_before_the_train_date():
    """Accumulating is all that happens; no model, no scoring, no orders."""
    ctx, strategy, stub = fit_run(-1, extra=5)
    # train_on is the final bar, so every earlier bar only collected.
    assert stub.calls == 1
    assert ctx.orders == []


def test_the_fit_happens_once_on_the_first_bar_at_or_after_the_date():
    ctx, strategy, stub = fit_run(-6, extra=5)
    assert strategy.model is stub
    # One seeding score at the fit, then one per bar for the five bars after it.
    assert stub.calls == 6


def test_the_training_bar_is_not_traded_but_the_next_one_is():
    """`train` stamps train_end with its own bar, and the guard skips it.

    No separate rule enforces the boundary — this is the same leak guard that
    kept the strategy off a loaded artifact's in-sample bars.
    """
    stamps = sessions(LOOKBACK + 2, start="2023-10-02")
    ctx = FakeContext(feed(stamps), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    strategy.train_on = yyyymmdd(stamps[LOOKBACK - 1])
    strategy.top_pct = 100.0
    stub = StubModel(signal_frame(ALL_PASS), train_end=stamps[LOOKBACK - 1])
    with stub_model(stub):
        strategy.on_start(ctx)
        drive(strategy, ctx, ticks=LOOKBACK)        # up to and incl. the fit bar
        assert strategy.model is not None, "the fit ran"
        assert buys(ctx) == [], "the bar it trained on must not be traded"
        drive(strategy, ctx, ticks=1)
    assert buys(ctx), "the bar after the fit does trade"


def test_the_fit_is_handed_every_symbol_including_one_halted_that_day():
    """The reason bars are accumulated rather than pulled in one history() call.

    A single `history(n)` at the training bar carries only the symbols that
    printed *that* bar, and `_liquid_universe` is computed from whatever it is
    given — so a name halted on the training day would be scored NaN for the
    rest of the run.
    """
    stamps = sessions(LOOKBACK, start="2023-10-02")
    bars = feed(stamps, symbols=["AAA", "BBB"])
    # BBB does not print on the final (training) bar.
    bars = [b for b in bars if not (b.symbol == "BBB" and b.timestamp == stamps[-1])]

    ctx, strategy, stub = fit_run(-1, bars=bars, symbols=["AAA", "BBB"])
    assert stub.trained is not None, "the fit ran"
    assert set(stub.trained["close"].columns) == {"AAA", "BBB"}


def test_the_accumulator_is_released_once_the_model_exists():
    """It is the whole feed, and there is no second fit to keep it for."""
    _, strategy, _ = fit_run(-2, extra=5)
    assert strategy.model is not None
    assert strategy._bars is None


def test_the_fit_scores_the_panel_it_trained_on_exactly_once():
    """The seeding call and the bar's own signal are the same call.

    Scoring the panel and then also scoring the window would advance
    `ScoringState` twice for one bar, silently corrupting `obv` and
    `days_since_past_extreme` from the first bar onward.
    """
    _, _, stub = fit_run(-1)
    assert stub.calls == 1


def test_the_seeding_call_gets_the_whole_history_not_a_window():
    """`_carry` seeds from the first window it is handed, so it must be the lot.

    A LOOKBACK-bounded window would rebase the two carried features; the panel
    the fit trains on is the whole accumulated feed, and it is that panel which
    is scored.
    """
    extra = 5
    _, _, stub = fit_run(-(extra + 1), extra=extra)
    seeded = stub.states[0]
    assert seeded is not None
    # The panel handed to `train` spans every bar collected so far, not LOOKBACK.
    assert len(stub.trained["close"]) == LOOKBACK


def test_train_on_is_an_int_so_the_gui_round_trips_it():
    """Param transport is numeric and `pythonstrategy.h` coerces to the current
    attribute's type — a float default would come back as 20241231.0."""
    assert isinstance(AlgoTradeStrategy.train_on, int)
    assert not isinstance(AlgoTradeStrategy.train_on, bool)
    assert AlgoTradeStrategy.train_on == algo_trade.TRAIN_ON


def test_moving_train_on_moves_the_bar_the_fit_happens_on():
    """The point of exposing it: a different date is a different fit."""
    early = fit_run(-6, extra=5)[2]
    late = fit_run(-2, extra=5)[2]
    assert len(early.trained["close"]) < len(late.trained["close"])


# ---------------------------------------------------------------------------
# The sigma gate — workspace.ipynb's selection, off by default
# ---------------------------------------------------------------------------

def wobble(vol, bars=algo_trade.SIGMA_VOL_LEN + 1, price=100.0):
    """`bars` closes whose log returns alternate +/-`vol`, so realised vol ~= vol.

    Kept well under the +10% band: a path that ends on a limit-up move would be
    refused by `skip_limit_locked` and the test would pass for the wrong reason.
    """
    steps = np.array([vol if i % 2 else -vol for i in range(bars - 1)])
    return list(price * np.exp(np.concatenate([[0.0], np.cumsum(steps)])))


def sigma_run(scores, vols, **overrides):
    """Drive one bar with per-symbol scores and per-symbol realised vols."""
    overrides.setdefault("sigma_gate", 1)
    overrides.setdefault("min_score", 0.0)
    ctx, strategy, _ = start_run(
        scores, paths={s: wobble(v) for s, v in vols.items()}, **overrides)
    drive(strategy, ctx)
    return ctx


def test_trailing_vol_is_the_std_of_the_windows_log_returns():
    closes = wobble(0.03)
    got = algo_trade._trailing_vol({"AAA": closes})["AAA"]
    want = float(np.std(np.diff(np.log(closes)), ddof=1))
    assert got == pytest.approx(want)


def test_trailing_vol_uses_only_the_last_length_returns():
    """A long tail must not drag in history the notebook's window excludes."""
    quiet = wobble(0.01, bars=algo_trade.SIGMA_VOL_LEN + 1)
    loud = wobble(0.09, bars=40, price=quiet[-1])
    combined = quiet + loud[1:]
    assert (algo_trade._trailing_vol({"AAA": combined})["AAA"]
            > algo_trade._trailing_vol({"AAA": quiet})["AAA"])


def test_trailing_vol_is_nan_below_min_periods():
    """Too little history to say what a name's noise is, so no sigma for it."""
    short = wobble(0.03, bars=algo_trade.SIGMA_VOL_MIN)
    assert np.isnan(algo_trade._trailing_vol({"AAA": short})["AAA"])


def test_trailing_vol_is_floored():
    """A price frozen across the window has ~0 vol; unfloored, `_to_sigma` would
    divide by nothing and manufacture a several-million-sigma pick."""
    frozen = [100.0] * (algo_trade.SIGMA_VOL_LEN + 1)
    assert (algo_trade._trailing_vol({"AAA": frozen})["AAA"]
            == algo_trade.SIGMA_VOL_FLOOR)


def test_to_sigma_divides_by_the_names_own_noise():
    assert algo_trade._to_sigma(0.06, 0.02) == pytest.approx(
        0.06 / (0.02 * np.sqrt(algo_trade.SIGMA_HOLD)))


def test_the_vol_window_counts_traded_bars_not_calendar_rows():
    """A halted day must not consume a slot of the volatility window.

    Same invariant the exit average relies on: `_tail_closes` slices the ragged
    window, so a name that missed sessions is measured over its own last traded
    bars rather than over a window shortened by its absences.
    """
    window = algo_trade.SIGMA_VOL_LEN + 1
    stamps = sessions(LOOKBACK, start="2023-10-02")
    stamps = [t for t in stamps if t <= TRAIN_END][-LOOKBACK:]
    path = wobble(0.03, bars=len(stamps))
    dense = feed(stamps, symbols=["AAA"], paths={"AAA": path})
    # BBB prints AAA's last `window` closes, but spread over twice as many
    # sessions — it sat out every other one. Both must still measure the same
    # noise, because both traded the same bars.
    absent = stamps[-(2 * window - 1)::2]
    sparse = [FakeKLine(t, "BBB", c, c, c, c, 1_000.0)
              for t, c in zip(absent, path[-window:])]
    assert len(sparse) == window and sparse[-1].timestamp == stamps[-1]

    ctx, strategy, _ = start_run({"AAA": 0.4, "BBB": 0.4},
                                 bars=dense + sparse, ticks_after_cutoff=0)
    w = ctx.history(LOOKBACK)
    tails = AlgoTradeStrategy._tail_closes(w, {"AAA", "BBB"},
                                           algo_trade.SIGMA_VOL_LEN + 1)
    vol = algo_trade._trailing_vol(tails)
    assert vol["AAA"] == pytest.approx(vol["BBB"])


def test_the_gate_is_off_by_default():
    """The opt-in promise: an untouched strategy ranks on the raw score."""
    assert AlgoTradeStrategy.sigma_gate == 0
    ctx = sigma_run({"AAA": 0.05, "BBB": 0.10},
                    {"AAA": 0.01, "BBB": 0.05}, sigma_gate=0)
    # BBB has the higher raw score and the lower sigma; without the gate it wins.
    assert [o.symbol for o in buys(ctx)][0] == "BBB"


def test_min_score_is_ignored_when_the_gate_is_off():
    ctx = sigma_run(ALL_PASS, {s: 0.02 for s in SYMBOLS},
                    sigma_gate=0, min_score=1e6)
    assert buys(ctx)


def test_the_gate_ranks_on_volatility_adjusted_score():
    """A quiet name with a smaller predicted move beats a loud name with a
    bigger one — the whole point of the notebook's unit."""
    ctx = sigma_run({"AAA": 0.05, "BBB": 0.10},
                    {"AAA": 0.01, "BBB": 0.05})
    assert [o.symbol for o in buys(ctx)][0] == "AAA"


def test_the_floor_can_empty_the_day():
    """Nothing clearing `min_score` means no entries, not a best-of-a-bad-lot."""
    ctx = sigma_run(ALL_PASS, {s: 0.05 for s in SYMBOLS}, min_score=1e6)
    assert buys(ctx) == []


def test_the_floor_admits_what_clears_it():
    ctx = sigma_run({"AAA": 0.05, "BBB": 0.001},
                    {"AAA": 0.01, "BBB": 0.05}, min_score=1.0)
    # AAA is 0.05/(0.01*3) ~= 1.67 sigma; BBB is ~0.007 and is cut.
    assert [o.symbol for o in buys(ctx)] == ["AAA"]


def test_the_gate_does_not_change_how_wide_the_slice_is():
    """`keep` counts every scored name, gate or no gate.

    Sizing the cut on the surviving subset instead would silently narrow the
    gate on any day with negative scores — three of the five below — and the
    slice would still look full.
    """
    scores = {"AAA": 0.05, "BBB": 0.04, "CCC": -0.1, "DDD": -0.2, "EEE": -0.3}
    vols = {s: 0.02 for s in scores}
    assert len(buys(sigma_run(scores, vols, top_pct=40.0, sigma_gate=0))) == 2
    assert len(buys(sigma_run(scores, vols, top_pct=40.0,
                              min_score=-1e6))) == 2


def test_a_negative_score_is_never_entered_under_the_gate():
    """Only the positive name trades, however low the floor is set.

    A negative predicted return is not a trade, and admitting one would also let
    the sigma inversion in: BBB's sigma (-0.33) beats AAA's (-1.33) purely
    because BBB is ten times noisier, even though BBB's predicted return is the
    worse of the two.
    """
    ctx = sigma_run({"AAA": -0.02, "BBB": -0.05, "CCC": 0.001},
                    {"AAA": 0.005, "BBB": 0.05, "CCC": 0.05},
                    min_score=-1e6)
    assert [o.symbol for o in buys(ctx)] == ["CCC"]


def test_a_board_with_no_positive_score_buys_nothing():
    """The anti-inversion guarantee, stated as an outcome: on a day the model
    likes nothing, the gate does not promote the noisiest name to fill a slot."""
    ctx = sigma_run({"AAA": -0.02, "BBB": -0.05},
                    {"AAA": 0.005, "BBB": 0.05}, min_score=-1e6)
    assert buys(ctx) == []


def test_the_gate_still_refuses_names_already_held():
    ctx = sigma_run({"AAA": 0.05, "BBB": 0.04}, {"AAA": 0.01, "BBB": 0.01})
    assert [o.symbol for o in buys(ctx)][0] == "AAA"
    fill_entries(ctx)
    before = len(buys(ctx))
    ctx.advance()
    assert len(buys(ctx)) == before


def test_declared_params_and_indicators():
    names = {p["name"] for p in stonks.param_specs(AlgoTradeStrategy)}
    assert names == {"train_on", "top_pct", "stop_pct", "exit_ma",
                     "max_positions", "cash_buffer", "skip_limit_locked",
                     "sigma_gate", "min_score"}
    assert {i["name"] for i in stonks.indicator_specs(AlgoTradeStrategy)} == {
        SCORE}


def test_the_entry_gate_is_a_percentage_of_the_universe():
    """`top_pct` is a percent, not a fraction and not a sigma.

    All three have been the gate's units at some point in this port's life — an
    early version used quantiles in [0, 1], bist uses raw sigmas — and each reads
    as a plausible number in the others' scale, so a silent unit swap would
    change what the strategy trades without failing anything else.
    """
    value = AlgoTradeStrategy.top_pct
    assert isinstance(value, float)
    assert 0.0 < value <= 100.0
    assert value > 0.05, "a fraction in [0,1] would read as a near-empty slice"


def test_the_fit_takes_the_runs_own_exit_average():
    """Both ends of the trade are one average, and now they cannot disagree.

    This used to be a warning: the entry gate read the loaded artifact's
    `exit_ma` while the exit read the run's, so overriding one without
    retraining entered on one average and left on another, and no report would
    show it. Fitting in-run removes the failure mode instead of reporting it —
    the model is constructed with whatever the run is using — so the test that
    asserted the warning is now a test that the two are the same object's value.
    """
    _, _, stub = start_run(exit_ma=10)
    assert stub.kwargs["exit_ma"] == 10


def test_the_fit_is_told_the_trade_it_is_for():
    _, strategy, stub = start_run()
    assert stub.kwargs["exit_ma"] == strategy.exit_ma
    assert stub.kwargs["max_hold"] == MAX_HOLD


def test_the_stop_is_risk_management_not_a_mirror_of_the_label():
    """The stop used to encode the label's -10% drawdown gate. It no longer can.

    The trade label has no drawdown floor — it holds until the average breaks —
    so the stop now cuts trades the label counts as winners. It is kept anyway,
    as the only exit that can act on an overnight gap. What is pinned here is
    that it still exists and that the target is still gone.
    """
    assert AlgoTradeStrategy.stop_pct > 0.0, "the gap defence must still exist"
    assert not hasattr(AlgoTradeStrategy, "target_pct"), "replaced by exit_ma"
    assert AlgoTradeStrategy.exit_ma >= 2, "an average needs at least two bars"


def test_every_tunable_binds_to_the_block_at_the_top_of_the_module():
    """Editing a constant up there must actually move the strategy's default.

    A stale literal left on the class would read as a working knob and silently
    ignore the block, which is the one failure this arrangement can have.
    """
    bound = {
        "train_on": "TRAIN_ON",
        "top_pct": "ENTRY_TOP_PCT",
        "stop_pct": "STOP_PCT",
        "exit_ma": "EXIT_MA",
        "max_positions": "MAX_POSITIONS",
        "cash_buffer": "CASH_BUFFER",
        "skip_limit_locked": "SKIP_LIMIT_LOCKED",
        "sigma_gate": "USE_SIGMA_GATE",
        "min_score": "ENTRY_MIN_SCORE",
    }
    declared = {p["name"] for p in stonks.param_specs(AlgoTradeStrategy)}
    assert declared == set(bound), "every param must come from the block"
    for attribute, constant in bound.items():
        assert getattr(AlgoTradeStrategy, attribute) == getattr(
            algo_trade, constant), f"{attribute} is not bound to {constant}"


# ---------------------------------------------------------------------------
# ScoringState — the two features a bounded window cannot carry
# ---------------------------------------------------------------------------

def test_state_accumulates_signed_volume():
    """Up bars add volume, down bars subtract, and NaN contributes nothing."""
    state = algo_trade.ScoringState()
    state.seed({"AAA": 0.0}, {"AAA": None})
    state.advance(["AAA"], np.array([1_000.0]), np.array([False]))
    state.advance(["AAA"], np.array([-1_000.0]), np.array([False]))
    state.advance(["AAA"], np.array([500.0]), np.array([False]))
    state.advance(["AAA"], np.array([np.nan]), np.array([False]))
    assert state.obv["AAA"] == pytest.approx(500.0)


def test_state_counter_steps_and_resets():
    """Bars since the last trigger, reset to 0 on the bar the trigger fires."""
    state = algo_trade.ScoringState()
    state.seed({}, {"AAA": None})

    for _ in range(3):
        state.advance(["AAA"], np.array([0.0]), np.array([False]))
    assert state.days_since["AAA"] is None, "no trigger on record yet"

    state.advance(["AAA"], np.array([0.0]), np.array([True]))
    assert state.days_since["AAA"] == 0.0
    state.advance(["AAA"], np.array([0.0]), np.array([False]))
    state.advance(["AAA"], np.array([0.0]), np.array([False]))
    assert state.days_since["AAA"] == 2.0

    state.advance(["AAA"], np.array([0.0]), np.array([True]))
    assert state.days_since["AAA"] == 0.0, "a fresh trigger restarts the count"


def test_state_counter_is_unbounded():
    """The whole point: it must outlive the 300-bar window, not cap at it."""
    state = algo_trade.ScoringState()
    state.seed({}, {"AAA": 0.0})
    for _ in range(LOOKBACK + 500):
        state.advance(["AAA"], np.array([0.0]), np.array([False]))
    assert state.days_since["AAA"] == float(LOOKBACK + 500)


def test_carry_seeds_from_the_window_then_advances(trained):
    """First call trusts the window; later calls trust the carry.

    On the first call the window IS the whole history, so its own obv and counter
    are right and become the seed. That is what makes the scheme exact rather than
    merely stable.
    """
    model, frame, cutoff = trained
    dates = np.sort(frame["timestamp"].unique())
    end = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff)))) + 1

    state = algo_trade.ScoringState()
    first = algo_trade._panel(frame[frame["timestamp"].isin(dates[:end - 1])])
    model.signal(first, state=state)
    assert state.bars == 1
    seeded = dict(state.obv)

    # One more bar: the carry must move by exactly that bar's signed volume.
    second = algo_trade._panel(frame[frame["timestamp"].isin(dates[:end])])
    model.signal(second, state=state)
    assert state.bars == 2

    close = second["close"]
    volume = second["volume"]
    step = (np.sign(np.log(close.iloc[-1] / close.iloc[-2])) * volume.iloc[-1])
    for symbol in seeded:
        if np.isfinite(step.get(symbol, np.nan)):
            assert state.obv[symbol] == pytest.approx(
                seeded[symbol] + step[symbol]), symbol


def test_carried_scoring_matches_batch_scoring(trained):
    """The reason the carry exists: window scoring must equal panel scoring.

    Scored off the full panel, obv and the counter carry real history. Scored off a
    300-bar window they do not — unless the state carries them. Driving the state
    bar by bar and comparing against the batch answer is the end-to-end check.
    """
    model, frame, cutoff = trained
    dates = np.sort(frame["timestamp"].unique())
    end = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff)))) + 1
    panel = algo_trade._panel(frame[frame["timestamp"].isin(dates[:end])])

    batch = model.signal(panel)                       # full history, no state

    # Walk every bar from the one where the window first fills, which is where the
    # strategy starts too. That first window IS the whole history, which is what
    # makes the seed correct; starting later would seed from a rebased obv and no
    # amount of subsequent carrying would recover it.
    state = algo_trade.ScoringState()
    windowed = None
    for stop in range(LOOKBACK, end + 1):
        sub = {k: v.iloc[:stop] for k, v in panel.items()}
        windowed = model.signal({k: v.iloc[-LOOKBACK:] for k, v in sub.items()},
                                state=state)
    assert state.bars == end + 1 - LOOKBACK

    both = batch[SCORE].notna() & windowed[SCORE].notna()
    assert both.any()
    pd.testing.assert_series_equal(batch.loc[both, SCORE],
                                   windowed.loc[both, SCORE])


def test_the_strategy_hands_its_state_to_the_model():
    """One object, passed every bar, owned by the strategy.

    Two calls, not one: the fit scores the panel it trained on to seed the
    carries, then the driven bar scores its window.
    """
    ctx, strategy, stub = start_run()
    drive(strategy, ctx)
    assert stub.calls == 2
    assert {id(s) for s in stub.states} == {id(strategy.state)}


def test_the_model_is_fit_for_the_trade_the_strategy_holds():
    """One head, and the exit average is shared by the label and the strategy.

    If these drift apart the model is ranking names for a trade nobody takes,
    which is exactly the state this target replaced.
    """
    assert algo_trade.EXIT_MA == AlgoTradeStrategy.exit_ma
    assert ManipulationModel().exit_ma == algo_trade.EXIT_MA
    assert ManipulationModel().max_hold == MAX_HOLD
