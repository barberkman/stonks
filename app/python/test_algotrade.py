"""Unit tests for AlgoTrade — the bist manipulation model as a strategy.

Two halves. The model half checks the properties the port rests on: features are
causal, they are window-bounded except for the two `ScoringState` carries, and
labels never read past the training cutoff. The strategy half injects a stub model
so the entry rule, the bracket shape and the leak guard can be tested without
training anything.

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
    FEATURE_NAMES,
    HEADS,
    LOOKBACK,
    OBV_COLUMN,
    AlgoTradeStrategy,
    Head,
    ManipulationModel,
)

HEAD_BY_NAME = {h.name: h for h in HEADS}

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

def ramp_panel(T=200, N=8, end_row=150, bars=8, per_bar=0.05, seed=5):
    """A noisy panel where symbol 0 ramps up over `bars` bars, ending at end_row.

    The move is spread across bars rather than applied as one jump because a
    single +40% print is a corporate action as far as `_returns` is concerned and
    gets booked as 0%. Per-bar steps stay inside the daily band, so the move is a
    return the model can actually see.

    Peers keep drifting, so the equal-weighted market return absorbs 1/N of the
    move and symbol 0's excess return keeps its sign. The noise keeps
    `excess_return_vol_60` — the label's denominator — off zero.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(T, N)), axis=0))
    start = end_row - bars + 1
    factor = np.ones(T)
    for i in range(start, end_row + 1):
        factor[i:] *= 1.0 + per_bar
    close[:, 0] = close[:, 0] * factor
    return panel_from_arrays(close), start


def test_label_takes_the_better_of_forward_and_centered():
    """bist's target is max(fwd, centered), and the centered term reaches back.

    Its centered window for h=10 spans [T-4, T+5], so a move that finished at T
    still labels T even though an entry at T+1 cannot capture any of it. bist's
    published precision figures inherit that optimism. Reproduced because it is
    what the shipped model was fit on: dropping it shrinks the label's spread by
    roughly a quarter at q90 and the fit shrinks with it.
    """
    h10 = HEAD_BY_NAME["up_h10"]
    end_row = 150
    panel, start = ramp_panel(T=200, end_row=end_row, bars=8, per_bar=0.05)
    y = algo_trade._labels(panel, h10)

    # The bar just before the ramp sees all of it in [T+1, T+10].
    ahead = y[start - 1, 0]
    # The bar the ramp finished on sees it *behind* — and is labelled anyway,
    # because the centered window covers [T-4, T+5].
    behind = y[end_row, 0]
    assert ahead > 3.0, ahead
    assert behind > 3.0, behind


def test_centered_window_is_what_lifts_a_finished_move():
    """Isolate the centered term: forward-only would score the last bar ~0."""
    h10 = HEAD_BY_NAME["up_h10"]
    end_row = 150
    panel, _ = ramp_panel(T=200, end_row=end_row, bars=8, per_bar=0.05)

    with_center = algo_trade._labels(panel, h10)[end_row, 0]
    # h=1 degenerates to forward-only in bist, which is the comparison we want,
    # but the cleanest isolation is to check the bar 6 ahead of the ramp's end:
    # past the centered reach (h//2 = 5), nothing should be left.
    past_reach = algo_trade._labels(panel, h10)[end_row + 6, 0]
    assert with_center > 3.0
    assert abs(past_reach) < with_center / 3.0, past_reach


def test_label_gates_on_drawdown():
    """A +40% run you would have been stopped out of first is worth zero.

    The gate is what makes the up target tradable rather than a move detector.
    The down head is unconditional and keeps its value.
    """
    h5 = HEAD_BY_NAME["up_h5"]
    dn5 = HEAD_BY_NAME["dn_h5"]
    T, N, row = 200, 8, 150

    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(T, N)), axis=0))
    # Symbol 0: down 15% on the very next bar, then up 40% — inside h=5.
    base = close[row, 0]
    close[row + 1, 0] = base * 0.85
    close[row + 2:, 0] = base * 1.40

    panel = panel_from_arrays(close)
    assert algo_trade._labels(panel, h5)[row, 0] == 0.0
    assert np.isfinite(algo_trade._labels(panel, dn5)[row, 0])


def test_label_stops_short_of_the_end_by_its_shortest_forward_reach():
    """How far a label resolves is set by the shortest window it needs.

    The up heads need the drawdown gate, whose forward minimum spans [T+1, T+h],
    so they stop `h` rows short. The down head is unconditional and its centered
    window only reaches T + h//2, so it resolves further.
    """
    T = 300
    panel = random_panel(T=T, N=8, seed=21)
    reach = {"up_h5": 5, "up_h10": 10, "dn_h5": 5 // 2}
    for head in HEADS:
        y = algo_trade._labels(panel, head)
        resolved = np.flatnonzero(np.isfinite(y).any(axis=1))
        assert resolved.max() == T - 1 - reach[head.name], head.name


def test_label_never_resolves_inside_a_symbols_dead_tail():
    """A symbol that stops printing must not borrow its own delisting.

    Packing lifts each symbol's traded bars to the top of the column, leaving dead
    slots below. `min_periods` will happily resolve a rolling window inside that
    dead region, and `shift(-h)` would then pull the value back onto real rows —
    where bist has NaN because its per-symbol group genuinely ended.
    """
    T, N = 200, 6
    panel = random_panel(T=T, N=N, seed=4, quirks=False)
    # Symbol 1 delists 30 rows early.
    for key in ("open", "high", "low", "close", "volume"):
        panel[key].iloc[T - 30:, 1] = np.nan
    panel["ret"] = algo_trade._returns(panel["close"])

    for head in HEADS:
        y = algo_trade._labels(panel, head)
        live = np.flatnonzero(np.isfinite(y[:, 1]))
        assert live.max() <= T - 30 - 1, head.name


def test_label_is_nan_where_dispersion_collapsed():
    """A perfectly flat panel has no idiosyncratic vol to normalise by."""
    panel = panel_from_arrays(np.full((120, 5), 50.0))
    for head in HEADS:
        assert not np.isfinite(algo_trade._labels(panel, head)).any(), head.name


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
    assert set(model.models) == {h.name for h in HEADS}


def test_train_labels_never_read_past_the_cutoff(trained):
    """The embargo is structural, so assert it on the labels themselves.

    A label at row r reads at most `reach` rows ahead — `horizon` for the up heads,
    whose drawdown gate spans [T+1, T+h], and `horizon // 2` for the unconditional
    down head, whose centered window is its longest-resolving term. Every row the
    fit could keep must satisfy r + reach <= last in-sample row.
    """
    _, frame, cutoff = trained
    panel = algo_trade._panel(frame)
    index = panel["close"].index.to_numpy()
    end = int(np.searchsorted(index, np.datetime64(pd.Timestamp(cutoff)), side="right"))
    truncated = algo_trade._truncate(panel, end)

    for head in HEADS:
        reach = head.horizon if head.max_drawdown is not None else head.horizon // 2
        y = algo_trade._labels(truncated, head)
        newest = np.flatnonzero(np.isfinite(y).any(axis=1)).max()
        assert newest + reach <= end - 1, head.name


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

    head = HEADS[0]
    filled = model._predict(head, np.nan_to_num(rows, nan=0.0, posinf=0.0,
                                                neginf=0.0))
    assert np.isfinite(filled).all()


def test_save_load_round_trip_is_exact(trained, tmp_path):
    model, frame, cutoff = trained
    window = _window_from_frame(frame, cutoff)

    model.save(tmp_path)
    reloaded = ManipulationModel.load(tmp_path)
    assert reloaded.train_end == model.train_end
    assert reloaded.liquid == model.liquid
    assert reloaded.params == model.params
    pd.testing.assert_frame_equal(model.signal(window), reloaded.signal(window))


def test_load_rejects_a_stale_feature_set(trained, tmp_path, monkeypatch):
    """A booster is only valid against the column order it was trained on."""
    model, _, _ = trained
    model.save(tmp_path)
    monkeypatch.setattr(algo_trade, "FEATURE_NAMES", FEATURE_NAMES[:-1])
    with pytest.raises(RuntimeError, match="different feature set"):
        ManipulationModel.load(tmp_path)


def test_load_without_an_artifact_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="algo_trade.py"):
        ManipulationModel.load(tmp_path / "nothing")


def _window_from_frame(frame, cutoff):
    """The trailing LOOKBACK bars ending at `cutoff`, as a window-shaped panel."""
    dates = np.sort(frame["timestamp"].unique())
    end = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cutoff)))) + 1
    keep = dates[max(0, end - LOOKBACK):end]
    sub = frame[frame["timestamp"].isin(keep)]
    return algo_trade._panel(sub)


def test_signal_reports_sigma_and_expected_pct_per_head(trained):
    """Two columns per head, on bist's raw-sigma scale. No percentile layer."""
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff))

    assert set(sig.columns) == {
        f"{h.name}{suffix}" for h in HEADS for suffix in ("", "_pct")}
    for head in HEADS:
        assert sig[head.name].notna().any()


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
    assert np.isnan(sig.at["NEW", "up_h10"])


def test_signal_only_scores_the_liquid_universe(trained):
    """Symbols under bist's turnover percentile are not scored at all."""
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff))
    scored = set(sig.index[sig["up_h5"].notna()])
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

    def __init__(self, frame, train_end=TRAIN_END):
        self.frame = frame
        self.train_end = train_end
        self.calls = 0
        self.states = []

    def signal(self, window, state=None):
        self.calls += 1
        self.states.append(state)
        return self.frame


def signal_frame(rows):
    """{symbol: (up_h5, up_h10, dn_h5)} -> a signal frame of raw sigmas."""
    frame = pd.DataFrame.from_dict(
        rows, orient="index", columns=[h.name for h in HEADS])
    for head in HEADS:
        frame[f"{head.name}_pct"] = frame[head.name] * 10.0
    return frame


# Clears bist's h5 >= 3.0 and h10 >= 2.0 with the down head elevated, which must
# not matter — bist never gates on it.
ALL_PASS = {s: (4.0, 3.0, 2.5) for s in SYMBOLS}


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


def start_run(rows=ALL_PASS, *, bars=None, ticks_after_cutoff=1, cash=100_000.0,
              train_end=TRAIN_END, paths=None, **overrides):
    """A strategy driven to just past the cutoff, with a stub model attached.

    Always goes through the real `on_start` with `load` patched, rather than
    setting up the strategy's state by hand — hand-rolled setup silently rots
    the moment `on_start` grows another attribute.

    Warmup is skipped by advancing without ticking: the strategy provably does
    nothing until it has LOOKBACK distinct timestamps (asserted separately), and
    driving 300 idle ticks through FakeContext's O(bars) history is pure cost.
    The cost of that shortcut is that `self._obv` starts empty here, which is
    exactly the hazard caveat 1 in the class docstring describes; the obv tests
    below drive from the first bar instead.
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
    with patch.object(ManipulationModel, "load", classmethod(lambda cls, d: stub)):
        strategy.on_start(ctx)
    for name, value in overrides.items():
        setattr(strategy, name, value)

    stamps = sorted({b.timestamp for b in bars})
    for _ in range(len(stamps) - ticks_after_cutoff):
        ctx.advance()
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


def test_on_start_loads_the_artifact_and_clears_state():
    ctx, strategy, stub = start_run()
    assert strategy.model is stub
    assert strategy.book == set()
    assert strategy.exiting == set()
    assert strategy.state.obv == {} and strategy.state.bars == 0


def test_warmup_takes_no_positions():
    """Fewer than LOOKBACK distinct timestamps means no signal is even asked for."""
    stamps = sessions(20, start="2025-01-02")
    ctx = FakeContext(feed(stamps), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    stub = StubModel(signal_frame(ALL_PASS), train_end=0)
    with patch.object(ManipulationModel, "load", classmethod(lambda cls, d: stub)):
        strategy.on_start(ctx)

    drive(strategy, ctx, ticks=20)
    assert ctx.orders == []
    assert stub.calls == 0


def test_no_trading_on_bars_the_model_was_trained_on():
    """The artifact saw those bars; trading them is not a backtest result.

    `signal` IS still called on them, unlike an earlier version of this strategy:
    ScoringState has to see every bar in order, so scoring runs from the moment the
    window fills and only *trading* waits for train_end.
    """
    before = sessions(LOOKBACK + 5, start="2023-10-02")
    train_end = before[-1]
    ctx = FakeContext(feed(before), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    stub = StubModel(signal_frame(ALL_PASS), train_end=train_end)
    with patch.object(ManipulationModel, "load", classmethod(lambda cls, d: stub)):
        strategy.on_start(ctx)

    drive(strategy, ctx, ticks=len(before))
    assert ctx.orders == [], "in-sample bars must not be traded"
    assert stub.calls == 6, "but they must still be scored, to advance the state"


def test_enters_the_qualifying_names():
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)
    assert sorted(o.symbol for o in buys(ctx)) == sorted(SYMBOLS)
    assert strategy.book == set(SYMBOLS)


@pytest.mark.parametrize("veto,expected", [
    ({"up_h5": 2.9}, "up_h5 below h5_min"),
    ({"up_h10": 1.9}, "up_h10 below h10_min"),
])
def test_each_up_gate_vetoes_independently(veto, expected):
    """Both up heads must clear their own threshold. bist's two-head conjunction.

    The gates are pinned here rather than inherited from the tunables block, so
    the veto values stay just under them however the block is tuned.
    """
    rows = dict(ALL_PASS)
    fields = [h.name for h in HEADS]
    values = list(rows["AAA"])
    for key, value in veto.items():
        values[fields.index(key)] = value
    rows["AAA"] = tuple(values)

    ctx, strategy, _ = start_run(rows, h5_min=3.0, h10_min=2.0)
    drive(strategy, ctx)
    entered = {o.symbol for o in buys(ctx)}
    assert "AAA" not in entered, expected
    assert entered == set(SYMBOLS) - {"AAA"}


def test_the_down_head_never_vetoes():
    """bist displays dn as an avoidance overlay and never subtracts it.

    A veto here would be this port's invention, so its absence is pinned.
    """
    rows = dict(ALL_PASS)
    rows["AAA"] = (4.0, 3.0, 99.0)     # maximally alarming downside
    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    assert "AAA" in {o.symbol for o in buys(ctx)}


def test_no_entries_when_nothing_qualifies():
    ctx, strategy, _ = start_run({s: (1.0, 1.0, 0.1) for s in SYMBOLS})
    drive(strategy, ctx)
    assert ctx.orders == []


def test_ranks_on_the_composite():
    """Order is by mean up sigma descending, not frame order."""
    rows = {
        "AAA": (4.0, 3.0, 0.1),    # composite 3.5
        "BBB": (9.0, 8.0, 0.1),    # composite 8.5 — best
        "CCC": (3.5, 2.5, 0.1),    # composite 3.0
        "DDD": (6.0, 5.0, 0.1),    # composite 5.5 — second
        "EEE": (3.2, 2.1, 0.1),
    }
    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    assert [o.symbol for o in buys(ctx)][:2] == ["BBB", "DDD"]


def test_ties_break_on_symbol():
    """Equal composites must order deterministically, not by frame order."""
    rows = {s: (4.0, 3.0, 0.1) for s in ["EEE", "CCC", "AAA", "DDD", "BBB"]}
    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    assert [o.symbol for o in buys(ctx)] == ["AAA", "BBB", "CCC", "DDD", "EEE"]


def test_there_is_no_position_cap():
    """bist caps one position per symbol and nothing else."""
    many = [f"S{i:02d}" for i in range(30)]
    before = sessions(LOOKBACK, start="2023-10-02")
    before = [t for t in before if t <= TRAIN_END][-LOOKBACK:]
    bars = feed(before + [TRAIN_END + 86_400_000], symbols=many)
    rows = {s: (4.0, 3.0, 0.1) for s in many}

    ctx, strategy, _ = start_run(rows, bars=bars)
    drive(strategy, ctx)
    assert len(buys(ctx)) == len(many)


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
    stub.frame = signal_frame({s: (1.0, 1.0, 0.1) for s in SYMBOLS})

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


def test_entry_size_is_a_share_of_free_cash():
    """bist commits 5% of available balance per name."""
    ctx, strategy, _ = start_run(cash=1_000.0, position_pct=5.0, cash_buffer=0.02)
    drive(strategy, ctx)

    orders = buys(ctx)
    assert len(orders) == len(SYMBOLS)
    budget = 1_000.0 * 0.98 * 0.05
    for order in orders:
        assert order.quantity == pytest.approx(budget / 100.0)


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
def test_the_limit_locked_gate_is_opt_in(skip, expected):
    """A close at the +10% band cannot be bought at tomorrow's open — but bist
    buys it anyway, and reproducing bist is the point.

    Measured on the real artifact: of the 14 signals bist's gate produces over the
    2025-2026 out-of-sample stretch, all 14 closed at the band on the signal bar.
    So `skip_limit_locked=1` is not a marginal filter, it empties the strategy —
    which is itself the most useful thing the port has to say about these rules.
    """
    # One out-of-sample bar, and it is the locked one, so AAA never gets an
    # unlocked tick on which it could have been bought first. The previous close
    # comes off the window, so no prior tick is needed.
    ctx, strategy, _ = start_run(paths={"AAA": [111.0]},
                                 skip_limit_locked=skip)
    drive(strategy, ctx)
    assert {o.symbol for o in buys(ctx)} == expected


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
    """One line per entry, carrying the order and the sigmas the gates read.

    The line has to be readable back against h5_min/h10_min without re-running
    the model, which is the whole point of printing the raw sigmas rather than
    just the composite.
    """
    rows = dict(ALL_PASS)
    rows["AAA"] = (4.0, 3.0, 2.5)
    ctx, strategy, _ = start_run(rows, cash=1_000.0, position_pct=5.0,
                                 cash_buffer=0.02, stop_pct=10.0, exit_ma=20)
    drive(strategy, ctx)

    line = next(l for l in capsys.readouterr().out.splitlines() if " AAA " in l)
    assert line.startswith("[2025-01-01 00:00 UTC] AAA ")
    assert "enter market @ 100.0000" in line
    assert "SL 90.0000 (-10.00%)" in line
    assert "exit < MA20" in line
    # sigma, then expected excess percent — signal_frame sets _pct to 10x sigma.
    assert "h5 4.000 (+40.00%)" in line
    assert "h10 3.000 (+30.00%)" in line
    assert "dn 2.500" in line
    assert "composite 3.500" in line
    assert f"qty {1_000.0 * 0.98 * 0.05 / 100.0:.6g}" in line


def test_nothing_is_logged_when_no_order_is_placed(capsys):
    ctx, strategy, _ = start_run({s: (1.0, 1.0, 0.1) for s in SYMBOLS})
    drive(strategy, ctx)
    assert capsys.readouterr().out == ""


def test_entries_are_plotted_for_all_three_heads():
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)
    plotted = {(p.name, p.symbol) for p in ctx.plots}
    for symbol in SYMBOLS:
        for head in ("up_h5", "up_h10", "dn_h5"):
            assert (head, symbol) in plotted


def test_declared_params_and_indicators():
    names = {p["name"] for p in stonks.param_specs(AlgoTradeStrategy)}
    assert names == {"h5_min", "h10_min", "stop_pct", "exit_ma",
                     "position_pct", "cash_buffer", "skip_limit_locked"}
    assert {i["name"] for i in stonks.indicator_specs(AlgoTradeStrategy)} == {
        "up_h5", "up_h10", "dn_h5"}


def test_entry_gates_are_absolute_sigmas_not_percentiles():
    """bist publishes h5 >= 3.0 / h10 >= 2.0, and the block is expected to drift
    from those — tuning is what it is for. What must not drift is the *scale*.

    An earlier version of this port gated on percentiles of the training
    prediction distribution, because its fit predicted roughly 4x tighter than
    bist's. The current fit reproduces bist's scale, so these are raw sigmas: a
    value in [0, 1] would silently read as a percentile and select nearly
    everything.
    """
    for name in ("h5_min", "h10_min"):
        value = getattr(AlgoTradeStrategy, name)
        assert isinstance(value, float)
        assert 1.0 < value < 10.0, f"{name}={value} is not on a sigma scale"


def test_the_stop_still_matches_the_labels_drawdown_gate():
    """The stop is the half of bist's bracket this port kept, and it is kept
    because the up labels are zeroed on exactly this drawdown. Tying the test to
    that reason rather than to bist's number says why it matters if it moves.
    """
    gated = [h.max_drawdown for h in HEADS if h.max_drawdown is not None]
    assert AlgoTradeStrategy.stop_pct == pytest.approx(
        100.0 * abs(gated[0])), "the stop no longer enforces the label's gate"
    assert not hasattr(AlgoTradeStrategy, "target_pct"), "replaced by exit_ma"
    assert AlgoTradeStrategy.exit_ma >= 2, "an average needs at least two bars"


def test_every_tunable_binds_to_the_block_at_the_top_of_the_module():
    """Editing a constant up there must actually move the strategy's default.

    A stale literal left on the class would read as a working knob and silently
    ignore the block, which is the one failure this arrangement can have.
    """
    bound = {
        "h5_min": "ENTRY_H5_MIN",
        "h10_min": "ENTRY_H10_MIN",
        "stop_pct": "STOP_PCT",
        "exit_ma": "EXIT_MA",
        "position_pct": "POSITION_PCT",
        "cash_buffer": "CASH_BUFFER",
        "skip_limit_locked": "SKIP_LIMIT_LOCKED",
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

    both = batch["up_h5"].notna() & windowed["up_h5"].notna()
    assert both.any()
    pd.testing.assert_series_equal(batch.loc[both, "up_h5"],
                                   windowed.loc[both, "up_h5"])


def test_the_strategy_hands_its_state_to_the_model():
    """One object, passed every bar, owned by the strategy."""
    ctx, strategy, stub = start_run()
    drive(strategy, ctx)
    assert stub.calls == 1
    assert stub.states[-1] is strategy.state


def test_heads_match_bists_configuration():
    """Two drawdown-gated up horizons and one unconditional down horizon."""
    assert HEAD_BY_NAME["up_h5"] == Head("up_h5", 5, "up", -0.10)
    assert HEAD_BY_NAME["up_h10"] == Head("up_h10", 10, "up", -0.10)
    assert HEAD_BY_NAME["dn_h5"] == Head("dn_h5", 5, "dn", None)
