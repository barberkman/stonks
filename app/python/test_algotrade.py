"""Unit tests for AlgoTrade — the bist manipulation model as a strategy.

Two halves. The model half checks the properties the port rests on: features are
causal and window-bounded, labels read forward only and never past the training
cutoff, and returns are cleaned before anything consumes them. The strategy half
injects a stub model so the entry rule, the bracket shape and the leak guard can
be tested without training anything.

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
    FEATURE_NAMES,
    HEADS,
    LOOKBACK,
    QUANTILE_GRID,
    AlgoTradeStrategy,
    Head,
    ManipulationModel,
)

HEAD_BY_NAME = {h.name: h for h in HEADS}


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
    """features(panel)[i] must equal features(panel[i-299:i+1])[-1], exactly.

    This is the invariant that lets one model be fit on a whole panel and scored
    one bar at a time. Not "close enough": a tree split sitting on zero turns a
    1e-17 disagreement into a different leaf, so the NaN pattern and the values
    both have to match bit for bit.
    """
    panel = random_panel(T=420, N=12, seed=7)
    full = algo_trade._features(panel)

    for i in (LOOKBACK - 1, 340, 400, 419):
        windowed = algo_trade._features(window_of(panel, i))[-1]
        row = full[i]
        assert np.array_equal(np.isnan(row), np.isnan(windowed)), (
            f"NaN pattern differs at row {i}")
        assert np.array_equal(row, windowed, equal_nan=True), (
            f"feature values differ at row {i}")


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

    assert np.array_equal(baseline, algo_trade._features(scrambled)[i], equal_nan=True)


def test_feature_layout_matches_the_contract():
    panel = random_panel(T=320, N=6, seed=3)
    assert algo_trade._features(panel).shape == (320, 6, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 72


# ---------------------------------------------------------------------------
# Return cleaning
# ---------------------------------------------------------------------------

def test_returns_book_corporate_actions_as_flat():
    """A bonus issue is not a 2800x return; a genuine limit day is a real move."""
    close = np.array([[100.0], [120.0], [100.0], [2_800.0], [2_800.0], [np.nan],
                      [2_900.0]])
    ret = algo_trade._returns(pd.DataFrame(close)).iloc[:, 0].to_numpy()

    assert np.isnan(ret[0])                       # no previous bar
    assert ret[1] == pytest.approx(0.20)          # +20%, inside the band
    assert ret[2] == pytest.approx(-1.0 / 6.0)    # -16.7%, inside the band
    assert ret[3] == 0.0                          # 2800% — a corporate action
    assert ret[4] == pytest.approx(0.0)           # unchanged
    assert np.isnan(ret[5])                       # halted: unknown, not flat
    assert np.isnan(ret[6])                       # no usable previous close


def test_returns_keep_nan_out_of_the_zero_bucket():
    """`.where(cond, 0.0)` would swallow NaN into 0.0; it must not."""
    close = pd.DataFrame([[100.0], [np.nan], [np.nan], [101.0]])
    ret = algo_trade._returns(close).iloc[:, 0]
    assert ret.isna().tolist() == [True, True, True, True]


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


def test_label_reads_forward_only():
    """A move that already happened must not label the bar it finished on.

    This is the deliberate divergence from bist, which takes
    max(forward, centered) — its centered window for h=10 spans [T-4, T+5], so a
    move finishing at T inflates the label at T even though an entry at T+1
    cannot capture any of it.
    """
    h10 = HEAD_BY_NAME["up_h10"]
    end_row = 150
    panel, start = ramp_panel(T=200, end_row=end_row, bars=8, per_bar=0.05)
    y = algo_trade._labels(panel, h10)

    # The bar just before the ramp sees all of it in [T+1, T+10].
    ahead = y[start - 1, 0]
    # The bar the ramp finished on sees nothing ahead of it.
    behind = y[end_row, 0]
    assert ahead > 3.0, ahead
    assert abs(behind) < 1.0, behind
    assert ahead > 4.0 * abs(behind)


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


def test_label_resolves_exactly_horizon_rows_short_of_the_end():
    """The forward window is the embargo: the last `horizon` rows have no label."""
    panel = random_panel(T=300, N=8, seed=21)
    for head in HEADS:
        y = algo_trade._labels(panel, head)
        resolved = np.flatnonzero(np.isfinite(y).any(axis=1))
        assert resolved.max() == 300 - 1 - head.horizon, head.name


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
    """A small real fit. Slow enough to share, fast enough to keep honest."""
    panel = random_panel(T=420, N=22, seed=13)
    frame = long_frame(panel)
    cutoff = sorted(frame["timestamp"].unique())[-40]
    model = ManipulationModel(rounds=20).train(frame, train_end=cutoff)
    return model, frame, cutoff


def test_train_stops_at_the_cutoff(trained):
    """train_end is a hard boundary: the fit records the last bar it could see."""
    model, frame, cutoff = trained
    assert model.train_end == int(pd.Timestamp(cutoff).value // 1_000_000)
    assert set(model.models) == {h.name for h in HEADS}


def test_train_labels_never_read_past_the_cutoff(trained):
    """The embargo is structural, so assert it on the labels themselves.

    A label at row r reads through r + horizon. Every row the fit could keep must
    therefore satisfy r + horizon <= last in-sample row.
    """
    _, frame, cutoff = trained
    panel = algo_trade._panel(frame)
    index = panel["close"].index.to_numpy()
    end = int(np.searchsorted(index, np.datetime64(pd.Timestamp(cutoff)), side="right"))
    truncated = algo_trade._truncate(panel, end)

    for head in HEADS:
        y = algo_trade._labels(truncated, head)
        newest = np.flatnonzero(np.isfinite(y).any(axis=1)).max()
        assert newest + head.horizon <= end - 1, head.name


def test_train_records_its_prediction_scale(trained):
    """Calibration is what makes the entry rule portable across retrains."""
    model, _, _ = trained
    for head in HEADS:
        q = model.quantiles[head.name]
        assert len(q) == len(QUANTILE_GRID)
        assert q == sorted(q), f"{head.name} quantiles must be non-decreasing"


def test_save_load_round_trip_is_exact(trained, tmp_path):
    model, frame, cutoff = trained
    window = _window_from_frame(frame, cutoff)

    model.save(tmp_path)
    reloaded = ManipulationModel.load(tmp_path)
    assert reloaded.train_end == model.train_end
    assert reloaded.quantiles == model.quantiles
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


def test_signal_reports_sigma_and_percentile_per_head(trained):
    model, frame, cutoff = trained
    sig = model.signal(_window_from_frame(frame, cutoff))

    for head in HEADS:
        assert head.name in sig
        assert f"{head.name}_pct" in sig
        q = sig[f"{head.name}_q"].dropna()
        assert ((q >= 0.0) & (q <= 1.0)).all()
    assert set(sig.columns) == {
        f"{h.name}{suffix}" for h in HEADS for suffix in ("", "_pct", "_q")}


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
    sig = model.signal(algo_trade._panel(pd.concat([sub, newborn])))
    assert np.isnan(sig.at["NEW", "up_h10"])
    assert np.isnan(sig.at["NEW", "up_h10_q"])


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

    def signal(self, window):
        self.calls += 1
        return self.frame


def signal_frame(rows):
    """{symbol: (up_h5_q, up_h10_q, dn_h5_q)} -> a signal frame."""
    frame = pd.DataFrame.from_dict(
        rows, orient="index", columns=["up_h5_q", "up_h10_q", "dn_h5_q"])
    for head in HEADS:
        # Raw sigma tracks the percentile monotonically, which is all the
        # strategy's composite ranking needs.
        frame[head.name] = frame[f"{head.name}_q"]
        frame[f"{head.name}_pct"] = frame[f"{head.name}_q"] * 10.0
    return frame


ALL_PASS = {s: (0.999, 0.999, 0.5) for s in SYMBOLS}


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


def ma_exits(ctx):
    """Market sells, i.e. moving-average exits — the stop is a resting order."""
    return [o for o in ctx.orders
            if o.side == OrderSide.Sell and o.order_type == "market"]


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
    """The artifact saw those bars; trading them is not a backtest result."""
    before = sessions(LOOKBACK + 5, start="2023-10-02")
    train_end = before[-1]
    ctx = FakeContext(feed(before), cash=100_000.0)
    strategy = AlgoTradeStrategy()
    stub = StubModel(signal_frame(ALL_PASS), train_end=train_end)
    with patch.object(ManipulationModel, "load", classmethod(lambda cls, d: stub)):
        strategy.on_start(ctx)

    drive(strategy, ctx, ticks=len(before))
    assert ctx.orders == []
    assert stub.calls == 0, "signal must not even be consulted in-sample"


def test_enters_the_qualifying_names():
    ctx, strategy, _ = start_run()
    drive(strategy, ctx)
    assert sorted(o.symbol for o in buys(ctx)) == sorted(SYMBOLS)
    assert strategy.book == set(SYMBOLS)


@pytest.mark.parametrize("veto,expected", [
    ({"up_h5_q": 0.5}, "up_h5 below h5_q"),
    ({"up_h10_q": 0.5}, "up_h10 below h10_q"),
    ({"dn_h5_q": 0.99}, "dn_h5 above dn_q_max"),
])
def test_each_gate_vetoes_independently(veto, expected):
    """Three conditions, each sufficient on its own to keep a name out."""
    rows = dict(ALL_PASS)
    fields = ["up_h5_q", "up_h10_q", "dn_h5_q"]
    values = list(rows["AAA"])
    for key, value in veto.items():
        values[fields.index(key)] = value
    rows["AAA"] = tuple(values)

    ctx, strategy, _ = start_run(rows)
    drive(strategy, ctx)
    entered = {o.symbol for o in buys(ctx)}
    assert "AAA" not in entered, expected
    assert entered == set(SYMBOLS) - {"AAA"}


def test_ranks_on_the_composite_and_respects_max_positions():
    """Two slots go to the two best composites, not to whoever sorts first."""
    rows = {
        "AAA": (0.995, 0.995, 0.1),   # composite 0.995
        "BBB": (0.999, 0.999, 0.1),   # composite 0.999 — best
        "CCC": (0.990, 0.990, 0.1),
        "DDD": (0.998, 0.998, 0.1),   # second best
        "EEE": (0.991, 0.991, 0.1),
    }
    ctx, strategy, _ = start_run(rows, max_positions=2)
    drive(strategy, ctx)
    assert sorted(o.symbol for o in buys(ctx)) == ["BBB", "DDD"]


def test_ties_break_on_symbol():
    """Equal composites must order deterministically, not by frame order."""
    rows = {s: (0.999, 0.999, 0.1) for s in ["EEE", "CCC", "AAA", "DDD", "BBB"]}
    ctx, strategy, _ = start_run(rows, max_positions=2)
    drive(strategy, ctx)
    assert sorted(o.symbol for o in buys(ctx)) == ["AAA", "BBB"]


def test_no_entries_when_nothing_qualifies():
    ctx, strategy, _ = start_run({s: (0.5, 0.5, 0.5) for s in SYMBOLS})
    drive(strategy, ctx)
    assert ctx.orders == []


def test_every_entry_carries_exactly_one_stop():
    """The stop is the only resting leg; the target is gone, replaced by the MA."""
    ctx, strategy, _ = start_run(max_positions=1, stop_pct=10.0)
    drive(strategy, ctx)

    entry = buys(ctx)[0]
    children = [o for o in ctx.orders if o.parent == entry.id]
    assert len(children) == 1
    stop = children[0]
    assert stop.order_type == "stop"
    assert stop.price == pytest.approx(100.0 * 0.90)
    assert stop.side == OrderSide.Sell
    assert stop.reduce_only is True
    assert stop.quantity == pytest.approx(entry.quantity)
    assert not [o for o in ctx.orders if o.order_type == "limit"]


def test_ma_exit_fires_when_the_close_breaks_below():
    """One bar at 90 against nineteen at 100 puts the close under the SMA(20)."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2, paths={"AAA": [100.0, 90.0]})
    drive(strategy, ctx)
    fill_entries(ctx)

    drive(strategy, ctx)
    exits = ma_exits(ctx)
    assert [o.symbol for o in exits] == ["AAA"]
    assert exits[0].reduce_only is True
    assert exits[0].quantity == pytest.approx(ctx.positions["AAA"].quantity)


def test_no_ma_exit_while_the_close_holds_above():
    """A name at or above its average is left alone — there is no time exit."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=3, paths={"AAA": [101.0, 102.0]})
    drive(strategy, ctx)
    fill_entries(ctx)

    drive(strategy, ctx, ticks=2)
    assert ma_exits(ctx) == []


def test_ma_exit_is_not_resent_while_it_is_pending():
    """The position survives one more bar before the sale settles; don't re-send."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=3,
                                 paths={"AAA": [100.0, 90.0, 89.0]})
    drive(strategy, ctx)
    fill_entries(ctx)

    drive(strategy, ctx)
    assert len(ma_exits(ctx)) == 1
    assert strategy.exiting == {"AAA"}

    drive(strategy, ctx)                       # still below, still not flat
    assert len(ma_exits(ctx)) == 1, "an exit must not be sent twice"


def test_ma_exit_runs_even_with_a_full_book():
    """The slot check must not short-circuit ahead of the exit rule."""
    ctx, strategy, _ = start_run(max_positions=len(SYMBOLS), ticks_after_cutoff=2,
                                 paths={"AAA": [100.0, 90.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    assert len(strategy.book) == strategy.max_positions   # no free slot

    drive(strategy, ctx)
    assert [o.symbol for o in ma_exits(ctx)] == ["AAA"]


def test_an_exiting_name_keeps_its_slot_until_it_is_flat():
    """Entries must not be funded out of proceeds that have not settled."""
    ctx, strategy, _ = start_run(max_positions=1, ticks_after_cutoff=3,
                                 paths={"AAA": [100.0, 90.0, 90.0]})
    drive(strategy, ctx)
    fill_entries(ctx)
    held = buys(ctx)[0].symbol

    drive(strategy, ctx)                       # exit placed, not yet filled
    assert len(ma_exits(ctx)) == 1
    assert held in strategy.book, "the slot is still occupied this bar"
    assert len(buys(ctx)) == 1, "no entry may be funded from an unsettled sale"

    ctx.positions.pop(held)                    # the sale settles
    drive(strategy, ctx)
    assert held not in strategy.exiting
    assert len(buys(ctx)) == 2, "the slot frees once the position is flat"


def test_ma_exit_needs_a_full_window():
    """A name with fewer than exit_ma bars has no average to break."""
    ctx, strategy, _ = start_run(ticks_after_cutoff=2, exit_ma=LOOKBACK + 50,
                                 paths={"AAA": [100.0, 1.0]})
    drive(strategy, ctx)
    fill_entries(ctx)

    drive(strategy, ctx)
    assert ma_exits(ctx) == []


def test_tail_closes_slices_each_symbol_independently():
    """The exit rule reads per-symbol tails off the ragged window, not a pivot."""
    bars = [
        FakeKLine(1_000, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0),
        FakeKLine(2_000, "AAA", 2.0, 2.0, 2.0, 2.0, 1.0),
        FakeKLine(3_000, "AAA", 3.0, 3.0, 3.0, 3.0, 1.0),
        FakeKLine(2_000, "BBB", 7.0, 7.0, 7.0, 7.0, 1.0),
        FakeKLine(3_000, "BBB", 8.0, 8.0, 8.0, 8.0, 1.0),
    ]
    ctx = FakeContext(bars)
    for _ in range(3):
        ctx.advance()
    w = ctx.history(5)

    tails = AlgoTradeStrategy._tail_closes(w, {"AAA", "BBB"}, 2)
    assert tails["AAA"].tolist() == [2.0, 3.0]
    assert tails["BBB"].tolist() == [7.0, 8.0]
    assert AlgoTradeStrategy._tail_closes(w, {"AAA"}, 99)["AAA"].tolist() == [1.0, 2.0, 3.0]
    assert AlgoTradeStrategy._tail_closes(w, set(), 2) == {}


def test_entry_size_respects_the_cash_buffer():
    """Sized on the lesser of an equal slot and what free cash can fund."""
    ctx, strategy, _ = start_run(max_positions=10, cash=1_000.0,
                                 cash_buffer=0.02)
    drive(strategy, ctx)

    orders = buys(ctx)
    assert len(orders) == len(SYMBOLS)
    budget = min(1_000.0 / 10, 1_000.0 * 0.98 / len(SYMBOLS))
    for order in orders:
        assert order.quantity == pytest.approx(budget / 100.0)


def test_a_full_book_does_not_ask_for_a_signal():
    """The signal is the run's dominant cost and there is nothing to do with it."""
    ctx, strategy, stub = start_run(max_positions=2, ticks_after_cutoff=2)
    drive(strategy, ctx)
    for order in buys(ctx):
        ctx.positions[order.symbol] = FakePosition(quantity=order.quantity, price=100.0)
    assert stub.calls == 1

    drive(strategy, ctx)
    assert stub.calls == 1, "a full book must short-circuit before signal()"


def test_a_position_closed_behind_our_back_frees_a_slot():
    """A bracket exit or a liquidation drops the name from the book.

    The freed slot is then refillable — including by the same name, since a
    fixed bracket is the only exit and nothing here embargoes a re-entry.
    """
    ctx, strategy, _ = start_run(max_positions=1, ticks_after_cutoff=2)
    drive(strategy, ctx)
    held = buys(ctx)[0].symbol
    ctx.positions[held] = FakePosition(quantity=1.0, price=100.0)
    assert strategy.book == {held}

    ctx.positions.pop(held)          # the stop filled
    drive(strategy, ctx)
    assert len(buys(ctx)) == 2, "the freed slot was not refilled"


def test_held_names_are_not_re_entered():
    ctx, strategy, _ = start_run(ticks_after_cutoff=2)
    drive(strategy, ctx)
    for order in buys(ctx):
        ctx.positions[order.symbol] = FakePosition(quantity=order.quantity, price=100.0)
    before = len(ctx.orders)

    drive(strategy, ctx)
    assert len(ctx.orders) == before, "the book is full and every name is held"


def test_entries_are_plotted():
    ctx, strategy, _ = start_run(max_positions=1)
    drive(strategy, ctx)
    assert [p.name for p in ctx.plots] == ["up_h10"]
    assert ctx.plots[0].symbol == buys(ctx)[0].symbol


def test_declared_params_and_indicators():
    names = {p["name"] for p in stonks.param_specs(AlgoTradeStrategy)}
    assert names == {"h5_q", "h10_q", "dn_q_max", "stop_pct", "exit_ma",
                     "max_positions", "cash_buffer"}
    assert [i["name"] for i in stonks.indicator_specs(AlgoTradeStrategy)] == ["up_h10"]


def test_heads_match_bists_configuration():
    """Two drawdown-gated up horizons and one unconditional down horizon."""
    assert HEAD_BY_NAME["up_h5"] == Head("up_h5", 5, "up", -0.10)
    assert HEAD_BY_NAME["up_h10"] == Head("up_h10", 10, "up", -0.10)
    assert HEAD_BY_NAME["dn_h5"] == Head("dn_h5", 5, "dn", None)
