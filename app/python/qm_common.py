"""Shared framework for the Qullamaggie momentum setups.

Ports the setups in ``app/pines/qullamaggie_momentum_swing.pine`` to the stonks
engine. The Pine indicator is intraday-aware and uses resting stop orders with
intrabar fills; the engine has neither, so every setup here follows one model:

  * Signals are read off the **closed** daily bar (the last row of the window).
  * Entries are **market orders that fill at the next bar's open**; stops, targets
    and the trailing-MA exit are *simulated* by inspecting each closed bar and
    emitting a market order when a level is crossed. This costs ~1 bar of lag vs
    the Pine intrabar model and is unavoidable given Market/Limit-only orders.
  * Stops/targets are anchored to the Pine reference price (breakout pivot, EP
    close), accepting the next-open fill gap.

There are no built-in indicators or position queries in the engine, so this
module hand-rolls the indicators and each strategy tracks its own per-symbol
position. ``QMBase`` holds the shared tick loop, sizing, and trade management;
each setup file subclasses it and implements only ``signal()``.
"""

from dataclasses import dataclass, field

import numpy as np

import stonks
from stonks import OrderSide

# The dataset carries no tick size, so the "buffer beyond the pivot" is zero.
MINTICK = 0.0


# ─── Parameters (mirror the Pine inputs) ──────────────────────────────────────
@dataclass
class Params:
    # Universe & trend filters
    min_price: float = 5.0
    min_avg_vol: float = 0.0
    adr_len: int = 20
    min_adr: float = 0.1
    mom_len: int = 24
    min_gain: float = 0.5
    require_mas: bool = True
    # Setup 1 — momentum breakout
    base_max_len: int = 40
    min_base_days: int = 3
    max_depth: float = 40.0
    use_vol_dry: bool = False
    vol_dry_ratio: float = 1.0
    buf_ticks: int = 0
    use_bo_vol: bool = True
    bo_vol_mult: float = 1.3
    wait_close: bool = True  # daily close-confirm the break (vs intrabar high)
    # Setup 1b — opening range breakout
    orb_bars: int = 1
    # Setup 2 — episodic pivot
    ep_min_gap: float = 0.5
    ep_vol_mult: float = 1.3
    ep_strong_close: bool = True
    # Setup 3 — parabolic short
    ps_lookback: int = 10
    ps_min_gain: float = 8.0
    ps_streak: int = 3
    ps_stop_lb: int = 3
    ps_max_hold: int = 5
    # Initial stop
    adr_stop_mult: float = 1.0
    use_lod_stop: bool = True
    # Trade management
    partial_rr: float = 2.0
    partial_days: int = 6
    move_be: bool = True
    trail_type: str = "EMA"
    trail_len: int = 20
    # Sizing (no Pine equivalent — the indicator is signal-only)
    risk_per_trade_pct: float = 0.5
    partial_fraction: float = 0.5


# ─── Indicators (operate on a 1-D numpy array, return the latest value) ───────
def sma(a, n):
    if a is None or n <= 0 or len(a) < n:
        return None
    return float(np.mean(a[-n:]))


def ema(a, n):
    """SMA-seeded EMA over the supplied window (seed = SMA of the first n)."""
    if a is None or n <= 0 or len(a) < n:
        return None
    alpha = 2.0 / (n + 1)
    e = float(np.mean(a[:n]))
    for x in a[n:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return e


def highest(a, n):
    if a is None or n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if a is None or n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def adr_pct(high, low, n):
    """Average per-bar range %, matching Pine's adrPct = sma(100*(high/low-1))."""
    if high is None or n <= 0 or len(high) < n:
        return None
    rng = 100.0 * (high[-n:] / low[-n:] - 1.0)
    return float(np.mean(rng))


def gain_pct(close, n):
    """Momentum: 100 * (close[-1] / close[-1-n] - 1)."""
    if close is None or len(close) < n + 1:
        return None
    base = float(close[-1 - n])
    if base == 0.0:
        return None
    return 100.0 * (float(close[-1]) / base - 1.0)


# ─── Per-symbol view of the multi-symbol window ──────────────────────────────
@dataclass
class Bars:
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self):
        return len(self.close)


def symbol_slices(window):
    """Split the long-frame MarketWindow into one Bars per symbol (order kept)."""
    syms = np.asarray(window.symbol)
    ts = np.asarray(window.timestamp)
    op = np.asarray(window.open)
    hi = np.asarray(window.high)
    lo = np.asarray(window.low)
    cl = np.asarray(window.close)
    vo = np.asarray(window.volume)
    out = {}
    for s in window.symbol:
        if s not in out:
            m = syms == s
            out[s] = Bars(ts[m], op[m], hi[m], lo[m], cl[m], vo[m])
    return out


# ─── Universe / trend gates (mirror Pine liqOK / adrOK / trendUp|DnOK) ────────
def liq_ok(bars, p):
    av20 = sma(bars.volume, 20)
    return float(bars.close[-1]) >= p.min_price and av20 is not None and av20 >= p.min_avg_vol


def adr_ok(bars, p):
    a = adr_pct(bars.high, bars.low, p.adr_len)
    return a is not None and a >= p.min_adr


def _ma_ok_up(bars, p):
    if not p.require_mas:
        return True
    s10, s20 = sma(bars.close, 10), sma(bars.close, 20)
    if s10 is None or s20 is None:
        return False
    return float(bars.close[-1]) > s20 and s10 > s20


def _ma_ok_dn(bars, p):
    if not p.require_mas:
        return True
    s10, s20 = sma(bars.close, 10), sma(bars.close, 20)
    if s10 is None or s20 is None:
        return False
    return float(bars.close[-1]) < s20 and s10 < s20


def universe_long(bars, p):
    g = gain_pct(bars.close, p.mom_len)
    return liq_ok(bars, p) and adr_ok(bars, p) and g is not None and g >= p.min_gain and _ma_ok_up(bars, p)


def universe_short(bars, p):
    g = gain_pct(bars.close, p.mom_len)
    return liq_ok(bars, p) and adr_ok(bars, p) and g is not None and g <= -p.min_gain and _ma_ok_dn(bars, p)


# ─── Breakout base detection (excludes the current/break bar) ─────────────────
def flag_base_long(bars, p):
    """Pivot = high of a prior consolidation; return its level if a valid base
    has formed and the current bar breaks above it, else None."""
    high, low, vol, close = bars.high, bars.low, bars.volume, bars.close
    n = p.base_max_len
    if len(high) < n + 1:
        return None
    base_high, base_low = high[:-1], low[:-1]
    w = base_high[-n:]
    mx = w.max()
    last_pos = int(np.where(w == mx)[0][-1])
    since_pk = (len(w) - 1) - last_pos
    pull_low = float(base_low[-max(since_pk, 1):].min())
    retrace = 100.0 * (float(mx) - pull_low) / float(mx) if mx > 0 else 1e9
    vol_dry = (not p.use_vol_dry) or _vol_dry_ok(vol, p)
    if not (since_pk >= p.min_base_days and retrace <= p.max_depth and vol_dry):
        return None
    pivot = float(mx) + p.buf_ticks * MINTICK
    broke = close[-1] >= pivot if p.wait_close else high[-1] >= pivot
    if not broke or not _break_vol_ok(vol, p):
        return None
    return EntryPlan(entry_ref=pivot)


def flag_base_short(bars, p):
    """Mirror of flag_base_long: pivot = low of a prior base, broken below."""
    high, low, vol, close = bars.high, bars.low, bars.volume, bars.close
    n = p.base_max_len
    if len(low) < n + 1:
        return None
    base_high, base_low = high[:-1], low[:-1]
    w = base_low[-n:]
    mn = w.min()
    last_pos = int(np.where(w == mn)[0][-1])
    since_tr = (len(w) - 1) - last_pos
    bounce_high = float(base_high[-max(since_tr, 1):].max())
    retrace = 100.0 * (bounce_high - float(mn)) / float(mn) if mn > 0 else 1e9
    vol_dry = (not p.use_vol_dry) or _vol_dry_ok(vol, p)
    if not (since_tr >= p.min_base_days and retrace <= p.max_depth and vol_dry):
        return None
    pivot = float(mn) - p.buf_ticks * MINTICK
    broke = close[-1] <= pivot if p.wait_close else low[-1] <= pivot
    if not broke or not _break_vol_ok(vol, p):
        return None
    return EntryPlan(entry_ref=pivot)


def _vol_dry_ok(vol, p):
    v5, v50 = sma(vol, 5), sma(vol, 50)
    return v5 is not None and v50 is not None and v5 < p.vol_dry_ratio * v50


def _break_vol_ok(vol, p):
    """Break bar volume >= boVolMult * avgVol50[1] (prior-bar 50-bar average)."""
    if not p.use_bo_vol:
        return True
    prev_avg50 = sma(vol[:-1], 50)
    return prev_avg50 is not None and float(vol[-1]) >= p.bo_vol_mult * prev_avg50


# ─── Sizing ──────────────────────────────────────────────────────────────────
def size_by_risk(equity, cash, entry, stop, p):
    """Risk a fixed % of equity to the stop; cap notional at available cash so a
    market buy can't exceed cash (an unfillable buy would linger and fill late)."""
    rps = abs(entry - stop)
    if rps <= 0.0 or entry <= 0.0:
        return 0.0
    qty = (equity * p.risk_per_trade_pct / 100.0) / rps
    cap = (cash * 0.99) / entry
    return max(0.0, min(qty, cap))


# ─── Entry plan + position state ─────────────────────────────────────────────
@dataclass
class EntryPlan:
    entry_ref: float           # price the stop/target/size are anchored to
    explicit_stop: float = None  # set by parabolic short; else ADR/LoD stop is used
    use_target: bool = True      # parabolic short has no R target


@dataclass
class Position:
    side: str                  # "LONG" / "SHORT"
    mgmt: str                  # "R" / "PARA"
    entry: float
    stop: float
    target: float
    qty: float
    bars_held: int = 0
    partial_done: bool = False
    filled: bool = False       # False until the bar after entry (the fill bar)


# ─── Base strategy: shared tick loop, entry, and management ──────────────────
class QMBase(stonks.Strategy):
    DIRECTION = "LONG"          # "LONG" / "SHORT"
    MGMT = "R"                  # "R" (stop/partial/BE/trail) / "PARA" (parabolic)
    SAME_BAR_BAIL = True        # bail on the fill bar if it closes past the stop
    PARAMS = Params()

    def on_start(self, ctx):
        self.p = type(self).PARAMS
        self.pos = {}           # symbol -> Position

    def signal(self, bars):
        """Return an EntryPlan if this symbol triggers an entry, else None."""
        raise NotImplementedError(f"{type(self).__name__} must implement signal(bars)")

    def on_tick(self, ctx):
        window = ctx.history(self._lookback())
        for symbol, bars in symbol_slices(window).items():
            if len(bars) < 2:
                continue
            if symbol in self.pos:
                self._manage(ctx, symbol, bars)
            else:
                plan = self.signal(bars)
                if plan is not None:
                    self._enter(ctx, symbol, bars, plan)

    # warmup / window sizing ----------------------------------------------------
    def _lookback(self):
        p = self.p
        return max(p.base_max_len, 51, p.mom_len) + 5

    # entry ---------------------------------------------------------------------
    def _enter(self, ctx, symbol, bars, plan):
        p = self.p
        entry = plan.entry_ref
        adr = adr_pct(bars.high, bars.low, p.adr_len)
        if adr is None:
            return
        if self.DIRECTION == "LONG":
            if plan.explicit_stop is not None:
                stop = plan.explicit_stop
            else:
                adr_stop = entry * (1.0 - p.adr_stop_mult * adr / 100.0)
                stop = max(adr_stop, float(bars.low[-1])) if p.use_lod_stop else adr_stop
            stop = min(stop, entry * 0.999)
            risk = entry - stop
            target = entry + p.partial_rr * risk if plan.use_target else float("nan")
            side_open = OrderSide.Buy
        else:
            if plan.explicit_stop is not None:
                stop = plan.explicit_stop
            else:
                adr_stop = entry * (1.0 + p.adr_stop_mult * adr / 100.0)
                stop = min(adr_stop, float(bars.high[-1])) if p.use_lod_stop else adr_stop
            stop = max(stop, entry * 1.001)
            risk = stop - entry
            target = entry - p.partial_rr * risk if plan.use_target else float("nan")
            side_open = OrderSide.Sell
        if risk <= 0.0:
            return
        qty = size_by_risk(ctx.equity(), ctx.cash(), entry, stop, p)
        if qty <= 0.0:
            return
        if ctx.place_market_order(symbol=symbol, side=side_open, quantity=qty):
            self.pos[symbol] = Position(self.DIRECTION, self.MGMT, entry, stop, target, qty)

    # management ----------------------------------------------------------------
    def _manage(self, ctx, symbol, bars):
        pos = self.pos[symbol]
        c, h, l = float(bars.close[-1]), float(bars.high[-1]), float(bars.low[-1])

        if not pos.filled:
            # The order placed last tick fills at this bar's open; management of
            # exits begins next bar (mirrors Pine's bar_index > entryBar), with an
            # optional same-bar bail if this bar already closes past the stop.
            pos.filled = True
            pos.bars_held = 0
            if pos.mgmt == "R" and self.SAME_BAR_BAIL:
                if pos.side == "LONG" and c < pos.stop:
                    self._close(ctx, symbol, pos, OrderSide.Sell)
                elif pos.side == "SHORT" and c > pos.stop:
                    self._close(ctx, symbol, pos, OrderSide.Buy)
            return

        pos.bars_held += 1

        if pos.mgmt == "PARA":
            prev_close = float(bars.close[-2])
            if h >= pos.stop or c > prev_close or pos.bars_held >= self.p.ps_max_hold:
                self._close(ctx, symbol, pos, OrderSide.Buy)
            return

        trail = ema(bars.close, self.p.trail_len) if self.p.trail_type == "EMA" \
            else sma(bars.close, self.p.trail_len)

        if pos.side == "LONG":
            if l <= pos.stop:
                self._close(ctx, symbol, pos, OrderSide.Sell)
                return
            if not pos.partial_done and (h >= pos.target or pos.bars_held >= self.p.partial_days):
                self._partial(ctx, symbol, pos, OrderSide.Sell)
                if self.p.move_be:
                    pos.stop = max(pos.stop, pos.entry)
            if pos.partial_done and trail is not None and c < trail:
                self._close(ctx, symbol, pos, OrderSide.Sell)
        else:
            if h >= pos.stop:
                self._close(ctx, symbol, pos, OrderSide.Buy)
                return
            if not pos.partial_done and (l <= pos.target or pos.bars_held >= self.p.partial_days):
                self._partial(ctx, symbol, pos, OrderSide.Buy)
                if self.p.move_be:
                    pos.stop = min(pos.stop, pos.entry)
            if pos.partial_done and trail is not None and c > trail:
                self._close(ctx, symbol, pos, OrderSide.Buy)

    def _partial(self, ctx, symbol, pos, side):
        part = pos.qty * self.p.partial_fraction
        if part > 0.0 and ctx.place_market_order(symbol=symbol, side=side, quantity=part):
            pos.qty -= part
            pos.partial_done = True

    def _close(self, ctx, symbol, pos, side):
        if pos.qty > 0.0:
            ctx.place_market_order(symbol=symbol, side=side, quantity=pos.qty)
        del self.pos[symbol]
