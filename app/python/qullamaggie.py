"""Qullamaggie complete ruleset — the three setups plus the management engine.

Mechanizes Kristjan "Qullamaggie" Kullamägi's swing-trading system on daily
bars, end to end:

  scan     weekly cross-sectional scan: top-decile 1/3/6-month momentum
           (percentile rank across the run's universe), ADR > 4%, dollar-
           volume and price floors, 10 > 20 EMA trend.
  regime   a synthetic equal-weight index of the run's symbols must hold
           SMA10 > SMA20 with both rising, or no NEW entries are taken
           (exits always run) and resting entries are cancelled.
  BO       breakout: a 30%+ leg into a pivot high, a 2-week-to-2-month
           consolidation that stays orderly and tightens, then a resting
           buy-stop just above the pivot.
  EP       episodic pivot: a 10%+ gap on huge volume in a neglected name
           with a strong close -> market entry at the close.
  PARA     parabolic short (off by default): 50%+ run-up, 3+ consecutive
           up closes, first down close -> market short, covered into the
           10/20-day SMAs.
  manage   initial stop 1 x ADR below entry, tightened to the entry day's
           low once known (the low-of-day stop, never wider than 1 x ADR);
           sell 1/3 into the first up close 3-5 days in (forced on day 5)
           and move the stop to breakeven; exit the remainder on the first
           close below the 10-day EMA. Risk 0.5% of equity per trade sized
           from the stop distance (fees included, notes §2); positions
           capped at 25% of equity and max_positions concurrent names.

Deviations from Qullamaggie's discretionary process (engine reality):

  1. Opening-range/intraday entries don't exist on daily bars: BO enters on
     a resting buy-stop above the pivot ("you can also simply watch the
     daily chart and buy as it breaks out"); EP/PARA enter at the signal
     close via a market order that fills at the NEXT open.
  2. "Scan weekly" = every scan_every-th trading bar.
  3. Months are trading bars: 21/63/126 for the 1/3/6-month windows.
  4. The market filter runs on a synthetic equal-weight index (cumulative
     mean 1-bar return of the run's symbols) — the data ships no NASDAQ
     series. Add an index symbol to the run to get closer.
  5. "Sell 1/3 into strength after 3-5 days" = the first up close between
     partial_min_bars and partial_max_bars after the fill bar, forced at
     partial_max_bars.
  6. The low-of-day stop uses the completed entry bar's low, known at that
     day's close; until then the stop rests 1 x ADR below entry, so day-0
     gap risk is the ADR budget, not the (unknowable) day low.
  7. No pyramiding: the broker rejects same-side adds — matching his "I buy
     everything at once".
  8. The regime filter gates every new entry including PARA shorts (a spec
     choice; enable_para is off by default, so it is moot until turned on).
  9. Leverage is fixed at 1.0 everywhere, stock-style; affordability comes
     from the notional and cash clamps, not from margin.
 10. A gapped fill re-anchors realized risk slightly: stops stay where they
     were planned (BO then re-tightens off the actual fill via the day-low
     rule).
 11. There is no earnings/news feed, so EP is a pure price/volume gap
     definition; "neglected" = no meaningful rally off the lows in the
     prior ~6 months, measured before the gap bar.
 12. PARA covers on the first CLOSE at/below the 10- or 20-day SMA — no
     intrabar MA touches on completed bars.

Execution timeline: decisions on bar close, fills from the next bar on.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import stonks
from stonks import OrderSide, OrderStatus

MOM_WINDOWS = (21, 63, 126)   # 1/3/6 months in trading bars
TREND_FAST, TREND_SLOW = 10, 20
REGIME_FAST, REGIME_SLOW = 10, 20
COVER_FAST, COVER_SLOW = 10, 20
CASH_USE = 0.99               # fraction of free cash an entry may consume
EMA_WINDOW = 100              # bars feeding ema_last


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.mean(a[-n:]))


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def bars_since_highest(a, n):
    """0 = the current bar; ties resolve to the most recent bar."""
    if n <= 0 or len(a) < n:
        return None
    return int(np.argmax(a[-n:][::-1]))


def ema_last(a, n):
    """The last value of an EMA(n): the exponentially-weighted mean of the
    most recent EMA_WINDOW bars (adjust-style EWM, truncated — the dropped
    tail weighs < 1e-4 of the total for n <= 20). Stateless and
    deterministic: no seeding, no state carried across ticks."""
    if n <= 0 or len(a) < n:
        return None
    k = min(len(a), EMA_WINDOW)
    w = np.power(1.0 - 2.0 / (n + 1.0), np.arange(k, dtype=np.float64))
    return float(np.dot(a[-k:][::-1], w) / np.sum(w))


def segments(ts):
    """Per-symbol slice bounds [starts[k], ends[k]] of the combined history
    frame. Rows are contiguous per symbol and every printing symbol's slice
    ends at the tick's timestamp, so the segment ends are exactly the rows
    stamped ts[-1]."""
    ends = np.flatnonzero(ts == ts[-1])
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    return starts, ends


@dataclass
class Armed:
    """One in-flight entry — a resting BO buy-stop, or an EP/PARA market
    order awaiting its fill bar — with the protective stop already
    bracketed under it."""

    setup: str        # "BO" | "EP" | "PARA"
    side: str         # "long" | "short"
    entry_id: int
    stop_id: int
    entry: float
    stop: float
    qty: float
    adr_pct: float
    armed_bar: int    # bar_count at the last (re)arm, for the TTL


@dataclass
class Trade:
    """Management state of one held position."""

    setup: str
    side: str
    entry_id: int
    stop_id: int
    fill_px: float
    stop_px: float
    adr_pct: float
    bars_held: int = 0            # symbol prints since the fill bar
    partial_done: bool = False
    exiting: bool = False         # trail/cover market exit in flight
    trail_ma: Optional[float] = None


@dataclass
class _Tick:
    """One tick's combined history frame, segmented per symbol."""

    now: int
    symbols: list
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    seg_of: dict      # symbol -> segment index

    def sym(self, k):
        return self.symbols[self.ends[k]]

    def bounds(self, k):
        return int(self.starts[k]), int(self.ends[k])


class QullamaggieStrategy(stonks.Strategy):
    # Scan (weekly cadence)
    scan_every = 5
    min_price = 1.0
    min_dollar_volume = 10_000_000.0
    adr_len = 20
    min_adr = 4.0
    rank_pct = 90.0

    # Market-regime filter (synthetic equal-weight index)
    use_regime = True

    # Setup 1 — breakout
    enable_breakout = True
    bo_leg_lookback = 63
    bo_min_leg_gain = 30.0
    bo_base_min = 10
    bo_base_max = 42
    bo_max_retrace = 25.0
    bo_require_tightening = True
    entry_buffer_bps = 10.0
    order_bars = 5

    # Setup 2 — episodic pivot
    enable_ep = True
    ep_min_gap = 10.0
    ep_vol_mult = 3.0
    ep_neglect_lookback = 126
    ep_max_prior_gain = 30.0
    ep_strong_close = True
    ep_max_stop_adr = 1.0

    # Setup 3 — parabolic short
    enable_para = False
    para_lookback = 10
    para_min_gain = 50.0
    para_streak = 3
    para_stop_lookback = 3
    para_max_stop_adr = 1.0

    # Risk / sizing / management
    risk_fraction = 0.005
    max_position_pct = 0.25
    max_positions = 15
    taker_fee_bps = 5.0
    adr_stop_mult = 1.0
    partial_min_bars = 3
    partial_max_bars = 5
    partial_fraction = 1.0 / 3.0
    trail_len = 10
    trail_use_ema = True

    params = {
        "scan_every": stonks.Param("cross-sectional scan cadence (weekly on daily bars)", unit="bars"),
        "min_price": stonks.Param("minimum close price", unit="$"),
        "min_dollar_volume": stonks.Param("20-bar average close x volume floor", unit="$"),
        "adr_len": stonks.Param("ADR length", unit="bars"),
        "min_adr": stonks.Param("minimum average daily range (mean high/low - 1)", unit="%"),
        "rank_pct": stonks.Param("momentum percentile-rank floor, any of the 1/3/6-month windows"),
        "use_regime": stonks.Param("index SMA10 > SMA20 and both rising gates new entries"),
        "enable_breakout": stonks.Param("enable the breakout setup"),
        "bo_leg_lookback": stonks.Param("prior-leg window ending at the pivot", unit="bars"),
        "bo_min_leg_gain": stonks.Param("minimum low-to-pivot gain of the prior leg", unit="%"),
        "bo_base_min": stonks.Param("minimum bars since the pivot high", unit="bars"),
        "bo_base_max": stonks.Param("maximum bars since the pivot high", unit="bars"),
        "bo_max_retrace": stonks.Param("maximum retracement from the pivot inside the base", unit="%"),
        "bo_require_tightening": stonks.Param("last-5-bar mean range < first-5-bar mean range of the base"),
        "entry_buffer_bps": stonks.Param("buy-stop buffer above the pivot", unit="bps"),
        "order_bars": stonks.Param("resting entry good for N bars after its last re-arm", unit="bars"),
        "enable_ep": stonks.Param("enable the episodic pivot setup"),
        "ep_min_gap": stonks.Param("minimum open gap over the prior close", unit="%"),
        "ep_vol_mult": stonks.Param("gap-day volume >= multiple of the prior 50-bar average"),
        "ep_neglect_lookback": stonks.Param("neglect window before the gap bar", unit="bars"),
        "ep_max_prior_gain": stonks.Param("maximum pre-gap rally off the window low", unit="%"),
        "ep_strong_close": stonks.Param("require close > open and close >= midrange on the gap bar"),
        "ep_max_stop_adr": stonks.Param("skip when (close - low)/close exceeds this x ADR", unit="x ADR"),
        "enable_para": stonks.Param("enable the parabolic short setup"),
        "para_lookback": stonks.Param("run-up window", unit="bars"),
        "para_min_gain": stonks.Param("minimum low-to-high run-up over the window", unit="%"),
        "para_streak": stonks.Param("minimum consecutive up closes before the first down close"),
        "para_stop_lookback": stonks.Param("stop above the highest high of the last N bars", unit="bars"),
        "para_max_stop_adr": stonks.Param("skip when the stop distance exceeds this x ADR", unit="x ADR"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade (fee-inclusive, notes §2)"),
        "max_position_pct": stonks.Param("entry notional cap as a fraction of equity"),
        "max_positions": stonks.Param("max open positions + armed entries"),
        "taker_fee_bps": stonks.Param("fee per sizing leg (entry and stop both taker)", unit="bps"),
        "adr_stop_mult": stonks.Param("initial stop distance below entry", unit="x ADR"),
        "partial_min_bars": stonks.Param("earliest partial-sell bar after the fill bar", unit="bars"),
        "partial_max_bars": stonks.Param("forced partial-sell bar after the fill bar", unit="bars"),
        "partial_fraction": stonks.Param("fraction of the position sold at the partial"),
        "trail_len": stonks.Param("trailing MA length for the remainder exit", unit="bars"),
        "trail_use_ema": stonks.Param("EMA (his 10-day) vs SMA for the trail"),
    }

    indicators = {
        "order_level": stonks.Indicator("armed resting entry level"),
        "stop_level": stonks.Indicator("resting stop: initial / day-low tightened / breakeven"),
        "trail_ma": stonks.Indicator("trailing exit MA while held (PARA: fast cover SMA)"),
    }

    def on_start(self, ctx):
        self._fee = self.taker_fee_bps / 10_000.0
        self._lookback = 2 + max(
            MOM_WINDOWS[-1] + 1,
            self.ep_neglect_lookback + 2,
            self.bo_leg_lookback + self.bo_base_max,
            51,                      # previous bar's 50-bar average volume
            self.adr_len + 1,
            EMA_WINDOW,
            max(self.para_lookback, self.para_streak + 2, self.para_stop_lookback) + 1,
            self.trail_len + 1,
        )
        self.bar_count = 0
        self.trades = {}             # symbol -> Trade
        self.armed = {}              # symbol -> Armed
        self._idx = 1.0
        self._idx_hist = deque(maxlen=REGIME_SLOW + 1)
        self.regime_on = False

    def on_tick(self, ctx):
        w = ctx.history(self._lookback)
        if len(w) == 0:
            return
        ts = w.timestamp
        starts, ends = segments(ts)
        t = _Tick(
            now=int(ts[-1]), symbols=w.symbol,
            open=w.open, high=w.high, low=w.low, close=w.close, volume=w.volume,
            starts=starts, ends=ends,
            seg_of={w.symbol[e]: k for k, e in enumerate(ends)},
        )
        self.bar_count += 1

        regime_on = self._update_regime(t)
        self._manage_trades(ctx, t)
        self._manage_armed(ctx, t, regime_on)
        if self.enable_ep:
            self._check_ep(ctx, t, regime_on)
        if self.enable_para:
            self._check_para(ctx, t, regime_on)
        if self.enable_breakout and self.bar_count % self.scan_every == 0:
            self._scan_and_arm(ctx, t, regime_on)
        self._plot_levels(ctx)

    # ─── Market regime ────────────────────────────────────────────────────────

    def _update_regime(self, t):
        """Fold this tick's equal-weight return into the synthetic index and
        gate on SMA10 > SMA20 with both rising. Always True when disabled."""
        live = t.ends > t.starts
        if live.any():
            e = t.ends[live]
            prev = t.close[e - 1]
            ok = prev > 0.0
            if ok.any():
                rets = t.close[e][ok] / prev[ok] - 1.0
                self._idx *= 1.0 + float(np.mean(rets))
        self._idx_hist.append(self._idx)
        if not self.use_regime:
            return True
        if len(self._idx_hist) < REGIME_SLOW + 1:
            return False
        a = np.fromiter(self._idx_hist, dtype=np.float64)
        fast = float(np.mean(a[-REGIME_FAST:]))
        slow = float(np.mean(a[-REGIME_SLOW:]))
        fast_prev = float(np.mean(a[-REGIME_FAST - 1:-1]))
        slow_prev = float(np.mean(a[-REGIME_SLOW - 1:-1]))
        on = fast > slow and fast > fast_prev and slow > slow_prev
        if on != self.regime_on:
            self._print(t.now, "*", f"regime {'ON' if on else 'OFF'} — index "
                                    f"SMA{REGIME_FAST} {fast:.4f} vs "
                                    f"SMA{REGIME_SLOW} {slow:.4f}")
            self.regime_on = on
        return on

    # ─── Trade management (exits always run, regime-independent) ─────────────

    def _manage_trades(self, ctx, t):
        for sym, tr in list(self.trades.items()):
            pos = ctx.position(sym)
            if pos is None:
                # stop / exit filled; the broker's OCO-on-flat already
                # cancelled the subtree — the explicit cancel is defensive
                ctx.cancel_order(tr.stop_id)
                self._print(t.now, sym, f"{tr.setup} position closed")
                del self.trades[sym]
                continue
            k = t.seg_of.get(sym)
            if k is None:
                continue   # symbol did not print; the resting stop keeps guarding
            s, e = t.bounds(k)
            cl = t.close[s:e + 1]
            tr.bars_held += 1
            qty = abs(pos.quantity)
            if tr.side == "long":
                ma = (ema_last(cl, self.trail_len) if self.trail_use_ema
                      else sma(cl, self.trail_len))
                tr.trail_ma = ma
                if tr.exiting:
                    continue
                if ma is not None and cl[-1] < ma:
                    ctx.place_market_order(symbol=sym, side=OrderSide.Sell,
                                           quantity=qty, parent=tr.entry_id,
                                           reduce_only=True)
                    tr.exiting = True
                    self._print(t.now, sym,
                                f"{tr.setup} trail exit: close {cl[-1]:.4f} < "
                                f"{'EMA' if self.trail_use_ema else 'SMA'}"
                                f"{self.trail_len} {ma:.4f}")
                    continue
                if not tr.partial_done and tr.bars_held >= self.partial_min_bars:
                    up = len(cl) >= 2 and cl[-1] > cl[-2]
                    if up or tr.bars_held >= self.partial_max_bars:
                        part = self.partial_fraction * qty
                        ctx.place_market_order(symbol=sym, side=OrderSide.Sell,
                                               quantity=part, parent=tr.entry_id,
                                               reduce_only=True)
                        self._replace_stop(ctx, sym, tr, tr.fill_px, qty - part)
                        tr.partial_done = True
                        self._print(t.now, sym,
                                    f"{tr.setup} partial {self.partial_fraction:.0%} "
                                    f"off after {tr.bars_held} bars, stop to "
                                    f"breakeven {tr.fill_px:.4f}")
            else:
                fast = sma(cl, COVER_FAST)
                slow = sma(cl, COVER_SLOW)
                tr.trail_ma = fast if fast is not None else slow
                if tr.exiting:
                    continue
                if ((fast is not None and cl[-1] <= fast)
                        or (slow is not None and cl[-1] <= slow)):
                    ctx.place_market_order(symbol=sym, side=OrderSide.Buy,
                                           quantity=qty, parent=tr.entry_id,
                                           reduce_only=True)
                    tr.exiting = True
                    self._print(t.now, sym,
                                f"{tr.setup} cover: close {cl[-1]:.4f} tagged "
                                f"the {COVER_FAST}/{COVER_SLOW} SMA")

    # ─── Armed-entry reconciliation ───────────────────────────────────────────

    def _manage_armed(self, ctx, t, regime_on):
        for sym, ar in list(self.armed.items()):
            entry = ctx.order(ar.entry_id)
            status = entry.status if entry is not None else OrderStatus.Cancelled
            if status == OrderStatus.Filled:
                del self.armed[sym]
                pos = ctx.position(sym)
                if pos is None:
                    self._print(t.now, sym, f"{ar.setup} filled and stopped out "
                                            "within the same bar")
                    continue
                fill_px = float(pos.price)
                tr = Trade(ar.setup, ar.side, ar.entry_id, ar.stop_id,
                           fill_px, ar.stop, ar.adr_pct)
                self.trades[sym] = tr
                self._print(t.now, sym, f"{ar.setup} entry filled @ {fill_px:.4f} "
                                        f"qty {abs(pos.quantity):.6g}")
                if ar.setup == "BO":
                    # low-of-day stop, known now that the fill bar is complete;
                    # never wider than adr_stop_mult x ADR below the fill
                    k = t.seg_of.get(sym)
                    if k is not None:
                        lod = float(t.low[t.ends[k]])
                        stop_floor = fill_px * (1.0 - self.adr_stop_mult
                                                * ar.adr_pct / 100.0)
                        new_stop = min(max(lod, stop_floor), fill_px * 0.999)
                        if new_stop > tr.stop_px:
                            self._replace_stop(ctx, sym, tr, new_stop,
                                               abs(pos.quantity))
                            self._print(t.now, sym, "stop tightened to the entry "
                                                    f"day's low: {new_stop:.4f}")
                continue
            if status == OrderStatus.Rejected:
                self._print(t.now, sym, f"{ar.setup} entry rejected by broker "
                                        "(margin + fee exceeded free cash)")
                del self.armed[sym]
                continue
            if status == OrderStatus.Cancelled:
                del self.armed[sym]
                continue
            if ar.setup != "BO":
                continue   # in-flight EP/PARA market: resolves on the next print
            if not regime_on:
                ctx.cancel_order(ar.entry_id)
                self._print(t.now, sym, f"BO order cancelled (regime off) "
                                        f"@ {ar.entry:.4f}")
                del self.armed[sym]
            elif self.bar_count - ar.armed_bar >= self.order_bars:
                ctx.cancel_order(ar.entry_id)
                self._print(t.now, sym, f"BO order expired unfilled @ {ar.entry:.4f}")
                del self.armed[sym]

    # ─── Setup 2: episodic pivot ──────────────────────────────────────────────

    def _check_ep(self, ctx, t, regime_on):
        if not regime_on:
            return
        lens = t.ends - t.starts + 1
        cand = np.flatnonzero(lens >= max(self.ep_neglect_lookback + 2, 51))
        if cand.size == 0:
            return
        e = t.ends[cand]
        prev = t.close[e - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = np.where(prev > 0.0, t.open[e] / prev - 1.0, -1.0)
        for k in cand[gap >= self.ep_min_gap / 100.0]:
            sym = t.sym(k)
            if sym in self.trades or ctx.position(sym) is not None:
                continue
            ar = self.armed.get(sym)
            if ar is not None and ar.setup != "BO":
                continue
            if ar is None and self._capacity_left() <= 0:
                continue
            s, en = t.bounds(k)
            op, hi = t.open[s:en + 1], t.high[s:en + 1]
            lo, cl, vo = t.low[s:en + 1], t.close[s:en + 1], t.volume[s:en + 1]
            if cl[-1] < self.min_price:
                continue
            dv = sma(cl * vo, 20)
            if dv is None or not np.isfinite(dv) or dv < self.min_dollar_volume:
                continue
            av50_prev = sma(vo[:-1], 50)
            if av50_prev is None or vo[-1] < self.ep_vol_mult * av50_prev:
                continue
            base = lowest(cl[:-1], self.ep_neglect_lookback)
            if base is None or base <= 0.0:
                continue
            if cl[-2] / base - 1.0 > self.ep_max_prior_gain / 100.0:
                continue   # not neglected: it already rallied into the news
            if self.ep_strong_close and not (cl[-1] > op[-1]
                                             and cl[-1] >= (hi[-1] + lo[-1]) / 2.0):
                continue
            adr = sma(100.0 * (hi / lo - 1.0), self.adr_len)
            if adr is None or not np.isfinite(adr) or adr <= 0.0:
                continue
            risk = (cl[-1] - lo[-1]) / cl[-1]
            if not 0.0 < risk <= self.ep_max_stop_adr * adr / 100.0:
                continue   # stop wider than the ADR budget: skip the trade
            if ar is not None:
                ctx.cancel_order(ar.entry_id)   # market entry supersedes the arm
                del self.armed[sym]
            stop = min(float(lo[-1]), float(cl[-1]) * 0.999)
            self._enter_market(ctx, t.now, sym, "EP", "long",
                               float(cl[-1]), stop, adr)

    # ─── Setup 3: parabolic short ─────────────────────────────────────────────

    def _check_para(self, ctx, t, regime_on):
        if not regime_on:
            return
        lens = t.ends - t.starts + 1
        need = max(self.para_lookback, self.para_streak + 2,
                   self.para_stop_lookback, self.adr_len) + 1
        cand = np.flatnonzero(lens >= need)
        if cand.size == 0:
            return
        e = t.ends[cand]
        sig = t.close[e] < t.close[e - 1]   # first down close ...
        for i in range(1, self.para_streak + 1):
            sig &= t.close[e - i] > t.close[e - i - 1]   # ... after the streak
        for k in cand[sig]:
            sym = t.sym(k)
            if (sym in self.trades or sym in self.armed
                    or ctx.position(sym) is not None):
                continue
            if self._capacity_left() <= 0:
                continue
            s, en = t.bounds(k)
            hi, lo = t.high[s:en + 1], t.low[s:en + 1]
            cl, vo = t.close[s:en + 1], t.volume[s:en + 1]
            if cl[-1] < self.min_price:
                continue
            dv = sma(cl * vo, 20)
            if dv is None or not np.isfinite(dv) or dv < self.min_dollar_volume:
                continue
            lo_lb = lowest(lo, self.para_lookback)
            hi_lb = highest(hi, self.para_lookback)
            if (lo_lb is None or lo_lb <= 0.0
                    or hi_lb / lo_lb - 1.0 < self.para_min_gain / 100.0):
                continue
            adr = sma(100.0 * (hi / lo - 1.0), self.adr_len)
            if adr is None or not np.isfinite(adr) or adr <= 0.0:
                continue
            stop_ref = highest(hi, self.para_stop_lookback)
            if (stop_ref - cl[-1]) / cl[-1] > self.para_max_stop_adr * adr / 100.0:
                continue
            stop = max(float(stop_ref), float(cl[-1]) * 1.001)
            self._enter_market(ctx, t.now, sym, "PARA", "short",
                               float(cl[-1]), stop, adr)

    # ─── Setup 1: weekly scan + breakout arming ───────────────────────────────

    def _scan_and_arm(self, ctx, t, regime_on):
        if not regime_on:
            return
        lens = t.ends - t.starts + 1
        n = len(t.ends)
        eligible = lens >= MOM_WINDOWS[-1] + 1
        best_rank = np.full(n, -1.0)
        for win in MOM_WINDOWS:
            idxs = np.flatnonzero(eligible & (lens > win))
            if idxs.size == 0:
                continue
            prev = t.close[t.ends[idxs] - win]
            idxs = idxs[prev > 0.0]
            if idxs.size == 0:
                continue
            roc = t.close[t.ends[idxs]] / prev[prev > 0.0] - 1.0
            m = idxs.size
            if m == 1:
                pct = np.array([100.0])
            else:
                order = np.argsort(roc, kind="stable")
                ranks = np.empty(m, dtype=np.float64)
                ranks[order] = np.arange(m, dtype=np.float64)
                pct = 100.0 * ranks / (m - 1)
            np.maximum.at(best_rank, idxs, pct)
        qual = np.flatnonzero(best_rank >= self.rank_pct)
        # best rank first, so the max_positions capacity bites deterministically
        qual = qual[np.argsort(-best_rank[qual], kind="stable")]
        for k in qual:
            sym = t.sym(k)
            if sym in self.trades or ctx.position(sym) is not None:
                continue
            ar = self.armed.get(sym)
            if ar is not None and ar.setup != "BO":
                continue
            s, e = t.bounds(k)
            hi, lo = t.high[s:e + 1], t.low[s:e + 1]
            cl, vo = t.close[s:e + 1], t.volume[s:e + 1]
            if cl[-1] < self.min_price:
                continue
            dv = sma(cl * vo, 20)
            if dv is None or not np.isfinite(dv) or dv < self.min_dollar_volume:
                continue
            adr = sma(100.0 * (hi / lo - 1.0), self.adr_len)
            if adr is None or not np.isfinite(adr) or adr < self.min_adr:
                continue
            e10 = ema_last(cl, TREND_FAST)
            e20 = ema_last(cl, TREND_SLOW)
            if e10 is None or e20 is None or e10 <= e20:
                continue
            pivot = self._detect_breakout(hi, lo, cl)
            if pivot is None:
                continue
            if ar is None and self._capacity_left() <= 0:
                continue
            self._arm_breakout(ctx, t.now, sym, pivot, adr, float(best_rank[k]))

    def _detect_breakout(self, hi, lo, cl):
        """Base geometry on the daily chart: a 30%+ leg into a pivot high,
        then a 2-week-to-2-month consolidation that stays orderly and
        tightens. Returns the pivot high, or None."""
        window = self.bo_leg_lookback + self.bo_base_max
        if len(hi) < window:
            return None
        since_pk = bars_since_highest(hi, window)
        if not self.bo_base_min <= since_pk <= self.bo_base_max:
            return None
        pk = len(hi) - 1 - since_pk
        pivot = float(hi[pk])
        leg_low = float(np.min(lo[pk + 1 - self.bo_leg_lookback:pk + 1]))
        if leg_low <= 0.0 or pivot / leg_low - 1.0 < self.bo_min_leg_gain / 100.0:
            return None
        if float(np.min(lo[pk + 1:])) < pivot * (1.0 - self.bo_max_retrace / 100.0):
            return None
        if self.bo_require_tightening and since_pk >= 10:
            rng = (hi[pk + 1:] - lo[pk + 1:]) / cl[pk + 1:]
            if float(np.mean(rng[-5:])) >= float(np.mean(rng[:5])):
                return None
        return pivot

    # ─── Order plumbing ───────────────────────────────────────────────────────

    def _capacity_left(self):
        return self.max_positions - len(self.trades) - len(self.armed)

    def _size(self, ctx, entry, stop):
        """notes §2 risk mode — a stop-out loses risk_fraction of equity
        including both taker fee legs — clamped by the notional cap and by
        free cash so the fill is not silently rejected."""
        rpu = abs(entry - stop) + entry * self._fee + stop * self._fee
        if entry <= 0.0 or rpu <= 0.0 or not np.isfinite(rpu):
            return 0.0
        equity = ctx.equity()
        qty = min(equity * self.risk_fraction / rpu,
                  self.max_position_pct * equity / entry,
                  CASH_USE * ctx.cash() / (entry * (1.0 + self._fee)))
        if qty <= 0.0 or not np.isfinite(qty):
            return 0.0
        return qty

    def _arm_breakout(self, ctx, now, sym, pivot, adr_pct, rank):
        entry = pivot * (1.0 + self.entry_buffer_bps / 10_000.0)
        stop = entry * (1.0 - self.adr_stop_mult * adr_pct / 100.0)
        stop = min(stop, entry * 0.999)
        ar = self.armed.get(sym)
        if ar is not None:
            if abs(ar.entry - entry) <= 1e-9 * max(1.0, abs(entry)):
                ar.armed_bar = self.bar_count   # same pivot: only the TTL resets
                return
            ctx.cancel_order(ar.entry_id)
            del self.armed[sym]
        qty = self._size(ctx, entry, stop)
        if qty <= 0.0:
            return
        entry_id = ctx.place_stop_order(symbol=sym, side=OrderSide.Buy,
                                        quantity=qty, price=entry)
        stop_id = ctx.place_stop_order(symbol=sym, side=OrderSide.Sell,
                                       quantity=qty, price=stop,
                                       parent=entry_id, reduce_only=True)
        self.armed[sym] = Armed("BO", "long", entry_id, stop_id, entry, stop,
                                qty, adr_pct, self.bar_count)
        self._print(now, sym, f"BO LONG arm stop-entry @ {entry:.4f} | "
                              f"SL {stop:.4f} ({100.0 * (stop / entry - 1.0):+.2f}%) | "
                              f"qty {qty:.6g} | rank {rank:.0f} | "
                              f"valid {self.order_bars} bars")

    def _enter_market(self, ctx, now, sym, setup, side, entry, stop, adr_pct):
        qty = self._size(ctx, entry, stop)
        if qty <= 0.0:
            return
        entry_side = OrderSide.Buy if side == "long" else OrderSide.Sell
        exit_side = OrderSide.Sell if side == "long" else OrderSide.Buy
        entry_id = ctx.place_market_order(symbol=sym, side=entry_side, quantity=qty)
        stop_id = ctx.place_stop_order(symbol=sym, side=exit_side, quantity=qty,
                                       price=stop, parent=entry_id,
                                       reduce_only=True)
        self.armed[sym] = Armed(setup, side, entry_id, stop_id, entry, stop,
                                qty, adr_pct, self.bar_count)
        self._print(now, sym, f"{setup} {side.upper()} enter market @ {entry:.4f} | "
                              f"SL {stop:.4f} ({100.0 * (stop / entry - 1.0):+.2f}%) | "
                              f"qty {qty:.6g}")

    def _replace_stop(self, ctx, sym, tr, price, qty):
        ctx.cancel_order(tr.stop_id)
        exit_side = OrderSide.Sell if tr.side == "long" else OrderSide.Buy
        tr.stop_id = ctx.place_stop_order(symbol=sym, side=exit_side, quantity=qty,
                                          price=price, parent=tr.entry_id,
                                          reduce_only=True)
        tr.stop_px = price

    def _plot_levels(self, ctx):
        for sym, ar in self.armed.items():
            if ar.setup == "BO":
                ctx.plot("order_level", sym, ar.entry)
            ctx.plot("stop_level", sym, ar.stop)
        for sym, tr in self.trades.items():
            ctx.plot("stop_level", sym, tr.stop_px)
            if tr.trail_ma is not None:
                ctx.plot("trail_ma", sym, tr.trail_ma)

    @staticmethod
    def _print(ts, symbol, msg):
        when = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        print(f"[{when.strftime('%Y-%m-%d %H:%M')} UTC] {symbol} {msg}", flush=True)
