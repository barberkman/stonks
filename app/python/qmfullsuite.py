"""Qullamaggie momentum swing with EVERYTHING switched on — the pine run as
its author shipped it would look with all five setup toggles enabled, both
directions live, one book per symbol.

Setups, scanned in the pine's own if/else-if priority:

  1. breakout        long resting buy-stop at the tight-flag pivot
  2. orb             long resting buy-stop above the UTC-day opening range
  3. short breakout  short resting sell-stop under the base low (downtrend)
  4. episodic pivot  long market entry on the gap bar's close
  5. parabolic short short market entry on the first red bar

The pine can keep a long and a short order resting at once because it only
simulates one hypothetical position; a real broker would let the second fill
REDUCE the first. So this port keeps ONE working entry: each tick the
highest-priority valid setup owns the slot, and a fresher/different signal
cancels and replaces a stale unfilled one (the reference strategy's rule).

Management dispatches on how the trade was opened:
  long/short swing  half partial at 2R or partial_bars, breakeven move, then
                    EMA20 close-trail (mirrored for shorts); weak-volume
                    breakout fills are scratched; gapped fills re-anchor
  parabolic         resting stop above the recent high; covered on the first
                    up-close or after ps_max_hold bars — never trailed
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus

MS_PER_DAY = 86_400_000


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def ema(a, n):
    if n <= 0 or len(a) < n:
        return None
    e = float(np.mean(a[:n]))
    alpha = 2.0 / (n + 1.0)
    for v in a[n:]:
        e = alpha * float(v) + (1.0 - alpha) * e
    return e


def adr_pct(high, low, n):
    if n <= 0 or len(high) < n:
        return None
    h = high[-n:]
    l = low[-n:]
    return float(np.sum(100.0 * (h / l - 1.0)) / n)


def gain_pct(close, n):
    if len(close) < n + 1:
        return None
    base = close[len(close) - 1 - n]
    if base == 0.0:
        return None
    return float(100.0 * (close[-1] / base - 1.0))


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


class QMFullSuiteStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Breakout bases (long and short mirror)
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Opening range breakout
    orb_bars = 1
    # Episodic pivot
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # Parabolic short
    ps_lookback = 10
    ps_min_gain = 8.0
    ps_streak = 3
    ps_stop_lb = 3
    ps_max_hold = 5
    # Stop & swing management
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "adr_stop_mult": stonks.Param("stop distance from entry, in ADRs", unit="ADR"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "ps_max_hold": stonks.Param("parabolic short max holding period", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
        "trail": stonks.Indicator("trailing EMA managing the swing setups"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.ps_lookback, self.ps_streak + 2, self.ps_stop_lb,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "setup": None, "trade": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
                "partial_done": False, "scratch": False, "check_break_vol": False,
                "day": None, "day_bars": 0, "or_high": None, "orb_taken_day": None,
                "was_in": False, "cooldown": 0}

    def on_tick(self, ctx):
        w = ctx.history(self.lookback())
        if len(w) == 0:
            return
        df = pd.DataFrame({
            "symbol": w.symbol, "timestamp": w.timestamp,
            "open": w.open, "high": w.high, "low": w.low,
            "close": w.close, "volume": w.volume,
        })
        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp")
            op = sub["open"].to_numpy()
            hi = sub["high"].to_numpy()
            lo = sub["low"].to_numpy()
            cl = sub["close"].to_numpy()
            vo = sub["volume"].to_numpy()
            ts = sub["timestamp"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1

            # Session bookkeeping for the ORB leg (runs from the first bar).
            day = int(ts[-1]) // MS_PER_DAY
            rollover = st["day"] is not None and day != st["day"]
            if st["day"] is None or rollover:
                st["day"] = day
                st["day_bars"] = 1
                st["or_high"] = float(hi[-1])
            else:
                st["day_bars"] += 1
                if st["day_bars"] <= self.orb_bars:
                    st["or_high"] = max(st["or_high"], float(hi[-1]))

            if len(cl) < self.lookback():
                continue

            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            trail = ema(cl, self.trail_len)
            if s10 is not None:
                ctx.plot("sma10", symbol, s10)
            if s20 is not None:
                ctx.plot("sma20", symbol, s20)
            if trail is not None:
                ctx.plot("trail", symbol, trail)

            # An unfilled ORB attempt dies with its session, like the pine's
            # per-session reset of the opening range.
            if rollover and st["pending"] is not None and st["setup"] == "orb":
                entry = ctx.order(st["pending"])
                if entry is not None and entry.status == OrderStatus.Open:
                    ctx.cancel_order(st["pending"])

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["entry_bar"] = st["bar_count"]
                        st["partial_done"] = False
                        if st["setup"] == "orb":
                            st["orb_taken_day"] = day
                        is_long = pos.quantity > 0.0
                        fill = pos.price
                        if (st["trade"] in ("long", "short") and st["planned_entry"] > 0.0
                                and fill != st["planned_entry"]):
                            ratio = fill / st["planned_entry"]
                            exit_side = OrderSide.Sell if is_long else OrderSide.Buy
                            if st["sl"] is not None:
                                ctx.cancel_order(st["sl"])
                            if st["tp"] is not None:
                                ctx.cancel_order(st["tp"])
                            st["planned_stop"] *= ratio
                            st["planned_target"] *= ratio
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=exit_side,
                                quantity=st["entry_qty"], price=st["planned_stop"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp"] = ctx.place_limit_order(
                                symbol=symbol, side=exit_side,
                                quantity=st["entry_qty"] / 2.0, price=st["planned_target"],
                                parent=st["entry_id"], reduce_only=True)
                        if st["check_break_vol"] and not self._break_vol_ok(vo):
                            ctx.place_market_order(
                                symbol=symbol,
                                side=OrderSide.Sell if is_long else OrderSide.Buy,
                                quantity=abs(pos.quantity),
                                parent=st["entry_id"], reduce_only=True)
                            st["scratch"] = True
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
                elif st["setup"] != "orb" and st["bar_count"] - st["armed_bar"] > self.order_ttl_bars:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["scratch"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
                st["trade"] = st["setup"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                self._manage(ctx, symbol, st, pos, cl, trail)
                continue
            if st["cooldown"] > 0:
                continue

            # ── The pine's priority chain owns the single working entry ───────
            sig = self._scan(op, hi, lo, cl, vo, st, day)
            if sig is None:
                continue
            setup, trade, kind, entry_px, stop_px, target_px = sig
            if st["pending"] is not None:
                if st["setup"] == setup and st["planned_entry"] == entry_px:
                    st["armed_bar"] = st["bar_count"]     # same signal: extend the TTL
                    continue
                ctx.cancel_order(st["pending"])           # replaced by the fresher signal
                st["pending"] = None
            risk = abs(entry_px - stop_px)
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_side = OrderSide.Buy if trade in ("long",) else OrderSide.Sell
            exit_side = OrderSide.Sell if trade in ("long",) else OrderSide.Buy
            if kind == "stop":
                entry_id = ctx.place_stop_order(symbol=symbol, side=entry_side,
                                                quantity=qty, price=entry_px)
            else:
                entry_id = ctx.place_market_order(symbol=symbol, side=entry_side,
                                                  quantity=qty)
            st["sl"] = ctx.place_stop_order(symbol=symbol, side=exit_side,
                                            quantity=qty, price=stop_px,
                                            parent=entry_id, reduce_only=True)
            st["tp"] = None
            if target_px is not None:
                st["tp"] = ctx.place_limit_order(symbol=symbol, side=exit_side,
                                                 quantity=qty / 2.0, price=target_px,
                                                 parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["setup"] = setup
            st["trade"] = trade
            st["entry_qty"] = qty
            st["planned_entry"] = entry_px
            st["planned_stop"] = stop_px
            st["planned_target"] = target_px if target_px is not None else 0.0
            st["check_break_vol"] = setup in ("breakout", "short_breakout", "orb")
            st["armed_bar"] = st["bar_count"]

    # ── Management ────────────────────────────────────────────────────────────
    def _manage(self, ctx, symbol, st, pos, cl, trail):
        if st["scratch"]:
            return
        held = abs(pos.quantity)
        bars_held = st["bar_count"] - st["entry_bar"]
        if bars_held < 1:
            return
        is_long = pos.quantity > 0.0
        exit_side = OrderSide.Sell if is_long else OrderSide.Buy
        if st["trade"] == "para":
            bounced = cl[-1] > cl[-2]
            if bounced or bars_held >= self.ps_max_hold:
                ctx.place_market_order(symbol=symbol, side=exit_side, quantity=held,
                                       parent=st["entry_id"], reduce_only=True)
                st["scratch"] = True
            return
        # Swing management, mirrored by direction.
        if not st["partial_done"] and st["tp"] is not None:
            tp_order = ctx.order(st["tp"])
            if tp_order is not None and tp_order.status == OrderStatus.Filled:
                st["partial_done"] = True
                st["tp"] = None
                if self.move_be:
                    self._move_stop_to_breakeven(ctx, symbol, st, pos, exit_side, is_long)
        if not st["partial_done"] and bars_held >= self.partial_bars:
            if st["tp"] is not None:
                ctx.cancel_order(st["tp"])
                st["tp"] = None
            ctx.place_market_order(symbol=symbol, side=exit_side, quantity=held / 2.0,
                                   parent=st["entry_id"], reduce_only=True)
            st["partial_done"] = True
            if self.move_be:
                self._move_stop_to_breakeven(ctx, symbol, st, pos, exit_side, is_long)
        if st["partial_done"] and trail is not None:
            crossed = cl[-1] < trail if is_long else cl[-1] > trail
            if crossed:
                ctx.place_market_order(symbol=symbol, side=exit_side, quantity=held,
                                       parent=st["entry_id"], reduce_only=True)
                st["scratch"] = True

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos, exit_side, is_long):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price) if is_long else min(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=exit_side,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    # ── The priority chain ────────────────────────────────────────────────────
    def _scan(self, op, hi, lo, cl, vo, st, day):
        """First valid setup wins: breakout > orb > short breakout > episodic
        pivot > parabolic short (the pine's if/else-if order). Returns
        (setup, trade, entry_kind, entry, stop, target|None)."""
        bo = self._breakout_arm(hi, lo, cl, vo)
        if bo is not None:
            return ("breakout", "long", "stop") + bo
        orb = self._orb_arm(hi, lo, cl, vo, st, day)
        if orb is not None:
            return ("orb", "long", "stop") + orb
        so = self._breakdown_arm(hi, lo, cl, vo)
        if so is not None:
            return ("short_breakout", "short", "stop") + so
        ep = self._episodic_pivot(op, hi, lo, cl, vo)
        if ep is not None:
            return ("episodic_pivot", "long", "market") + ep
        ps = self._parabolic_short(hi, lo, cl, vo)
        if ps is not None:
            return ("parabolic_short", "para", "market") + ps
        return None

    def _breakout_arm(self, hi, lo, cl, vo):
        if len(cl) < self.base_max_len + 1 or not self._universe(hi, lo, cl, vo, up=True):
            return None
        window = hi[-self.base_max_len:]
        mx = window[0]
        last_pos = 0
        for i in range(self.base_max_len):
            if window[i] >= mx:
                mx = window[i]
                last_pos = i
        since_pk = (self.base_max_len - 1) - last_pos
        pull_low = lowest(lo, max(since_pk, 1))
        if mx <= 0.0 or pull_low is None:
            return None
        retrace = 100.0 * (mx - pull_low) / mx
        if since_pk < self.min_base_days or retrace > self.max_depth:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        entry = float(mx) * (1.0 + self.entry_buffer_bps / 10_000.0)
        stop, target = self._long_levels(entry, adr, lo[-1])
        return entry, stop, target

    def _orb_arm(self, hi, lo, cl, vo, st, day):
        if st["day_bars"] <= self.orb_bars or st["or_high"] is None:
            return None
        if st["orb_taken_day"] == day:
            return None
        if not self._universe(hi, lo, cl, vo, up=True):
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        entry = st["or_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
        if entry <= cl[-1]:
            return None                # the range already gave way; no chase
        stop, target = self._long_levels(entry, adr, lo[-1])
        return entry, stop, target

    def _breakdown_arm(self, hi, lo, cl, vo):
        if len(cl) < self.base_max_len + 1 or not self._universe(hi, lo, cl, vo, up=False):
            return None
        window = lo[-self.base_max_len:]
        mn = window[0]
        last_pos = 0
        for i in range(self.base_max_len):
            if window[i] <= mn:
                mn = window[i]
                last_pos = i
        since_tr = (self.base_max_len - 1) - last_pos
        bounce_high = highest(hi, max(since_tr, 1))
        if mn <= 0.0 or bounce_high is None:
            return None
        retrace_up = 100.0 * (bounce_high - mn) / mn
        if since_tr < self.min_base_days or retrace_up > self.max_depth:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        entry = float(mn) * (1.0 - self.entry_buffer_bps / 10_000.0)
        stop, target = self._short_levels(entry, adr, hi[-1])
        return entry, stop, target

    def _episodic_pivot(self, op, hi, lo, cl, vo):
        if len(cl) < 52 or not self._liq_ok(cl, vo):
            return None
        prev_close = cl[-2]
        if prev_close <= 0.0:
            return None
        if 100.0 * (op[-1] / prev_close - 1.0) < self.ep_min_gap:
            return None
        prev_avg50 = sma(vo[:-1], 50)
        if prev_avg50 is None or vo[-1] < self.ep_vol_mult * prev_avg50:
            return None
        if self.ep_strong_close and not (cl[-1] > op[-1] and cl[-1] >= (hi[-1] + lo[-1]) / 2.0):
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        if adr is None:
            return None
        risk_ps = cl[-1] - lo[-1]
        if not (risk_ps > 0.0 and risk_ps <= self.adr_stop_mult * adr / 100.0 * cl[-1]):
            return None
        entry = float(cl[-1])
        stop, target = self._long_levels(entry, adr, lo[-1])
        return entry, stop, target

    def _parabolic_short(self, hi, lo, cl, vo):
        need = max(self.ps_lookback, self.ps_streak + 1, self.ps_stop_lb) + 1
        if len(cl) < need or not self._liq_ok(cl, vo):
            return None
        hh = highest(hi, self.ps_lookback)
        ll = lowest(lo, self.ps_lookback)
        if hh is None or ll is None or ll <= 0.0:
            return None
        if 100.0 * (hh / ll - 1.0) < self.ps_min_gain:
            return None
        up_streak = 0
        i = len(cl) - 2
        while i >= 1 and cl[i] > cl[i - 1]:
            up_streak += 1
            i -= 1
        if not (cl[-1] < cl[-2] and up_streak >= self.ps_streak):
            return None
        ps_stop = highest(hi, self.ps_stop_lb)
        if ps_stop is None:
            return None
        entry = float(cl[-1])
        stop = ps_stop * (1.0 + self.entry_buffer_bps / 10_000.0)
        return entry, float(stop), None      # the parabolic track has no target

    # ── Levels & gates ────────────────────────────────────────────────────────
    def _long_levels(self, entry, adr, bar_low):
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(bar_low)) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return float(stop), float(entry + self.partial_rr * (entry - stop))

    def _short_levels(self, entry, adr, bar_high):
        adr_stop = entry * (1.0 + self.adr_stop_mult * adr / 100.0)
        stop = min(adr_stop, float(bar_high)) if self.use_lod_stop else adr_stop
        stop = max(stop, entry * 1.001)
        return float(stop), float(entry - self.partial_rr * (stop - entry))

    def _universe(self, hi, lo, cl, vo, up):
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None or not self._liq_ok(cl, vo) or adr < self.min_adr:
            return False
        if up and g < self.min_gain:
            return False
        if not up and g > -self.min_gain:
            return False
        if self.require_mas:
            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is None or s20 is None:
                return False
            if up and not (cl[-1] > s20 and s10 > s20):
                return False
            if not up and not (cl[-1] < s20 and s10 < s20):
                return False
        return True

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def _break_vol_ok(self, vo):
        if not self.use_bo_vol:
            return True
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
