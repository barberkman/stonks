"""Qullamaggie momentum swing — the faithful port, pine defaults.

Ports app/pines/qullamaggie_momentum_swing.pine with its default switches:
long-only breakout (Setup 1) + episodic pivot (Setup 2); ORB, short breakout
and parabolic short stay off (they get their own interpretation files).

What is literal:
  * universe/trend gates (price floor, ADR floor, momentum gain, close>SMA20
    with SMA10>SMA20), the tight-flag base scan (pivot high of the last 40
    bars, >=3 bars of digestion, retracement <=40%), the resting buy-stop at
    the pivot with a 10-bar TTL refreshed while the setup holds, the EP gap
    bar (gap%, volume multiple, strong close, risk within one ADR), the ADR
    initial stop tightened to the signal bar's low, the half partial at 2R or
    after 6 bars, the move to breakeven, and the EMA20 close-trail exit.

What is reinterpreted for a real broker:
  * pine's "buffer ticks" becomes a fraction-of-price buffer (no tick size).
  * pine gates the resting order's fill on the break bar's volume; a real
    resting stop cannot see volume, so the gate becomes a post-fill check —
    if the fill bar closed with volume < bo_vol_mult x the 50-bar average,
    the trade is scratched at the next open.
  * a gapped fill re-anchors the protective legs proportionally to the actual
    fill price (same pattern the engine docs describe).

Execution timeline: decisions on bar close, fills from the next bar on.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


# ─── Stateless TA helpers (recomputed from the pulled window each tick) ───────
def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def ema(a, n):
    """EMA seeded with the SMA of the first n window values, then iterated
    across the rest. The pulled window is >= 3x the period, so the truncated
    tail's weight is negligible."""
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


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


class QMLiteralStrategy(stonks.Strategy):
    # Universe & trend filters (pine defaults)
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Setup 1 — momentum breakout base
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    use_vol_dry = False
    vol_dry_ratio = 1.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Setup 2 — episodic pivot
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # Initial stop & management (pine's "Trade management")
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
        "order_ttl_bars": stonks.Param("resting buy-stop lifetime", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
                "partial_done": False, "scratch": False, "check_break_vol": False,
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

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1
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

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos

            # ── Entry-order fill / TTL bookkeeping ───────────────────────────
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["entry_bar"] = st["bar_count"]
                        st["partial_done"] = False
                        fill = pos.price
                        if st["planned_entry"] > 0.0 and fill != st["planned_entry"]:
                            # Gapped fill: re-anchor the bracket proportionally.
                            ratio = fill / st["planned_entry"]
                            if st["sl"] is not None:
                                ctx.cancel_order(st["sl"])
                            if st["tp"] is not None:
                                ctx.cancel_order(st["tp"])
                            st["planned_stop"] *= ratio
                            st["planned_target"] *= ratio
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=st["planned_stop"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp"] = ctx.place_limit_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"] / 2.0, price=st["planned_target"],
                                parent=st["entry_id"], reduce_only=True)
                        if st["check_break_vol"] and not self._break_vol_ok(vo):
                            # The fill bar closed on weak volume: scratch.
                            ctx.place_market_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=abs(pos.quantity),
                                parent=st["entry_id"], reduce_only=True)
                            st["scratch"] = True
                    else:
                        closed = True   # same-bar round trip
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
                elif st["bar_count"] - st["armed_bar"] > self.order_ttl_bars:
                    ctx.cancel_order(st["pending"])   # children die with it
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["scratch"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            # ── Manage the open trade (pine: only after the entry bar) ───────
            if in_pos:
                if st["scratch"]:
                    continue
                held = abs(pos.quantity)
                bars_held = st["bar_count"] - st["entry_bar"]
                if bars_held < 1:
                    continue
                if not st["partial_done"] and st["tp"] is not None:
                    tp_order = ctx.order(st["tp"])
                    if tp_order is not None and tp_order.status == OrderStatus.Filled:
                        st["partial_done"] = True
                        st["tp"] = None
                        if self.move_be:
                            self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if not st["partial_done"] and bars_held >= self.partial_bars:
                    # Time-based partial: sell half at the next open.
                    if st["tp"] is not None:
                        ctx.cancel_order(st["tp"])
                        st["tp"] = None
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    if self.move_be:
                        self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if st["partial_done"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["scratch"] = True   # exit pending; stop managing
                continue
            if st["cooldown"] > 0:
                continue

            # ── Setup 1: (re)arm the resting buy-stop while the flag holds ───
            arm = self._breakout_arm(op, hi, lo, cl, vo)
            if arm is not None:
                entry_px, stop_px, target_px = arm
                if st["pending"] is not None and st["planned_entry"] != entry_px:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                if st["pending"] is None:
                    self._place_bracket(ctx, symbol, st, entry_px, stop_px, target_px,
                                        market=False, check_break_vol=True)
                st["armed_bar"] = st["bar_count"]   # TTL extends while setup holds
                continue

            # ── Setup 2: episodic pivot fires at the close, market entry ─────
            if st["pending"] is None:
                ep = self._episodic_pivot(op, hi, lo, cl, vo)
                if ep is not None:
                    entry_px, stop_px, target_px = ep
                    self._place_bracket(ctx, symbol, st, entry_px, stop_px, target_px,
                                        market=True, check_break_vol=False)
                    st["armed_bar"] = st["bar_count"]

    # ── Order placement ───────────────────────────────────────────────────────
    def _place_bracket(self, ctx, symbol, st, entry_px, stop_px, target_px,
                       market, check_break_vol):
        risk = entry_px - stop_px
        if risk <= 0.0 or not np.isfinite(risk):
            return
        qty = ctx.equity() * self.risk_fraction / risk
        if qty <= 0.0 or not np.isfinite(qty):
            return
        if market:
            entry_id = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                              quantity=qty)
        else:
            entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                            quantity=qty, price=entry_px)
        # Protective legs: dormant children of the entry, reduce-only. The SL
        # keeps the full entry quantity for the life of the trade (fills clamp
        # to what is held); the TP takes the pine's half partial.
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=qty, price=stop_px,
                                        parent=entry_id, reduce_only=True)
        st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                         quantity=qty / 2.0, price=target_px,
                                         parent=entry_id, reduce_only=True)
        st["pending"] = entry_id
        st["entry_qty"] = qty
        st["planned_entry"] = entry_px
        st["planned_stop"] = stop_px
        st["planned_target"] = target_px
        st["check_break_vol"] = check_break_vol

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price)   # pine: stopPx := max(stopPx, entryPx)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    # ── Setup detection ───────────────────────────────────────────────────────
    def _universe_up(self, hi, lo, cl, vo):
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        if not self._liq_ok(cl, vo) or adr < self.min_adr or g < self.min_gain:
            return None
        if self.require_mas:
            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is None or s20 is None or not (cl[-1] > s20 and s10 > s20):
                return None
        return adr

    def _breakout_arm(self, op, hi, lo, cl, vo):
        """(entry, stop, target) for the resting buy-stop while the tight flag
        qualifies — armed BEFORE any break, like the pine's boSetup."""
        if len(cl) < self.base_max_len + 1:
            return None
        adr = self._universe_up(hi, lo, cl, vo)
        if adr is None:
            return None
        window = hi[-self.base_max_len:]
        mx = window[0]
        last_pos = 0
        for i in range(self.base_max_len):
            if window[i] >= mx:   # keep the LAST extreme, like ta.highestbars
                mx = window[i]
                last_pos = i
        since_pk = (self.base_max_len - 1) - last_pos
        pull_n = max(since_pk, 1)
        pull_low = lowest(lo, pull_n)
        if mx <= 0.0 or pull_low is None:
            return None
        retrace = 100.0 * (mx - pull_low) / mx
        if since_pk < self.min_base_days or retrace > self.max_depth:
            return None
        if self.use_vol_dry and not self._vol_dry_ok(vo):
            return None
        entry = float(mx) * (1.0 + self.entry_buffer_bps / 10_000.0)
        stop, target = self._long_levels(entry, adr, lo[-1])
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

    def _long_levels(self, entry, adr, bar_low):
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(bar_low)) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return float(stop), float(entry + self.partial_rr * (entry - stop))

    # ── Gate predicates ───────────────────────────────────────────────────────
    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def _vol_dry_ok(self, vo):
        v5 = sma(vo, 5)
        v50 = sma(vo, 50)
        return v5 is not None and v50 is not None and v5 < self.vol_dry_ratio * v50

    def _break_vol_ok(self, vo):
        if not self.use_bo_vol:
            return True
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
