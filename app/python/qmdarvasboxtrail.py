"""QM entries, a two-tripwire runner: the LAST box bottom or the slow EMA,
whichever gives out first.

The pine trails its runner on one moving average. Moving averages are slow
to admit a trend died; box bottoms are fast but noisy. This interpretation
wires BOTH tripwires under the post-partial runner and exits on the first
to break:

  entries    the QM pair from the literal port — tight-flag resting buy-stop
             (TTL, weak-volume scratch, gapped-fill re-anchor) and the
             episodic-pivot market entry
  partial    half at 2R or after partial_bars bars, stop to breakeven
  the runner exits at the next open when a close prints BELOW the most
             recently completed Darvas box's bottom (structure failed) OR
             below the EMA50 (the slow trend failed) — whichever first.
             The box machine runs continuously so "the last box bottom"
             is always current; before any box has completed only the EMA
             leg is armed.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


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


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def step_box(st, hi_now, lo_now, roll_low):
    prev_state = st["bx_state"]
    prev_high = st["bx_high"]
    prev_low = st["bx_low"]
    state, high_v, low_v = 1, float(hi_now), float(roll_low)
    if prev_high is not None:
        if prev_state == 1 and hi_now < prev_high:
            state, high_v = 2, prev_high
        elif prev_state == 2 and hi_now < prev_high:
            state, high_v = 3, prev_high
        elif prev_state in (3, 4) and hi_now < prev_high:
            if lo_now > prev_low:
                state, high_v, low_v = 5, prev_high, prev_low
            else:
                state, high_v = 4, prev_high
        elif prev_state == 5 and hi_now <= prev_high and lo_now >= prev_low:
            state, high_v, low_v = 5, prev_high, prev_low
    broke_up = prev_state == 5 and state != 5 and hi_now > prev_high
    broke_dn = prev_state == 5 and state != 5 and not broke_up and lo_now < prev_low
    new_box = state == 5 and prev_state != 5
    st["bx_state"] = state
    st["bx_high"] = high_v
    st["bx_low"] = low_v
    return state == 5, new_box, broke_up, broke_dn


class QMDarvasBoxTrailStrategy(stonks.Strategy):
    # QM universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # QM base & entries
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # Stop / partial
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    # The two tripwires
    length = 4                # box machine feeding "the last box bottom"
    slow_ema_len = 50
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "slow_ema_len": stonks.Param("slow EMA tripwire under the runner", unit="bars"),
        "length": stonks.Param("box lookback feeding the structure tripwire", unit="bars"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "slow_ema": stonks.Indicator("EMA50 tripwire"),
        "last_box_bottom": stonks.Indicator("bottom of the most recently completed box"),
    }

    def lookback(self):
        # slow_ema_len + 25 (not 3x): the EMA50 is a coarse tripwire, and an
        # SMA-seeded EMA with 25 extra iterations is converged enough for it.
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.slow_ema_len + 25) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "last_box_bottom": None,
                "pending": None, "entry_id": None, "sl": None, "tp": None,
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
            ts = sub["timestamp"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1
            if len(hi) < self.length:
                continue
            if st["last_ts"] == int(ts[-1]):
                continue
            st["last_ts"] = int(ts[-1])
            roll_low = lowest(lo, self.length)
            active, new_box, broke_up, broke_dn = step_box(st, hi[-1], lo[-1], roll_low)
            if new_box:
                st["last_box_bottom"] = float(st["bx_low"])
            if len(cl) < self.lookback():
                continue

            slow = ema(cl, self.slow_ema_len)
            if slow is not None:
                ctx.plot("slow_ema", symbol, slow)
            if st["last_box_bottom"] is not None:
                ctx.plot("last_box_bottom", symbol, st["last_box_bottom"])

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
                        fill = pos.price
                        if st["planned_entry"] > 0.0 and fill != st["planned_entry"]:
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
                            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                                   quantity=abs(pos.quantity),
                                                   parent=st["entry_id"], reduce_only=True)
                            st["scratch"] = True
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
                elif st["bar_count"] - st["armed_bar"] > self.order_ttl_bars:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["scratch"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

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
                    if st["tp"] is not None:
                        ctx.cancel_order(st["tp"])
                        st["tp"] = None
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    if self.move_be:
                        self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if st["partial_done"]:
                    # Two tripwires; first to break flattens the runner.
                    structure_broke = (st["last_box_bottom"] is not None
                                       and cl[-1] < st["last_box_bottom"])
                    trend_broke = slow is not None and cl[-1] < slow
                    if structure_broke or trend_broke:
                        ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                               quantity=held,
                                               parent=st["entry_id"], reduce_only=True)
                        st["scratch"] = True
                continue
            if st["cooldown"] > 0:
                continue

            arm = self._breakout_arm(hi, lo, cl, vo)
            if arm is not None:
                entry_px, stop_px, target_px = arm
                if st["pending"] is not None and st["planned_entry"] != entry_px:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                if st["pending"] is None:
                    self._place_bracket(ctx, symbol, st, entry_px, stop_px, target_px,
                                        market=False, check_break_vol=True)
                st["armed_bar"] = st["bar_count"]
                continue
            if st["pending"] is None:
                ep = self._episodic_pivot(op, hi, lo, cl, vo)
                if ep is not None:
                    entry_px, stop_px, target_px = ep
                    self._place_bracket(ctx, symbol, st, entry_px, stop_px, target_px,
                                        market=True, check_break_vol=False)
                    st["armed_bar"] = st["bar_count"]

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
        be = max(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _breakout_arm(self, hi, lo, cl, vo):
        if len(cl) < self.base_max_len + 1:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        if not self._liq_ok(cl, vo) or adr < self.min_adr or g < self.min_gain:
            return None
        s10 = sma(cl, 10)
        s20 = sma(cl, 20)
        if s10 is None or s20 is None or not (cl[-1] > s20 and s10 > s20):
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
