"""QM entry, Darvas exit — sell into strength once, then let the boxes walk
the stop up.

The pine's post-partial trail is a moving average of closes; Darvas' trail
is structure. This hybrid takes QM's whole entry stack and first partial,
then hands the runner to the box staircase instead of the MA:

  entry       QM tight-flag scan (pivot of the last 40 bars, base age,
              retracement depth, full universe gates), resting buy-stop at
              the pivot with a TTL, weak-volume fills scratched, gapped
              fills re-anchored — identical to the literal port
  partial     half at 2R (or after partial_bars bars), stop to breakeven
  the runner  NO moving-average trail at all: whenever a Darvas box
              completes with a bottom above the current stop, the stop-loss
              is cancelled and re-placed under that box (works before or
              after the partial — structure outranks the calendar). The
              trade ends only by stop — initial, breakeven, or box-ratchet.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


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


class QMDarvasExitStrategy(stonks.Strategy):
    # QM universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # QM base & entry
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Stop / partial
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    # The Darvas trail
    length = 4
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback of the ratchet trail", unit="bars"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "adr_stop_mult": stonks.Param("initial stop distance, in ADRs", unit="ADR"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
        "box_bottom": stonks.Indicator("active box bottom (the ratchet trail feed)"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.length) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
                "partial_done": False, "scratch": False,
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
            if len(cl) < self.lookback():
                continue

            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is not None:
                ctx.plot("sma10", symbol, s10)
            if s20 is not None:
                ctx.plot("sma20", symbol, s20)
            if active:
                ctx.plot("box_bottom", symbol, st["bx_low"])

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
                        if self.use_bo_vol and not self._break_vol_ok(vo):
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
                            self._replace_stop(ctx, symbol, st,
                                               max(st["planned_stop"], pos.price))
                if not st["partial_done"] and bars_held >= self.partial_bars:
                    if st["tp"] is not None:
                        ctx.cancel_order(st["tp"])
                        st["tp"] = None
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    if self.move_be:
                        self._replace_stop(ctx, symbol, st,
                                           max(st["planned_stop"], pos.price))
                # Darvas trail: a completed box with a higher bottom is the
                # only thing that ever tightens the runner's stop.
                if new_box and st["bx_low"] > st["planned_stop"]:
                    self._replace_stop(ctx, symbol, st, float(st["bx_low"]))
                continue
            if st["cooldown"] > 0:
                continue

            arm = self._breakout_arm(hi, lo, cl, vo)
            if arm is None:
                continue
            entry_px, stop_px, target_px = arm
            if st["pending"] is not None and st["planned_entry"] != entry_px:
                ctx.cancel_order(st["pending"])
                st["pending"] = None
            if st["pending"] is None:
                risk = entry_px - stop_px
                if risk <= 0.0 or not np.isfinite(risk):
                    continue
                qty = ctx.equity() * self.risk_fraction / risk
                if qty <= 0.0 or not np.isfinite(qty):
                    continue
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
            st["armed_bar"] = st["bar_count"]

    def _replace_stop(self, ctx, symbol, st, price):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=price,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = price

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
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return entry, float(stop), float(entry + self.partial_rr * (entry - stop))

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def _break_vol_ok(self, vo):
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
