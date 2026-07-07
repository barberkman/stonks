"""Darvas box with the box's OWN geometry as the whole risk plan — a
measured-move interpretation.

Classic Darvas stops under the box bottom and never takes profit. This
reading treats the box as a measuring instrument instead: a consolidation's
height is the market's own estimate of the tradeable swing, so both the risk
and the reward are read off the box itself.

  entry   resting buy-stop just above the completed box top (as classic)
  stop    the box MIDPOINT — half a box of risk. A breakout that falls back
          into the lower half of its own box has failed the measured-move
          premise; there is no reason to wait for the full bottom
  target  the measured move: top + one box height, full size

One bracket, no partials, no ratchet, no trail. The resting entry lives as
long as its box; levels are structural, so nothing re-anchors on a gap.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


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


class DarvasBoxRiskStrategy(stonks.Strategy):
    length = 4
    entry_buffer_bps = 5.0
    move_mult = 1.0           # target = top + move_mult x box height
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "move_mult": stonks.Param("measured-move target, in box heights", unit="x"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
        "box_mid": stonks.Indicator("box midpoint — the stop of the measured-move bracket"),
    }

    def lookback(self):
        return self.length + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "pending": None, "entry_id": None,
                "was_in": False, "cooldown": 0, "bar_count": 0}

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

            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
                ctx.plot("box_bottom", symbol, st["bx_low"])
                ctx.plot("box_mid", symbol, (st["bx_high"] + st["bx_low"]) / 2.0)

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                continue        # the bracket is the trade
            if st["cooldown"] > 0:
                continue

            if st["pending"] is not None and not active:
                ctx.cancel_order(st["pending"])
                st["pending"] = None
            if st["pending"] is None and new_box:
                top = st["bx_high"]
                bottom = st["bx_low"]
                height = top - bottom
                if height <= 0.0:
                    continue
                entry_px = top * (1.0 + self.entry_buffer_bps / 10_000.0)
                stop_px = (top + bottom) / 2.0
                target_px = top + self.move_mult * height
                risk = entry_px - stop_px
                if risk <= 0.0 or not np.isfinite(risk):
                    continue
                qty = ctx.equity() * self.risk_fraction / risk
                if qty <= 0.0 or not np.isfinite(qty):
                    continue
                entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                quantity=qty, price=entry_px)
                ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell, quantity=qty,
                                     price=stop_px, parent=entry_id, reduce_only=True)
                ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell, quantity=qty,
                                      price=target_px, parent=entry_id, reduce_only=True)
                st["pending"] = entry_id
