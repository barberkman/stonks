"""Darvas box traded in BOTH directions — the box end IS the signal, exactly
as the pine's arrows paint it.

The pine draws an up arrow or a down arrow the bar the box ends, colored by
which edge gave way. This interpretation trades that arrow directly: no
resting orders at all. By the time a closed bar shows the box broke (the
strategy decides on closed bars, after the broker has settled), the break
already happened — a resting stop at the edge would be a fiction. So the
entry is an honest MARKET order at the next open, in the break's direction:

  break up    buy;  stop at the box BOTTOM, target = entry + 1 box height
  break down  sell; stop at the box TOP,    target = entry - 1 box height

One symmetric R-style bracket sized off the box's own geometry, full size on
both legs, no partials, no trailing — the simplest possible bidirectional
baseline against which the fancier variants can be judged.
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


class DarvasBothStrategy(stonks.Strategy):
    length = 4
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
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
            cl = sub["close"].to_numpy()
            ts = sub["timestamp"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1
            if len(hi) < self.length:
                continue
            if st["last_ts"] == int(ts[-1]):
                continue
            st["last_ts"] = int(ts[-1])

            # The just-ended box's edges are needed AFTER the step overwrites
            # them, so snapshot first.
            box_top = st["bx_high"]
            box_bot = st["bx_low"]
            roll_low = lowest(lo, self.length)
            active, new_box, broke_up, broke_dn = step_box(st, hi[-1], lo[-1], roll_low)

            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
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

            if in_pos or st["cooldown"] > 0 or st["pending"] is not None:
                continue
            if not (broke_up or broke_dn) or box_top is None or box_bot is None:
                continue

            height = box_top - box_bot
            entry = float(cl[-1])                     # planned; fills at the next open
            if broke_up:
                side, exit_side = OrderSide.Buy, OrderSide.Sell
                stop_px = float(box_bot)
                target_px = entry + height
                risk = entry - stop_px
            else:
                side, exit_side = OrderSide.Sell, OrderSide.Buy
                stop_px = float(box_top)
                target_px = entry - height
                risk = stop_px - entry
            if risk <= 0.0 or height <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_id = ctx.place_market_order(symbol=symbol, side=side, quantity=qty)
            ctx.place_stop_order(symbol=symbol, side=exit_side, quantity=qty,
                                 price=stop_px, parent=entry_id, reduce_only=True)
            ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=qty,
                                  price=target_px, parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
