"""Darvas box, the UNMODIFIED reading — the pine's own header admits its
"logic changed to transition to state 5 after one higher low only"; this file
undoes that modification and demands symmetry: three lower highs to confirm
the top AND three consecutive higher lows to confirm the bottom.

The top half of the machine is identical to the classic port. The bottom
half replaces the single-shot higher-low check with a streak counter: while
the frozen top keeps holding (every bar's high below it), each bar whose low
beats the previous bar's rolling low extends the streak; a bar that undercuts
it resets the streak to zero WITHOUT invalidating the top (the machine keeps
waiting in state 4). Only a three-bar streak completes the box, whose bottom
freezes at the rolling low the streak was measured against. Fewer, cleaner
boxes — at the price of missing the fast ones.

Trading is deliberately identical to darvasclassic (buy-stop at the top,
stop-loss at the bottom, later boxes ratchet the stop, never a take-profit),
so any performance difference between the two files is attributable to the
state machine alone.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def step_box_strict(st, hi_now, lo_now, roll_low):
    """The strict machine: states 1-3 as in the pine; the 3->5 transition
    needs THREE consecutive higher lows (streak in st["low_streak"])."""
    prev_state = st["bx_state"]
    prev_high = st["bx_high"]
    prev_low = st["bx_low"]
    streak = st["low_streak"]
    state, high_v, low_v, new_streak = 1, float(hi_now), float(roll_low), 0
    if prev_high is not None:
        if prev_state == 1 and hi_now < prev_high:
            state, high_v = 2, prev_high
        elif prev_state == 2 and hi_now < prev_high:
            state, high_v = 3, prev_high
        elif prev_state in (3, 4) and hi_now < prev_high:
            if lo_now > prev_low:
                if streak + 1 >= 3:
                    state, high_v, low_v = 5, prev_high, prev_low
                else:
                    state, high_v, new_streak = 4, prev_high, streak + 1
            else:
                state, high_v = 4, prev_high      # streak resets, top still holds
        elif prev_state == 5 and hi_now <= prev_high and lo_now >= prev_low:
            state, high_v, low_v = 5, prev_high, prev_low
    broke_up = prev_state == 5 and state != 5 and hi_now > prev_high
    broke_dn = prev_state == 5 and state != 5 and not broke_up and lo_now < prev_low
    new_box = state == 5 and prev_state != 5
    st["bx_state"] = state
    st["bx_high"] = high_v
    st["bx_low"] = low_v
    st["low_streak"] = new_streak
    return state == 5, new_box, broke_up, broke_dn


class DarvasStrictStrategy(stonks.Strategy):
    length = 4
    entry_buffer_bps = 5.0
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "entry_buffer_bps": stonks.Param("buy-stop offset above the box top", unit="bps"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active strict-Darvas box top"),
        "box_bottom": stonks.Indicator("active strict-Darvas box bottom"),
    }

    def lookback(self):
        return self.length + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "low_streak": 0,
                "last_ts": None, "pending": None, "entry_id": None, "sl": None,
                "entry_qty": 0.0, "stop_px": 0.0,
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
            active, new_box, broke_up, broke_dn = step_box_strict(st, hi[-1], lo[-1], roll_low)

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
                st["sl"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if new_box and st["bx_low"] > st["stop_px"]:
                    if st["sl"] is not None:
                        ctx.cancel_order(st["sl"])
                    st["stop_px"] = st["bx_low"]
                    st["sl"] = ctx.place_stop_order(
                        symbol=symbol, side=OrderSide.Sell,
                        quantity=st["entry_qty"], price=st["stop_px"],
                        parent=st["entry_id"], reduce_only=True)
                continue
            if st["cooldown"] > 0:
                continue

            if st["pending"] is not None and not active:
                ctx.cancel_order(st["pending"])
                st["pending"] = None
            if st["pending"] is None and new_box:
                entry_px = st["bx_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
                stop_px = st["bx_low"]
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
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["stop_px"] = stop_px
