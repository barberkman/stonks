"""Darvas box — the literal port of app/pines/darvas_box.pine.

The pine's 5-state machine, index for index (including its documented
modification: the bottom confirms after ONE higher low):

  state 1  fresh reference: box top candidate = this bar's high
  state 2  one lower high (top frozen)
  state 3  two lower highs
  state 4  three+ lower highs but the low undercut the rolling lookback low
  state 5  three lower highs AND a higher low -> the box is active, top and
           bottom frozen; it stays active while price holds inside and ends
           the bar the range breaks either way

`low_value` mirrors the pine exactly: the rolling `ta.lowest(length)` low
until the state-5 transition freezes the PREVIOUS bar's value as the bottom.

Trading (classic Darvas, long-only): the tick a box completes, park a
buy-stop just above the frozen top with a stop-loss child at the box bottom.
No take-profit — the runner is managed Darvas-style: each LATER box that
completes with a higher bottom ratchets the stop-loss up underneath it
(cancel + re-place, never loosened). The resting entry lives exactly as long
as its box: if the box ends without a fill, the order is cancelled.

The stop stays anchored to box geometry, so a gapped fill does not re-anchor
anything (unlike the R-calibrated QM ports).
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
    """Advance the pine state machine by one bar. Returns
    (active, new_box, broke_up, broke_dn). Persisted per symbol — a state-1
    run can outlast any pulled window, so this is never window-recomputed."""
    prev_state = st["bx_state"]
    prev_high = st["bx_high"]
    prev_low = st["bx_low"]
    state, high_v, low_v = 1, float(hi_now), float(roll_low)   # pine defaults
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


class DarvasClassicStrategy(stonks.Strategy):
    length = 4                # pine: box lookback period (rolling low)
    entry_buffer_bps = 5.0    # buy-stop offset above the box top
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "entry_buffer_bps": stonks.Param("buy-stop offset above the box top", unit="bps"),
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
                "pending": None, "entry_id": None, "sl": None,
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

            # Step the machine once per new bar.
            if st["last_ts"] == int(ts[-1]):
                continue
            st["last_ts"] = int(ts[-1])
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
                        closed = True   # same-bar round trip
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
                # Darvas trail: a fresh box with a higher bottom ratchets the
                # stop up underneath it. Never loosened.
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

            # The resting entry lives exactly as long as its box.
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
