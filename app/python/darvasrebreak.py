"""Darvas box with the re-entry ladder — Darvas re-tried a level that
stopped him out; the pine (and the classic port) gives each box exactly one
chance.

The book describes buying back the SAME breakout after a shakeout, at the
same price, because the box logic hadn't changed — only the fill had been
unlucky. This file models that persistence with a hard cap:

  attempt 1  a completed box arms the classic buy-stop at its top, stop-loss
             at its bottom
  attempt 2  a STOP-OUT (a real filled loss, not a TTL expiry) immediately
             re-arms the identical levels once — no cooldown between the
             attempts; the shakeout thesis is time-sensitive
  done       a second stop-out retires that box: nothing arms again until a
             genuinely NEW box completes (which also resets the attempt
             counter)

Because re-arms happen after the original box has typically ended, resting
orders here live on a bar TTL (order_ttl_bars) instead of the classic
box-lifetime rule. There is no bar cooldown at all: the attempt cap and the
wait-for-a-new-box rule ARE the cooldown. No take-profit, no ratchet — the
file isolates the re-entry behavior.
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


class DarvasRebreakStrategy(stonks.Strategy):
    length = 4
    entry_buffer_bps = 5.0
    max_attempts = 2
    order_ttl_bars = 10
    risk_fraction = 0.02

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "max_attempts": stonks.Param("entries allowed per box (1 + retries)", unit="tries"),
        "order_ttl_bars": stonks.Param("resting entry lifetime", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
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
                "entry_qty": 0.0,
                "armed_level": None, "armed_stop": None,   # the box being traded
                "attempts": 0, "filled_once": False, "armed_bar": 0,
                "was_in": False, "bar_count": 0}

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

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["filled_once"] = True
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
                elif st["bar_count"] - st["armed_bar"] > self.order_ttl_bars:
                    ctx.cancel_order(st["pending"])   # TTL expiry: not an attempt
                    st["pending"] = None
            st["was_in"] = in_pos

            if closed:
                # A real filled loss consumes an attempt; the shakeout re-try
                # (if any remains) goes straight back to the SAME levels.
                st["attempts"] += 1
                st["sl"] = st["entry_id"] = None
                if (st["attempts"] < self.max_attempts
                        and st["armed_level"] is not None):
                    self._arm(ctx, symbol, st, st["armed_level"], st["armed_stop"])
                continue

            if in_pos:
                continue                      # one stop, no management
            if new_box:
                # A genuinely new box: fresh levels, fresh attempt budget.
                st["attempts"] = 0
                st["filled_once"] = False
                st["armed_level"] = st["bx_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
                st["armed_stop"] = st["bx_low"]
                if st["pending"] is not None:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                self._arm(ctx, symbol, st, st["armed_level"], st["armed_stop"])

    def _arm(self, ctx, symbol, st, entry_px, stop_px):
        risk = entry_px - stop_px
        if risk <= 0.0 or not np.isfinite(risk):
            return
        qty = ctx.equity() * self.risk_fraction / risk
        if qty <= 0.0 or not np.isfinite(qty):
            return
        entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                        quantity=qty, price=entry_px)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=qty, price=float(stop_px),
                                        parent=entry_id, reduce_only=True)
        st["pending"] = entry_id
        st["entry_qty"] = qty
        st["armed_bar"] = st["bar_count"]
