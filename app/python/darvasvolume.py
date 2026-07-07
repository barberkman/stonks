"""Darvas box the way the BOOK describes it — no volume, no trade.

"How I Made $2,000,000 in the Stock Market" is explicit: Darvas only bought
a box breakout when it came with a marked increase in volume; a quiet drift
through the top was a trap to him. The pine ignores volume entirely. This
interpretation restores the book's filter — and because a resting stop order
cannot see the break bar's volume (stop triggers are price-only, and the
volume isn't known until that bar closes), the entry becomes close-confirmed:

  signal   the box ends upward AND the break bar CLOSES above the top AND
           prints at least break_vol_mult x the 50-bar average volume
  entry    market order at the next open (no resting orders)
  no-go    a break on quiet volume simply discards that box — no scratch
           logic needed, the trade is never taken

Management is classic Darvas: stop-loss at the broken box's bottom, each
later completed box with a higher bottom ratchets it up, no take-profit.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


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


class DarvasVolumeStrategy(stonks.Strategy):
    length = 4
    break_vol_mult = 1.5      # the book's "marked increase" in volume
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "break_vol_mult": stonks.Param("break-bar volume vs the 50-bar average", unit="x"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
    }

    def lookback(self):
        return max(self.length, 51) + 5

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
            if st["cooldown"] > 0 or st["pending"] is not None:
                continue

            # The book's entry: an upward box end, closed above the top, on
            # a marked volume expansion — or no trade at all.
            if not broke_up or box_top is None or box_bot is None:
                continue
            if cl[-1] <= box_top:
                continue
            prev_avg50 = sma(vo[:-1], 50)
            if prev_avg50 is None or vo[-1] < self.break_vol_mult * prev_avg50:
                continue
            entry = float(cl[-1])
            stop_px = float(box_bot)
            risk = entry - stop_px
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_id = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                              quantity=qty)
            st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                            quantity=qty, price=stop_px,
                                            parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["entry_qty"] = qty
            st["stop_px"] = stop_px
