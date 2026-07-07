"""Darvas box gated by the tide — "I only buy in a rising market."

Darvas' techno-fundamentalist method had a market filter the box machine
alone doesn't capture: he wanted the stock itself in a persistent advance
before a box mattered. The pine has no such gate. This reading adds it,
re-expressed per symbol with QM-style momentum measures:

  gate    at arm time the close must sit above its 50-bar SMA and the gain
          over the momentum lookback must be at least min_gain% — a box that
          completes in a flat or falling tape is ignored entirely
  trade   otherwise exactly classic Darvas: resting buy-stop above the box
          top, stop-loss at the box bottom, later boxes ratchet the stop up,
          no take-profit

One filter, nothing else changed — the file measures what the tide alone is
worth on top of darvasclassic.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


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


class DarvasTrendStrategy(stonks.Strategy):
    length = 4
    trend_ma_len = 50
    mom_len = 24
    min_gain = 0.5
    entry_buffer_bps = 5.0
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "trend_ma_len": stonks.Param("SMA the close must hold above at arm time", unit="bars"),
        "min_gain": stonks.Param("minimum gain over the momentum lookback", unit="%"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
        "trend_ma": stonks.Indicator("the rising-tide SMA gate"),
    }

    def lookback(self):
        return max(self.length, self.trend_ma_len, self.mom_len + 1) + 5

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

            trend_ma = sma(cl, self.trend_ma_len)
            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
                ctx.plot("box_bottom", symbol, st["bx_low"])
            if trend_ma is not None:
                ctx.plot("trend_ma", symbol, trend_ma)

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
                # The tide gate: no rising market, no box trade.
                g = gain_pct(cl, self.mom_len)
                if (trend_ma is None or g is None
                        or cl[-1] <= trend_ma or g < self.min_gain):
                    continue
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
