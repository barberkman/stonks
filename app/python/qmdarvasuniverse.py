"""Darvas box trading QM's universe — the filters travel, the exits stay home.

The mirror-image composition of qmdarvasbase.py: there, Darvas supplies the
base and QM manages the trade; HERE, QM only decides which symbols/moments
are worth trading at all, and the trade itself is pure Nicolas Darvas:

  gates   at arm time, QM's full universe must hold: price floor, ADR floor,
          momentum gain over the lookback, close above the 20-SMA with the
          10-SMA above it — a box completing in a name that fails any gate
          is ignored
  trade   classic Darvas mechanics, unchanged from darvasclassic.py: resting
          buy-stop above the completed box top (alive while the box is),
          stop-loss at the box bottom, each later box with a higher bottom
          ratchets the stop up, and there is NO take-profit — the staircase
          decides when it is over

No partials, no breakeven shuffle, no MA trail, no volume scratch: exit
style is exactly what separates this file from qmdarvasbase.py.
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


class QMDarvasUniverseStrategy(stonks.Strategy):
    # QM universe & trend gates
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # Darvas trade
    length = 4
    entry_buffer_bps = 5.0
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "min_gain": stonks.Param("QM momentum gate over the lookback", unit="%"),
        "min_adr": stonks.Param("QM average-range floor", unit="%"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
        "sma20": stonks.Indicator("20-bar SMA (QM trend gate)"),
    }

    def lookback(self):
        return max(self.length, self.mom_len + 1, self.adr_len, 21) + 5

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
            roll_low = lowest(lo, self.length)
            active, new_box, broke_up, broke_dn = step_box(st, hi[-1], lo[-1], roll_low)

            s20 = sma(cl, 20)
            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
                ctx.plot("box_bottom", symbol, st["bx_low"])
            if s20 is not None:
                ctx.plot("sma20", symbol, s20)

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
                # Darvas' own exit engine: the box-bottom staircase.
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
            if st["pending"] is None and new_box and self._universe_up(hi, lo, cl, vo):
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

    def _universe_up(self, hi, lo, cl, vo):
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return False
        av20 = sma(vo, 20)
        if cl[-1] < self.min_price or av20 is None or av20 < self.min_avg_vol:
            return False
        if adr < self.min_adr or g < self.min_gain:
            return False
        s10 = sma(cl, 10)
        s20 = sma(cl, 20)
        return (s10 is not None and s20 is not None
                and cl[-1] > s20 and s10 > s20)
