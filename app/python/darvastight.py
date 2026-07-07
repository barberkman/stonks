"""Darvas box with a TIGHTNESS gate — only compressed boxes are worth
breaking, in the spirit of Qullamaggie's "the tighter the flag, the better
the break" applied to Darvas geometry.

The pine draws every box the machine finds; most are sloppy. This reading
takes only the coiled ones: a completed box qualifies only if its height is
at most tight_mult x the average bar range (ADR% of price). A wide box is a
trading range; a tight box is a spring.

  entry   resting buy-stop just above a QUALIFIED box top
  stop    the box bottom (tight box -> tight stop, that is the point)
  trail   nothing until the measured move pays for the box: only after some
          close reaches entry + one box height does the EMA20 close-trail
          arm; from then on a close below the EMA20 flattens the runner at
          the next open. No partial, no take-profit, no ratchet.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


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


class DarvasTightStrategy(stonks.Strategy):
    length = 4
    adr_len = 20
    tight_mult = 2.0          # box height <= tight_mult x ADR% x price
    entry_buffer_bps = 5.0
    trail_len = 20
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "tight_mult": stonks.Param("max box height, in average bar ranges", unit="ADR"),
        "trail_len": stonks.Param("EMA trail armed after +1 box height", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active qualified box top"),
        "box_bottom": stonks.Indicator("active qualified box bottom"),
        "trail": stonks.Indicator("EMA trail (live only after the measured move pays)"),
    }

    def lookback(self):
        return max(self.length, self.adr_len, 3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "pending": None, "entry_id": None, "sl": None,
                "entry_qty": 0.0, "entry_px": 0.0, "height": 0.0,
                "trail_armed": False, "exiting": False,
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

            trail = ema(cl, self.trail_len)
            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
                ctx.plot("box_bottom", symbol, st["bx_low"])
            if trail is not None:
                ctx.plot("trail", symbol, trail)

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["trail_armed"] = False
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["trail_armed"] = False
                st["exiting"] = False
                st["sl"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if st["exiting"]:
                    continue
                # The trail arms only once the measured move has paid.
                if not st["trail_armed"] and cl[-1] >= st["entry_px"] + st["height"]:
                    st["trail_armed"] = True
                if st["trail_armed"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=abs(pos.quantity),
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0:
                continue

            if st["pending"] is not None and not active:
                ctx.cancel_order(st["pending"])
                st["pending"] = None
            if st["pending"] is None and new_box:
                top = st["bx_high"]
                bottom = st["bx_low"]
                height = top - bottom
                adr = adr_pct(hi, lo, self.adr_len)
                if adr is None or height <= 0.0:
                    continue
                if height > self.tight_mult * adr / 100.0 * float(cl[-1]):
                    continue                    # too sloppy to be a spring
                entry_px = top * (1.0 + self.entry_buffer_bps / 10_000.0)
                risk = entry_px - bottom
                if risk <= 0.0 or not np.isfinite(risk):
                    continue
                qty = ctx.equity() * self.risk_fraction / risk
                if qty <= 0.0 or not np.isfinite(qty):
                    continue
                entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                quantity=qty, price=entry_px)
                st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                                quantity=qty, price=float(bottom),
                                                parent=entry_id, reduce_only=True)
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["entry_px"] = entry_px
                st["height"] = height
