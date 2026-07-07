"""Qullamaggie breakout with the risk engine re-read through ATR — a
volatility-unit interpretation of the pine's "x ADR" language.

The pine measures stop distance in ADR (average daily range percent) and
trails with a moving average of closes. This reading swaps both for the other
standard volatility yardstick: Wilder's ATR(14).

  entry   the same tight-flag base and resting buy-stop at the pivot as the
          literal port (universe gates included), but with NO volume gates —
          volatility does all the qualifying here
  stop    entry - atr_mult x ATR at placement
  trail   a chandelier: the stop-loss order is cancelled and re-placed at
          highest-high-since-entry - atr_mult x ATR whenever that level rises
          above the current stop. It only ever ratchets up. There is no
          take-profit, no partial, no breakeven step, and no close-based
          check: the resting stop order IS the entire exit engine.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def atr(hi, lo, cl, n):
    """Wilder ATR seeded with the SMA of the first n true ranges of the
    window, then smoothed across the rest. The window is ~3x the period, so
    the seed's influence has decayed to noise by the last bar."""
    if len(cl) < n + 1:
        return None
    trs = np.maximum(hi[1:] - lo[1:],
                     np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    a = float(np.mean(trs[:n]))
    for tr in trs[n:]:
        a = (a * (n - 1) + float(tr)) / n
    return a


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


class QMATRStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    mom_len = 24
    min_gain = 0.5
    # Base
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    # ATR risk engine
    atr_len = 14
    atr_mult = 2.0
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "atr_len": stonks.Param("ATR period", unit="bars"),
        "atr_mult": stonks.Param("stop and chandelier distance, in ATRs", unit="ATR"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "order_ttl_bars": stonks.Param("resting buy-stop lifetime", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "chandelier": stonks.Indicator("highest-high-since-entry - atr_mult x ATR (while holding)"),
    }

    def lookback(self):
        return max(self.base_max_len, self.mom_len, 3 * self.atr_len + 1, 21) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None,
                "planned_entry": 0.0, "stop_px": 0.0, "entry_qty": 0.0,
                "armed_bar": 0, "bar_count": 0, "hh": 0.0,
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

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1
            if len(cl) < self.lookback():
                continue

            a = atr(hi, lo, cl, self.atr_len)
            if a is None:
                continue

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["hh"] = float(hi[-1])
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
                st["sl"] = st["entry_id"] = None
                st["hh"] = 0.0
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                # Chandelier ratchet: cancel + re-place, only ever upward.
                st["hh"] = max(st["hh"], float(hi[-1]))
                candidate = st["hh"] - self.atr_mult * a
                ctx.plot("chandelier", symbol, candidate)
                if candidate > st["stop_px"]:
                    if st["sl"] is not None:
                        ctx.cancel_order(st["sl"])
                    st["stop_px"] = candidate
                    st["sl"] = ctx.place_stop_order(
                        symbol=symbol, side=OrderSide.Sell,
                        quantity=st["entry_qty"], price=candidate,
                        parent=st["entry_id"], reduce_only=True)
                continue
            if st["cooldown"] > 0:
                continue

            arm = self._breakout_arm(hi, lo, cl, vo, a)
            if arm is None:
                continue
            entry_px, stop_px = arm
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
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["planned_entry"] = entry_px
                st["stop_px"] = stop_px
            st["armed_bar"] = st["bar_count"]

    def _breakout_arm(self, hi, lo, cl, vo, a):
        if len(cl) < self.base_max_len + 1:
            return None
        g = gain_pct(cl, self.mom_len)
        if g is None or g < self.min_gain or not self._liq_ok(cl, vo):
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
        stop = entry - self.atr_mult * a
        if stop <= 0.0:
            return None
        return entry, float(stop)

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
