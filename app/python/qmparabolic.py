"""Qullamaggie parabolic short — Setup 3 standalone, short-only.

The pine keeps this on a "separate track" from the swing setups, and off by
default; here it IS the strategy. After a parabolic run-up (at least
ps_min_gain% trough-to-peak inside ps_lookback bars, with at least ps_streak
consecutive up-closes), the FIRST RED BAR — a close below the prior close —
is the trigger: short the close (market order, fills at the next open).

The exit is asymmetric by design, exactly as in the pine:

  stop   a resting buy-stop just above the highest high of the last
         ps_stop_lb bars — parabolic moves die fast or they kill you
  cover  no take-profit order at all (the pine never assigns one on this
         path): the short is covered at the next open on the first up-close
         (the bounce is the profit signal) or after ps_max_hold bars

Only the liquidity floor gates the universe — a parabolic, by definition,
already violated every trend filter. The stop is anchored to structure (the
recent high), so a gapped fill re-anchors nothing.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


class QMParabolicStrategy(stonks.Strategy):
    # Universe (liquidity only — see docstring)
    min_price = 5.0
    min_avg_vol = 0.0
    # Parabolic qualification
    ps_lookback = 10
    ps_min_gain = 8.0
    ps_streak = 3
    ps_stop_lb = 3
    ps_max_hold = 5
    entry_buffer_bps = 5.0
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "ps_min_gain": stonks.Param("minimum run-up over the lookback", unit="%"),
        "ps_streak": stonks.Param("consecutive up-closes before the red bar", unit="bars"),
        "ps_stop_lb": stonks.Param("stop above the highest high of N bars", unit="bars"),
        "ps_max_hold": stonks.Param("max holding period", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "ps_stop": stonks.Indicator("highest high of the stop lookback (the cover stop level)"),
    }

    def lookback(self):
        return max(self.ps_lookback, self.ps_streak + 2, self.ps_stop_lb, 21) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None,
                "entry_qty": 0.0, "bar_count": 0, "entry_bar": 0,
                "exiting": False, "was_in": False, "cooldown": 0}

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

            ps_stop = highest(hi, self.ps_stop_lb)
            if ps_stop is not None:
                ctx.plot("ps_stop", symbol, ps_stop)

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos
            if st["pending"] is not None:
                entry = ctx.order(st["pending"])
                status = entry.status if entry is not None else OrderStatus.Cancelled
                if status == OrderStatus.Filled:
                    if in_pos:
                        st["entry_id"] = st["pending"]
                        st["entry_bar"] = st["bar_count"]
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["exiting"] = False
                st["sl"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if st["exiting"]:
                    continue
                bars_held = st["bar_count"] - st["entry_bar"]
                if bars_held < 1:
                    continue                      # pine: only after the entry bar
                bounced = cl[-1] > cl[-2]
                if bounced or bars_held >= self.ps_max_hold:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                           quantity=abs(pos.quantity),
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0 or st["pending"] is not None:
                continue

            sig = self._first_red_bar(hi, lo, cl, vo, ps_stop)
            if sig is None:
                continue
            entry_px, stop_px = sig
            risk = stop_px - entry_px
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_id = ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                              quantity=qty)
            st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                            quantity=qty, price=stop_px,
                                            parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["entry_qty"] = qty

    def _first_red_bar(self, hi, lo, cl, vo, ps_stop):
        need = max(self.ps_lookback, self.ps_streak + 1, self.ps_stop_lb) + 1
        if len(cl) < need or ps_stop is None:
            return None
        if not self._liq_ok(cl, vo):
            return None
        hh = highest(hi, self.ps_lookback)
        ll = lowest(lo, self.ps_lookback)
        if hh is None or ll is None or ll <= 0.0:
            return None
        if 100.0 * (hh / ll - 1.0) < self.ps_min_gain:
            return None
        up_streak = 0
        i = len(cl) - 2
        while i >= 1 and cl[i] > cl[i - 1]:
            up_streak += 1
            i -= 1
        if not (cl[-1] < cl[-2] and up_streak >= self.ps_streak):
            return None
        entry = float(cl[-1])
        stop = ps_stop * (1.0 + self.entry_buffer_bps / 10_000.0)
        return entry, float(stop)

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
