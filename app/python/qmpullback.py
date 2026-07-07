"""Qullamaggie base traded by anticipation — buy the pullback reset INSIDE
the flag instead of the breakout above it.

The pine only ever buys strength (the pivot break). Qullamaggie's own
commentary describes the alternative lower-risk entry: when a qualified base
pulls into the rising 10-SMA and reclaims it, you can start the position
early with a much tighter stop and let the pivot itself become the first
sell point. This file is that reading:

  setup   the same qualified tight flag as the literal port (pivot high of
          the last 40 bars, >=3 bars of digestion, retracement <= max_depth,
          full universe gates)
  dip     while flat, closes under the 10-SMA mark a dip; its running lowest
          low is remembered
  entry   the first close BACK ABOVE the 10-SMA while the base still
          qualifies -> market buy at the next open; stop at the dip low
  sells   half at the PIVOT (a limit at the structure the breakout crowd is
          watching), stop to breakeven after it fills, remainder trailed on
          EMA20 closes

Levels here are structural (dip low, pivot), so a gapped fill does not
re-scale them — unlike the R-calibrated ports.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.sum(a[-n:]) / n)


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


class QMPullbackStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # Base qualification
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    # Reset entry
    reset_ma_len = 10         # the MA whose reclaim triggers the buy
    # Management
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "reset_ma_len": stonks.Param("SMA whose reclaim triggers the entry", unit="bars"),
        "max_depth": stonks.Param("max retracement inside the base", unit="%"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "reset_ma": stonks.Indicator("10-bar SMA — dips below it arm the reset buy"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.base_max_len, self.mom_len, self.adr_len, 21,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_stop": 0.0, "entry_qty": 0.0, "dip_low": None,
                "bar_count": 0, "entry_bar": 0,
                "partial_done": False, "exiting": False,
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

            reset_ma = sma(cl, self.reset_ma_len)
            trail = ema(cl, self.trail_len)
            if reset_ma is not None:
                ctx.plot("reset_ma", symbol, reset_ma)
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
                        st["entry_bar"] = st["bar_count"]
                        st["partial_done"] = False
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["exiting"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if st["exiting"]:
                    continue
                held = abs(pos.quantity)
                bars_held = st["bar_count"] - st["entry_bar"]
                if bars_held < 1:
                    continue
                if not st["partial_done"] and st["tp"] is not None:
                    tp_order = ctx.order(st["tp"])
                    if tp_order is not None and tp_order.status == OrderStatus.Filled:
                        st["partial_done"] = True
                        st["tp"] = None
                        if self.move_be:
                            if st["sl"] is not None:
                                ctx.cancel_order(st["sl"])
                            be = max(st["planned_stop"], pos.price)
                            st["planned_stop"] = be
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=be,
                                parent=st["entry_id"], reduce_only=True)
                if st["partial_done"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0 or st["pending"] is not None:
                st["dip_low"] = None
                continue
            if reset_ma is None:
                continue

            # ── Dip bookkeeping and the reclaim trigger ───────────────────────
            if cl[-1] < reset_ma:
                low_now = float(lo[-1])
                st["dip_low"] = low_now if st["dip_low"] is None else min(st["dip_low"], low_now)
                continue
            dip_low = st["dip_low"]
            st["dip_low"] = None                # any close back above resets the dip
            if dip_low is None:
                continue                        # no dip preceded this close

            pivot = self._qualified_pivot(hi, lo, cl, vo)
            if pivot is None:
                continue
            entry = float(cl[-1])
            if not (dip_low < entry < pivot):
                continue
            risk = entry - dip_low
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            entry_id = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                              quantity=qty)
            st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                            quantity=qty, price=dip_low,
                                            parent=entry_id, reduce_only=True)
            st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                             quantity=qty / 2.0, price=pivot,
                                             parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["entry_qty"] = qty
            st["planned_stop"] = dip_low

    def _qualified_pivot(self, hi, lo, cl, vo):
        """The base pivot if the tight flag qualifies right now, else None."""
        if len(cl) < self.base_max_len + 1:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        if not self._liq_ok(cl, vo) or adr < self.min_adr or g < self.min_gain:
            return None
        s20 = sma(cl, 20)
        if s20 is None or cl[-1] <= s20:        # reclaim bar sits above the slow MA
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
        return float(mx)

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
