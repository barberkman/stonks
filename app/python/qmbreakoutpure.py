"""Qullamaggie breakout, distilled to the textbook tight flag — Setup 1 only,
with every quality gate the pine leaves optional turned ON and everything
else stripped away.

Interpretation: the pine defaults volume dry-up OFF and tolerates a 40%
retracement because it wants signals; this reading wants only the A+ base.
The base must digest on drying volume (5-bar average below the 50-bar
average), retrace at most 25%, and the break bar must print expanded volume.
There is no episodic pivot, no partial, no breakeven shuffle, no trail: one
resting buy-stop at the pivot, one full-size stop-loss, one full-size
take-profit at 3R. The bracket is the whole trade.

As in the other resting-order ports, the break-bar volume gate becomes a
post-fill check (a real resting stop cannot see volume): a fill whose bar
closed below bo_vol_mult x the 50-bar average volume is scratched at the
next open. A gapped fill re-anchors both legs proportionally.
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


class QMBreakoutPureStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # The A+ base: tighter and quieter than the pine's defaults
    base_max_len = 40
    min_base_days = 3
    max_depth = 25.0          # pine default 40
    vol_dry_ratio = 1.0       # dry-up is MANDATORY here (pine defaults it off)
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    bo_vol_mult = 1.3         # break-bar expansion is mandatory too
    # One bracket, no management
    adr_stop_mult = 1.0
    use_lod_stop = True
    target_rr = 3.0           # single full take-profit
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "max_depth": stonks.Param("max retracement inside the base", unit="%"),
        "vol_dry_ratio": stonks.Param("5-bar avg volume must be under this x 50-bar avg", unit="x"),
        "bo_vol_mult": stonks.Param("break-bar volume vs the 50-bar average", unit="x"),
        "target_rr": stonks.Param("single full take-profit, in R multiples", unit="R"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0,
                "scratch": False, "was_in": False, "cooldown": 0}

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

            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is not None:
                ctx.plot("sma10", symbol, s10)
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
                        fill = pos.price
                        if st["planned_entry"] > 0.0 and fill != st["planned_entry"]:
                            ratio = fill / st["planned_entry"]
                            if st["sl"] is not None:
                                ctx.cancel_order(st["sl"])
                            if st["tp"] is not None:
                                ctx.cancel_order(st["tp"])
                            st["planned_stop"] *= ratio
                            st["planned_target"] *= ratio
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=st["planned_stop"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp"] = ctx.place_limit_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=st["planned_target"],
                                parent=st["entry_id"], reduce_only=True)
                        if not self._break_vol_ok(vo):
                            # Weak-volume break: scratch the fill at the next open.
                            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                                   quantity=abs(pos.quantity),
                                                   parent=st["entry_id"], reduce_only=True)
                            st["scratch"] = True
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
                st["scratch"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                continue        # the resting bracket IS the management
            if st["cooldown"] > 0:
                continue

            arm = self._breakout_arm(hi, lo, cl, vo)
            if arm is None:
                continue
            entry_px, stop_px, target_px = arm
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
                st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                                 quantity=qty, price=target_px,
                                                 parent=entry_id, reduce_only=True)
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["planned_entry"] = entry_px
                st["planned_stop"] = stop_px
                st["planned_target"] = target_px
            st["armed_bar"] = st["bar_count"]

    def _breakout_arm(self, hi, lo, cl, vo):
        if len(cl) < self.base_max_len + 1:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        if not self._liq_ok(cl, vo) or adr < self.min_adr or g < self.min_gain:
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
        if not self._vol_dry_ok(vo):            # mandatory dry-up
            return None
        entry = float(mx) * (1.0 + self.entry_buffer_bps / 10_000.0)
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return entry, float(stop), float(entry + self.target_rr * (entry - stop))

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def _vol_dry_ok(self, vo):
        v5 = sma(vo, 5)
        v50 = sma(vo, 50)
        return v5 is not None and v50 is not None and v5 < self.vol_dry_ratio * v50

    def _break_vol_ok(self, vo):
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
