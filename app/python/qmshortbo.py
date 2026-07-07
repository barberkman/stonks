"""Qullamaggie short breakout (breakdown) — Setup 1c standalone, short-only.

The exact mirror of the long breakout, which the pine ships disabled by
default: in a qualified DOWNTREND (momentum down at least min_gain%, close
under the 20-SMA with the 10-SMA under the 20-SMA), find a base whose LOW
has held for at least min_base_days bars with the bounce off it retracing at
most max_depth%, and park a resting SELL-stop just under that base low. The
breakdown fills it; a bounce leaves it untouched until the TTL expires.

Management mirrors the long side: stop-loss above (one ADR from entry,
tightened to the signal bar's high if closer), half cover at 2R or after
partial_bars bars, stop dropped to breakeven, then the rest is covered when
a close crosses back ABOVE the EMA20 trail. The pine applies its break-bar
volume gate to shorts as well, so a fill whose bar closed on weak volume is
scratched at the next open, and a gapped fill re-anchors the legs.
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


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


class QMShortBOStrategy(stonks.Strategy):
    # Universe & trend (downtrend mirror)
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Base (mirror of the long flag)
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Stop & management
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "adr_stop_mult": stonks.Param("stop distance above entry, in ADRs", unit="ADR"),
        "partial_rr": stonks.Param("half cover target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "order_ttl_bars": stonks.Param("resting sell-stop lifetime", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
        "trail": stonks.Indicator("trailing EMA the post-partial cover checks"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
                "partial_done": False, "scratch": False,
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

            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            trail = ema(cl, self.trail_len)
            if s10 is not None:
                ctx.plot("sma10", symbol, s10)
            if s20 is not None:
                ctx.plot("sma20", symbol, s20)
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
                                symbol=symbol, side=OrderSide.Buy,
                                quantity=st["entry_qty"], price=st["planned_stop"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp"] = ctx.place_limit_order(
                                symbol=symbol, side=OrderSide.Buy,
                                quantity=st["entry_qty"] / 2.0, price=st["planned_target"],
                                parent=st["entry_id"], reduce_only=True)
                        if not self._break_vol_ok(vo):
                            ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
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
                st["partial_done"] = False
                st["scratch"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if st["scratch"]:
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
                            self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if not st["partial_done"] and bars_held >= self.partial_bars:
                    if st["tp"] is not None:
                        ctx.cancel_order(st["tp"])
                        st["tp"] = None
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    if self.move_be:
                        self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if st["partial_done"] and trail is not None and cl[-1] > trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["scratch"] = True
                continue
            if st["cooldown"] > 0:
                continue

            arm = self._breakdown_arm(hi, lo, cl, vo)
            if arm is not None:
                entry_px, stop_px, target_px = arm
                if st["pending"] is not None and st["planned_entry"] != entry_px:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                if st["pending"] is None:
                    risk = stop_px - entry_px
                    if risk <= 0.0 or not np.isfinite(risk):
                        continue
                    qty = ctx.equity() * self.risk_fraction / risk
                    if qty <= 0.0 or not np.isfinite(qty):
                        continue
                    entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                                    quantity=qty, price=entry_px)
                    st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                    quantity=qty, price=stop_px,
                                                    parent=entry_id, reduce_only=True)
                    st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Buy,
                                                     quantity=qty / 2.0, price=target_px,
                                                     parent=entry_id, reduce_only=True)
                    st["pending"] = entry_id
                    st["entry_qty"] = qty
                    st["planned_entry"] = entry_px
                    st["planned_stop"] = stop_px
                    st["planned_target"] = target_px
                st["armed_bar"] = st["bar_count"]

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = min(st["planned_stop"], pos.price)   # mirror: stop drops to entry
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _breakdown_arm(self, hi, lo, cl, vo):
        if len(cl) < self.base_max_len + 1:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        if not self._liq_ok(cl, vo) or adr < self.min_adr or g > -self.min_gain:
            return None
        if self.require_mas:
            s10 = sma(cl, 10)
            s20 = sma(cl, 20)
            if s10 is None or s20 is None or not (cl[-1] < s20 and s10 < s20):
                return None
        window = lo[-self.base_max_len:]
        mn = window[0]
        last_pos = 0
        for i in range(self.base_max_len):
            if window[i] <= mn:                 # keep the LAST extreme
                mn = window[i]
                last_pos = i
        since_tr = (self.base_max_len - 1) - last_pos
        bounce_high = highest(hi, max(since_tr, 1))
        if mn <= 0.0 or bounce_high is None:
            return None
        retrace_up = 100.0 * (bounce_high - mn) / mn
        if since_tr < self.min_base_days or retrace_up > self.max_depth:
            return None
        entry = float(mn) * (1.0 - self.entry_buffer_bps / 10_000.0)
        adr_stop = entry * (1.0 + self.adr_stop_mult * adr / 100.0)
        stop = min(adr_stop, float(hi[-1])) if self.use_lod_stop else adr_stop
        stop = max(stop, entry * 1.001)
        target = entry - self.partial_rr * (stop - entry)
        return entry, float(stop), float(target)

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol

    def _break_vol_ok(self, vo):
        if not self.use_bo_vol:
            return True
        if len(vo) < 51:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        return prev_avg50 is not None and vo[-1] >= self.bo_vol_mult * prev_avg50
