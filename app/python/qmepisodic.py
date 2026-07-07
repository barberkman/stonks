"""Qullamaggie episodic pivot, isolated — Setup 2 as a standalone strategy.

The pine's EP is one branch of a five-way priority chain; this file reads it
as its own strategy: a gap up of at least ep_min_gap% against the prior
close, printed on ep_vol_mult x the 50-bar average volume, closing strong
(green and in the upper half of the range), with the close-to-low risk within
one ADR — nothing else. No base, no pivot, no trend-MA gate (the pine's EP
only requires the liquidity floor: the gap IS the signal).

Entry is a market order at the signal close (the pine buys the close of the
gap bar; a real order fills at the next open, so the protective legs are
re-anchored proportionally to the actual fill). Management is the full QM
exit engine: ADR stop tightened to the signal bar's low, half partial at 2R
or after partial_bars, breakeven move, then an EMA20 close-trail on the rest.
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


class QMEpisodicStrategy(stonks.Strategy):
    # Universe (pine: EP only gates on liquidity)
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    # Episodic pivot
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
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
        "ep_min_gap": stonks.Param("minimum gap up vs the prior close", unit="%"),
        "ep_vol_mult": stonks.Param("gap-bar volume vs the 50-bar average", unit="x"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(51, self.adr_len, 3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "bar_count": 0, "entry_bar": 0,
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
            op = sub["open"].to_numpy()
            hi = sub["high"].to_numpy()
            lo = sub["low"].to_numpy()
            cl = sub["close"].to_numpy()
            vo = sub["volume"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1
            if len(cl) < self.lookback():
                continue

            trail = ema(cl, self.trail_len)
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
                            # The market entry filled at the open, not the
                            # signal close: re-anchor the legs proportionally.
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
                                quantity=st["entry_qty"] / 2.0, price=st["planned_target"],
                                parent=st["entry_id"], reduce_only=True)
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
                            self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if not st["partial_done"] and bars_held >= self.partial_bars:
                    if st["tp"] is not None:
                        ctx.cancel_order(st["tp"])
                        st["tp"] = None
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    if self.move_be:
                        self._move_stop_to_breakeven(ctx, symbol, st, pos)
                if st["partial_done"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0 or st["pending"] is not None:
                continue

            ep = self._episodic_pivot(op, hi, lo, cl, vo)
            if ep is None:
                continue
            entry_px, stop_px, target_px = ep
            risk = entry_px - stop_px
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
            st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                             quantity=qty / 2.0, price=target_px,
                                             parent=entry_id, reduce_only=True)
            st["pending"] = entry_id
            st["entry_qty"] = qty
            st["planned_entry"] = entry_px
            st["planned_stop"] = stop_px
            st["planned_target"] = target_px

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _episodic_pivot(self, op, hi, lo, cl, vo):
        if len(cl) < 52 or not self._liq_ok(cl, vo):
            return None
        prev_close = cl[-2]
        if prev_close <= 0.0:
            return None
        if 100.0 * (op[-1] / prev_close - 1.0) < self.ep_min_gap:
            return None
        prev_avg50 = sma(vo[:-1], 50)
        if prev_avg50 is None or vo[-1] < self.ep_vol_mult * prev_avg50:
            return None
        if self.ep_strong_close and not (cl[-1] > op[-1] and cl[-1] >= (hi[-1] + lo[-1]) / 2.0):
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        if adr is None:
            return None
        risk_ps = cl[-1] - lo[-1]
        if not (risk_ps > 0.0 and risk_ps <= self.adr_stop_mult * adr / 100.0 * cl[-1]):
            return None
        entry = float(cl[-1])
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return entry, float(stop), float(entry + self.partial_rr * (entry - stop))

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
