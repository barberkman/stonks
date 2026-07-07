"""Qullamaggie breakout scaled out in thirds — the "sell into strength"
reading taken further than the pine's single partial.

The pine sells one half at one target; interviews with the trader describe
peeling off pieces as the move extends. This interpretation formalizes that:

  at signal   resting buy-stop at the pivot, PLUS three protective children:
                SL   stop, FULL size, at the ADR/low stop
                TP1  limit, 1/3 size, at 2R
                TP2  limit, 1/3 size, at 4R
              All three are siblings under the entry. Only one path ever
              closes the whole trade, so their quantities deliberately sum
              past 100% — opposite-side fills clamp to what is held, and the
              broker cancels the survivors the instant the position is flat.
  after TP1   the stop is re-placed at breakeven, STILL full size (a stop-out
              from here closes whatever remains, 2/3 or 1/3).
  after TP2   the last third has no resting target: it trails on closes and
              flattens at the next open once a close drops under the SMA20.

Entry mechanics match the literal port (TTL, weak-volume scratch, gapped-fill
re-anchor of all three legs).
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


class QMThirdsStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # Base & entry
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Scale-out ladder
    adr_stop_mult = 1.0
    use_lod_stop = True
    tp1_rr = 2.0
    tp2_rr = 4.0
    trail_len = 20            # SMA trail for the final third
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "tp1_rr": stonks.Param("first third's target, in R multiples", unit="R"),
        "tp2_rr": stonks.Param("second third's target, in R multiples", unit="R"),
        "trail_len": stonks.Param("SMA length trailing the final third", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "trail": stonks.Indicator("SMA20 — the final third exits on a close below it"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp1": None, "tp2": None,
                "planned_entry": 0.0, "planned_stop": 0.0,
                "tp1_px": 0.0, "tp2_px": 0.0, "entry_qty": 0.0,
                "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
                "be_done": False, "trail_live": False, "exiting": False,
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

            trail = sma(cl, self.trail_len)
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
                        fill = pos.price
                        if st["planned_entry"] > 0.0 and fill != st["planned_entry"]:
                            ratio = fill / st["planned_entry"]
                            for key in ("sl", "tp1", "tp2"):
                                if st[key] is not None:
                                    ctx.cancel_order(st[key])
                            st["planned_stop"] *= ratio
                            st["tp1_px"] *= ratio
                            st["tp2_px"] *= ratio
                            third = st["entry_qty"] / 3.0
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=st["planned_stop"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp1"] = ctx.place_limit_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=third, price=st["tp1_px"],
                                parent=st["entry_id"], reduce_only=True)
                            st["tp2"] = ctx.place_limit_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=third, price=st["tp2_px"],
                                parent=st["entry_id"], reduce_only=True)
                        if not self._break_vol_ok(vo):
                            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                                   quantity=abs(pos.quantity),
                                                   parent=st["entry_id"], reduce_only=True)
                            st["exiting"] = True
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
                st["be_done"] = False
                st["trail_live"] = False
                st["exiting"] = False
                st["sl"] = st["tp1"] = st["tp2"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                if st["exiting"]:
                    continue
                bars_held = st["bar_count"] - st["entry_bar"]
                if bars_held < 1:
                    continue
                if not st["be_done"] and st["tp1"] is not None:
                    tp1 = ctx.order(st["tp1"])
                    if tp1 is not None and tp1.status == OrderStatus.Filled:
                        # First third banked: stop to breakeven, still full size.
                        st["be_done"] = True
                        st["tp1"] = None
                        if st["sl"] is not None:
                            ctx.cancel_order(st["sl"])
                        be = max(st["planned_stop"], pos.price)
                        st["planned_stop"] = be
                        st["sl"] = ctx.place_stop_order(
                            symbol=symbol, side=OrderSide.Sell,
                            quantity=st["entry_qty"], price=be,
                            parent=st["entry_id"], reduce_only=True)
                if not st["trail_live"] and st["tp2"] is not None:
                    tp2 = ctx.order(st["tp2"])
                    if tp2 is not None and tp2.status == OrderStatus.Filled:
                        st["trail_live"] = True
                        st["tp2"] = None
                if st["trail_live"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=abs(pos.quantity),
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0:
                continue

            arm = self._breakout_arm(hi, lo, cl, vo)
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
                third = qty / 3.0
                tp1_px = entry_px + self.tp1_rr * risk
                tp2_px = entry_px + self.tp2_rr * risk
                entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                quantity=qty, price=entry_px)
                st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                                quantity=qty, price=stop_px,
                                                parent=entry_id, reduce_only=True)
                st["tp1"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                                  quantity=third, price=tp1_px,
                                                  parent=entry_id, reduce_only=True)
                st["tp2"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                                  quantity=third, price=tp2_px,
                                                  parent=entry_id, reduce_only=True)
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["planned_entry"] = entry_px
                st["planned_stop"] = stop_px
                st["tp1_px"] = tp1_px
                st["tp2_px"] = tp2_px
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
        entry = float(mx) * (1.0 + self.entry_buffer_bps / 10_000.0)
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return entry, float(stop)

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
