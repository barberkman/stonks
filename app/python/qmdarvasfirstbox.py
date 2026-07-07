"""Ignition, then the FIRST box — Qullamaggie's "wait for the first
consolidation after the catalyst," formalized with the Darvas machine.

An episodic pivot marks a stock as interesting; chasing the gap bar itself
is the pine's EP entry. Qullamaggie's higher-quality variant is to let the
ignition digest and buy the FIRST proper consolidation that forms afterward.
This file uses the Darvas box as the definition of "proper consolidation":

  ignition   an EP-quality bar (gap >= ep_min_gap%, volume >= ep_vol_mult x
             the 50-bar average, strong close) opens a WATCH WINDOW of
             window_bars bars — no order is placed yet
  the box    the first Darvas box that COMPLETES inside the window arms a
             resting buy-stop at its top, stop-loss at its bottom; the box
             ending unfilled, or the window expiring boxless, resets cleanly
             to idle (that ignition's chance has passed)
  exits      QM management: half partial at 2R or after partial_bars,
             breakeven move, EMA20 close-trail on the rest

New ignitions are ignored while a window is open or a trade is working —
one thesis at a time per symbol.
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


class QMDarvasFirstBoxStrategy(stonks.Strategy):
    # Universe (liquidity, as the pine's EP)
    min_price = 5.0
    min_avg_vol = 0.0
    # Ignition
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    window_bars = 20
    # The box
    length = 4
    entry_buffer_bps = 5.0
    # QM management
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "window_bars": stonks.Param("bars after ignition to wait for the first box", unit="bars"),
        "ep_min_gap": stonks.Param("ignition gap vs the prior close", unit="%"),
        "ep_vol_mult": stonks.Param("ignition volume vs the 50-bar average", unit="x"),
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active box top (armed only inside a watch window)"),
        "box_bottom": stonks.Indicator("active box bottom"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.length, 51, 3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "phase": "idle",              # idle -> watching -> armed
                "deadline": 0,
                "pending": None, "entry_id": None, "sl": None, "tp": None,
                "planned_stop": 0.0, "entry_qty": 0.0,
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
            op = sub["open"].to_numpy()
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
            if len(cl) < self.lookback():
                continue

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
                st["phase"] = "idle"
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
            if st["cooldown"] > 0:
                continue

            # ── The two-phase thesis machine ──────────────────────────────────
            if st["phase"] == "armed":
                if st["pending"] is None:
                    st["phase"] = "idle"      # cancelled/rejected elsewhere
                elif not active:
                    ctx.cancel_order(st["pending"])   # THE first box has ended unfilled
                    st["pending"] = None
                    st["phase"] = "idle"
                continue
            if st["phase"] == "watching":
                if st["bar_count"] > st["deadline"]:
                    st["phase"] = "idle"      # the window closed boxless
                elif new_box:
                    entry_px = st["bx_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
                    stop_px = float(st["bx_low"])
                    risk = entry_px - stop_px
                    if risk <= 0.0 or not np.isfinite(risk):
                        st["phase"] = "idle"
                        continue
                    qty = ctx.equity() * self.risk_fraction / risk
                    if qty <= 0.0 or not np.isfinite(qty):
                        st["phase"] = "idle"
                        continue
                    target_px = entry_px + self.partial_rr * risk
                    entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                    quantity=qty, price=entry_px)
                    st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                                    quantity=qty, price=stop_px,
                                                    parent=entry_id, reduce_only=True)
                    st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                                     quantity=qty / 2.0, price=target_px,
                                                     parent=entry_id, reduce_only=True)
                    st["pending"] = entry_id
                    st["entry_qty"] = qty
                    st["planned_stop"] = stop_px
                    st["phase"] = "armed"
                continue
            # idle: scan for an ignition
            if self._ignition(op, hi, lo, cl, vo):
                st["phase"] = "watching"
                st["deadline"] = st["bar_count"] + self.window_bars

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _ignition(self, op, hi, lo, cl, vo):
        if len(cl) < 52:
            return False
        av20 = sma(vo, 20)
        if cl[-1] < self.min_price or av20 is None or av20 < self.min_avg_vol:
            return False
        prev_close = cl[-2]
        if prev_close <= 0.0:
            return False
        if 100.0 * (op[-1] / prev_close - 1.0) < self.ep_min_gap:
            return False
        prev_avg50 = sma(vo[:-1], 50)
        if prev_avg50 is None or vo[-1] < self.ep_vol_mult * prev_avg50:
            return False
        if self.ep_strong_close and not (cl[-1] > op[-1]
                                         and cl[-1] >= (hi[-1] + lo[-1]) / 2.0):
            return False
        return True
