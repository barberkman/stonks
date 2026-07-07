"""Episodic pivot INTO a Darvas box top — the gap only counts if it clears a
proven ceiling.

QM's episodic pivot takes any qualifying gap; most gaps die because they
launch from nowhere. This hybrid demands structure under the gap: the gap
bar must blow through the top of a box that was ACTIVE on the PRIOR bar.

The prior-bar gate is load-bearing, not a nicety: re-running the box machine
on the gap bar itself would already mark the box as broken (its high exceeds
the top), so the box state and edges are snapshotted BEFORE the step and the
gap is judged against those frozen values.

  signal   prior bar had an active box; this bar gaps up >= ep_min_gap%,
           OPENS above that box's frozen top, prints >= ep_vol_mult x the
           50-bar average volume, and closes strong (green, upper half)
  entry    market at the next open (the pine buys the gap-bar close)
  stop     the box BOTTOM — the structure that made the gap credible
  exits    QM management: half partial at 2R (of the box-anchored risk) or
           after partial_bars, breakeven move, EMA20 close-trail on the rest

Levels are structural (box edges), so a gapped fill re-anchors nothing.
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


class QMDarvasEPBoxStrategy(stonks.Strategy):
    # Universe (liquidity, as the pine's EP)
    min_price = 5.0
    min_avg_vol = 0.0
    # The box under the gap
    length = 4
    # Episodic pivot gates
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # QM management
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "ep_min_gap": stonks.Param("minimum gap up vs the prior close", unit="%"),
        "ep_vol_mult": stonks.Param("gap-bar volume vs the 50-bar average", unit="x"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active box top (the ceiling a gap must clear)"),
        "box_bottom": stonks.Indicator("active box bottom (the stop)"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.length, 51, 3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
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

            # Snapshot the box as of BEFORE this bar: the gap is judged
            # against the ceiling that existed when the market gapped.
            was_active = st["bx_state"] == 5
            prior_top = st["bx_high"]
            prior_bot = st["bx_low"]
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

            # ── The gated gap ─────────────────────────────────────────────────
            if not was_active or prior_top is None or prior_bot is None:
                continue
            if len(cl) < 52 or not self._liq_ok(cl, vo):
                continue
            prev_close = cl[-2]
            if prev_close <= 0.0:
                continue
            if 100.0 * (op[-1] / prev_close - 1.0) < self.ep_min_gap:
                continue
            if op[-1] <= prior_top:                # the gap must CLEAR the ceiling
                continue
            prev_avg50 = sma(vo[:-1], 50)
            if prev_avg50 is None or vo[-1] < self.ep_vol_mult * prev_avg50:
                continue
            if self.ep_strong_close and not (cl[-1] > op[-1]
                                             and cl[-1] >= (hi[-1] + lo[-1]) / 2.0):
                continue
            entry = float(cl[-1])
            stop_px = float(prior_bot)
            risk = entry - stop_px
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            target_px = entry + self.partial_rr * risk
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
            st["planned_stop"] = stop_px

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
