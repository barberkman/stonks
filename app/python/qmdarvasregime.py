"""Regime switch: QM when the tape is fast, Darvas when it is quiet.

The two pines suit different volatility regimes. QM's momentum breakout
wants range expansion (its own universe has an ADR floor); Darvas' boxes
describe patient, low-noise consolidations. Instead of picking one, this
strategy reads the current ADR% and dispatches WHILE FLAT:

  ADR% >= regime_adr   QM mode: tight-flag scan, resting buy-stop at the
                       pivot, ADR stop, half partial at 2R or after
                       partial_bars, breakeven, EMA20 close-trail
  ADR% <  regime_adr   Darvas mode: completed box arms a buy-stop at its
                       top, stop at the bottom, later boxes ratchet the
                       stop, no take-profit

Two rules keep the handoff honest:
  * every trade is TAGGED with the regime that armed it, and its management
    dispatches on the stored tag — never on the live regime, which may have
    flipped mid-trade;
  * an armed-but-unfilled entry whose regime flips underneath it is stale
    evidence and is cancelled; the next tick re-scans under the new regime.
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


class QMDarvasRegimeStrategy(stonks.Strategy):
    # Regime dial
    regime_adr = 1.0          # ADR% at/above which the QM engine runs
    adr_len = 20
    # Shared universe basics
    min_price = 5.0
    min_avg_vol = 0.0
    mom_len = 24
    min_gain = 0.5
    # QM mode
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Darvas mode
    length = 4
    # Shared entry/gating
    entry_buffer_bps = 5.0
    order_ttl_bars = 10
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "regime_adr": stonks.Param("ADR% threshold splitting QM (fast) from Darvas (quiet)", unit="%"),
        "partial_rr": stonks.Param("QM-mode half partial target, in R multiples", unit="R"),
        "trail_len": stonks.Param("QM-mode EMA trail length", unit="bars"),
        "length": stonks.Param("Darvas-mode box lookback", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active Darvas box top (quiet-regime engine)"),
        "box_bottom": stonks.Indicator("active Darvas box bottom"),
        "trail": stonks.Indicator("EMA trail (fast-regime engine)"),
    }

    def lookback(self):
        return max(self.base_max_len, self.mom_len, self.adr_len,
                   self.length, 3 * self.trail_len, 21) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "pending": None, "entry_id": None, "sl": None, "tp": None,
                "regime": None,   # "qm" | "darvas", stamped at arm time
                "planned_entry": 0.0, "planned_stop": 0.0, "planned_target": 0.0,
                "entry_qty": 0.0, "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
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

            adr = adr_pct(hi, lo, self.adr_len)
            trail = ema(cl, self.trail_len)
            if active:
                ctx.plot("box_top", symbol, st["bx_high"])
                ctx.plot("box_bottom", symbol, st["bx_low"])
            if trail is not None:
                ctx.plot("trail", symbol, trail)
            live_regime = "qm" if (adr is not None and adr >= self.regime_adr) else "darvas"

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
                elif st["regime"] != live_regime:
                    # Stale evidence: the market the order was priced for is gone.
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                elif st["bar_count"] - st["armed_bar"] > self.order_ttl_bars:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["exiting"] = False
                st["sl"] = st["tp"] = st["entry_id"] = None
                st["regime"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            if in_pos:
                # Management dispatches on the STORED tag, never the live regime.
                if st["exiting"]:
                    continue
                held = abs(pos.quantity)
                bars_held = st["bar_count"] - st["entry_bar"]
                if st["regime"] == "darvas":
                    if new_box and st["bx_low"] > st["planned_stop"]:
                        if st["sl"] is not None:
                            ctx.cancel_order(st["sl"])
                        st["planned_stop"] = float(st["bx_low"])
                        st["sl"] = ctx.place_stop_order(
                            symbol=symbol, side=OrderSide.Sell,
                            quantity=st["entry_qty"], price=st["planned_stop"],
                            parent=st["entry_id"], reduce_only=True)
                    continue
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

            # ── Flat: scan under the LIVE regime ─────────────────────────────
            if live_regime == "qm":
                arm = self._qm_arm(hi, lo, cl, vo, adr)
                if arm is None:
                    continue
                entry_px, stop_px, target_px = arm
                if st["pending"] is not None and st["planned_entry"] != entry_px:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                if st["pending"] is None:
                    self._place(ctx, symbol, st, entry_px, stop_px, target_px, "qm")
                st["armed_bar"] = st["bar_count"]
            else:
                if st["pending"] is not None and not active:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                if st["pending"] is None and new_box:
                    entry_px = st["bx_high"] * (1.0 + self.entry_buffer_bps / 10_000.0)
                    stop_px = float(st["bx_low"])
                    self._place(ctx, symbol, st, entry_px, stop_px, None, "darvas")
                    st["armed_bar"] = st["bar_count"]

    def _place(self, ctx, symbol, st, entry_px, stop_px, target_px, regime):
        risk = entry_px - stop_px
        if risk <= 0.0 or not np.isfinite(risk):
            return
        qty = ctx.equity() * self.risk_fraction / risk
        if qty <= 0.0 or not np.isfinite(qty):
            return
        entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                        quantity=qty, price=entry_px)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=qty, price=stop_px,
                                        parent=entry_id, reduce_only=True)
        st["tp"] = None
        if target_px is not None:
            st["tp"] = ctx.place_limit_order(symbol=symbol, side=OrderSide.Sell,
                                             quantity=qty / 2.0, price=target_px,
                                             parent=entry_id, reduce_only=True)
        st["pending"] = entry_id
        st["regime"] = regime
        st["entry_qty"] = qty
        st["planned_entry"] = entry_px
        st["planned_stop"] = stop_px
        st["planned_target"] = target_px if target_px is not None else 0.0

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price)
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                        quantity=st["entry_qty"], price=be,
                                        parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be

    def _qm_arm(self, hi, lo, cl, vo, adr):
        if len(cl) < self.base_max_len + 1 or adr is None:
            return None
        g = gain_pct(cl, self.mom_len)
        av20 = sma(vo, 20)
        if g is None or g < self.min_gain:
            return None
        if cl[-1] < self.min_price or av20 is None or av20 < self.min_avg_vol:
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
        return entry, float(stop), float(entry + self.partial_rr * (entry - stop))
