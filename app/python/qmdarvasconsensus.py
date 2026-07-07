"""QM and Darvas must AGREE — the double-confirmation breakout.

The QM pivot scan and the Darvas box machine each nominate a resistance
level. They usually disagree slightly; when they nominate the SAME price,
two independent definitions of the base have converged on one ceiling and
the break of it means more. This file trades only that consensus:

  consensus  a Darvas box is ACTIVE, the QM tight flag qualifies (base age,
             retracement, full universe gates), and the box top sits within
             consensus_bps of the QM pivot
  entry      resting buy-stop above the HIGHER of the two levels (break
             both ceilings or nothing), alive while the consensus holds —
             the box ending or the flag degrading cancels it
  stop       the tightest available: max(QM ADR stop, box bottom)
  exits      QM management: half partial at 2R or after partial_bars,
             breakeven move, EMA20 close-trail on the rest

Rarer signals by construction; that selectivity is the interpretation.
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


class QMDarvasConsensusStrategy(stonks.Strategy):
    # QM universe & flag
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    # Darvas box + consensus
    length = 4
    consensus_bps = 50.0      # box top within this of the QM pivot
    entry_buffer_bps = 5.0
    # Stop & management
    adr_stop_mult = 1.0
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "consensus_bps": stonks.Param("max distance between box top and QM pivot", unit="bps"),
        "length": stonks.Param("box lookback period (rolling low)", unit="bars"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "box_top": stonks.Indicator("active box top (one of the two ceilings)"),
        "box_bottom": stonks.Indicator("active box bottom"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.length, 3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"bx_state": 1, "bx_high": None, "bx_low": None, "last_ts": None,
                "pending": None, "entry_id": None, "sl": None, "tp": None,
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

            consensus = self._consensus(hi, lo, cl, vo, st)
            if consensus is None:
                # The agreement broke (box gone or flag degraded): stand down.
                if st["pending"] is not None:
                    ctx.cancel_order(st["pending"])
                    st["pending"] = None
                continue
            entry_px, stop_px, target_px = consensus
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

    def _consensus(self, hi, lo, cl, vo, st):
        """(entry, stop, target) when the QM flag and the ACTIVE box agree on
        the ceiling, else None."""
        if st["bx_state"] != 5 or len(cl) < self.base_max_len + 1:
            return None
        adr = adr_pct(hi, lo, self.adr_len)
        g = gain_pct(cl, self.mom_len)
        if adr is None or g is None:
            return None
        av20 = sma(vo, 20)
        if cl[-1] < self.min_price or av20 is None or av20 < self.min_avg_vol:
            return None
        if adr < self.min_adr or g < self.min_gain:
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
        box_top = float(st["bx_high"])
        box_bot = float(st["bx_low"])
        if abs(box_top - mx) / mx > self.consensus_bps / 10_000.0:
            return None                      # two ceilings, no agreement
        entry = max(float(mx), box_top) * (1.0 + self.entry_buffer_bps / 10_000.0)
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, box_bot)        # the tightest available stop
        stop = min(stop, entry * 0.999)
        target = entry + self.partial_rr * (entry - stop)
        return entry, float(stop), float(target)
