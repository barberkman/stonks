"""Qullamaggie breakout run as a pure runner — no partial, no breakeven,
trailed on the fast MA from bar one.

Qullamaggie's own description of trade management is "sell into strength,
trail the rest on the 10- or 20-day MA". The pine models the first half of
that; this interpretation models a trader who skips the sell-into-strength
leg entirely: take the full position through the breakout and hold it until
either the hard stop or the FAST (10-bar) EMA gives out. Two exits, full
size, nothing else:

  * the initial ADR stop rests at the broker for the life of the trade,
    never tightened, never moved to breakeven;
  * a confirmed close below the EMA10 flattens everything at the next open
    (active immediately, not gated behind a partial like the pine's trail).

Entry mechanics match the literal port: resting buy-stop at the tight-flag
pivot, TTL while unfilled, post-fill volume scratch on a weak break bar.
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


class QMRunnerStrategy(stonks.Strategy):
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
    # Exits: hard stop + fast trail, both full size
    adr_stop_mult = 1.0
    use_lod_stop = True
    trail_len = 10            # the aggressive "10 MA" reading
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "trail_len": stonks.Param("EMA length of the full-size close-trail", unit="bars"),
        "adr_stop_mult": stonks.Param("hard stop distance from entry, in ADRs", unit="ADR"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "order_ttl_bars": stonks.Param("resting buy-stop lifetime", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "trail": stonks.Indicator("EMA10 — a close below it flattens the runner"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None,
                "planned_entry": 0.0, "planned_stop": 0.0, "entry_qty": 0.0,
                "armed_bar": 0, "bar_count": 0, "entry_bar": 0,
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
                        fill = pos.price
                        if st["planned_entry"] > 0.0 and fill != st["planned_entry"]:
                            ratio = fill / st["planned_entry"]
                            if st["sl"] is not None:
                                ctx.cancel_order(st["sl"])
                            st["planned_stop"] *= ratio
                            st["sl"] = ctx.place_stop_order(
                                symbol=symbol, side=OrderSide.Sell,
                                quantity=st["entry_qty"], price=st["planned_stop"],
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
                    continue
                if trail is not None and cl[-1] < trail:
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
                entry_id = ctx.place_stop_order(symbol=symbol, side=OrderSide.Buy,
                                                quantity=qty, price=entry_px)
                st["sl"] = ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                                quantity=qty, price=stop_px,
                                                parent=entry_id, reduce_only=True)
                st["pending"] = entry_id
                st["entry_qty"] = qty
                st["planned_entry"] = entry_px
                st["planned_stop"] = stop_px
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
