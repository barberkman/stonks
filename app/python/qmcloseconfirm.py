"""Qullamaggie breakout in the pine's "Wait for bar close" mode, read
strictly: NOTHING rests at the broker — no stop-entry, no stop-loss, no
take-profit orders. Every decision is made on a confirmed close and executed
with a market order at the next open.

The pine offers waitClose as an entry option only; this interpretation
extends the same philosophy to the whole trade, the way a close-only
end-of-bar trader would run it:

  entry    close >= pivot of the PRIOR 40-bar base (the poke that only wicks
           through the level never triggers anything), break-bar volume
           checked directly on the signal bar — no post-fill scratch needed
  stop     evaluated on closes: close <= stop level -> flatten next open
  partial  close >= 2R target, or partial_bars bars in the trade -> half off
           next open, stop level lifted to breakeven (internally)
  trail    after the partial: close < EMA20 -> flatten next open

Stops evaluated on closes are looser than resting stops (an intrabar flush
does not stop this strategy out) — that is the point of the interpretation,
not an accident. Levels live in strategy state; the broker only ever sees
reduce-only market orders on the way out.
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


class QMCloseConfirmStrategy(stonks.Strategy):
    # Universe & trend
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    # Base & break (evaluated on the confirmed close)
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    entry_buffer_bps = 5.0
    bo_vol_mult = 1.3
    # Close-driven management
    adr_stop_mult = 1.0
    use_lod_stop = True
    partial_rr = 2.0
    partial_bars = 6
    trail_len = 20
    # Sizing & gating
    risk_fraction = 0.02
    cooldown_bars = 5

    params = {
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "adr_stop_mult": stonks.Param("close-based stop distance, in ADRs", unit="ADR"),
        "partial_rr": stonks.Param("half partial target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "trail_len": stonks.Param("EMA length of the post-partial trail", unit="bars"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
    }

    indicators = {
        "sma10": stonks.Indicator("10-bar SMA of close (trend filter, fast)"),
        "sma20": stonks.Indicator("20-bar SMA of close (trend filter, slow)"),
        "trail": stonks.Indicator("trailing EMA the post-partial exit checks"),
    }

    def lookback(self):
        return max(self.base_max_len + 1, 51, self.mom_len, self.adr_len,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None,
                "stop_px": 0.0, "target_px": 0.0, "planned_entry": 0.0,
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
                            # Fill landed at the open, not the signal close:
                            # rescale the INTERNAL levels (nothing rests).
                            ratio = fill / st["planned_entry"]
                            st["stop_px"] *= ratio
                            st["target_px"] *= ratio
                    else:
                        closed = True
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None
            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["exiting"] = False
                st["entry_id"] = None
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
                if cl[-1] <= st["stop_px"]:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                    continue
                if not st["partial_done"] and (cl[-1] >= st["target_px"]
                                               or bars_held >= self.partial_bars):
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held / 2.0,
                                           parent=st["entry_id"], reduce_only=True)
                    st["partial_done"] = True
                    st["stop_px"] = max(st["stop_px"], pos.price)   # breakeven, internally
                    continue
                if st["partial_done"] and trail is not None and cl[-1] < trail:
                    ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                           quantity=held,
                                           parent=st["entry_id"], reduce_only=True)
                    st["exiting"] = True
                continue
            if st["cooldown"] > 0 or st["pending"] is not None:
                continue

            sig = self._close_confirmed_break(hi, lo, cl, vo)
            if sig is None:
                continue
            entry_px, stop_px, target_px = sig
            risk = entry_px - stop_px
            if risk <= 0.0 or not np.isfinite(risk):
                continue
            qty = ctx.equity() * self.risk_fraction / risk
            if qty <= 0.0 or not np.isfinite(qty):
                continue
            st["pending"] = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                                   quantity=qty)
            st["planned_entry"] = entry_px
            st["stop_px"] = stop_px
            st["target_px"] = target_px

    def _close_confirmed_break(self, hi, lo, cl, vo):
        """The pivot comes from the PRIOR base_max_len bars (the signal bar is
        excluded so the break bar cannot be its own pivot), and only a CLOSE
        beyond it counts — the wait-for-bar-close reading."""
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
        window = hi[-(self.base_max_len + 1):-1]
        mx = window[0]
        last_pos = 0
        for i in range(self.base_max_len):
            if window[i] >= mx:
                mx = window[i]
                last_pos = i
        since_pk = (self.base_max_len - 1) - last_pos
        pull_n = max(since_pk, 1)
        pull_low = float(np.min(lo[-1 - pull_n:-1]))
        if mx <= 0.0:
            return None
        retrace = 100.0 * (mx - pull_low) / mx
        if since_pk < self.min_base_days or retrace > self.max_depth:
            return None
        entry = float(mx) * (1.0 + self.entry_buffer_bps / 10_000.0)
        if cl[-1] < entry:                      # the CLOSE must clear the level
            return None
        prev_avg50 = sma(vo[:-1], 50)           # break-bar volume, checked directly
        if prev_avg50 is None or vo[-1] < self.bo_vol_mult * prev_avg50:
            return None
        signal_close = float(cl[-1])
        adr_stop = signal_close * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(lo[-1])) if self.use_lod_stop else adr_stop
        stop = min(stop, signal_close * 0.999)
        target = signal_close + self.partial_rr * (signal_close - stop)
        return signal_close, float(stop), float(target)

    def _liq_ok(self, cl, vo):
        av20 = sma(vo, 20)
        return cl[-1] >= self.min_price and av20 is not None and av20 >= self.min_avg_vol
