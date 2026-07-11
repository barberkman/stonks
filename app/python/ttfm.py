"""TTrades Fractal Model (TTFM) — the mechanical playbook, one file.

The model rests on a single law: price cannot reverse without first building a
swing point, and a swing is only tradable once a HIGHER-timeframe C2 closure
(sweep the prior candle's extreme, then close back inside its range) AND a
LOWER-timeframe CISD (change in state of delivery) both confirm it, in the same
direction. HTF closure with no LTF CISD, or a CISD with no HTF closure behind
it, is a skip. Everything else — the POI in the discount/premium half of the
prior range, the equilibrium filter, the 2R-to-DOL target, the stop beyond the
protected swing, the C4-extended stand-aside — is machinery around that gate.

Timeframe mapping (this engine feeds exactly one timeframe; higher ones are
derived in Python):
  * Run on a DAILY feed (binance_1d, or any us_*_1d equities file).
  * HTF = Weekly, aggregated from the daily stream by Monday-start week buckets.
  * LTF = Daily (native) — carries the CISD, the POI and the entry.
This is the deck's "Weekly Profile — Classic Expansion" / Weekly→Daily pairing.
The finer 5m/15m execution step the model normally nests has no data to run on
(no sub-daily bars here), so it collapses onto the daily CISD — an honest
reduction, not a silent one.

Execution timeline: decisions on the daily close, fills from the next bar's open
(the engine's no-lookahead contract). Weekly C2 is read provisionally — we do
not wait for the week to close; the daily CISD is what confirms it.
"""

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus

MS_PER_DAY = 86_400_000


# ─── Calendar helpers (epoch day 0 = Thursday 1970-01-01) ─────────────────────
def week_index(ts_ms):
    """Monday-start week bucket. +3 shifts Thursday-anchored epoch days so a
    week rolls on Monday; identical index for every day of the same Mon–Sun week."""
    return (int(ts_ms) // MS_PER_DAY + 3) // 7


def day_of_week(ts_ms):
    """0 = Monday … 6 = Sunday."""
    return (int(ts_ms) // MS_PER_DAY + 3) % 7


def weekly_bars(ts, op, hi, lo, cl):
    """Aggregate the pulled daily window into per-week OHLC, in time order.
    Each entry carries the window indices of the week's extremes so the swing
    anchor and its timestamp can be recovered. Recomputed every tick — only the
    trade lifecycle is persisted."""
    wk = (ts // MS_PER_DAY + 3) // 7
    weeks = []
    start = 0
    n = len(wk)
    for i in range(1, n + 1):
        if i == n or wk[i] != wk[start]:
            seg = slice(start, i)
            weeks.append({
                "idx": int(wk[start]),
                "open": float(op[start]),
                "high": float(np.max(hi[seg])),
                "low": float(np.min(lo[seg])),
                "close": float(cl[i - 1]),
                "low_i": start + int(np.argmin(lo[seg])),
                "high_i": start + int(np.argmax(hi[seg])),
            })
            start = i
    return weeks


class TTFMStrategy(stonks.Strategy):
    # Risk & management
    risk_fraction = 0.01
    partial_rr = 2.0
    stop_buffer_bps = 10.0
    cooldown_bars = 3
    # The core-law gate
    atr_len = 14
    min_displacement = 0.5      # CISD break bar body, in ATRs (rejects slow chop)
    max_expansion_bars = 4      # C4 stand-aside: no entry >N daily bars past the swing
    require_early_extreme = False   # strict Classic Expansion: weekly extreme on Mon/Tue
    allow_c3_closure = True     # accept a C3-style close when the weekly C2 fails
    require_poi_in_half = True  # POI must sit in discount (long) / premium (short)
    require_eq_filter = True    # skip if price has already traded through EQ
    # Direction
    allow_long = True
    allow_short = True

    params = {
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "partial_rr": stonks.Param("first partial + minimum draw, in R multiples", unit="R"),
        "stop_buffer_bps": stonks.Param("stop distance beyond the protected swing", unit="bps"),
        "cooldown_bars": stonks.Param("bars to sit out after a close", unit="bars"),
        "atr_len": stonks.Param("ATR length for the displacement filter", unit="bars"),
        "min_displacement": stonks.Param("CISD break-bar body floor, in ATRs", unit="ATR"),
        "max_expansion_bars": stonks.Param("no new entry once this many bars past the swing (C4)", unit="bars"),
        "require_early_extreme": stonks.Param("require the weekly extreme on Mon/Tue (Classic Expansion)"),
        "allow_c3_closure": stonks.Param("accept a C3 close when the weekly C2 fails"),
        "require_poi_in_half": stonks.Param("POI must be in discount (long) / premium (short)"),
        "require_eq_filter": stonks.Param("skip entries after price has crossed equilibrium"),
        "allow_long": stonks.Param("take long setups"),
        "allow_short": stonks.Param("take short setups"),
    }

    indicators = {
        "wk_eq": stonks.Indicator("prior-week equilibrium (premium/discount split)"),
        "wk_hi": stonks.Indicator("prior-week high (buy-side draw on liquidity)"),
        "wk_lo": stonks.Indicator("prior-week low (sell-side draw on liquidity)"),
    }

    def lookback(self):
        return max(60, self.atr_len + 30)

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {"pending": None, "entry_id": None, "sl": None, "tp1": None, "tp2": None,
                "is_long": True, "entry_qty": 0.0,
                "planned_entry": 0.0, "planned_stop": 0.0,
                "planned_tp1": 0.0, "planned_tp2": 0.0,
                "entry_bar": 0, "bar_count": 0, "partial_done": False,
                "was_in": False, "cooldown": 0}

    def on_tick(self, ctx):
        w = ctx.history(self.lookback())
        if len(w) == 0:
            return
        df = pd.DataFrame({
            "symbol": w.symbol, "timestamp": w.timestamp,
            "open": w.open, "high": w.high, "low": w.low, "close": w.close,
        })
        for symbol, sub in df.groupby("symbol", sort=False):
            sub = sub.sort_values("timestamp")
            op = sub["open"].to_numpy()
            hi = sub["high"].to_numpy()
            lo = sub["low"].to_numpy()
            cl = sub["close"].to_numpy()
            ts = sub["timestamp"].to_numpy()

            st = self.state.setdefault(symbol, self._fresh())
            st["bar_count"] += 1

            self._plot_weekly(ctx, symbol, ts, op, hi, lo, cl)

            pos = ctx.position(symbol)
            in_pos = pos is not None
            closed = st["was_in"] and not in_pos

            # ── Entry-order fill bookkeeping ─────────────────────────────────
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
                            self._reanchor(ctx, symbol, st, fill)
                    else:
                        closed = True   # same-bar round trip
                    st["pending"] = None
                elif status != OrderStatus.Open:
                    st["pending"] = None

            if closed:
                st["cooldown"] = self.cooldown_bars
                st["partial_done"] = False
                st["sl"] = st["tp1"] = st["tp2"] = st["entry_id"] = None
            elif not in_pos and st["cooldown"] > 0:
                st["cooldown"] -= 1
            st["was_in"] = in_pos

            # ── Manage the open trade: first partial → breakeven ─────────────
            if in_pos:
                bars_held = st["bar_count"] - st["entry_bar"]
                if bars_held < 1:
                    continue
                if not st["partial_done"] and st["tp1"] is not None:
                    tp1 = ctx.order(st["tp1"])
                    if tp1 is not None and tp1.status == OrderStatus.Filled:
                        st["partial_done"] = True
                        st["tp1"] = None
                        self._move_stop_to_breakeven(ctx, symbol, st, pos)
                continue

            if st["cooldown"] > 0 or st["pending"] is not None:
                continue
            if len(cl) < self.lookback():
                continue

            # ── The gate: weekly C2 + daily CISD + POI, else stand aside ─────
            sig = self._signal(op, hi, lo, cl, ts)
            if sig is None:
                continue
            is_long, entry_px, stop_px, target_px = sig
            if (is_long and not self.allow_long) or (not is_long and not self.allow_short):
                continue
            self._place_bracket(ctx, symbol, st, is_long, entry_px, stop_px, target_px)

    # ── Signal detection (recomputed each tick; fires on the POI-close bar) ──────
    def _signal(self, op, hi, lo, cl, ts):
        n = len(cl)
        if n < self.atr_len + 6:
            return None
        weeks = weekly_bars(ts, op, hi, lo, cl)
        if len(weeks) < 2:
            return None
        cur, prev = weeks[-1], weeks[-2]

        rng = hi[-self.atr_len:] - lo[-self.atr_len:]
        atr = float(np.mean(rng))
        if atr <= 0.0 or not np.isfinite(atr):
            return None
        eq = (prev["high"] + prev["low"]) / 2.0

        # 1. Weekly manipulation → one-sided bias (Step 1).
        bull_sweep = cur["low"] < prev["low"]
        bear_sweep = cur["high"] > prev["high"]
        long_bias = bull_sweep and (cl[-1] > prev["low"] or self.allow_c3_closure) and not bear_sweep
        short_bias = bear_sweep and (cl[-1] < prev["high"] or self.allow_c3_closure) and not bull_sweep
        if long_bias == short_bias:      # both or neither → not one-sided
            return None
        is_long = long_bias

        # 2. Protected swing = the week's extreme; optional Mon/Tue timing.
        swing_i = cur["low_i"] if is_long else cur["high_i"]
        if self.require_early_extreme and day_of_week(int(ts[swing_i])) > 1:
            return None

        # 3. C4 stand-aside: don't initiate once the leg is extended.
        if (n - 1) - swing_i > self.max_expansion_bars:
            return None

        # 4. Daily CISD (the LTF confirmation — no CISD, no trade).
        cisd = self._cisd(op, cl, swing_i, is_long, atr)
        if cisd is None:
            return None
        _, cisd_bar = cisd

        # 5. POI (FVG, else order block) and the close-into-POI trigger.
        poi = self._poi(op, hi, lo, cl, swing_i, cisd_bar, is_long)
        if poi is None:
            return None
        poi_bot, poi_top = poi
        if not (poi_bot <= cl[-1] <= poi_top):   # this bar must close INTO the POI
            return None

        # 6. Premium/discount + equilibrium filter.
        poi_mid = (poi_bot + poi_top) / 2.0
        if is_long:
            if self.require_poi_in_half and poi_mid >= eq:
                return None
            if self.require_eq_filter and cl[-1] >= eq:
                return None
        else:
            if self.require_poi_in_half and poi_mid <= eq:
                return None
            if self.require_eq_filter and cl[-1] <= eq:
                return None

        # 7. Entry / stop / target, with the 2R-minimum-to-DOL gate.
        entry = float(cl[-1])
        buf = self.stop_buffer_bps / 10_000.0
        if is_long:
            stop = cur["low"] * (1.0 - buf)
            target = prev["high"]          # buy-side DOL
        else:
            stop = cur["high"] * (1.0 + buf)
            target = prev["low"]           # sell-side DOL
        risk = abs(entry - stop)
        if risk <= 0.0 or not np.isfinite(risk):
            return None
        if is_long and target <= entry:
            return None
        if not is_long and target >= entry:
            return None
        if abs(target - entry) < self.partial_rr * risk:   # DOL closer than 2R → skip
            return None
        return is_long, entry, stop, target

    def _cisd(self, op, cl, swing_i, is_long, atr):
        """(CISD line, break-bar index). Mark the open of the same-color run that
        delivered price into the swing; confirm when a later bar closes beyond it
        with displacement. None if delivery never flips."""
        n = len(cl)
        red = cl < op    # down-close
        start = swing_i
        if is_long:
            # step off the reversal candle if the low bar itself closed up
            if start >= 0 and not red[start]:
                start -= 1
            run_start = start
            while run_start >= 0 and red[run_start]:
                run_start -= 1
            run_start += 1
            line = float(op[run_start]) if run_start <= start else float(op[swing_i])
            for k in range(swing_i + 1, n):
                if cl[k] > line and (cl[k] - op[k]) >= self.min_displacement * atr:
                    return line, k
        else:
            green = cl > op
            if start >= 0 and not green[start]:
                start -= 1
            run_start = start
            while run_start >= 0 and green[run_start]:
                run_start -= 1
            run_start += 1
            line = float(op[run_start]) if run_start <= start else float(op[swing_i])
            for k in range(swing_i + 1, n):
                if cl[k] < line and (op[k] - cl[k]) >= self.min_displacement * atr:
                    return line, k
        return None

    def _poi(self, op, hi, lo, cl, swing_i, cisd_bar, is_long):
        """The point of interest left by the CISD move: the most recent fair-value
        gap in the impulse, else the last opposite-color candle (order block).
        Returns (bottom, top) or None."""
        n = len(hi)
        best = None
        for i in range(max(swing_i, 2), min(cisd_bar + 1, n - 1) + 1):
            if is_long and lo[i] > hi[i - 2]:
                best = (float(hi[i - 2]), float(lo[i]))      # bullish FVG
            elif not is_long and hi[i] < lo[i - 2]:
                best = (float(hi[i]), float(lo[i - 2]))      # bearish FVG
        if best is not None:
            return best
        # Order-block fallback: last opposite-color candle at/before the break.
        want_red = is_long
        for i in range(min(cisd_bar, n - 1), -1, -1):
            if (cl[i] < op[i]) == want_red and cl[i] != op[i]:
                return float(lo[i]), float(hi[i])
        return None

    def _plot_weekly(self, ctx, symbol, ts, op, hi, lo, cl):
        weeks = weekly_bars(ts, op, hi, lo, cl)
        if len(weeks) < 2:
            return
        prev = weeks[-2]
        ctx.plot("wk_eq", symbol, (prev["high"] + prev["low"]) / 2.0)
        ctx.plot("wk_hi", symbol, prev["high"])
        ctx.plot("wk_lo", symbol, prev["low"])

    # ── Order placement & management ────────────────────────────────────────────
    def _place_bracket(self, ctx, symbol, st, is_long, entry, stop, target):
        risk = abs(entry - stop)
        if risk <= 0.0 or not np.isfinite(risk):
            return
        qty = ctx.equity() * self.risk_fraction / risk
        if qty <= 0.0 or not np.isfinite(qty):
            return
        entry_side = OrderSide.Buy if is_long else OrderSide.Sell
        exit_side = OrderSide.Sell if is_long else OrderSide.Buy
        tp1 = entry + (self.partial_rr * risk if is_long else -self.partial_rr * risk)
        entry_id = ctx.place_market_order(symbol=symbol, side=entry_side, quantity=qty)
        # Protective legs: dormant children of the entry, reduce-only. SL holds the
        # full quantity; the two TPs split it (half to the 2R partial, half rides
        # to the draw on liquidity).
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=exit_side, quantity=qty,
                                        price=stop, parent=entry_id, reduce_only=True)
        st["tp1"] = ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=qty / 2.0,
                                          price=tp1, parent=entry_id, reduce_only=True)
        st["tp2"] = ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=qty / 2.0,
                                          price=target, parent=entry_id, reduce_only=True)
        st["pending"] = entry_id
        st["is_long"] = is_long
        st["entry_qty"] = qty
        st["planned_entry"] = entry
        st["planned_stop"] = stop
        st["planned_tp1"] = tp1
        st["planned_tp2"] = target

    def _reanchor(self, ctx, symbol, st, fill):
        """A gapped fill re-anchors the protective legs proportionally to the
        actual fill price — the pattern the engine docs describe."""
        ratio = fill / st["planned_entry"]
        for leg in ("sl", "tp1", "tp2"):
            if st[leg] is not None:
                ctx.cancel_order(st[leg])
        st["planned_stop"] *= ratio
        st["planned_tp1"] *= ratio
        st["planned_tp2"] *= ratio
        exit_side = OrderSide.Sell if st["is_long"] else OrderSide.Buy
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=exit_side, quantity=st["entry_qty"],
                                        price=st["planned_stop"], parent=st["entry_id"], reduce_only=True)
        st["tp1"] = ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=st["entry_qty"] / 2.0,
                                          price=st["planned_tp1"], parent=st["entry_id"], reduce_only=True)
        st["tp2"] = ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=st["entry_qty"] / 2.0,
                                          price=st["planned_tp2"], parent=st["entry_id"], reduce_only=True)

    def _move_stop_to_breakeven(self, ctx, symbol, st, pos):
        if st["sl"] is not None:
            ctx.cancel_order(st["sl"])
        be = max(st["planned_stop"], pos.price) if st["is_long"] else min(st["planned_stop"], pos.price)
        exit_side = OrderSide.Sell if st["is_long"] else OrderSide.Buy
        st["sl"] = ctx.place_stop_order(symbol=symbol, side=exit_side, quantity=st["entry_qty"],
                                        price=be, parent=st["entry_id"], reduce_only=True)
        st["planned_stop"] = be
