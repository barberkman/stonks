"""Qullamaggie momentum swing — port of app/pines/qullamaggie_momentum_swing.pine (v12).

Signal-first port: every tick the strategy re-evaluates the pine's setup
conditions on the just-confirmed bar and turns them into engine orders.
The breakout setups run in one of two execution models, switched by
`use_bo_vol` (the pine's useBOVol break-bar volume gate):

  gate ON (default)  VIRTUAL arms. A real resting stop cannot check its
                     fill bar's volume — the broker fills on price alone —
                     so nothing is parked. The strategy tracks the armed
                     level and expiry timer itself, evaluates the pine's
                     fill condition (break AND volume >= bo_vol_mult x
                     avgVol50[1]) on each confirmed bar, and enters at
                     market when it holds. Low-volume crossings are
                     skipped exactly like the pine: no fill, no fees.
  gate OFF           A real stop order rests at the level with its bracket
                     children and fills on the first touch — the pine with
                     useBOVol off, and the numbers can be parked at the
                     exchange in advance.

Setups (one position at a time per symbol, like the pine):

  BO        breakout above the pivot high of a tight flag/base
  ORB       breakout above the high of the first N bars of a UTC day
  SHORT_BO  breakdown below the base low in a downtrend (mirror)
  EP        episodic pivot: gap up on volume with a strong close (market)
  PARA      parabolic "first red bar" short (market)

Deviations from the pine (everything else follows it condition for
condition):

  1. bufTicks x syminfo.mintick -> `entry_buffer_bps` (the engine has no
     tick size; bps is scale-free across symbols).
  2. entryMode is stop-at-level only; "Wait for bar close" dropped.
  3. useBOVol switches the execution model (see above) instead of gating
     resting-order fills, which the broker cannot do. Gate-on reproduces
     the pine's trade list; gate-off reproduces its first-touch fills.
  4. use_lod_stop: virtual fills tighten the stop to the break bar's
     low (high for shorts) exactly like the pine — the break bar IS the
     entry bar. The gate-off parked path cannot (its entry bar is unknown
     at arm time), so those stops stay pure ADR-multiples off the level.
     EP always applies it (signal bar = entry bar).
  5. Virtual fills recompute stop/target from the pine's entry reference
     max(open, level) (min for shorts) and the break bar's ADR, like the
     pine. The gate-off parked path keeps SL/TP anchored to the armed
     level so the numbers are known beforehand; the engine still fills
     the parked stop at max(level, open), like the pine.
  6. Partials, the breakeven move, the trail-MA exit, and the PARA
     bounce/time covers are dropped: one stop-loss + one take-profit for
     the whole position (partialRR -> `target_rr`). PARA keeps the pine's
     na target -> stop-loss only.
  7. The pine fills intrabar (breakouts at the level, EP/PARA at the
     close); the engine has no same-bar fills. EP/PARA market orders fill
     at the next open. Virtual breakout fills share this: the signal is
     decided on the confirmed break bar — numbers anchored to its
     max(open, level) — and the market order fills at the NEXT bar's
     open, one bar after the pine's intrabar fill.
  8. Session = UTC calendar day (24/7 crypto data has no sessions).
  9. Sizing and leverage added (the pine is an indicator), following
     app/notes/position-calculator-formulas.md: risk-mode quantity (its §2)
     qty = equity * risk_fraction / (|entry - stop| + entry*fee + stop*fee)
     with both legs taker, so a stop-out loses exactly risk_fraction of
     equity INCLUDING the entry and stop fees. The entry order carries the
     §9 max-safe isolated leverage (the largest L that keeps the
     liquidation price just beyond the stop; floored, one step further
     down on exact integers) capped at `max_leverage` — the margin a trade
     locks is then roughly the dollars it risks, and tight-stop signals
     stay affordable instead of being rejected at 1x.
     maintenance_margin_pct defaults to 0 to match the backtest broker's
     run config. Caveat: a gapped entry fill re-anchors the engine's
     liquidation price off the fill price (slightly past the planned
     stop), but the dollars lost in that path stay roughly the risked
     amount.
 10. The pine's same-bar "filled then collapsed through the stop" bail is
     native here: the broker's settle rounds let the stop child fill on
     the entry bar.
 11. Gate-on: virtual bo/so arms survive EP/PARA holds like the pine's
     boArmed (fills are blocked by canEnter while positioned; the expiry
     timer keeps ticking), the two arms can coexist, and fills follow the
     pine's chain priority BO > ORB > SHORT_BO > EP > PARA. Gate-off keeps
     a one-parked-order-per-symbol model: an EP/PARA market entry cancels
     a still-armed resting entry (the pine leaves it armed, but its fill
     would be rejected as a same-side add anyway), and when several setups
     hold at once the first in priority wins the slot.
 12. Signals print on new arm / level change / market entry / expiry /
     skipped low-volume break / broker rejection — not on every bar a
     setup silently re-arms at the same level.
 13. `require_ma_stack` (off by default) ports the pine's `requireStack`: when
     on it requires the MA stack (sma10>sma20>sma50 long,
     sma10<sma20<sma50 short) as part of the universe trend gate, so it
     constrains only the breakout setups that flow through it — BO / ORB /
     SHORT_BO. EP and PARA ride the liquidity gate and are not affected, exactly
     as in the pine code (its header comment claims EP is gated, but the code
     omits it). Needs 50 bars of history for sma50; those breakout setups are
     suppressed until then.
 14. A rejected market entry consumes the virtual arm (the pine cannot
     reject an entry); the setup re-arms organically while it still holds.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import stonks
from stonks import OrderSide, OrderStatus


def sma(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.mean(a[-n:]))


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def bars_since_highest(a, n):
    """pine `-ta.highestbars(a, n)`: 0 = the current bar; ties resolve to
    the most recent bar."""
    if n <= 0 or len(a) < n:
        return None
    return int(np.argmax(a[-n:][::-1]))


def bars_since_lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return int(np.argmin(a[-n:][::-1]))


def max_safe_leverage(entry, stop, m, cap):
    """notes §9: the largest isolated leverage that keeps the liquidation
    price just beyond the stop (P_liq = S solved for L), floored — one step
    further down when it lands on an exact integer — so P_liq sits strictly
    beyond the stop. Long when stop < entry, short when stop > entry; `m`
    is the maintenance margin rate as a fraction."""
    denom = entry - stop * (1.0 - m) if stop < entry else stop * (1.0 + m) - entry
    if denom <= 0.0:
        return max(1, int(cap))
    raw = entry / denom
    lev = math.floor(raw)
    if lev == raw:
        lev -= 1
    return max(1, min(lev, int(cap)))


@dataclass
class Signal:
    """One actionable trade, generated before its entry condition trades.

    `action` is "arm" (park a resting stop entry at `entry`) or "enter"
    (market entry now; `entry` is the pine's entry reference on the signal
    bar). `target` is None when the setup has no take-profit (PARA).
    `leverage` is the max-safe isolated leverage the entry order will
    carry. `valid_bars` is how long an armed order stays parked after its
    last re-arm (parked BO/SHORT_BO only)."""

    setup: str                  # "BO" | "ORB" | "SHORT_BO" | "EP" | "PARA"
    side: str                   # "long" | "short"
    action: str                 # "arm" | "enter"
    entry: float
    stop: float
    target: Optional[float]
    leverage: int
    valid_bars: Optional[int]


class QMMomentumSwingStrategy(stonks.Strategy):
    # Universe & trend filters
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    require_ma_stack = False

    # Setup 1 — momentum breakout (long)
    enable_bo = True
    base_max_len = 40
    min_base_bars = 3
    max_depth = 40.0
    use_vol_dry = False
    vol_dry_ratio = 1.0
    entry_buffer_bps = 5.0
    order_bars = 10
    use_bo_vol = True
    bo_vol_mult = 1.3

    # Setup 1b — opening range breakout (intraday, long)
    enable_orb = False
    orb_bars = 1

    # Setup 1c — short breakout (breakdown)
    enable_short_bo = False

    # Setup 2 — episodic pivot (gap bar, long)
    enable_ep = True
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True

    # Setup 3 — parabolic short
    enable_para = False
    ps_lookback = 10
    ps_min_gain = 8.0
    ps_streak = 3
    ps_stop_lookback = 3

    # Initial stop / target / sizing / leverage
    adr_stop_mult = 1.0
    use_lod_stop = True
    target_rr = 2.0
    risk_fraction = 0.01
    taker_fee_bps = 5.0
    maintenance_margin_pct = 0.0
    max_leverage = 100.0

    params = {
        "min_price": stonks.Param("minimum close price", unit="$"),
        "min_avg_vol": stonks.Param("minimum 20-bar average volume, 0 = disabled"),
        "adr_len": stonks.Param("ADR length", unit="bars"),
        "min_adr": stonks.Param("minimum average bar range", unit="%"),
        "mom_len": stonks.Param("momentum lookback", unit="bars"),
        "min_gain": stonks.Param("minimum gain over the lookback", unit="%"),
        "require_mas": stonks.Param("require close vs 20 SMA and 10 SMA vs 20 SMA trend gate"),
        "require_ma_stack": stonks.Param(
            "require MA stack 10>20>50 long / inverse short (gates BO/ORB/SHORT_BO)"),
        "enable_bo": stonks.Param("enable long breakout signals"),
        "base_max_len": stonks.Param("max base lookback", unit="bars"),
        "min_base_bars": stonks.Param("min bars since the base high/low", unit="bars"),
        "max_depth": stonks.Param("max retracement inside the base", unit="%"),
        "use_vol_dry": stonks.Param("require volume contraction in the base"),
        "vol_dry_ratio": stonks.Param("5-bar avg vol < ratio x 50-bar avg vol"),
        "entry_buffer_bps": stonks.Param("entry buffer beyond the pivot", unit="bps"),
        "order_bars": stonks.Param("breakout arm good for N bars after its last re-arm", unit="bars"),
        "use_bo_vol": stonks.Param(
            "require volume expansion on the break bar (virtual arm + market entry)"),
        "bo_vol_mult": stonks.Param("break-bar volume >= x 50-bar avg"),
        "enable_orb": stonks.Param("enable opening range breakout (UTC-day session)"),
        "orb_bars": stonks.Param("opening range length", unit="bars"),
        "enable_short_bo": stonks.Param("enable short breakout (breakdown) signals"),
        "enable_ep": stonks.Param("enable episodic pivot signals"),
        "ep_min_gap": stonks.Param("minimum gap up", unit="%"),
        "ep_vol_mult": stonks.Param("bar volume >= multiple of 50-bar avg"),
        "ep_strong_close": stonks.Param("require close > open in the upper half of the range"),
        "enable_para": stonks.Param("enable parabolic first-red-bar short signals"),
        "ps_lookback": stonks.Param("run-up lookback", unit="bars"),
        "ps_min_gain": stonks.Param("minimum run-up over the lookback", unit="%"),
        "ps_streak": stonks.Param("min consecutive up-closes before the reversal"),
        "ps_stop_lookback": stonks.Param("stop above the highest high of the last N bars", unit="bars"),
        "adr_stop_mult": stonks.Param("max initial stop distance", unit="x ADR"),
        "use_lod_stop": stonks.Param("tighten the stop to the entry-bar extreme if closer"),
        "target_rr": stonks.Param("take-profit distance for the whole position", unit="R"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "taker_fee_bps": stonks.Param("fee per sizing leg (entry and stop-loss execute as taker)", unit="bps"),
        "maintenance_margin_pct": stonks.Param("maintenance margin rate in the liquidation formula", unit="%"),
        "max_leverage": stonks.Param("cap on the per-trade isolated leverage", unit="x"),
    }

    indicators = {
        "order_level": stonks.Indicator("armed entry level (parked or virtual)"),
        "stop_level": stonks.Indicator("stop-loss level (in flight or held)"),
        "target_level": stonks.Indicator("take-profit level (in flight or held)"),
    }

    def lookback(self):
        # avgVol50 of the PREVIOUS bar needs 51 bars; gainPct needs close[mom_len];
        # the MA-stack gate needs sma50 -> 50 bars (already covered by 51).
        base = max(self.base_max_len, 51, self.mom_len + 1, self.adr_len)
        return base + 2

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {
            "bar_count": 0, "last_ts": None,
            # entry currently at the broker: a parked stop (gate off) or an
            # in-flight market entry (gate on)
            "pending": None,       # entry OrderID
            "pending_sig": None,   # the Signal it was placed from
            "armed_bar": None,     # bar_count at the last (re)arm
            # virtual arms (gate on): {"level": float, "armed_bar": int}
            "bo_arm": None, "so_arm": None,
            "orb_plot": None,      # active virtual ORB level, for the plot
            # last filled entry's signal, for the level plots while held
            "held_sig": None,
            "was_in_pos": False,
            # PARA: pine's upStk, previous bar's value kept for `frd`
            "up_streak": 0, "prev_up_streak": 0,
            # ORB: opening range of the current UTC day
            "day": None, "or_high": None, "or_count": 0,
            "bars_into_day": 0, "orb_taken": False,
        }

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
            st = self.state.setdefault(symbol, self._fresh())
            ts = int(sub["timestamp"].iloc[-1])
            if st["last_ts"] == ts:
                continue   # symbol did not print a new bar this tick
            st["last_ts"] = ts
            st["bar_count"] += 1
            bars = {k: sub[k].to_numpy()
                    for k in ("open", "high", "low", "close", "volume")}

            self._update_rolling(st, ts, bars)
            in_pos = ctx.position(symbol) is not None
            round_trip = self._sync_orders(ctx, symbol, st, ts, in_pos)
            exited_now = round_trip or (st["was_in_pos"] and not in_pos)
            st["was_in_pos"] = in_pos

            if self.use_bo_vol:
                self._tick_virtual(ctx, symbol, st, ts, bars, in_pos, exited_now)
            else:
                evaluated = not in_pos and not exited_now   # pine: canEnter
                sig = self.signal(st, bars) if evaluated else None
                self._act(ctx, symbol, st, ts, sig, evaluated)
            self._plot_levels(ctx, symbol, st)

    # ─── Per-bar rolling state (needs no window recompute) ───────────────────

    def _update_rolling(self, st, ts, bars):
        c, h = bars["close"], bars["high"]
        # PARA: upStk := close > close[1] ? upStk + 1 : 0
        st["prev_up_streak"] = st["up_streak"]
        if len(c) >= 2 and c[-1] > c[-2]:
            st["up_streak"] += 1
        else:
            st["up_streak"] = 0
        # ORB: opening range of the first orb_bars bars of each UTC day.
        day = ts // 86_400_000
        if day != st["day"]:
            st["day"] = day
            st["or_high"] = float(h[-1])
            st["or_count"] = 1
            st["bars_into_day"] = 0
            st["orb_taken"] = False
        else:
            if st["or_count"] < self.orb_bars:
                st["or_high"] = max(st["or_high"], float(h[-1]))
                st["or_count"] += 1
            st["bars_into_day"] += 1

    # ─── Broker reconciliation ────────────────────────────────────────────────

    def _sync_orders(self, ctx, symbol, st, ts, in_pos):
        """Track the entry order at the broker and expire a stale parked arm.
        Returns True when the entry filled and its bracket already closed the
        position again within the same bar (the pine's exitedNow bail)."""
        round_trip = False
        if st["pending"] is not None:
            entry = ctx.order(st["pending"])
            status = entry.status if entry is not None else OrderStatus.Cancelled
            if status == OrderStatus.Filled:
                sig = st["pending_sig"]
                if in_pos:
                    st["held_sig"] = sig
                else:
                    round_trip = True
                if sig.setup == "ORB" and sig.action == "arm":
                    st["orb_taken"] = True   # pine sets orbTaken only on a real fill
                self._clear_pending(st)
            elif status != OrderStatus.Open:
                if status == OrderStatus.Rejected:
                    self._print(ts, symbol,
                                f"{st['pending_sig'].setup} entry rejected by "
                                "broker (margin + fee exceeded free cash)")
                self._clear_pending(st)
            elif st["pending_sig"].valid_bars is not None \
                    and st["bar_count"] - st["armed_bar"] >= self.order_bars:
                # pine cancels the order boOrderBars bars after the LAST
                # re-arm; fills were possible on bars arm+1 .. arm+N (the
                # broker's fill sweep for this bar ran before on_tick, so
                # >= here equals the pine's strict >).
                ctx.cancel_order(st["pending"])
                self._print(ts, symbol,
                            f"{st['pending_sig'].setup} order expired unfilled "
                            f"@ {st['pending_sig'].entry:.4f}")
                self._clear_pending(st)
        if not in_pos and st["pending"] is None:
            st["held_sig"] = None
        return round_trip

    @staticmethod
    def _clear_pending(st):
        st["pending"] = None
        st["pending_sig"] = None
        st["armed_bar"] = None

    def _act(self, ctx, symbol, st, ts, sig, evaluated=True):
        """Gate-off path: park/refresh/cancel the resting entry, or fire the
        EP/PARA market entries."""
        if sig is None:
            # An ORB arm lapses with its setup (the pine re-checks orbActive
            # every bar) — but only when the setup was actually evaluated.
            if (evaluated and st["pending"] is not None
                    and st["pending_sig"].setup == "ORB"):
                ctx.cancel_order(st["pending"])
                self._print(ts, symbol, "ORB order cancelled (setup lapsed) "
                                        f"@ {st['pending_sig'].entry:.4f}")
                self._clear_pending(st)
            return
        if sig.action == "enter":
            if st["pending"] is not None:   # deviation 11
                ctx.cancel_order(st["pending"])
                self._clear_pending(st)
            self._place(ctx, symbol, st, ts, sig)
            return
        # sig.action == "arm"
        if st["pending"] is not None:
            same = (st["pending_sig"].setup == sig.setup
                    and abs(st["pending_sig"].entry - sig.entry)
                    <= 1e-9 * max(1.0, abs(sig.entry)))
            if same:
                # pine re-arms every bar the setup holds: only the expiry
                # timer resets; the parked bracket stays as printed.
                st["armed_bar"] = st["bar_count"]
                return
            ctx.cancel_order(st["pending"])
            self._clear_pending(st)
        self._place(ctx, symbol, st, ts, sig)

    def _place(self, ctx, symbol, st, ts, sig):
        # notes §2 risk mode: a stop-out loses exactly risk_fraction of
        # equity including the entry and stop-loss fees (both taker).
        fee = self.taker_fee_bps / 10_000.0
        risk_per_unit = abs(sig.entry - sig.stop) + sig.entry * fee + sig.stop * fee
        if risk_per_unit <= 0.0 or not np.isfinite(risk_per_unit):
            return
        qty = ctx.equity() * self.risk_fraction / risk_per_unit
        if qty <= 0.0 or not np.isfinite(qty):
            return
        entry_side = OrderSide.Buy if sig.side == "long" else OrderSide.Sell
        exit_side = OrderSide.Sell if sig.side == "long" else OrderSide.Buy
        if sig.action == "arm":
            entry_id = ctx.place_stop_order(symbol=symbol, side=entry_side,
                                            quantity=qty, price=sig.entry,
                                            leverage=float(sig.leverage))
        else:
            entry_id = ctx.place_market_order(symbol=symbol, side=entry_side,
                                              quantity=qty,
                                              leverage=float(sig.leverage))
        # closing legs ignore their own leverage — the position's is used
        ctx.place_stop_order(symbol=symbol, side=exit_side, quantity=qty,
                             price=sig.stop, parent=entry_id, reduce_only=True)
        if sig.target is not None:
            ctx.place_limit_order(symbol=symbol, side=exit_side, quantity=qty,
                                  price=sig.target, parent=entry_id,
                                  reduce_only=True)
        st["pending"] = entry_id
        st["pending_sig"] = sig
        st["armed_bar"] = st["bar_count"]
        self._print_signal(ts, symbol, sig, qty, qty * sig.entry / sig.leverage)

    def _plot_levels(self, ctx, symbol, st):
        if self.use_bo_vol:
            arm = st["bo_arm"] if st["bo_arm"] is not None else st["so_arm"]
            if arm is not None:
                ctx.plot("order_level", symbol, arm["level"])
            elif st["orb_plot"] is not None:
                ctx.plot("order_level", symbol, st["orb_plot"])
        sig = st["pending_sig"] if st["pending_sig"] is not None else st["held_sig"]
        if sig is None:
            return
        if st["pending_sig"] is not None and sig.action == "arm":
            ctx.plot("order_level", symbol, sig.entry)
        ctx.plot("stop_level", symbol, sig.stop)
        if sig.target is not None:
            ctx.plot("target_level", symbol, sig.target)

    # ─── Virtual arms (gate on): the pine's fill condition, self-evaluated ───

    def _tick_virtual(self, ctx, symbol, st, ts, bars, in_pos, exited_now):
        """pine per-bar order for the volume-gated mode: refresh the arm
        registers, tick the expiry timer, then evaluate the fill chain
        boFill -> orbFill -> soFill -> EP -> PARA on the confirmed bar and
        enter at market. `flight` extends the pine's instant inLong through
        the one-bar market-order limbo."""
        g = self._gates(bars)
        flight = st["pending"] is not None
        can_enter = not in_pos and not exited_now and not flight

        bo_setup = so_setup = False
        if can_enter:
            level = self._bo_setup_level(g, bars)
            if level is not None:
                bo_setup = True
                self._rearm_virtual(st, ts, symbol, "bo_arm", "BO LONG", level)
            level = self._so_setup_level(g, bars)
            if level is not None:
                so_setup = True
                self._rearm_virtual(st, ts, symbol, "so_arm", "SHORT_BO SHORT", level)

        # the timer ticks unconditionally, like the pine's (strict >: the
        # last fillable bar is armed_bar + order_bars, checked below in the
        # same tick after this sweep)
        for slot, setup in (("bo_arm", "BO"), ("so_arm", "SHORT_BO")):
            arm = st[slot]
            if arm is not None and st["bar_count"] - arm["armed_bar"] > self.order_bars:
                self._print(ts, symbol,
                            f"{setup} order expired unfilled @ {arm['level']:.4f}")
                st[slot] = None

        orb_level = self._orb_active_level(g, st) if can_enter else None
        st["orb_plot"] = orb_level
        if not can_enter:
            return

        high_now = float(bars["high"][-1])
        low_now = float(bars["low"][-1])
        vol_ok = self._break_volume_ok(bars)
        sig = None
        if st["bo_arm"] is not None and high_now >= st["bo_arm"]["level"]:
            if vol_ok:
                sig = self._long_fill_signal("BO", st["bo_arm"]["level"], g, bars)
                st["bo_arm"] = None
            else:
                self._print_skip(ts, symbol, "BO", st["bo_arm"]["level"], bars)
        if sig is None and orb_level is not None and high_now >= orb_level:
            if vol_ok:
                sig = self._long_fill_signal("ORB", orb_level, g, bars)
                st["orb_taken"] = True   # pine sets orbTaken at the fill
            else:
                self._print_skip(ts, symbol, "ORB", orb_level, bars)
        if sig is None and st["so_arm"] is not None and low_now <= st["so_arm"]["level"]:
            if vol_ok:
                sig = self._short_fill_signal("SHORT_BO", st["so_arm"]["level"], g, bars)
                st["so_arm"] = None
            else:
                self._print_skip(ts, symbol, "SHORT_BO", st["so_arm"]["level"], bars)
        if sig is None:
            sig = self._ep_signal(g, bars)
        if sig is None and not bo_setup and not so_setup:   # pine psSetup exclusions
            sig = self._para_signal(g, st, bars)
        if sig is not None:
            self._place(ctx, symbol, st, ts, sig)

    def _rearm_virtual(self, st, ts, symbol, slot, label, level):
        arm = st[slot]
        changed = arm is None or \
            abs(arm["level"] - level) > 1e-9 * max(1.0, abs(level))
        st[slot] = {"level": level, "armed_bar": st["bar_count"]}
        if changed:
            self._print(ts, symbol, f"{label} arm virtual @ {level:.4f} | "
                                    f"valid {self.order_bars} bars")

    def _long_fill_signal(self, setup, level, g, bars):
        entry = max(float(bars["open"][-1]), level)   # pine: max(open, level)
        stop = entry * (1.0 - self.adr_stop_mult * g["adr_pct"] / 100.0)
        if self.use_lod_stop:
            stop = max(stop, float(bars["low"][-1]))
        stop = min(stop, entry * 0.999)
        target = entry + self.target_rr * (entry - stop)
        return Signal(setup, "long", "enter", entry, stop, target,
                      self._leverage(entry, stop), None)

    def _short_fill_signal(self, setup, level, g, bars):
        entry = min(float(bars["open"][-1]), level)
        stop = entry * (1.0 + self.adr_stop_mult * g["adr_pct"] / 100.0)
        if self.use_lod_stop:
            stop = min(stop, float(bars["high"][-1]))
        stop = max(stop, entry * 1.001)
        target = entry - self.target_rr * (stop - entry)
        return Signal(setup, "short", "enter", entry, stop, target,
                      self._leverage(entry, stop), None)

    def _print_skip(self, ts, symbol, setup, level, bars):
        v = bars["volume"]
        avg = sma(v[:-1], 50)
        avg_txt = f"{avg:.0f}" if avg is not None else "n/a"
        self._print(ts, symbol,
                    f"{setup} break @ {level:.4f} skipped "
                    f"(vol {float(v[-1]):.0f} < {self.bo_vol_mult:g} x {avg_txt})")

    # ─── Signal generation (the pine, condition for condition) ───────────────

    def _gates(self, bars):
        """The pine's universe/trend gates and shared series on the
        just-confirmed bar. None-valued series behave like pine na: every
        comparison against them is False."""
        o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))
        close_now = float(c[-1])
        sma10 = sma(c, 10)
        sma20 = sma(c, 20)
        sma50 = sma(c, 50)
        adr_pct = sma(100.0 * (h / l - 1.0), self.adr_len)
        avg_vol5 = sma(v, 5)
        avg_vol20 = sma(v, 20)
        avg_vol50 = sma(v, 50)
        avg_vol50_prev = sma(v[:-1], 50)
        gain_pct = (100.0 * (close_now / float(c[-self.mom_len - 1]) - 1.0)
                    if len(c) > self.mom_len else None)

        liq_ok = (avg_vol20 is not None and close_now >= self.min_price
                  and avg_vol20 >= self.min_avg_vol)
        adr_ok = adr_pct is not None and adr_pct >= self.min_adr
        vol_dry_ok = (not self.use_vol_dry) or (
            avg_vol5 is not None and avg_vol50 is not None
            and avg_vol5 < self.vol_dry_ratio * avg_vol50)
        ma_up_ok = (not self.require_mas) or (
            sma20 is not None and close_now > sma20 and sma10 > sma20)
        ma_dn_ok = (not self.require_mas) or (
            sma20 is not None and close_now < sma20 and sma10 < sma20)
        stack_up_ok = (not self.require_ma_stack) or (
            sma50 is not None and sma10 > sma20 > sma50)
        stack_dn_ok = (not self.require_ma_stack) or (
            sma50 is not None and sma10 < sma20 < sma50)
        trend_up = (gain_pct is not None and gain_pct >= self.min_gain
                    and ma_up_ok and stack_up_ok)
        trend_dn = (gain_pct is not None and gain_pct <= -self.min_gain
                    and ma_dn_ok and stack_dn_ok)
        return {
            "close": close_now,
            "buf": self.entry_buffer_bps / 10_000.0,
            "adr_pct": adr_pct,
            "avg_vol50_prev": avg_vol50_prev,
            "liq_ok": liq_ok,
            "vol_dry_ok": vol_dry_ok,
            "universe_up": liq_ok and adr_ok and trend_up,
            "universe_dn": liq_ok and adr_ok and trend_dn,
        }

    def _bo_setup_level(self, g, bars):
        """pine boSetup: the armed level of a tight flag/base, or None."""
        if not (self.enable_bo and g["universe_up"]):
            return None
        h, l = bars["high"], bars["low"]
        base_high = highest(h, self.base_max_len)
        since_pk = bars_since_highest(h, self.base_max_len)
        if base_high is None:
            return None
        pull_low = lowest(l, max(since_pk, 1))
        retrace_pct = 100.0 * (base_high - pull_low) / base_high
        if (since_pk >= self.min_base_bars
                and retrace_pct <= self.max_depth and g["vol_dry_ok"]):
            return base_high * (1.0 + g["buf"])
        return None

    def _so_setup_level(self, g, bars):
        """pine soSetup: the armed level of a base low in a downtrend."""
        if not (self.enable_short_bo and g["universe_dn"]):
            return None
        h, l = bars["high"], bars["low"]
        base_low = lowest(l, self.base_max_len)
        since_tr = bars_since_lowest(l, self.base_max_len)
        if base_low is None:
            return None
        bounce_high = highest(h, max(since_tr, 1))
        retrace_up_pct = 100.0 * (bounce_high - base_low) / base_low
        if (since_tr >= self.min_base_bars
                and retrace_up_pct <= self.max_depth and g["vol_dry_ok"]):
            return base_low * (1.0 - g["buf"])
        return None

    def _orb_active_level(self, g, st):
        """pine orbActive: the opening-range level while the setup holds."""
        if (self.enable_orb and g["universe_up"] and st["or_high"] is not None
                and st["bars_into_day"] >= self.orb_bars and not st["orb_taken"]):
            return st["or_high"] * (1.0 + g["buf"])
        return None

    def _ep_signal(self, g, bars):
        """pine epSetup: gap up on volume with a strong close, entry at the
        close (fills next open, deviation 7)."""
        o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))
        close_now = g["close"]
        if not (self.enable_ep and g["liq_ok"] and len(c) >= 2
                and g["avg_vol50_prev"] is not None and g["adr_pct"] is not None):
            return None
        ep_gap = 100.0 * (float(o[-1]) / float(c[-2]) - 1.0)
        ep_vol_ok = float(v[-1]) >= self.ep_vol_mult * g["avg_vol50_prev"]
        ep_cls_ok = (not self.ep_strong_close) or (
            c[-1] > o[-1] and c[-1] >= (h[-1] + l[-1]) / 2.0)
        risk_ps = close_now - float(l[-1])
        adr_budget = self.adr_stop_mult * g["adr_pct"] / 100.0 * close_now
        if (ep_gap >= self.ep_min_gap and ep_vol_ok and ep_cls_ok
                and 0.0 < risk_ps <= adr_budget):
            entry = close_now
            stop = entry * (1.0 - self.adr_stop_mult * g["adr_pct"] / 100.0)
            if self.use_lod_stop:
                stop = max(stop, float(l[-1]))
            stop = min(stop, entry * 0.999)
            target = entry + self.target_rr * (entry - stop)
            return Signal("EP", "long", "enter", entry, stop, target,
                          self._leverage(entry, stop), None)
        return None

    def _para_signal(self, g, st, bars):
        """pine psSetup: parabolic run-up, first red bar, entry at the close.
        Callers apply the pine's not-boSetup/epSetup/soSetup exclusions."""
        h, l, c = bars["high"], bars["low"], bars["close"]
        if not (self.enable_para and g["liq_ok"] and len(c) >= 2):
            return None
        hi_lb = highest(h, self.ps_lookback)
        lo_lb = lowest(l, self.ps_lookback)
        frd = c[-1] < c[-2] and st["prev_up_streak"] >= self.ps_streak
        if hi_lb is None or lo_lb is None or lo_lb <= 0.0 or not frd:
            return None
        runup = 100.0 * (hi_lb / lo_lb - 1.0)
        stop_ref = highest(h, self.ps_stop_lookback)
        if runup >= self.ps_min_gain and stop_ref is not None:
            stop = stop_ref * (1.0 + g["buf"])
            close_now = g["close"]
            return Signal("PARA", "short", "enter", close_now, stop, None,
                          self._leverage(close_now, stop), None)
        return None

    def signal(self, st, bars):
        """Gate-off path: the trade to park/fire on the just-confirmed bar,
        or None. Priority mirrors the pine's realized behavior: an EP entry
        beats a same-bar arm; resting arms rank BO > ORB > SHORT_BO; PARA
        only fires when no other setup did."""
        g = self._gates(bars)
        sig = self._ep_signal(g, bars)
        if sig is not None:
            return sig
        level = self._bo_setup_level(g, bars)
        if level is not None:
            stop, target = self._long_levels(level, g["adr_pct"])
            return Signal("BO", "long", "arm", level, stop, target,
                          self._leverage(level, stop), self.order_bars)
        level = self._orb_active_level(g, st)
        if level is not None:
            stop, target = self._long_levels(level, g["adr_pct"])
            return Signal("ORB", "long", "arm", level, stop, target,
                          self._leverage(level, stop), None)
        level = self._so_setup_level(g, bars)
        if level is not None:
            stop, target = self._short_levels(level, g["adr_pct"])
            return Signal("SHORT_BO", "short", "arm", level, stop, target,
                          self._leverage(level, stop), self.order_bars)
        return self._para_signal(g, st, bars)

    def _break_volume_ok(self, bars):
        """pine volBreakOK on the current (break) bar:
        volume >= boVolMult * avgVol50[1]; na -> false, like the pine."""
        v = bars["volume"]
        avg_vol50_prev = sma(v[:-1], 50)
        return (avg_vol50_prev is not None
                and float(v[-1]) >= self.bo_vol_mult * avg_vol50_prev)

    def _leverage(self, entry, stop):
        return max_safe_leverage(entry, stop, self.maintenance_margin_pct / 100.0,
                                 self.max_leverage)

    def _long_levels(self, entry, adr_pct):
        stop = entry * (1.0 - self.adr_stop_mult * adr_pct / 100.0)
        stop = min(stop, entry * 0.999)
        target = entry + self.target_rr * (entry - stop)
        return stop, target

    def _short_levels(self, entry, adr_pct):
        stop = entry * (1.0 + self.adr_stop_mult * adr_pct / 100.0)
        stop = max(stop, entry * 1.001)
        target = entry - self.target_rr * (stop - entry)
        return stop, target

    # ─── Signal printing ──────────────────────────────────────────────────────

    def _print_signal(self, ts, symbol, sig, qty, margin):
        stop_pct = 100.0 * (sig.stop / sig.entry - 1.0)
        action = "arm stop-entry" if sig.action == "arm" else "enter market"
        line = (f"{sig.setup} {sig.side.upper()} {action} @ {sig.entry:.4f} | "
                f"SL {sig.stop:.4f} ({stop_pct:+.2f}%)")
        if sig.target is not None:
            target_pct = 100.0 * (sig.target / sig.entry - 1.0)
            line += f" TP {sig.target:.4f} ({target_pct:+.2f}%, {self.target_rr:g}R)"
        else:
            line += " TP none"
        line += f" | qty {qty:.6g} L {sig.leverage}x margin {margin:.2f}"
        if sig.valid_bars is not None:
            line += f" | valid {sig.valid_bars} bars"
        self._print(ts, symbol, line)

    @staticmethod
    def _print(ts, symbol, msg):
        when = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
        print(f"[{when} UTC] {symbol} {msg}", flush=True)
