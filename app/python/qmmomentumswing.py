"""Qullamaggie momentum swing — port of app/pines/qullamaggie_momentum_swing.pine (v12).

Signal-first port: every tick, `signal()` re-evaluates the pine's setup
conditions on the just-confirmed bar and returns the trade the pine would
take (setup name, side, entry, stop, target) BEFORE the entry condition
trades — the pine's resting-order model, so the same numbers can be parked
at the exchange in advance. `on_tick` only reconciles that signal with the
broker: park/refresh/expire the resting stop-entry with its bracket
children, or fire the market entries (EP/PARA), and print each signal as
it is generated.

Setups (one position at a time per symbol, like the pine):

  BO        resting buy-stop at the pivot high of a tight flag/base
  ORB       resting buy-stop at the high of the first N bars of a UTC day
  SHORT_BO  resting sell-stop at the base low in a downtrend (mirror)
  EP        episodic pivot: gap up on volume with a strong close (market)
  PARA      parabolic "first red bar" short (market)

Deviations from the pine (everything else follows it condition for
condition):

  1. bufTicks x syminfo.mintick -> `entry_buffer_bps` (the engine has no
     tick size; bps is scale-free across symbols).
  2. entryMode is stop-at-level only; "Wait for bar close" dropped — the
     whole point here is the resting order.
  3. useBOVol (break-bar volume expansion gate) is enforced by
     fill-check-and-bail: a real resting order cannot inspect the fill
     bar's volume, so when the parked entry fills on a bar whose volume
     misses `bo_vol_mult` x the previous bar's 50-bar average, the
     position is immediately closed at market and the same order is
     re-parked with its original expiry timer — reproducing the pine,
     where such a touch never fills and the order stays armed for a
     later volume-confirmed break. The bail round trip (two taker fees
     plus any gap to the next open) is the price of this emulation.
  4. useLodStop (tighten the stop to the entry-bar extreme) applies only
     to EP, where the signal bar IS the entry bar; for the resting setups
     the entry bar is unknown at arm time, so their stops are pure
     ADR-multiples off the level.
  5. The pine recomputes stop/target from the actual fill (entryPx =
     max(open, level)); here SL/TP stay anchored to the armed level so the
     numbers are known beforehand. The engine still fills the stop entry
     at max(level, open), like the pine.
  6. Partials, the breakeven move, the trail-MA exit, and the PARA
     bounce/time covers are dropped: one stop-loss + one take-profit for
     the whole position (partialRR -> `target_rr`). PARA keeps the pine's
     na target -> stop-loss only.
  7. EP/PARA enter "at the close" in the pine; the engine has no same-bar
     fills, so their market order fills at the next open. The printed
     signal is anchored to the signal bar's close.
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
 11. An EP/PARA market entry cancels a still-armed resting entry (the pine
     leaves it armed, but its fill would be rejected as a same-side add
     anyway). One resting order is parked per symbol at a time; when
     several setups hold at once the first in the pine's fill priority
     (EP entry, then BO > ORB > SHORT_BO) wins the slot.
 12. Signals print on new arm / level change / market entry / expiry /
     volume bail / broker rejection — not on every bar a setup silently
     re-arms at the same level.
 13. `require_ma_stack` (off by default) ports the pine's `requireStack`: when
     on it requires the full MA stack (sma10>sma20>sma50>sma200 long,
     sma10<sma20<sma50<sma200 short) as part of the universe trend gate, so it
     constrains only the breakout setups that flow through it — BO / ORB /
     SHORT_BO. EP and PARA ride the liquidity gate and are not affected, exactly
     as in the pine code (its header comment claims EP is gated, but the code
     omits it). Needs 200 bars of history for sma200; those breakout setups are
     suppressed until then.
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
    (market entry now; `entry` is the signal bar's close). `target` is None
    when the setup has no take-profit (PARA). `leverage` is the max-safe
    isolated leverage the entry order will carry. `valid_bars` is how long
    an armed order stays parked after its last re-arm (BO/SHORT_BO only)."""

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
            "require MA stack 10>20>50>200 long / inverse short (gates BO/ORB/SHORT_BO)"),
        "enable_bo": stonks.Param("enable long breakout signals"),
        "base_max_len": stonks.Param("max base lookback", unit="bars"),
        "min_base_bars": stonks.Param("min bars since the base high/low", unit="bars"),
        "max_depth": stonks.Param("max retracement inside the base", unit="%"),
        "use_vol_dry": stonks.Param("require volume contraction in the base"),
        "vol_dry_ratio": stonks.Param("5-bar avg vol < ratio x 50-bar avg vol"),
        "entry_buffer_bps": stonks.Param("entry buffer beyond the pivot", unit="bps"),
        "order_bars": stonks.Param("resting order good for N bars after its last re-arm", unit="bars"),
        "use_bo_vol": stonks.Param("require volume expansion on the break bar (bail + re-arm when missing)"),
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
        "use_lod_stop": stonks.Param("tighten the EP stop to the signal-bar low if closer"),
        "target_rr": stonks.Param("take-profit distance for the whole position", unit="R"),
        "risk_fraction": stonks.Param("fraction of equity risked per trade"),
        "taker_fee_bps": stonks.Param("fee per sizing leg (entry and stop-loss execute as taker)", unit="bps"),
        "maintenance_margin_pct": stonks.Param("maintenance margin rate in the liquidation formula", unit="%"),
        "max_leverage": stonks.Param("cap on the per-trade isolated leverage", unit="x"),
    }

    indicators = {
        "order_level": stonks.Indicator("armed resting entry level"),
        "stop_level": stonks.Indicator("stop-loss level (armed or held)"),
        "target_level": stonks.Indicator("take-profit level (armed or held)"),
    }

    def lookback(self):
        # avgVol50 of the PREVIOUS bar needs 51 bars; gainPct needs close[mom_len];
        # any enabled MA-stack gate needs sma200 -> 200 bars.
        base = max(self.base_max_len, 51, self.mom_len + 1, self.adr_len)
        if self.require_ma_stack:
            base = max(base, 200)
        return base + 2

    def on_start(self, ctx):
        self.state = {}

    def _fresh(self):
        return {
            "bar_count": 0, "last_ts": None,
            # resting/in-flight entry currently at the broker
            "pending": None,       # entry OrderID
            "pending_sig": None,   # the Signal it was placed from
            "armed_bar": None,     # bar_count at the last (re)arm
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
            round_trip = self._sync_orders(ctx, symbol, st, ts, in_pos, bars)
            exited_now = round_trip or (st["was_in_pos"] and not in_pos)
            st["was_in_pos"] = in_pos

            sig = None
            evaluated = not in_pos and not exited_now   # pine: canEnter
            if evaluated:
                sig = self.signal(st, bars)
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

    def _sync_orders(self, ctx, symbol, st, ts, in_pos, bars):
        """Track the parked entry and expire a stale arm. Returns True when
        the entry filled and its bracket already closed the position again
        within the same bar (the pine's exitedNow bail)."""
        round_trip = False
        if st["pending"] is not None:
            entry = ctx.order(st["pending"])
            status = entry.status if entry is not None else OrderStatus.Cancelled
            if status == OrderStatus.Filled:
                sig = st["pending_sig"]
                if (in_pos and sig.action == "arm" and self.use_bo_vol
                        and not self._break_volume_ok(bars)):
                    # pine's volBreakOK failed on the break bar: that touch
                    # never fills there and the order stays armed. Emulate:
                    # bail out at market and re-park the same bracket with
                    # the ORIGINAL expiry timer.
                    pos = ctx.position(symbol)
                    exit_side = OrderSide.Sell if pos.quantity > 0 else OrderSide.Buy
                    ctx.place_market_order(symbol=symbol, side=exit_side,
                                           quantity=abs(pos.quantity),
                                           parent=st["pending"], reduce_only=True)
                    self._print(ts, symbol,
                                f"{sig.setup} break lacked volume — bail, "
                                f"re-arm @ {sig.entry:.4f}")
                    saved_armed_bar = st["armed_bar"]
                    self._clear_pending(st)
                    st["held_sig"] = None
                    self._place(ctx, symbol, st, ts, sig)
                    st["armed_bar"] = saved_armed_bar   # pine keeps the old timer
                    return round_trip
                if in_pos:
                    st["held_sig"] = sig
                else:
                    round_trip = True
                if sig.setup == "ORB":
                    st["orb_taken"] = True   # pine sets orbTaken only on a real fill
                self._clear_pending(st)
            elif status != OrderStatus.Open:
                if status == OrderStatus.Rejected:
                    self._print(ts, symbol,
                                f"{st['pending_sig'].setup} entry rejected by "
                                "broker (margin + fee exceeded free cash)")
                self._clear_pending(st)
            elif st["pending_sig"].setup in ("BO", "SHORT_BO") \
                    and st["bar_count"] - st["armed_bar"] >= self.order_bars:
                # pine cancels the order boOrderBars bars after the LAST
                # re-arm; fills were possible on bars arm+1 .. arm+N.
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
        if sig is None:
            # An ORB arm lapses with its setup (the pine re-checks orbActive
            # every bar) — but only when the setup was actually evaluated;
            # while a volume-bail is in flight the re-armed order must live.
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
        sig = st["pending_sig"] if st["pending_sig"] is not None else st["held_sig"]
        if sig is None:
            return
        if st["pending_sig"] is not None:
            ctx.plot("order_level", symbol, sig.entry)
        ctx.plot("stop_level", symbol, sig.stop)
        if sig.target is not None:
            ctx.plot("target_level", symbol, sig.target)

    # ─── Signal generation (the pine, condition for condition) ───────────────

    def signal(self, st, bars):
        """Evaluate the pine's setups on the just-confirmed bar and return the
        trade to park/fire, or None. Priority mirrors the pine's realized
        behavior: an EP entry beats a same-bar arm; resting arms rank
        BO > ORB > SHORT_BO; PARA only fires when no other setup did."""
        o, h, l, c, v = (bars[k] for k in ("open", "high", "low", "close", "volume"))
        close_now = float(c[-1])
        buf = self.entry_buffer_bps / 10_000.0

        sma10 = sma(c, 10)
        sma20 = sma(c, 20)
        sma50 = sma(c, 50)
        sma200 = sma(c, 200)
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
            sma200 is not None and sma10 > sma20 > sma50 > sma200)
        stack_dn_ok = (not self.require_ma_stack) or (
            sma200 is not None and sma10 < sma20 < sma50 < sma200)
        trend_up = (gain_pct is not None and gain_pct >= self.min_gain
                    and ma_up_ok and stack_up_ok)
        trend_dn = (gain_pct is not None and gain_pct <= -self.min_gain
                    and ma_dn_ok and stack_dn_ok)
        universe_up = liq_ok and adr_ok and trend_up
        universe_dn = liq_ok and adr_ok and trend_dn

        # ── EP: gap up on volume with a strong close, entry at the close ──
        if (self.enable_ep and liq_ok and len(c) >= 2
                and avg_vol50_prev is not None and adr_pct is not None):
            ep_gap = 100.0 * (float(o[-1]) / float(c[-2]) - 1.0)
            ep_vol_ok = float(v[-1]) >= self.ep_vol_mult * avg_vol50_prev
            ep_cls_ok = (not self.ep_strong_close) or (
                c[-1] > o[-1] and c[-1] >= (h[-1] + l[-1]) / 2.0)
            risk_ps = close_now - float(l[-1])
            adr_budget = self.adr_stop_mult * adr_pct / 100.0 * close_now
            if (ep_gap >= self.ep_min_gap and ep_vol_ok and ep_cls_ok
                    and 0.0 < risk_ps <= adr_budget):
                entry = close_now
                stop = entry * (1.0 - self.adr_stop_mult * adr_pct / 100.0)
                if self.use_lod_stop:
                    stop = max(stop, float(l[-1]))
                stop = min(stop, entry * 0.999)
                target = entry + self.target_rr * (entry - stop)
                return Signal("EP", "long", "enter", entry, stop, target,
                              self._leverage(entry, stop), None)

        # ── BO: resting buy-stop at the pivot high of a tight flag/base ──
        if self.enable_bo and universe_up:
            base_high = highest(h, self.base_max_len)
            since_pk = bars_since_highest(h, self.base_max_len)
            if base_high is not None:
                pull_low = lowest(l, max(since_pk, 1))
                retrace_pct = 100.0 * (base_high - pull_low) / base_high
                if (since_pk >= self.min_base_bars
                        and retrace_pct <= self.max_depth and vol_dry_ok):
                    entry = base_high * (1.0 + buf)
                    stop, target = self._long_levels(entry, adr_pct)
                    return Signal("BO", "long", "arm", entry, stop, target,
                                  self._leverage(entry, stop), self.order_bars)

        # ── ORB: resting buy-stop at the opening range high of the UTC day ──
        if (self.enable_orb and universe_up and st["or_high"] is not None
                and st["bars_into_day"] >= self.orb_bars and not st["orb_taken"]):
            entry = st["or_high"] * (1.0 + buf)
            stop, target = self._long_levels(entry, adr_pct)
            return Signal("ORB", "long", "arm", entry, stop, target,
                          self._leverage(entry, stop), None)

        # ── SHORT_BO: resting sell-stop at the base low in a downtrend ──
        if self.enable_short_bo and universe_dn:
            base_low = lowest(l, self.base_max_len)
            since_tr = bars_since_lowest(l, self.base_max_len)
            if base_low is not None:
                bounce_high = highest(h, max(since_tr, 1))
                retrace_up_pct = 100.0 * (bounce_high - base_low) / base_low
                if (since_tr >= self.min_base_bars
                        and retrace_up_pct <= self.max_depth and vol_dry_ok):
                    entry = base_low * (1.0 - buf)
                    stop, target = self._short_levels(entry, adr_pct)
                    return Signal("SHORT_BO", "short", "arm", entry, stop, target,
                                  self._leverage(entry, stop), self.order_bars)

        # ── PARA: parabolic run-up, first red bar, entry at the close ──
        if self.enable_para and liq_ok and len(c) >= 2:
            hi_lb = highest(h, self.ps_lookback)
            lo_lb = lowest(l, self.ps_lookback)
            frd = c[-1] < c[-2] and st["prev_up_streak"] >= self.ps_streak
            if hi_lb is not None and lo_lb is not None and lo_lb > 0.0 and frd:
                runup = 100.0 * (hi_lb / lo_lb - 1.0)
                stop_ref = highest(h, self.ps_stop_lookback)
                if runup >= self.ps_min_gain and stop_ref is not None:
                    stop = stop_ref * (1.0 + buf)
                    return Signal("PARA", "short", "enter", close_now, stop, None,
                                  self._leverage(close_now, stop), None)
        return None

    def _break_volume_ok(self, bars):
        """pine volBreakOK on the current (fill) bar:
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
