"""Qullamaggie momentum swing — the 1:1 SIGNAL PRINTER port of the pine.

Ports app/pines/qullamaggie_momentum_swing.pine (v12) whole: all five setups,
every input, and the pine's own hypothetical one-long/one-short book — but as
a pure signal indicator. It NEVER places an engine order; it simulates the
book per symbol exactly like the pine does on a chart and

  * prints ADVANCE notice at each bar's close of every entry that could fire
    on a later bar — the armed BO/SBO stop levels with their TTL, the ORB
    level, the exact EP gap-open/volume thresholds for the next bar, and the
    PARA "close below X" trigger — so a manual trader can park real orders at
    a broker before the break happens;
  * prints the entry / partial / exit events with entry, stop and target
    prices when the simulated book acts (the pine's arrows + value labels);
  * publishes the pine's chart overlays (resting levels, the active trade's
    entry/stop/target lines, the trailing MA) via ctx.plot.

Every signal is also appended to `self.signals` (a list of Signal records,
never trimmed) so tests and tooling read the stream without parsing stdout;
`print_signals` gates only the console lines. `show_order_levels` and
`show_trail_ma` keep their pine meaning and gate only the chart plots.

Advance-notice exactness: the BO/SBO/ORB levels, their break-bar volume
thresholds, and the EP minimum-open/volume thresholds are exact for a fill
on the NEXT bar (the pine evaluates those against the prior bar's 50-bar
average volume, which is fully known today; the lines reprint every bar, so
the latest line always carries the exact numbers). The PARA stop preview is
approximate — it finalizes on the signal bar, whose own high it includes.

What is reinterpreted (same conventions as the sibling qm* ports):
  * pine's "buffer ticks" becomes a fraction-of-price buffer (no tick size).
  * session.isfirstbar_regular becomes a UTC-day boundary, like qmorb.py.
  * ta.ema is seeded with the SMA of the first n window values.
  * arrows/labels/alerts become print lines; plot() becomes ctx.plot().
  * pine's `enableShort` input is named `enable_para` here (it toggles the
    parabolic short and would read ambiguously next to `enable_sbo`).

Faithful pine oddities kept on purpose:
  * armed BO/SBO orders live their own life — they keep refreshing, printing
    and expiring while an unrelated position is open; they just cannot fill
    until the book is flat again (the pine's canEnter gate).
  * the ORB level still prints/plots on the very bar it fills (the pine
    computes orbActive before the entry block and never reassigns it).

One deliberate deviation: the pine's parabolic entry never resets sTargetPx,
so a stale target from an earlier short-breakout trade can leak into its
value labels. This port always reports the parabolic short with no target.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import stonks

MS_PER_DAY = 86_400_000


# ─── Stateless TA helpers (recomputed from the pulled window each tick) ───────
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


def highest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.max(a[-n:]))


def lowest(a, n):
    if n <= 0 or len(a) < n:
        return None
    return float(np.min(a[-n:]))


def since_extreme(window, use_max):
    """(extreme value, bars since it), keeping the most RECENT tie — the
    pine's ta.highestbars / ta.lowestbars convention."""
    ext = window[0]
    pos = 0
    for i in range(len(window)):
        better = window[i] >= ext if use_max else window[i] <= ext
        if better:
            ext = window[i]
            pos = i
    return float(ext), (len(window) - 1) - pos


def up_streak_at(cl, end_idx):
    """Consecutive up-closes ending at end_idx inclusive — the pine's upStk
    as read on that bar."""
    n = 0
    i = end_idx
    while i >= 1 and cl[i] > cl[i - 1]:
        n += 1
        i -= 1
    return n


def _fmt_ts(ms):
    dt = datetime.datetime.fromtimestamp(ms / 1000.0, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Signal:
    """One emitted signal: an advance notice (arm_bo / arm_so / arm_orb /
    watch_ep / watch_para) or a book event (entry_bo / entry_orb / entry_sbo /
    entry_ep / entry_para / partial / exit). `text` is the rendered console
    line without the leading "[timestamp] symbol:" prefix."""

    timestamp: int
    symbol: str
    kind: str
    side: str = ""
    price: Optional[float] = None
    stop: Optional[float] = None
    target: Optional[float] = None
    ttl_bars: Optional[int] = None
    vol_threshold: Optional[float] = None
    reason: str = ""
    text: str = ""


class QMSwingStrategy(stonks.Strategy):
    # Universe & trend filters (pine grpU)
    min_price = 5.0
    min_avg_vol = 0.0
    adr_len = 20
    min_adr = 0.1
    mom_len = 24
    min_gain = 0.5
    require_mas = True
    # Setup 1 — momentum breakout, long (pine grpB)
    enable_bo = True
    base_max_len = 40
    min_base_days = 3
    max_depth = 40.0
    use_vol_dry = False
    vol_dry_ratio = 1.0
    entry_buffer_bps = 5.0
    wait_for_close = False
    order_ttl_bars = 10
    show_order_levels = True
    use_bo_vol = True
    bo_vol_mult = 1.3
    # Setup 1b — opening range breakout (pine grpO)
    enable_orb = False
    orb_bars = 1
    # Setup 1c — short breakout / breakdown (pine grpSB)
    enable_sbo = False
    # Setup 2 — episodic pivot (pine grpE)
    enable_ep = True
    ep_min_gap = 0.5
    ep_vol_mult = 1.3
    ep_strong_close = True
    # Setup 3 — parabolic short (pine grpS; pine calls the toggle enableShort)
    enable_para = False
    ps_lookback = 10
    ps_min_gain = 8.0
    ps_streak = 3
    ps_stop_lb = 3
    ps_max_hold = 5
    # Initial stop (pine grpR)
    adr_stop_mult = 1.0
    use_lod_stop = True
    # Trade management (pine grpX)
    partial_rr = 2.0
    partial_bars = 6
    move_be = True
    trail_use_ema = True
    trail_len = 20
    show_trail_ma = True
    # Printer (not in the pine)
    print_signals = True

    params = {
        "min_price": stonks.Param("minimum price floor for the universe gate", unit="$"),
        "min_avg_vol": stonks.Param("minimum 20-bar average volume; 0 disables the floor", unit="shares"),
        "adr_len": stonks.Param("ADR (average bar range) lookback", unit="bars"),
        "min_adr": stonks.Param("minimum ADR / average bar range", unit="%"),
        "mom_len": stonks.Param("momentum lookback", unit="bars"),
        "min_gain": stonks.Param("minimum gain (longs) / decline (shorts) over the lookback", unit="%"),
        "require_mas": stonks.Param("require close vs 20-SMA and 10-SMA vs 20-SMA trend alignment"),
        "enable_bo": stonks.Param("enable the long breakout setup"),
        "base_max_len": stonks.Param("max lookback for the base pivot", unit="bars"),
        "min_base_days": stonks.Param("min bars since the base high/low before arming", unit="bars"),
        "max_depth": stonks.Param("max retracement inside the base", unit="%"),
        "use_vol_dry": stonks.Param("require volume contraction inside the base"),
        "vol_dry_ratio": stonks.Param("5-bar avg volume < ratio x 50-bar avg volume", unit="x"),
        "entry_buffer_bps": stonks.Param("entry buffer beyond the level (pine's ticks, as bps of price)", unit="bps"),
        "wait_for_close": stonks.Param("require the bar to CLOSE beyond the level instead of a touch"),
        "order_ttl_bars": stonks.Param("resting BO/SBO order lifetime before it expires", unit="bars"),
        "show_order_levels": stonks.Param("plot the live resting BO/ORB/SBO levels"),
        "use_bo_vol": stonks.Param("require volume expansion on the break bar"),
        "bo_vol_mult": stonks.Param("break-bar volume >= multiple of the 50-bar average", unit="x"),
        "enable_orb": stonks.Param("enable the opening range breakout setup (intraday)"),
        "orb_bars": stonks.Param("opening range length, bars from the UTC-day open", unit="bars"),
        "enable_sbo": stonks.Param("enable the short breakout (breakdown) setup"),
        "enable_ep": stonks.Param("enable the episodic pivot setup"),
        "ep_min_gap": stonks.Param("minimum gap up vs the prior close", unit="%"),
        "ep_vol_mult": stonks.Param("gap-bar volume >= multiple of the 50-bar average", unit="x"),
        "ep_strong_close": stonks.Param("require a strong close (green, upper half of the range)"),
        "enable_para": stonks.Param("enable the parabolic short setup (pine: enableShort)"),
        "ps_lookback": stonks.Param("parabolic run-up lookback", unit="bars"),
        "ps_min_gain": stonks.Param("minimum run-up over the lookback", unit="%"),
        "ps_streak": stonks.Param("min consecutive up-closes before the reversal", unit="bars"),
        "ps_stop_lb": stonks.Param("stop above the highest high of the last N bars", unit="bars"),
        "ps_max_hold": stonks.Param("parabolic short max holding period", unit="bars"),
        "adr_stop_mult": stonks.Param("max initial stop distance, in ADRs", unit="ADR"),
        "use_lod_stop": stonks.Param("tighten the stop to the entry bar's extreme if closer"),
        "partial_rr": stonks.Param("partial profit target, in R multiples", unit="R"),
        "partial_bars": stonks.Param("...or take the partial after N bars", unit="bars"),
        "move_be": stonks.Param("move the stop to breakeven after the partial"),
        "trail_use_ema": stonks.Param("trail with an EMA (on) or an SMA (off)"),
        "trail_len": stonks.Param("trailing MA length", unit="bars"),
        "show_trail_ma": stonks.Param("plot the trailing MA the exits check"),
        "print_signals": stonks.Param("print signal lines to stdout (self.signals always records)"),
    }

    indicators = {
        "bo_level": stonks.Indicator("resting BO buy-stop (pivot high + buffer)", color="#4caf50"),
        "orb_level": stonks.Indicator("resting ORB level (opening-range high + buffer)", color="#00e676"),
        "so_level": stonks.Indicator("resting SBO sell-stop (base low - buffer)", color="#ff5252"),
        "entry_level": stonks.Indicator("active trade entry price", color="#2962ff"),
        "stop_level": stonks.Indicator("active trade stop / exit price", color="#ff5252"),
        "target_level": stonks.Indicator("active trade partial / target price", color="#ff9800"),
        "trail_ma": stonks.Indicator("trailing MA the post-partial exits check", color="#2962ff"),
    }

    def lookback(self):
        return max(self.base_max_len, 51, self.mom_len, self.adr_len,
                   self.ps_lookback, self.ps_streak + 2, self.ps_stop_lb,
                   3 * self.trail_len) + 5

    def on_start(self, ctx):
        self.state = {}
        self.signals = []

    def _fresh(self):
        return {"bar_count": 0,
                # long book (pine: inLong, entryPx, stopPx, targetPx, entryBar, partialDone)
                "in_long": False, "entry_px": None, "stop_px": None, "target_px": None,
                "entry_bar": None, "partial_done": False,
                # short book (pine: inShort, shortType, sEntryPx, sStopPx, sTargetPx, ...)
                "in_short": False, "short_type": None, "s_entry_px": None, "s_stop_px": None,
                "s_target_px": None, "s_entry_bar": None, "s_partial_done": False,
                # resting orders (pine: boArmed/boOrderPx/boArmedBar, soArmed/...)
                "bo_armed": False, "bo_order_px": None, "bo_armed_bar": 0,
                "so_armed": False, "so_order_px": None, "so_armed_bar": 0,
                # ORB session (pine: orHigh/orCount/orbTaken, UTC-day reading)
                "day": None, "day_bars": 0, "or_high": None, "orb_taken_day": None}

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
            now = int(ts[-1])

            # Session bookkeeping for the ORB leg (runs from the first bar).
            day = now // MS_PER_DAY
            if st["day"] is None or day != st["day"]:
                st["day"] = day
                st["day_bars"] = 1
                st["or_high"] = float(hi[-1])
            else:
                st["day_bars"] += 1
                if st["day_bars"] <= self.orb_bars:
                    st["or_high"] = max(st["or_high"], float(hi[-1]))

            if len(cl) < self.lookback():
                continue
            self._process(ctx, symbol, st, op, hi, lo, cl, vo, now, day)

    def _process(self, ctx, symbol, st, op, hi, lo, cl, vo, now, day):
        buf = self.entry_buffer_bps / 10_000.0

        # ── Core calculations (pine 97-108) ─────────────────────────────────
        s10 = sma(cl, 10)
        s20 = sma(cl, 20)
        trail = ema(cl, self.trail_len) if self.trail_use_ema else sma(cl, self.trail_len)
        adr = adr_pct(hi, lo, self.adr_len)
        gain = gain_pct(cl, self.mom_len)
        avg_vol5 = sma(vo, 5)
        avg_vol20 = sma(vo, 20)
        avg_vol50_now = sma(vo, 50)          # tomorrow's avgVol50[1], known today
        avg_vol50_prev = sma(vo[:-1], 50)    # today's avgVol50[1]

        # ── Universe & trend gates (pine 110-118) ───────────────────────────
        liq_ok = cl[-1] >= self.min_price and avg_vol20 >= self.min_avg_vol
        adr_ok = adr >= self.min_adr
        vol_dry_ok = not self.use_vol_dry or avg_vol5 < self.vol_dry_ratio * avg_vol50_now
        ma_ok_up = not self.require_mas or (cl[-1] > s20 and s10 > s20)
        ma_ok_dn = not self.require_mas or (cl[-1] < s20 and s10 < s20)
        universe_up = liq_ok and adr_ok and gain >= self.min_gain and ma_ok_up
        universe_dn = liq_ok and adr_ok and gain <= -self.min_gain and ma_ok_dn

        # The pine's setup formulas read the position state left over from the
        # PREVIOUS bar: they sit before the exit-management block in script
        # order. Snapshot it; exits/entries mutate the live state afterward.
        flat_prior = not st["in_long"] and not st["in_short"]

        # ── Long base / setup (pine 138-144) ────────────────────────────────
        base_high, since_pk = since_extreme(hi[-self.base_max_len:], use_max=True)
        pull_low = lowest(lo, max(since_pk, 1))
        flag_up = (since_pk >= self.min_base_days and base_high > 0.0
                   and 100.0 * (base_high - pull_low) / base_high <= self.max_depth
                   and vol_dry_ok)
        entry_level = base_high * (1.0 + buf)
        bo_setup = self.enable_bo and universe_up and flag_up and flat_prior

        # ── Short base / setup, mirror (pine 147-153) ───────────────────────
        base_low, since_tr = since_extreme(lo[-self.base_max_len:], use_max=False)
        bounce_high = highest(hi, max(since_tr, 1))
        flag_dn = (since_tr >= self.min_base_days and base_low > 0.0
                   and 100.0 * (bounce_high - base_low) / base_low <= self.max_depth
                   and vol_dry_ok)
        entry_level_dn = base_low * (1.0 - buf)
        so_setup = self.enable_sbo and universe_dn and flag_dn and flat_prior

        # ── Episodic pivot (pine 156-161) ───────────────────────────────────
        ep_gap_ok = cl[-2] > 0.0 and 100.0 * (op[-1] / cl[-2] - 1.0) >= self.ep_min_gap
        ep_vol_ok = vo[-1] >= self.ep_vol_mult * avg_vol50_prev
        ep_cls_ok = not self.ep_strong_close or (cl[-1] > op[-1] and cl[-1] >= (hi[-1] + lo[-1]) / 2.0)
        ep_risk = cl[-1] - lo[-1]
        ep_within = 0.0 < ep_risk <= self.adr_stop_mult * adr / 100.0 * cl[-1]
        ep_setup = (self.enable_ep and liq_ok and ep_gap_ok and ep_vol_ok
                    and ep_cls_ok and ep_within and flat_prior)

        # ── Parabolic short (pine 164-169) ──────────────────────────────────
        streak_prev = up_streak_at(cl, len(cl) - 2)   # pine's upStk[1]
        streak_now = up_streak_at(cl, len(cl) - 1)    # today's, for the watch line
        ps_ll = lowest(lo, self.ps_lookback)
        ps_runup = 100.0 * (highest(hi, self.ps_lookback) / ps_ll - 1.0) if ps_ll > 0.0 else 0.0
        ps_stop_level = highest(hi, self.ps_stop_lb) * (1.0 + buf)
        frd = cl[-1] < cl[-2] and streak_prev >= self.ps_streak
        ps_setup = (self.enable_para and liq_ok and ps_runup >= self.ps_min_gain and frd
                    and not bo_setup and not ep_setup and not so_setup and flat_prior)

        # ── Opening range breakout (pine 172-184) — computed ONCE, before the
        # exit/entry blocks, and never refreshed: it is not a pine `var`. ─────
        orb_active = (self.enable_orb and universe_up and st["or_high"] is not None
                      and st["day_bars"] - 1 >= self.orb_bars
                      and st["orb_taken_day"] != day and flat_prior)
        orb_level = st["or_high"] * (1.0 + buf) if st["or_high"] is not None else None

        # ── Resting long order: arm/refresh, then TTL expiry (pine 187-196) ──
        if bo_setup:
            st["bo_armed"] = True
            st["bo_order_px"] = float(entry_level)
            st["bo_armed_bar"] = st["bar_count"]
        if st["bo_armed"] and st["bar_count"] - st["bo_armed_bar"] > self.order_ttl_bars:
            st["bo_armed"] = False
            st["bo_order_px"] = None

        # ── Resting short order, mirror (pine 199-208) ──────────────────────
        if so_setup:
            st["so_armed"] = True
            st["so_order_px"] = float(entry_level_dn)
            st["so_armed_bar"] = st["bar_count"]
        if st["so_armed"] and st["bar_count"] - st["so_armed_bar"] > self.order_ttl_bars:
            st["so_armed"] = False
            st["so_order_px"] = None

        exited_now = False

        # ── Long exit management (pine 244-269) ─────────────────────────────
        if st["in_long"] and st["bar_count"] > st["entry_bar"]:
            close_long = False
            if lo[-1] <= st["stop_px"]:
                self._emit(symbol, now, "exit", side="long", price=st["stop_px"],
                           reason="stop", text=f"long exit (stop) @ {st['stop_px']:.6g}")
                close_long = True
            else:
                if not st["partial_done"] and (hi[-1] >= st["target_px"]
                                               or st["bar_count"] - st["entry_bar"] >= self.partial_bars):
                    reason = "target" if hi[-1] >= st["target_px"] else "time"
                    price = st["target_px"] if reason == "target" else float(cl[-1])
                    st["partial_done"] = True
                    be_txt = "; stop unchanged"
                    if self.move_be:
                        st["stop_px"] = max(st["stop_px"], st["entry_px"])
                        be_txt = f"; stop -> breakeven {st['stop_px']:.6g}"
                    self._emit(symbol, now, "partial", side="long", price=price,
                               stop=st["stop_px"], reason=reason,
                               text=f"long partial ({reason}) near {price:.6g}{be_txt}")
                if st["partial_done"] and cl[-1] < trail:
                    self._emit(symbol, now, "exit", side="long", price=float(cl[-1]),
                               reason="trail",
                               text=f"long exit (trail) @ {cl[-1]:.6g} — closed below the trail MA {trail:.6g}")
                    close_long = True
            if close_long:
                self._reset_long(st)
                exited_now = True

        # ── Short exit management (pine 272-303) ────────────────────────────
        if st["in_short"] and st["bar_count"] > st["s_entry_bar"]:
            close_short = False
            if st["short_type"] == "PARA":
                if (hi[-1] >= st["s_stop_px"] or cl[-1] > cl[-2]
                        or st["bar_count"] - st["s_entry_bar"] >= self.ps_max_hold):
                    if hi[-1] >= st["s_stop_px"]:
                        reason, price = "stop", st["s_stop_px"]
                    elif cl[-1] > cl[-2]:
                        reason, price = "bounce", float(cl[-1])
                    else:
                        reason, price = "time", float(cl[-1])
                    self._emit(symbol, now, "exit", side="short", price=price, reason=reason,
                               text=f"parabolic cover ({reason}) @ {price:.6g}")
                    close_short = True
            else:
                if hi[-1] >= st["s_stop_px"]:
                    self._emit(symbol, now, "exit", side="short", price=st["s_stop_px"],
                               reason="stop", text=f"short cover (stop) @ {st['s_stop_px']:.6g}")
                    close_short = True
                else:
                    if not st["s_partial_done"] and (lo[-1] <= st["s_target_px"]
                                                     or st["bar_count"] - st["s_entry_bar"] >= self.partial_bars):
                        reason = "target" if lo[-1] <= st["s_target_px"] else "time"
                        price = st["s_target_px"] if reason == "target" else float(cl[-1])
                        st["s_partial_done"] = True
                        be_txt = "; stop unchanged"
                        if self.move_be:
                            st["s_stop_px"] = min(st["s_stop_px"], st["s_entry_px"])
                            be_txt = f"; stop -> breakeven {st['s_stop_px']:.6g}"
                        self._emit(symbol, now, "partial", side="short", price=price,
                                   stop=st["s_stop_px"], reason=reason,
                                   text=f"short partial cover ({reason}) near {price:.6g}{be_txt}")
                    if st["s_partial_done"] and cl[-1] > trail:
                        self._emit(symbol, now, "exit", side="short", price=float(cl[-1]),
                                   reason="trail",
                                   text=f"short cover (trail) @ {cl[-1]:.6g} — closed above the trail MA {trail:.6g}")
                        close_short = True
            if close_short:
                self._reset_short(st)
                exited_now = True

        # ── Entries (pine 305-392) ──────────────────────────────────────────
        can_enter = not st["in_long"] and not st["in_short"] and not exited_now
        vol_break_ok = not self.use_bo_vol or vo[-1] >= self.bo_vol_mult * avg_vol50_prev

        def broke_up(level):
            return cl[-1] >= level if self.wait_for_close else hi[-1] >= level

        def broke_dn(level):
            return cl[-1] <= level if self.wait_for_close else lo[-1] <= level

        bo_fill = (can_enter and st["bo_armed"] and st["bo_order_px"] is not None
                   and broke_up(st["bo_order_px"]) and vol_break_ok)
        orb_fill = (can_enter and orb_active and orb_level is not None
                    and broke_up(orb_level) and vol_break_ok and not bo_fill)
        so_fill = (can_enter and st["so_armed"] and st["so_order_px"] is not None
                   and broke_dn(st["so_order_px"]) and vol_break_ok)

        if bo_fill or orb_fill:
            level = st["bo_order_px"] if bo_fill else orb_level
            entry = max(float(op[-1]), level)   # resting stop fills at the level, or the open on a gap
            stop, target = self._long_levels(entry, adr, lo[-1])
            st.update(in_long=True, entry_px=entry, stop_px=stop, target_px=target,
                      entry_bar=st["bar_count"], partial_done=False)
            if bo_fill:
                st["bo_armed"] = False
                kind, label = "entry_bo", "Buy BO (stop filled)"
            else:
                st["orb_taken_day"] = day
                kind, label = "entry_orb", "Buy ORB (stop filled)"
            self._emit(symbol, now, kind, side="long", price=entry, stop=stop, target=target,
                       text=f"{label}: entry={entry:.6g} stop={stop:.6g} target={target:.6g}")
            if cl[-1] < stop:   # filled then collapsed below the stop same bar — bail
                self._emit(symbol, now, "exit", side="long", price=float(cl[-1]),
                           reason="collapse",
                           text=f"long exit (collapse) @ {cl[-1]:.6g} — closed through the fresh stop {stop:.6g}")
                self._reset_long(st)
        elif so_fill:
            entry = min(float(op[-1]), st["so_order_px"])
            stop, target = self._short_levels(entry, adr, hi[-1])
            st.update(in_short=True, short_type="BO", s_entry_px=entry, s_stop_px=stop,
                      s_target_px=target, s_entry_bar=st["bar_count"], s_partial_done=False)
            st["so_armed"] = False
            self._emit(symbol, now, "entry_sbo", side="short", price=entry, stop=stop, target=target,
                       text=f"Short BO (stop filled): entry={entry:.6g} stop={stop:.6g} target={target:.6g}")
            if cl[-1] > stop:
                self._emit(symbol, now, "exit", side="short", price=float(cl[-1]),
                           reason="collapse",
                           text=f"short cover (collapse) @ {cl[-1]:.6g} — closed through the fresh stop {stop:.6g}")
                self._reset_short(st)
        elif can_enter and ep_setup:
            entry = float(cl[-1])
            stop, target = self._long_levels(entry, adr, lo[-1])
            st.update(in_long=True, entry_px=entry, stop_px=stop, target_px=target,
                      entry_bar=st["bar_count"], partial_done=False)
            self._emit(symbol, now, "entry_ep", side="long", price=entry, stop=stop, target=target,
                       text=f"Buy EP (at the close): entry={entry:.6g} stop={stop:.6g} target={target:.6g}")
        elif can_enter and ps_setup:
            entry = float(cl[-1])
            st.update(in_short=True, short_type="PARA", s_entry_px=entry,
                      s_stop_px=float(ps_stop_level), s_target_px=None,
                      s_entry_bar=st["bar_count"], s_partial_done=False)
            self._emit(symbol, now, "entry_para", side="short", price=entry,
                       stop=float(ps_stop_level),
                       text=f"Short PARA (at the close): entry={entry:.6g} stop={ps_stop_level:.6g} (no target on this track)")

        # ── Advance notices — where the pine's level plots sit (395-414) ─────
        cmp_up = "close>=" if self.wait_for_close else "high>="
        cmp_dn = "close<=" if self.wait_for_close else "low<="
        vt_break = self.bo_vol_mult * avg_vol50_now if self.use_bo_vol else None
        vol_txt = f" | fill needs vol>={vt_break:,.0f}" if vt_break is not None else ""
        if st["bo_armed"]:
            left = self.order_ttl_bars - (st["bar_count"] - st["bo_armed_bar"])
            self._emit(symbol, now, "arm_bo", side="long", price=st["bo_order_px"],
                       ttl_bars=left, vol_threshold=vt_break,
                       text=f"BO buy-stop armed @ {st['bo_order_px']:.6g} | trigger {cmp_up}{st['bo_order_px']:.6g}"
                            f" | TTL {left}/{self.order_ttl_bars} bars{vol_txt}")
        if st["so_armed"]:
            left = self.order_ttl_bars - (st["bar_count"] - st["so_armed_bar"])
            self._emit(symbol, now, "arm_so", side="short", price=st["so_order_px"],
                       ttl_bars=left, vol_threshold=vt_break,
                       text=f"SBO sell-stop armed @ {st['so_order_px']:.6g} | trigger {cmp_dn}{st['so_order_px']:.6g}"
                            f" | TTL {left}/{self.order_ttl_bars} bars{vol_txt}")
        if orb_active:
            self._emit(symbol, now, "arm_orb", side="long", price=orb_level,
                       vol_threshold=vt_break,
                       text=f"ORB level @ {orb_level:.6g} | trigger {cmp_up}{orb_level:.6g}"
                            f" | one attempt this UTC day{vol_txt}")

        flat_final = not st["in_long"] and not st["in_short"]
        if self.enable_ep and liq_ok and flat_final:
            min_open = float(cl[-1]) * (1.0 + self.ep_min_gap / 100.0)
            vt = self.ep_vol_mult * avg_vol50_now
            self._emit(symbol, now, "watch_ep", side="long", price=min_open, vol_threshold=vt,
                       text=f"EP watch | next bar needs open>={min_open:.6g} (gap>={self.ep_min_gap:g}%)"
                            f" and vol>={vt:,.0f}; strong close and risk checked on the signal bar")
        if (self.enable_para and liq_ok and flat_final
                and streak_now >= self.ps_streak and ps_runup >= self.ps_min_gain):
            self._emit(symbol, now, "watch_para", side="short", price=float(cl[-1]),
                       stop=float(ps_stop_level),
                       text=f"PARA watch | up-streak {streak_now} (need {self.ps_streak}) | shorts at its close"
                            f" if the next bar closes < {cl[-1]:.6g}"
                            f" (stop preview ~{ps_stop_level:.6g}, finalizes on the signal bar)")

        # ── Chart overlays (pine 395-414) ───────────────────────────────────
        if self.show_order_levels:
            if st["bo_armed"] and st["bo_order_px"] is not None:
                ctx.plot("bo_level", symbol, st["bo_order_px"])
            if orb_active and orb_level is not None:
                ctx.plot("orb_level", symbol, orb_level)
            if st["so_armed"] and st["so_order_px"] is not None:
                ctx.plot("so_level", symbol, st["so_order_px"])
        if st["in_long"] or st["in_short"]:
            ctx.plot("entry_level", symbol, st["entry_px"] if st["in_long"] else st["s_entry_px"])
            ctx.plot("stop_level", symbol, st["stop_px"] if st["in_long"] else st["s_stop_px"])
            target_val = st["target_px"] if st["in_long"] else st["s_target_px"]
            if target_val is not None:
                ctx.plot("target_level", symbol, target_val)
        if self.show_trail_ma and trail is not None:
            ctx.plot("trail_ma", symbol, trail)

    # ── Levels (pine 323-327 / 350-354) ───────────────────────────────────────
    def _long_levels(self, entry, adr, bar_low):
        adr_stop = entry * (1.0 - self.adr_stop_mult * adr / 100.0)
        stop = max(adr_stop, float(bar_low)) if self.use_lod_stop else adr_stop
        stop = min(stop, entry * 0.999)
        return float(stop), float(entry + self.partial_rr * (entry - stop))

    def _short_levels(self, entry, adr, bar_high):
        adr_stop = entry * (1.0 + self.adr_stop_mult * adr / 100.0)
        stop = min(adr_stop, float(bar_high)) if self.use_lod_stop else adr_stop
        stop = max(stop, entry * 1.001)
        return float(stop), float(entry - self.partial_rr * (stop - entry))

    # ── Book resets / signal emission ─────────────────────────────────────────
    def _reset_long(self, st):
        st.update(in_long=False, entry_px=None, stop_px=None, target_px=None,
                  entry_bar=None, partial_done=False)

    def _reset_short(self, st):
        st.update(in_short=False, short_type=None, s_entry_px=None, s_stop_px=None,
                  s_target_px=None, s_entry_bar=None, s_partial_done=False)

    def _emit(self, symbol, ts_ms, kind, text, side="", price=None, stop=None,
              target=None, ttl_bars=None, vol_threshold=None, reason=""):
        self.signals.append(Signal(
            timestamp=ts_ms, symbol=symbol, kind=kind, side=side,
            price=None if price is None else float(price),
            stop=None if stop is None else float(stop),
            target=None if target is None else float(target),
            ttl_bars=ttl_bars,
            vol_threshold=None if vol_threshold is None else float(vol_threshold),
            reason=reason, text=text))
        if self.print_signals:
            print(f"[{_fmt_ts(ts_ms)}] {symbol}: {text}")
