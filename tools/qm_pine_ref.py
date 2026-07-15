"""Exact replay of app/pines/qullamaggie_momentum_swing.pine (v12) over local bars.

Reference implementation of the pine INDICATOR, bar for bar and in pine's
own evaluation order — including the volume-gated breakout fills
(useBOVol/volBreakOK) and the full trade management the strategy port
deliberately does not carry (partial at R or after N bars, breakeven move,
trail-MA exit, PARA stop/bounce/time covers). Those exits matter here even
though they are not ported: they define pine's canEnter windows, i.e. which
breaks pine skips because it is still holding.

The replay emits one event per pine action so an engine backtest can be
diffed against it (tools/qm_trade_diff.py):

    arm         resting order armed at a new level (or the level changed)
    rearm       setup held: timer reset, same level (silent in pine)
    expire      armed order lapsed bo_order_bars after the last (re)arm
    fill        an entry printed (setup BO/ORB/SHORT_BO/EP/PARA)
    skip_volume armed level was crossed but the break bar failed volBreakOK
    skip_in_pos armed level was crossed with volume, but canEnter was false
    partial     partial profit point (R target or partial_days)
    be_move     stop moved to breakeven after the partial
    exit        position closed (reason: stop / trail / same_bar_collapse /
                para_stop / para_bounce / para_time)

Deliberate stand-ins for chart-only pine facilities:
  - syminfo.mintick        -> --mintick (default 0.01, US equities)
  - session.isfirstbar_regular -> first bar of a new UTC calendar day; on a
    daily file every bar starts a session, so ORB is structurally dead there
    (barsIntoSession is always 0), exactly like the pine on a daily chart.
  - ta.highestbars tie-break is undocumented on TradingView; --tie picks
    which of two equal extremes counts (default "recent", matching the
    strategy port's assumption). Bars with non-positive lows poison
    high/low ratios; they are treated as na (TV data has no such rows).

Usage (from the project root, with the app venv for pandas/pyarrow):

    app/python/.venv/bin/python tools/qm_pine_ref.py \
        app/data/us_1d.parquet MU --report-start 2025-01-01 \
        --json mu_ref.json [--out mu_events.csv] [--<param> value ...]

Every pine input is a CLI flag with the pine default (e.g. --use-bo-vol,
--bo-vol-mult, --min-gain). The replay always runs the symbol's full history
(TV-style warmup); --report-start/--report-end only bound the output.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

NAN = float("nan")


@dataclass
class PineParams:
    # Universe & trend filters
    min_price: float = 5.0
    min_avg_vol: float = 0.0
    adr_len: int = 20
    min_adr: float = 0.1
    mom_len: int = 24
    min_gain: float = 0.5
    require_mas: bool = True
    # Setup 1 — momentum breakout (long)
    enable_bo: bool = True
    base_max_len: int = 40
    min_base_bars: int = 3
    max_depth: float = 40.0
    use_vol_dry: bool = False
    vol_dry_ratio: float = 1.0
    buf_ticks: int = 1
    bo_order_bars: int = 10
    use_bo_vol: bool = True
    bo_vol_mult: float = 1.3
    # Setup 1b — opening range breakout
    enable_orb: bool = False
    orb_bars: int = 1
    # Setup 1c — short breakout
    enable_short_bo: bool = False
    # Setup 2 — episodic pivot
    enable_ep: bool = True
    ep_min_gap: float = 0.5
    ep_vol_mult: float = 1.3
    ep_strong_close: bool = True
    # Setup 3 — parabolic short
    enable_para: bool = False
    ps_lookback: int = 10
    ps_min_gain: float = 8.0
    ps_streak: int = 3
    ps_stop_lookback: int = 3
    ps_max_hold: int = 5
    # Initial stop
    adr_stop_mult: float = 1.0
    use_lod_stop: bool = True
    # Trade management
    partial_rr: float = 2.0
    partial_days: int = 6
    move_be: bool = True
    trail_type: str = "EMA"
    trail_len: int = 20
    # Chart stand-ins (not pine inputs)
    mintick: float = 0.01
    tie: str = "recent"   # ta.highestbars tie-break: "recent" | "oldest"


def sma_series(a, n):
    """pine ta.sma: na until n bars, na when the window contains na."""
    return pd.Series(a).rolling(n, min_periods=n).mean().to_numpy()


def ema_series(a, n):
    """pine ta.ema: na for the first n-1 bars, seeded with SMA(n), then
    alpha = 2/(n+1) recursion."""
    out = np.full(len(a), NAN)
    if len(a) < n:
        return out
    alpha = 2.0 / (n + 1.0)
    out[n - 1] = np.mean(a[:n])
    for i in range(n, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def highest(a, i, n):
    if n <= 0 or i + 1 < n:
        return NAN
    return float(np.max(a[i - n + 1: i + 1]))


def lowest(a, i, n):
    if n <= 0 or i + 1 < n:
        return NAN
    return float(np.min(a[i - n + 1: i + 1]))


def bars_since_extreme(a, i, n, tie, kind):
    """pine -ta.highestbars / -ta.lowestbars as a non-negative offset
    (0 = current bar), or None during warmup (pine returns na there).
    tie="recent" resolves equal extremes to the most recent bar,
    tie="oldest" to the oldest."""
    if n <= 0 or i + 1 < n:
        return None
    win = a[i - n + 1: i + 1]
    pick = np.argmax if kind == "high" else np.argmin
    if tie == "recent":
        return int(pick(win[::-1]))
    return n - 1 - int(pick(win))


def timestamps_ms(df):
    ts = df["timestamp"]
    if ts.dtype == "int64":
        return ts.to_numpy(np.int64)
    return ts.astype("datetime64[ns, UTC]").astype("int64").to_numpy() // 1_000_000


def replay(df, p):
    """Run the pine over one symbol's bars (chronological). Returns
    (events, position_intervals)."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    ts = timestamps_ms(df)
    n = len(df)

    sma10 = sma_series(c, 10)
    sma20 = sma_series(c, 20)
    trail = ema_series(c, p.trail_len) if p.trail_type == "EMA" \
        else sma_series(c, p.trail_len)
    range_pct = np.where(l > 0.0, 100.0 * (h / np.where(l > 0.0, l, 1.0) - 1.0), NAN)
    adr = sma_series(range_pct, p.adr_len)
    av5 = sma_series(v, 5)
    av20 = sma_series(v, 20)
    av50 = sma_series(v, 50)
    av50_prev = np.concatenate([[NAN], av50[:-1]])
    gain = np.full(n, NAN)
    if n > p.mom_len:
        gain[p.mom_len:] = 100.0 * (c[p.mom_len:] / c[:-p.mom_len] - 1.0)

    events = []
    intervals = []

    def emit(i, event, setup="", side="", level=NAN, entry=NAN, stop=NAN,
             target=NAN, reason="", detail=""):
        events.append({
            "ts": int(ts[i]), "bar_index": i, "event": event, "setup": setup,
            "side": side, "level": level, "entry": entry, "stop": stop,
            "target": target, "reason": reason, "detail": detail,
        })

    # position state (pine's var block)
    in_long = False
    entry_px = stop_px = target_px = NAN
    entry_bar = -1
    partial_done = False
    in_short = False
    short_type = ""
    s_entry_px = s_stop_px = s_target_px = NAN
    s_entry_bar = -1
    s_partial_done = False
    interval = None   # the open position's summary dict

    up_stk = 0
    or_high = NAN
    or_count = 0
    orb_taken = False
    prev_day = None
    bars_into_session = 0
    bo_armed = False
    bo_order_px = NAN
    bo_armed_bar = -1
    so_armed = False
    so_order_px = NAN
    so_armed_bar = -1

    def open_interval(i, setup, side, e_px, reason_slot):
        return {"entry_ts": int(ts[i]), "entry_bar": i, "setup": setup,
                "side": side, "entry": e_px, "exit_ts": None, "exit_bar": None,
                "exit_reason": reason_slot}

    def close_interval(i, reason):
        nonlocal interval
        interval["exit_ts"] = int(ts[i])
        interval["exit_bar"] = i
        interval["exit_reason"] = reason
        intervals.append(interval)
        interval = None

    for i in range(n):
        # ── core calculations & gates (pine 97-119) ──────────────────────────
        close_now, open_now, high_now, low_now, vol_now = c[i], o[i], h[i], l[i], v[i]
        prev_close = c[i - 1] if i > 0 else NAN
        liq_ok = close_now >= p.min_price and av20[i] >= p.min_avg_vol
        adr_ok = adr[i] >= p.min_adr
        vol_dry_ok = (not p.use_vol_dry) or (av5[i] < p.vol_dry_ratio * av50[i])
        ma_up_ok = (not p.require_mas) or (close_now > sma20[i] and sma10[i] > sma20[i])
        ma_dn_ok = (not p.require_mas) or (close_now < sma20[i] and sma10[i] < sma20[i])
        trend_up = gain[i] >= p.min_gain and ma_up_ok
        trend_dn = gain[i] <= -p.min_gain and ma_dn_ok
        universe_up = liq_ok and adr_ok and trend_up
        universe_dn = liq_ok and adr_ok and trend_dn

        # ── long base / setup (pine 138-144) ─────────────────────────────────
        base_high = highest(h, i, p.base_max_len)
        since_pk_off = bars_since_extreme(h, i, p.base_max_len, p.tie, "high")
        since_pk = since_pk_off if since_pk_off is not None else 0   # nz()
        pull_low = lowest(l, i, max(since_pk, 1))
        retrace_pct = 100.0 * (base_high - pull_low) / base_high
        flag_up_ok = since_pk >= p.min_base_bars and retrace_pct <= p.max_depth \
            and vol_dry_ok
        entry_level = base_high + p.buf_ticks * p.mintick
        bo_setup = p.enable_bo and universe_up and flag_up_ok \
            and not in_long and not in_short

        # ── short base / setup (pine 147-153) ────────────────────────────────
        base_low = lowest(l, i, p.base_max_len)
        since_tr_off = bars_since_extreme(l, i, p.base_max_len, p.tie, "low")
        since_tr = since_tr_off if since_tr_off is not None else 0
        bounce_high = highest(h, i, max(since_tr, 1))
        retrace_up_pct = 100.0 * (bounce_high - base_low) / base_low \
            if base_low > 0.0 else NAN
        flag_dn_ok = since_tr >= p.min_base_bars and retrace_up_pct <= p.max_depth \
            and vol_dry_ok
        entry_level_dn = base_low - p.buf_ticks * p.mintick
        so_setup = p.enable_short_bo and universe_dn and flag_dn_ok \
            and not in_long and not in_short

        # ── episodic pivot (pine 156-161) ────────────────────────────────────
        ep_gap = 100.0 * (open_now / prev_close - 1.0)
        ep_vol_ok = vol_now >= p.ep_vol_mult * av50_prev[i]
        ep_cls_ok = (not p.ep_strong_close) or \
            (close_now > open_now and close_now >= (high_now + low_now) / 2.0)
        ep_risk_ps = close_now - low_now
        ep_within = ep_risk_ps > 0.0 and \
            ep_risk_ps <= p.adr_stop_mult * adr[i] / 100.0 * close_now
        ep_setup = p.enable_ep and liq_ok and ep_gap >= p.ep_min_gap and ep_vol_ok \
            and ep_cls_ok and ep_within and not in_long and not in_short

        # ── parabolic short (pine 164-169) ───────────────────────────────────
        up_stk_prev = up_stk   # pine upStk[1]
        up_stk = up_stk + 1 if close_now > prev_close else 0
        ps_runup_hi = highest(h, i, p.ps_lookback)
        ps_runup_lo = lowest(l, i, p.ps_lookback)
        ps_runup = 100.0 * (ps_runup_hi / ps_runup_lo - 1.0) \
            if ps_runup_lo > 0.0 else NAN
        ps_stop_lv = highest(h, i, p.ps_stop_lookback) + p.buf_ticks * p.mintick
        frd = close_now < prev_close and up_stk_prev >= p.ps_streak
        ps_setup = p.enable_para and liq_ok and ps_runup >= p.ps_min_gain and frd \
            and not bo_setup and not ep_setup and not so_setup \
            and not in_long and not in_short

        # ── opening range (pine 172-184); session = UTC calendar day ─────────
        day = ts[i] // 86_400_000
        if day != prev_day:
            prev_day = day
            or_high = high_now
            or_count = 1
            orb_taken = False
            bars_into_session = 0
        else:
            if or_count < p.orb_bars:
                or_high = max(or_high, high_now)
                or_count += 1
            bars_into_session += 1
        orb_active = p.enable_orb and universe_up and not np.isnan(or_high) \
            and bars_into_session >= p.orb_bars and not orb_taken \
            and not in_long and not in_short
        orb_level = (or_high if not np.isnan(or_high) else 0.0) \
            + p.buf_ticks * p.mintick

        # ── resting order registers (pine 187-208) ───────────────────────────
        if bo_setup:
            if not bo_armed or bo_order_px != entry_level:
                emit(i, "arm", "BO", "long", level=entry_level)
            else:
                emit(i, "rearm", "BO", "long", level=entry_level)
            bo_armed = True
            bo_order_px = entry_level
            bo_armed_bar = i
        if bo_armed and i - bo_armed_bar > p.bo_order_bars:
            emit(i, "expire", "BO", "long", level=bo_order_px)
            bo_armed = False
            bo_order_px = NAN
        if so_setup:
            if not so_armed or so_order_px != entry_level_dn:
                emit(i, "arm", "SHORT_BO", "short", level=entry_level_dn)
            else:
                emit(i, "rearm", "SHORT_BO", "short", level=entry_level_dn)
            so_armed = True
            so_order_px = entry_level_dn
            so_armed_bar = i
        if so_armed and i - so_armed_bar > p.bo_order_bars:
            emit(i, "expire", "SHORT_BO", "short", level=so_order_px)
            so_armed = False
            so_order_px = NAN

        # ── exit management (pine 244-303) ───────────────────────────────────
        exited_now = False
        if in_long and i > entry_bar:
            close_long = False
            if low_now <= stop_px:
                emit(i, "exit", interval["setup"], "long", entry=entry_px,
                     stop=stop_px, reason="stop")
                close_long = True
                close_interval(i, "stop")
            else:
                if not partial_done and (high_now >= target_px
                                         or i - entry_bar >= p.partial_days):
                    emit(i, "partial", interval["setup"], "long",
                         entry=entry_px, target=target_px)
                    partial_done = True
                    if p.move_be:
                        stop_px = max(stop_px, entry_px)
                        emit(i, "be_move", interval["setup"], "long", stop=stop_px)
                if partial_done and close_now < trail[i]:
                    emit(i, "exit", interval["setup"], "long", entry=entry_px,
                         stop=stop_px, reason="trail")
                    close_long = True
                    close_interval(i, "trail")
            if close_long:
                in_long = False
                entry_px = stop_px = target_px = NAN
                entry_bar = -1
                partial_done = False
                exited_now = True
        if in_short and i > s_entry_bar:
            close_short = False
            if short_type == "PARA":
                if high_now >= s_stop_px or close_now > prev_close \
                        or i - s_entry_bar >= p.ps_max_hold:
                    reason = "para_stop" if high_now >= s_stop_px else \
                        "para_bounce" if close_now > prev_close else "para_time"
                    emit(i, "exit", "PARA", "short", entry=s_entry_px,
                         stop=s_stop_px, reason=reason)
                    close_short = True
                    close_interval(i, reason)
            else:
                if high_now >= s_stop_px:
                    emit(i, "exit", interval["setup"], "short", entry=s_entry_px,
                         stop=s_stop_px, reason="stop")
                    close_short = True
                    close_interval(i, "stop")
                else:
                    if not s_partial_done and (low_now <= s_target_px
                                               or i - s_entry_bar >= p.partial_days):
                        emit(i, "partial", interval["setup"], "short",
                             entry=s_entry_px, target=s_target_px)
                        s_partial_done = True
                        if p.move_be:
                            s_stop_px = min(s_stop_px, s_entry_px)
                            emit(i, "be_move", interval["setup"], "short",
                                 stop=s_stop_px)
                    if s_partial_done and close_now > trail[i]:
                        emit(i, "exit", interval["setup"], "short",
                             entry=s_entry_px, stop=s_stop_px, reason="trail")
                        close_short = True
                        close_interval(i, "trail")
            if close_short:
                in_short = False
                short_type = ""
                s_entry_px = s_stop_px = s_target_px = NAN
                s_entry_bar = -1
                s_partial_done = False
                exited_now = True

        # ── entries (pine 306-392) ───────────────────────────────────────────
        can_enter = not in_long and not in_short and not exited_now
        vol_break_ok = (not p.use_bo_vol) or vol_now >= p.bo_vol_mult * av50_prev[i]
        bo_break = high_now >= bo_order_px
        bo_fill = can_enter and bo_armed and not np.isnan(bo_order_px) \
            and bo_break and vol_break_ok
        orb_break = high_now >= orb_level
        orb_fill = can_enter and orb_active and orb_break and vol_break_ok \
            and not bo_fill
        so_break = low_now <= so_order_px
        so_fill = can_enter and so_armed and not np.isnan(so_order_px) \
            and so_break and vol_break_ok

        # diagnosis-only events: crossings that did NOT print in pine
        if bo_armed and not np.isnan(bo_order_px) and bo_break:
            if not can_enter:
                emit(i, "skip_in_pos", "BO", "long", level=bo_order_px)
            elif not vol_break_ok:
                emit(i, "skip_volume", "BO", "long", level=bo_order_px,
                     detail=f"vol {vol_now:.0f} < {p.bo_vol_mult:g}x{av50_prev[i]:.0f}")
        if orb_active and orb_break and not orb_fill and not bo_fill:
            if not can_enter:
                emit(i, "skip_in_pos", "ORB", "long", level=orb_level)
            elif not vol_break_ok:
                emit(i, "skip_volume", "ORB", "long", level=orb_level)
        if so_armed and not np.isnan(so_order_px) and so_break:
            if not can_enter:
                emit(i, "skip_in_pos", "SHORT_BO", "short", level=so_order_px)
            elif not vol_break_ok:
                emit(i, "skip_volume", "SHORT_BO", "short", level=so_order_px,
                     detail=f"vol {vol_now:.0f} < {p.bo_vol_mult:g}x{av50_prev[i]:.0f}")

        if bo_fill or orb_fill:
            level = bo_order_px if bo_fill else orb_level
            setup = "BO" if bo_fill else "ORB"
            in_long = True
            entry_px = max(open_now, level)
            adr_stop = entry_px * (1.0 - p.adr_stop_mult * adr[i] / 100.0)
            stop_px = max(adr_stop, low_now) if p.use_lod_stop else adr_stop
            stop_px = min(stop_px, entry_px * 0.999)
            target_px = entry_px + p.partial_rr * (entry_px - stop_px)
            entry_bar = i
            partial_done = False
            if bo_fill:
                bo_armed = False
            else:
                orb_taken = True
            emit(i, "fill", setup, "long", level=level, entry=entry_px,
                 stop=stop_px, target=target_px)
            interval = open_interval(i, setup, "long", entry_px, None)
            if close_now < stop_px:   # same-bar collapse bail (pine 338)
                emit(i, "exit", setup, "long", entry=entry_px, stop=stop_px,
                     reason="same_bar_collapse")
                close_interval(i, "same_bar_collapse")
                in_long = False
                entry_px = stop_px = target_px = NAN
                entry_bar = -1
                partial_done = False
        elif so_fill:
            in_short = True
            short_type = "BO"
            s_entry_px = min(open_now, so_order_px)
            adr_stop_s = s_entry_px * (1.0 + p.adr_stop_mult * adr[i] / 100.0)
            s_stop_px = min(adr_stop_s, high_now) if p.use_lod_stop else adr_stop_s
            s_stop_px = max(s_stop_px, s_entry_px * 1.001)
            s_target_px = s_entry_px - p.partial_rr * (s_stop_px - s_entry_px)
            s_entry_bar = i
            s_partial_done = False
            so_armed = False
            emit(i, "fill", "SHORT_BO", "short", level=so_order_px,
                 entry=s_entry_px, stop=s_stop_px, target=s_target_px)
            interval = open_interval(i, "SHORT_BO", "short", s_entry_px, None)
            if close_now > s_stop_px:   # same-bar snap-back bail (pine 361)
                emit(i, "exit", "SHORT_BO", "short", entry=s_entry_px,
                     stop=s_stop_px, reason="same_bar_collapse")
                close_interval(i, "same_bar_collapse")
                in_short = False
                short_type = ""
                s_entry_px = s_stop_px = s_target_px = NAN
                s_entry_bar = -1
                s_partial_done = False
        elif can_enter and ep_setup:
            in_long = True
            entry_px = close_now
            adr_stop = entry_px * (1.0 - p.adr_stop_mult * adr[i] / 100.0)
            stop_px = max(adr_stop, low_now) if p.use_lod_stop else adr_stop
            stop_px = min(stop_px, entry_px * 0.999)
            target_px = entry_px + p.partial_rr * (entry_px - stop_px)
            entry_bar = i
            partial_done = False
            emit(i, "fill", "EP", "long", entry=entry_px, stop=stop_px,
                 target=target_px)
            interval = open_interval(i, "EP", "long", entry_px, None)
        elif can_enter and ps_setup:
            in_short = True
            short_type = "PARA"
            s_entry_px = close_now
            s_stop_px = ps_stop_lv
            s_entry_bar = i
            s_partial_done = False
            emit(i, "fill", "PARA", "short", entry=s_entry_px, stop=s_stop_px)
            interval = open_interval(i, "PARA", "short", s_entry_px, None)

    if interval is not None:
        intervals.append(interval)   # still open at the end of history
    return events, intervals


def cli_type(default):
    if isinstance(default, bool):
        return lambda s: s.lower() in ("1", "true", "yes", "on")
    return type(default)


def main():
    ap = argparse.ArgumentParser(
        description="Replay the qullamaggie_momentum_swing pine over local bars.")
    ap.add_argument("parquet")
    ap.add_argument("symbol")
    ap.add_argument("--report-start", default=None,
                    help="only output events at/after this date (state still "
                         "runs the full history)")
    ap.add_argument("--report-end", default=None)
    ap.add_argument("--out", default=None, help="write events CSV here")
    ap.add_argument("--json", default=None, help="write the summary JSON here")
    for f in fields(PineParams):
        ap.add_argument("--" + f.name.replace("_", "-"), dest=f.name,
                        type=cli_type(f.default), default=f.default,
                        help=f"pine default: {f.default}")
    args = ap.parse_args()
    p = PineParams(**{f.name: getattr(args, f.name) for f in fields(PineParams)})

    df = pd.read_parquet(args.parquet)
    df = df[df["symbol"] == args.symbol].sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        sys.exit(f"symbol {args.symbol} not found in {args.parquet}")

    events, intervals = replay(df, p)

    start_ms = int(pd.Timestamp(args.report_start, tz="UTC").value // 1_000_000) \
        if args.report_start else None
    end_ms = int(pd.Timestamp(args.report_end, tz="UTC").value // 1_000_000) \
        if args.report_end else None

    def in_window(ms):
        return (start_ms is None or ms >= start_ms) and (end_ms is None or ms <= end_ms)

    windowed = [e for e in events if in_window(e["ts"])]
    win_intervals = [iv for iv in intervals
                     if (end_ms is None or iv["entry_ts"] <= end_ms)
                     and (start_ms is None or iv["exit_ts"] is None
                          or iv["exit_ts"] >= start_ms)]

    counts = {}
    for e in windowed:
        counts[e["event"]] = counts.get(e["event"], 0) + 1
    print(f"{args.symbol}: {len(df)} bars replayed, "
          f"{len(windowed)} events in window")
    for name in ("arm", "rearm", "expire", "fill", "skip_volume", "skip_in_pos",
                 "partial", "be_move", "exit"):
        if name in counts:
            print(f"  {name:12} {counts[name]}")
    for e in windowed:
        if e["event"] == "fill":
            when = pd.Timestamp(e["ts"], unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
            print(f"  FILL {when} {e['setup']:8} {e['side']:5} "
                  f"entry {e['entry']:.4f} stop {e['stop']:.4f}")

    if args.out:
        cols = ["ts", "bar_index", "event", "setup", "side", "level", "entry",
                "stop", "target", "reason", "detail"]
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(windowed)
        print(f"events CSV written to {args.out}")
    if args.json:
        summary = {
            "symbol": args.symbol,
            "params": asdict(p),
            "window": {"start": args.report_start, "end": args.report_end},
            "counts": counts,
            "fills": [e for e in windowed if e["event"] == "fill"],
            "skipped_breaks": [e for e in windowed
                               if e["event"] in ("skip_volume", "skip_in_pos")],
            "arms": [e for e in windowed if e["event"] in ("arm", "expire")],
            "position_intervals": win_intervals,
        }
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"summary JSON written to {args.json}")


if __name__ == "__main__":
    main()
