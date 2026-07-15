"""Diff a stonks backtest of qmmomentumswing against the pine replayer.

Takes a report JSON (the engine's per-fill trade list) and a qm_pine_ref.py
summary JSON for ONE symbol, reconstructs the engine's entry->exit round
trips, matches engine entries against the pine's fills, and classifies every
mismatch into a named cause bucket — the quantified answer to "why does the
backtest enter trades the pine doesn't print?".

Engine-extra buckets (entries the pine did not take), first hit wins:

  churn-repeat            stop entry at ~the same level right after a round
                          trip that exited via a reduce-only market bail —
                          the parked-order volume-bail churn loop
  volume-failed-crossing  the pine skipped this break bar (volBreakOK false)
  pine-in-position        pine was still holding (its partial/BE/trail exits
                          hold longer than the port's all-out bracket)
  arm-level-mismatch      stop entry at a level the pine never armed near
                          that time (highestbars tie / window-drift suspect)
  unexplained             none of the above (a real logic bug, or data drift
                          vs whatever chart you eyeballed — engine and
                          replayer share this parquet, so TV-feed drift can
                          only surface here)

Pine-extra buckets (pine fills the engine missed): engine-in-position /
entry-rejected / entry-buffer (port arms base_high x (1+bps) vs pine
base_high + mintick; shallow crossings fit between the two) /
never-armed / unexplained.

Everything after the first divergence on either side is additionally tagged
`post-divergence`: position lifecycles drift apart from that point by design
(the port deliberately does not carry pine's exit management), so later
in-position mismatches are structural, not new bugs.

Usage (from the project root, with the app venv):

    app/python/.venv/bin/python tools/qm_trade_diff.py \
        app/reports/report-YYYYMMDD-HHMMSS.json mu_ref.json [--parquet ...]

The parquet defaults to the report's own run.data_file; the symbol and the
comparison window default to the replayer summary's. Exit code is always 0 —
this is a diagnosis tool, not a gate.
"""

import argparse
import json
import sys
from collections import defaultdict

import pandas as pd

BUY, SELL = "Buy", "Sell"


def iso_ms(s):
    return int(pd.Timestamp(s).value // 1_000_000)


def when(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def load_bar_index(parquet, symbol):
    df = pd.read_parquet(parquet)
    df = df[df["symbol"] == symbol]
    ts = df["timestamp"]
    if ts.dtype != "int64":
        ts = ts.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    ordered = sorted(ts.tolist())
    return {t: i for i, t in enumerate(ordered)}


def reconstruct_round_trips(trades, orders, symbol):
    """Replay the signed position from the per-fill trade list; one round
    trip per flat->nonflat->flat cycle. Fill ids are monotonic in fill order."""
    trips = []
    qty = 0.0
    trip = None
    for t in sorted(trades, key=lambda t: t["id"]):
        if t["symbol"] != symbol:
            continue
        signed = t["quantity"] if t["side"] == BUY else -t["quantity"]
        was_flat = abs(qty) < 1e-12
        qty += signed
        order = orders.get(t["order_id"], {})
        if was_flat and abs(qty) > 1e-12:
            trip = {
                "side": "long" if signed > 0 else "short",
                "entry_ts": t["ts_ms"], "entry_price": t["price"],
                "entry_order": order,
                "placed_ts": iso_ms(order["timestamp"]) if order else t["ts_ms"],
                "exit_ts": None, "exit_kind": None,
            }
        elif trip is not None and abs(qty) < 1e-12:
            if t["liquidation"]:
                kind = "liquidation"
            elif order.get("type") == "Market" and order.get("reduce_only") \
                    and order.get("parent_id") == trip["entry_order"].get("id"):
                kind = "volume-bail"
            elif order.get("type") == "Stop":
                kind = "stop"
            elif order.get("type") == "Limit":
                kind = "target"
            else:
                kind = "market"
            trip["exit_ts"] = t["ts_ms"]
            trip["exit_kind"] = kind
            trips.append(trip)
            trip = None
            qty = 0.0   # flat-snap float dust
    if trip is not None:
        trips.append(trip)   # still open at run end
    return trips


def close_level(a, b, rel=2e-3):
    return a is not None and b is not None \
        and abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def main():
    ap = argparse.ArgumentParser(
        description="Classify engine-vs-pine entry mismatches for one symbol.")
    ap.add_argument("report")
    ap.add_argument("ref_json", help="qm_pine_ref.py --json output")
    ap.add_argument("--parquet", default=None,
                    help="bars parquet (default: the report's run.data_file)")
    ap.add_argument("--symbol", default=None,
                    help="default: the replayer summary's symbol")
    args = ap.parse_args()

    report = json.load(open(args.report))
    ref = json.load(open(args.ref_json))
    symbol = args.symbol or ref["symbol"]
    parquet = args.parquet or report.get("run", {}).get("data_file")
    if not parquet:
        sys.exit("no --parquet given and the report has no run.data_file")

    bar_of = load_bar_index(parquet, symbol)
    if not bar_of:
        sys.exit(f"symbol {symbol} not found in {parquet}")

    orders = {o["id"]: o for o in report.get("orders", [])}
    trades = [dict(t, ts_ms=iso_ms(t["timestamp"])) for t in report.get("trades", [])]
    trips = reconstruct_round_trips(trades, orders, symbol)

    window = ref.get("window", {})
    start_ms = iso_ms(window["start"]) if window.get("start") else None
    end_ms = iso_ms(window["end"]) if window.get("end") else None

    def in_window(ms):
        return (start_ms is None or ms >= start_ms) and (end_ms is None or ms <= end_ms)

    entries = [t for t in trips if in_window(t["entry_ts"])]
    pine_fills = [dict(f) for f in ref.get("fills", []) if in_window(f["ts"])]
    skip_vol = ref.get("skipped_breaks", [])
    intervals = ref.get("position_intervals", [])
    buf_bps = report.get("strategy", {}).get("params", {}) \
        .get("entry_buffer_bps", 5.0)
    mintick = ref.get("params", {}).get("mintick", 0.01)

    # decision bar: market entries act on the signal bar (order placement ts);
    # parked stops act on the bar that traded through them (fill ts)
    for t in entries:
        kind = t["entry_order"].get("type")
        t["bar"] = bar_of.get(t["placed_ts"] if kind == "Market" else t["entry_ts"])
        t["level"] = t["entry_order"].get("price")
    for f in pine_fills:
        f["bar"] = f["bar_index"] if f["bar_index"] in range(len(bar_of)) else \
            bar_of.get(f["ts"])

    # ── greedy 1:1 match: exact bar first, then ±1 ───────────────────────────
    unmatched_pine = list(pine_fills)
    for t in entries:
        t["match"] = None
        for dist in (0, 1):
            cands = [f for f in unmatched_pine
                     if f["side"] == t["side"] and t["bar"] is not None
                     and f["bar"] is not None and abs(f["bar"] - t["bar"]) <= dist]
            if cands:
                best = min(cands, key=lambda f: (abs(f["bar"] - t["bar"]),
                                                 abs(f["entry"] - t["entry_price"])))
                t["match"] = best
                t["timing"] = abs(best["bar"] - t["bar"])
                unmatched_pine.remove(best)
                break

    # ── classify engine extras ───────────────────────────────────────────────
    def pine_in_position(ms, bar):
        for iv in intervals:
            entry_bar = iv.get("entry_bar")
            exit_bar = iv.get("exit_bar")
            if entry_bar is not None and bar is not None:
                if entry_bar <= bar and (exit_bar is None or bar <= exit_bar):
                    return True
        return False

    prev_trip_by_entry = {}
    for i, t in enumerate(trips):
        if i > 0:
            prev_trip_by_entry[id(t)] = trips[i - 1]

    extras = []
    for t in entries:
        if t["match"] is not None:
            continue
        prev = prev_trip_by_entry.get(id(t))
        entry_type = t["entry_order"].get("type")
        bucket = None
        if (entry_type == "Stop" and prev is not None
                and prev["exit_kind"] == "volume-bail"
                and close_level(prev["level"], t["level"])
                and prev["exit_ts"] is not None
                and t["bar"] is not None and bar_of.get(prev["exit_ts"]) is not None
                and t["bar"] - bar_of[prev["exit_ts"]] <= 1):
            bucket = "churn-repeat"
        elif any(t["bar"] is not None and abs(e["bar_index"] - t["bar"]) <= 1
                 and (t["level"] is None or close_level(e["level"], t["level"]))
                 for e in skip_vol if e["event"] == "skip_volume"):
            bucket = "volume-failed-crossing"
        elif pine_in_position(t["entry_ts"], t["bar"]) or \
                any(t["bar"] is not None and abs(e["bar_index"] - t["bar"]) <= 1
                    for e in skip_vol if e["event"] == "skip_in_pos"):
            bucket = "pine-in-position"
        elif entry_type == "Stop" and not any(
                close_level(a["level"], t["level"])
                for a in ref.get("arms", []) if a["event"] == "arm"):
            bucket = "arm-level-mismatch"
        else:
            bucket = "unexplained"
        extras.append((t, bucket))

    # ── classify pine extras (fills the engine missed) ───────────────────────
    def engine_in_position(bar):
        for t in trips:
            eb = bar_of.get(t["entry_ts"])
            xb = bar_of.get(t["exit_ts"]) if t["exit_ts"] else None
            if eb is not None and bar is not None and eb <= bar \
                    and (xb is None or bar <= xb):
                return True
        return False

    rejected_bars = set()
    stop_levels = []
    for o in orders.values():
        if o["symbol"] != symbol or o.get("reduce_only"):
            continue
        b = bar_of.get(iso_ms(o["timestamp"]))
        if o.get("status") == "Rejected" and b is not None:
            rejected_bars.add(b)
        if o.get("type") == "Stop" and o.get("price") is not None and b is not None:
            stop_levels.append((b, o["price"]))

    pine_extras = []
    for f in unmatched_pine:
        bucket = None
        port_level = None
        if f.get("level"):
            port_level = (f["level"] - mintick) * (1.0 + buf_bps / 10_000.0)
        if engine_in_position(f["bar"]):
            bucket = "engine-in-position"
        elif f["bar"] in rejected_bars or (f["bar"] is not None
                                           and f["bar"] - 1 in rejected_bars):
            bucket = "entry-rejected"
        elif port_level is not None and f["bar"] is not None and any(
                b <= f["bar"] and f["bar"] - b <= 12 and close_level(px, port_level)
                for b, px in stop_levels) and f["entry"] < port_level:
            bucket = "entry-buffer"
        elif f.get("level") and not any(
                b <= (f["bar"] or 0) and (f["bar"] or 0) - b <= 12
                and close_level(px, port_level or f["level"], rel=5e-3)
                for b, px in stop_levels):
            bucket = "never-armed"
        else:
            bucket = "unexplained"
        pine_extras.append((f, bucket))

    # ── post-divergence tagging ──────────────────────────────────────────────
    mismatch_bars = [t["bar"] for t, _ in extras if t["bar"] is not None]
    mismatch_bars += [f["bar"] for f, _ in pine_extras if f["bar"] is not None]
    first_div = min(mismatch_bars) if mismatch_bars else None

    # ── report ───────────────────────────────────────────────────────────────
    matched = [t for t in entries if t["match"] is not None]
    exact = sum(1 for t in matched if t["timing"] == 0)
    print(f"symbol {symbol}  window {window.get('start')} .. {window.get('end')}")
    print(f"engine entries: {len(entries)}   pine fills: {len(pine_fills)}")
    print(f"matched: {len(matched)} ({exact} same-bar, {len(matched) - exact} ±1 bar)")

    counts = defaultdict(int)
    for _, b in extras:
        counts["engine-extra/" + b] += 1
    for _, b in pine_extras:
        counts["pine-extra/" + b] += 1
    if counts:
        print("\nmismatch buckets:")
        for name in sorted(counts):
            print(f"  {name:38} {counts[name]}")

    if extras:
        print("\nengine entries the pine did not take:")
        for t, bucket in sorted(extras, key=lambda x: x[0]["entry_ts"]):
            tag = " [post-divergence]" \
                if first_div is not None and t["bar"] is not None \
                and t["bar"] > first_div else ""
            lvl = f" level {t['level']:.4f}" if t["level"] else ""
            print(f"  {when(t['entry_ts'])} {t['side']:5} "
                  f"{t['entry_order'].get('type', '?'):6} entry "
                  f"{t['entry_price']:.4f}{lvl} exit={t['exit_kind']} "
                  f"-> {bucket}{tag}")
    if pine_extras:
        print("\npine fills the engine missed:")
        for f, bucket in sorted(pine_extras, key=lambda x: x[0]["ts"]):
            tag = " [post-divergence]" \
                if first_div is not None and f["bar"] is not None \
                and f["bar"] > first_div else ""
            print(f"  {when(f['ts'])} {f['setup']:8} {f['side']:5} entry "
                  f"{f['entry']:.4f} -> {bucket}{tag}")

    print("\nnote: engine and replayer read the same parquet; drift vs a "
          "TradingView chart's own feed can only appear as 'unexplained'.")


if __name__ == "__main__":
    main()
