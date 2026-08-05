"""Independent trade-by-trade audit of a stonks backtest run.

Replays the run's recorded orders (from the report JSON) and strategy
cancellations (from an optional STONKS_LOG capture) against the raw parquet
bars, using a from-spec Python reimplementation of the broker mechanics —
deliberately written from the documented semantics, not ported from the C++.
Every fill, liquidation, final order status, and equity-curve point must
reproduce exactly; each fill is additionally checked against first-principles
invariants (no lookahead, trigger touched, fill-price rule, fees, reduce-only,
netting). Strategy-level gating (one trade at a time, cooldown, single pending
entry) is audited from the replayed state.

The verifier is self-configuring: starting cash, the tick timeline, and the
traded-symbol set come from the report itself, and the broker knobs (fees,
maintenance margin, loss cap, epsilon, floors) from its `config` block.

Usage (from the project root, with the app venv for pandas/pyarrow):

    app/python/.venv/bin/python tools/verify_backtest.py \
        app/reports/report-YYYYMMDD-HHMMSS.json app/data/binance_1d.parquet [run.log]

The log argument supplies strategy-cancel timing (`ev=cancel_req` lines from a
`-DSTONKS_LOG=ON` build, captured via `2> run.log`). Without it, a run whose
strategy called cancel_order() cannot replay faithfully and will report
violations — runs that never cancel verify fine log-less.

Flags: --cooldown-bars N (default 5) tunes the qmsignals gating audit;
--skip-gating disables the strategy-gating checks for non-qmsignals runs.

Exit code 0 = CLEAN, 1 = violations found.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("parquet")
ap.add_argument("log", nargs="?", default=None)
ap.add_argument("--cooldown-bars", type=int, default=5)
ap.add_argument("--skip-gating", action="store_true")
args = ap.parse_args()

violations = []


def flag(check, detail):
    violations.append((check, detail))
    print(f"VIOLATION [{check}] {detail}")


# ─── Inputs ──────────────────────────────────────────────────────────────────
report = json.load(open(args.report))

CFG = report.get("config", {})
MAKER_BPS = CFG.get("maker_fee_bps", 0.0)
TAKER_BPS = CFG.get("taker_fee_bps", 0.0)
FEE_PER_FILL = CFG.get("fee_per_fill", 0.0)
MAINT_M = CFG.get("maintenance_margin_rate", 0.0)
LOSS_CAP = CFG.get("isolated_loss_cap", False)
FLAT_EPS = CFG.get("flat_epsilon", 0.0)
MIN_EQUITY = CFG.get("min_equity", 0.0)
MIN_NOTIONAL = CFG.get("min_notional", 0.0)
STARTING_CASH = report["metrics"]["starting_cash"]
assert CFG.get("fill_policy", "Conservative") == "Conservative", \
    "this verifier models the Conservative intrabar fill policy only"

# The gating audit's cooldown: the report's own strategy params win when
# present (self-configuring, like the broker knobs); otherwise the CLI flag.
STRATEGY_PARAMS = report.get("strategy", {}).get("params", {})
if "cooldown_bars" in STRATEGY_PARAMS:
    COOLDOWN_BARS = int(STRATEGY_PARAMS["cooldown_bars"])
    print(f"cooldown-bars audit setting: {COOLDOWN_BARS} (from report)")
else:
    COOLDOWN_BARS = args.cooldown_bars
    print(f"cooldown-bars audit setting: {COOLDOWN_BARS} (CLI default)")


def fee_of(notional, maker):
    return notional * (MAKER_BPS if maker else TAKER_BPS) / 10_000.0 + FEE_PER_FILL


def iso_ms(s):
    return int(pd.Timestamp(s).value // 1_000_000)


orders = {o["id"]: o for o in report["orders"]}
for o in orders.values():
    o["ts_ms"] = iso_ms(o["timestamp"])
trades = sorted(report["trades"], key=lambda t: t["id"])
for t in trades:
    t["ts_ms"] = iso_ms(t["timestamp"])
synthetic_ids = {t["order_id"] for t in trades if t["liquidation"]}
strategy_orders = {i: o for i, o in orders.items() if i not in synthetic_ids}

# Strategy cancels: pair "[ctx] ev=cancel_req id=N now=T" with the following
# "[broker] ev=cancel_req id=N result=cancelled".
cancels = []
log_fill_ids = []
log_liq_count = 0
if args.log:
    pending_req = None
    for line in open(args.log):
        m = re.match(r"\[ctx\] ev=cancel_req id=(\d+) now=(\d+)", line)
        if m:
            pending_req = (int(m.group(2)), int(m.group(1)))
            continue
        m = re.match(r"\[broker\] ev=cancel_req id=(\d+) result=(\w+)", line)
        if m and pending_req and int(m.group(1)) == pending_req[1]:
            if m.group(2) == "cancelled":
                cancels.append(pending_req)
            pending_req = None
        if "ev=fill" in line:
            fm = re.search(r"ev=fill trade=\d+ id=(\d+)", line)
            if fm:
                log_fill_ids.append(int(fm.group(1)))
        if "ev=liquidate" in line:
            log_liq_count += 1

# The run's tick timeline is exactly the equity curve's timestamps; only the
# symbols the strategy actually touched need replaying.
timeline_set = {iso_ms(p["timestamp"]) for p in report["equity_curve"]}
traded = {o["symbol"] for o in orders.values()} | {t["symbol"] for t in trades}

df = pd.read_parquet(args.parquet)
if df["timestamp"].dtype != "int64":
    # The equity/bist panels store tz-naive timestamps where binance_1d.parquet
    # stores tz-aware ones, and .astype cannot add a zone. KLineFeed casts the
    # column to timestamp[ms, UTC] regardless, so naive values are UTC already.
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    # Pin nanoseconds before the integer cast: these files store microseconds,
    # and //1e6 on a microsecond count would yield seconds, not milliseconds.
    df["timestamp"] = ts.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
df = df[df["timestamp"].isin(timeline_set) & df["symbol"].isin(traded)]
df = df.reset_index(drop=True)
bars_by_ts = defaultdict(list)          # ts -> bars in file order
for row in df.itertuples(index=False):
    bars_by_ts[row.timestamp].append(row)
timeline = sorted(timeline_set)
printed_index = {s: {} for s in traded}   # symbol -> ts -> its printed-bar index
for s in traded:
    for i, ts in enumerate(sorted(df[df["symbol"] == s]["timestamp"])):
        printed_index[s][ts] = i

# ─── Broker replay (independent reimplementation) ────────────────────────────
cash = STARTING_CASH
positions = {}                    # sym -> dict(qty signed, entry, entry_id, lev)
book = {}                         # id -> dict(spec + live status)
open_ids = []                     # placement order
last_close = {}
bankrupt = False
fills = []
liqs = []
closes = []                       # (ts, symbol) position-close events
inject_at = defaultdict(list)
for i in sorted(strategy_orders):
    inject_at[strategy_orders[i]["ts_ms"]].append(i)
cancel_at = defaultdict(list)
for ts, oid in cancels:
    cancel_at[ts].append(oid)

PRIORITY = {"Market": 0, "Stop": 1, "Limit": 2}   # Conservative policy


def cancel_subtree(parent_id, keep=None):
    for oid, o in book.items():
        if o.get("parent_id") == parent_id and o["status"] == "Open":
            if oid != keep:
                o["status"] = "Cancelled"
            cancel_subtree(oid, keep)


def equity():
    e = cash
    for sym, p in positions.items():
        reserved = abs(p["qty"]) * p["entry"] / p["lev"]
        mark = last_close.get(sym, p["entry"])
        upnl = (mark - p["entry"] if p["qty"] > 0 else p["entry"] - mark) * abs(p["qty"])
        e += reserved + upnl
    return e


def liquidate(sym, price, ts):
    global cash
    p = positions[sym]
    qty = abs(p["qty"])
    side = "Sell" if p["qty"] > 0 else "Buy"
    pnl = (price - p["entry"] if p["qty"] > 0 else p["entry"] - price) * qty
    margin = qty * p["entry"] / p["lev"]
    if LOSS_CAP:
        pnl = max(pnl, -margin)
    fee = fee_of(qty * price, maker=False)   # forced closes take liquidity
    cash += margin + pnl - fee
    liqs.append(dict(ts=ts, symbol=sym, side=side, qty=qty, price=price, fee=fee))
    closes.append((ts, sym))
    cancel_subtree(p["entry_id"])
    del positions[sym]


def try_fill(oid, bar):
    """Mirror of BacktestBroker::try_fill. Returns True when the order left Open."""
    global cash
    o = book[oid]
    ts = bar.timestamp
    # fill price / trigger; a limit filled at its own price rested -> maker
    maker = False
    if o["type"] == "Market":
        price = bar.open
    elif o["type"] == "Limit":
        lim = o["price"]
        if o["side"] == "Buy":
            if bar.low > lim:
                return False
            price = min(lim, bar.open)
        else:
            if bar.high < lim:
                return False
            price = max(lim, bar.open)
        maker = (price == lim)
    else:                                     # Stop
        trig = o["price"]
        if o["side"] == "Buy":
            if bar.high < trig:
                return False
            price = max(trig, bar.open)
        else:
            if bar.low > trig:
                return False
            price = min(trig, bar.open)

    if o["ts_ms"] >= ts:
        flag("lookahead", f"order {oid} marketable on its own placement bar {ts}")

    # reduce-only guard: cancel rather than open or add
    if o.get("reduce_only"):
        p = positions.get(o["symbol"])
        reduces = p is not None and ((p["qty"] > 0) != (o["side"] == "Buy"))
        if not reduces:
            o["status"] = "Cancelled"
            cancel_subtree(oid)
            return True

    p = positions.get(o["symbol"])
    if p is None:
        notional = o["quantity"] * price
        if notional < MIN_NOTIONAL:
            o["status"] = "Rejected"
            cancel_subtree(oid)
            return True
        cost = notional / o["leverage"]
        fee = fee_of(notional, maker)
        if cost + fee > cash:
            o["status"] = "Rejected"
            cancel_subtree(oid)
            return True
        cash -= cost + fee
        positions[o["symbol"]] = dict(
            qty=o["quantity"] if o["side"] == "Buy" else -o["quantity"],
            entry=price, entry_id=oid, lev=o["leverage"])
        filled_qty = o["quantity"]
    else:
        long = p["qty"] > 0
        if long == (o["side"] == "Buy"):
            o["status"] = "Rejected"          # same-side add
            cancel_subtree(oid)
            return True
        filled_qty = min(o["quantity"], abs(p["qty"]))
        qty_before = abs(p["qty"])
        pnl = (price - p["entry"] if long else p["entry"] - price) * filled_qty
        fee = fee_of(filled_qty * price, maker)
        cash += filled_qty * p["entry"] / p["lev"] + pnl - fee
        p["qty"] += -filled_qty if long else filled_qty
        # dust snap, relative to the pre-close size
        if abs(p["qty"]) <= FLAT_EPS * max(qty_before, 1.0):
            p["qty"] = 0.0
        if p["qty"] == 0.0:
            cancel_subtree(p["entry_id"], keep=oid)
            del positions[o["symbol"]]
            closes.append((ts, o["symbol"]))
    o["status"] = "Filled"
    fills.append(dict(ts=ts, order_id=oid, symbol=o["symbol"], side=o["side"],
                      qty=filled_qty, price=price, fee=fee))
    return True


def process_bar(bar):
    global bankrupt
    sym = bar.symbol
    last_close[sym] = bar.close
    if bankrupt:
        return
    progressed = True
    while progressed:
        progressed = False
        candidates = []
        for oid in open_ids:
            o = book[oid]
            if o["status"] != "Open" or o["symbol"] != sym:
                continue
            pid = o.get("parent_id")
            if pid is not None and book.get(pid, {}).get("status") != "Filled":
                continue
            if o["ts_ms"] >= bar.timestamp:
                continue
            candidates.append(oid)
        candidates.sort(key=lambda i: PRIORITY[book[i]["type"]])   # stable
        for oid in candidates:
            if book[oid]["status"] != "Open":
                continue
            progressed = try_fill(oid, bar) or progressed
    # per-position liquidation, after the sweep (formulas §8 with maintenance m)
    p = positions.get(sym)
    if p is not None:
        long = p["qty"] > 0
        lp = (p["entry"] * (1.0 - 1.0 / p["lev"]) / (1.0 - MAINT_M) if long
              else p["entry"] * (1.0 + 1.0 / p["lev"]) / (1.0 + MAINT_M))
        breached = bar.low <= lp if long else bar.high >= lp
        if breached:
            fill_price = min(lp, bar.open) if long else max(lp, bar.open)
            liquidate(sym, fill_price, bar.timestamp)
    # account bankruptcy stop (floored at min_equity)
    if equity() <= MIN_EQUITY and not bankrupt:
        for s in list(positions):
            liquidate(s, last_close.get(s, positions[s]["entry"]), bar.timestamp)
        for oid, o in book.items():
            if o["status"] == "Open":
                o["status"] = "Cancelled"
        bankrupt = True


# ─── Run the replay ──────────────────────────────────────────────────────────
equity_curve = []
for ts in timeline:
    for bar in bars_by_ts.get(ts, []):
        process_bar(bar)
    equity_curve.append((ts, equity()))
    # strategy phase: cancels first, then this tick's placements
    for oid in cancel_at.get(ts, []):
        o = book.get(oid)
        if bankrupt or o is None or o["status"] != "Open":
            flag("cancel-replay", f"logged cancel of order {oid} at {ts} not applicable in replay")
            continue
        o["status"] = "Cancelled"
        cancel_subtree(oid)
    for oid in inject_at.get(ts, []):
        src = strategy_orders[oid]
        book[oid] = dict(status="Open",
                         symbol=src["symbol"], side=src["side"], type=src["type"],
                         price=src["price"], quantity=src["quantity"],
                         leverage=src["leverage"], reduce_only=src.get("reduce_only", False),
                         parent_id=src.get("parent_id"), ts_ms=src["ts_ms"])
        # registration-time validation (mirrors register_order)
        valid = (book[oid]["quantity"] > 0.0
                 and (book[oid]["type"] == "Market" or (book[oid]["price"] or 0) > 0.0)
                 and book[oid]["leverage"] >= 1.0 and not bankrupt)
        pid = book[oid]["parent_id"]
        if pid is not None and book.get(pid, {}).get("status") not in ("Open", "Filled"):
            valid = False
        if not valid:
            book[oid]["status"] = "Rejected"
        else:
            open_ids.append(oid)

# ─── Verification passes ─────────────────────────────────────────────────────
print(f"replay: {len(fills)} fills, {len(liqs)} liquidations, "
      f"{len(cancels)} strategy cancels, final cash {cash:.6f}")

# A. fills == non-liquidation trades, one by one
rep = [t for t in trades if not t["liquidation"]]
if len(rep) != len(fills):
    flag("fill-count", f"replay {len(fills)} vs report {len(rep)}")
for f, t in zip(sorted(fills, key=lambda x: (x['ts'], x['order_id'])),
                sorted(rep, key=lambda x: (x['ts_ms'], x['order_id']))):
    if (f["order_id"] != t["order_id"] or f["ts"] != t["ts_ms"] or f["side"] != t["side"]
            or abs(f["qty"] - t["quantity"]) > 1e-9 or abs(f["price"] - t["price"]) > 1e-9
            or abs(f["fee"] - t.get("fee", 0.0)) > 1e-9):
        flag("fill-mismatch", f"replay {f} vs report trade id {t['id']} "
                              f"(order {t['order_id']} ts {t['ts_ms']} {t['side']} "
                              f"{t['quantity']} @ {t['price']} fee {t.get('fee')})")

# B. liquidations, one by one
rep_liq = [t for t in trades if t["liquidation"]]
if len(rep_liq) != len(liqs):
    flag("liq-count", f"replay {len(liqs)} vs report {len(rep_liq)}")
for f, t in zip(sorted(liqs, key=lambda x: x["ts"]), sorted(rep_liq, key=lambda x: x["ts_ms"])):
    if (f["ts"] != t["ts_ms"] or f["symbol"] != t["symbol"] or f["side"] != t["side"]
            or abs(f["qty"] - t["quantity"]) > 1e-9 or abs(f["price"] - t["price"]) > 1e-9
            or abs(f["fee"] - t.get("fee", 0.0)) > 1e-9):
        flag("liq-mismatch", f"replay {f} vs report {t}")

# C. final order statuses
for oid, src in strategy_orders.items():
    got = book.get(oid, {}).get("status")
    if got != src["status"]:
        flag("status", f"order {oid}: replay {got} vs report {src['status']}")

# D. cash + equity curve, point by point
if abs(cash - report["metrics"]["ending_cash"]) > 1e-6:
    flag("cash", f"replay {cash} vs report {report['metrics']['ending_cash']}")
rep_curve = [(iso_ms(p["timestamp"]), p["equity"]) for p in report["equity_curve"]]
if len(rep_curve) != len(equity_curve):
    flag("equity-count", f"replay {len(equity_curve)} vs report {len(rep_curve)}")
else:
    bad = sum(1 for (t1, e1), (t2, e2) in zip(equity_curve, rep_curve)
              if t1 != t2 or abs(e1 - e2) > 1e-6)
    if bad:
        flag("equity-curve", f"{bad}/{len(rep_curve)} points differ")

# E. per-fill trigger/price invariants against raw bars (belt and braces)
bar_lookup = {(b.timestamp, b.symbol): b for ts in timeline for b in bars_by_ts.get(ts, [])}
for f in fills:
    o = strategy_orders[f["order_id"]]
    b = bar_lookup[(f["ts"], f["symbol"])]
    typ, side, ref = o["type"], o["side"], o["price"]
    ok = True
    if typ == "Market":
        ok = f["price"] == b.open
    elif typ == "Limit":
        ok = (b.low <= ref and f["price"] == min(ref, b.open)) if side == "Buy" \
            else (b.high >= ref and f["price"] == max(ref, b.open))
    else:
        ok = (b.high >= ref and f["price"] == max(ref, b.open)) if side == "Buy" \
            else (b.low <= ref and f["price"] == min(ref, b.open))
    if not ok:
        flag("price-rule", f"order {f['order_id']} {typ} {side} ref {ref} "
                           f"filled {f['price']} on bar o={b.open} h={b.high} l={b.low}")
    if o["ts_ms"] >= f["ts"]:
        flag("lookahead", f"order {f['order_id']} filled at {f['ts']} but placed at {o['ts_ms']}")

# F. strategy gating audit: entries vs positions, cooldown, single pending
entry_ids = [i for i, o in strategy_orders.items()
             if o.get("parent_id") is None and not o.get("reduce_only", False)]
if not args.skip_gating:
    open_intervals = defaultdict(list)
    events = sorted(
        [(f["ts"], "fill", f) for f in fills] + [(l["ts"], "liq", l) for l in liqs],
        key=lambda e: (e[0], e[2].get("order_id", 1 << 62)))
    pos_state = {}
    for ts, kind, ev in events:
        sym = ev["symbol"]
        if sym not in pos_state:
            pos_state[sym] = [ts, ev["qty"] if ev["side"] == "Buy" else -ev["qty"]]
        else:
            opened_ts, q = pos_state[sym]
            q += ev["qty"] if ev["side"] == "Buy" else -ev["qty"]
            if abs(q) < 1e-12:
                open_intervals[sym].append((opened_ts, ts))
                del pos_state[sym]
            else:
                pos_state[sym][1] = q
    for oid in entry_ids:
        o = strategy_orders[oid]
        sym, placed = o["symbol"], o["ts_ms"]
        for (a, b) in open_intervals[sym]:
            if a <= placed < b:
                flag("gate-positioned", f"entry {oid} placed at {placed} inside open interval "
                                        f"[{a}, {b}) on {sym}")
        closes_before = [b for (_, b) in open_intervals[sym] if b <= placed]
        if closes_before:
            last_close_ts = max(closes_before)
            gap = printed_index[sym][placed] - printed_index[sym][last_close_ts]
            if gap < COOLDOWN_BARS:
                flag("gate-cooldown", f"entry {oid} on {sym} placed {gap} printed bars after "
                                      f"the close at {last_close_ts} (cooldown {COOLDOWN_BARS})")
    inject_order = sorted(entry_ids, key=lambda i: (strategy_orders[i]["ts_ms"], i))
    for idx, oid in enumerate(inject_order):
        o = strategy_orders[oid]
        for prev in inject_order[:idx]:
            po = strategy_orders[prev]
            if po["symbol"] == o["symbol"] and po["status"] == "Open":
                flag("gate-pending", f"entry {oid} placed while earlier entry {prev} "
                                     f"is still Open at end of run")

# G. log cross-check
if args.log:
    if len(log_fill_ids) != len(fills):
        flag("log-fills", f"log has {len(log_fill_ids)} ev=fill vs replay {len(fills)}")
    if log_liq_count != len(liqs):
        flag("log-liqs", f"log has {log_liq_count} ev=liquidate vs replay {len(liqs)}")

print()
print(f"checked: {len(rep)} fills | {len(rep_liq)} liquidations | "
      f"{len(strategy_orders)} order statuses | {len(rep_curve)} equity points | "
      f"{len(entry_ids)} entries against gating rules")
if violations:
    print(f"\nRESULT: {len(violations)} VIOLATIONS")
    sys.exit(1)
print("\nRESULT: CLEAN — replay reproduces the run exactly; all invariants hold")
