"""Backtest every chart pattern in app/python/patterns.py and tabulate the results.

`PatternsStrategy` trades exactly one pattern per run, chosen by its `pattern`
param (an index into `patterns.PATTERNS`). This driver runs the real engine once
per pattern — same broker, fills and fees as any other backtest — collects the
per-run report JSONs, and renders a league table.

Usage (from the project root, with the app venv for pandas/pyarrow):

    # sweep one dataset
    app/python/.venv/bin/python tools/pattern_sweep.py run \
        --data app/data/us_ndx100_2020_1d.parquet \
        --out-dir app/reports/pattern-sweep/ndx100 \
        --cash 100000 --jobs 8

    # render the table from whatever has finished
    app/python/.venv/bin/python tools/pattern_sweep.py table \
        app/reports/pattern-sweep/ndx100

`run` is restartable: --resume skips any pattern whose report already exists and
parses, so an interrupted sweep picks up where it stopped.

The engine's report carries closed_trades / winning_trades / win_rate_pct /
return_pct / max_drawdown_pct but no per-trade statistics, so avg win, avg loss,
profit factor and expectancy are reconstructed here from the `trades` array by
replaying fills into round trips. That replay is a port of compute_metrics() in
app/src/report.h; `table` asserts it reproduces the engine's own closed_trades
and winning_trades for every report, so a divergence is caught rather than
quietly tabulated.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_APP = REPO / "build" / "macos-release" / "app" / "app"


# ─── the pattern registry ────────────────────────────────────────────────────


def load_patterns():
    """(index, name, side, tradeable) for every registered pattern."""
    sys.path.insert(0, str(REPO / "app" / "python"))
    sys.path.insert(0, str(REPO / "python"))
    import patterns as mod
    return [(i, s.name, s.side, s.tradeable) for i, s in enumerate(mod.PATTERNS)]


# ─── running ────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    index: int
    name: str
    status: str          # "ok" | "FAILED" | "TIMEOUT"
    seconds: float
    detail: str = ""


def report_path(out_dir: Path, index: int, name: str) -> Path:
    return out_dir / f"{index:03d}-{name}.json"


def run_one(app: Path, out_dir: Path, index: int, name: str, args) -> RunResult:
    report = report_path(out_dir, index, name)
    log = out_dir / "logs" / f"{index:03d}-{name}.log.gz"
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [str(app),
           "--strategy", "patterns:PatternsStrategy",
           "--param", f"pattern={index}",
           "--cash", str(args.cash),
           "--data", args.data,
           "--out", str(report)]
    for flag, value in (("--start", args.start), ("--end", args.end),
                        ("--symbols", args.symbols)):
        if value:
            cmd += [flag, value]

    t0 = time.time()
    try:
        # The strategy narrates every arm and exit, which on a 632-symbol panel
        # is a lot of text nobody reads unless a run looks wrong — keep it, but
        # compressed and out of the terminal.
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return RunResult(index, name, "TIMEOUT", time.time() - t0,
                         f"exceeded {args.timeout}s")
    elapsed = time.time() - t0

    with gzip.open(log, "wt") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n=== stderr ===\n")
            f.write(proc.stderr)

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return RunResult(index, name, "FAILED", elapsed, " / ".join(tail))
    if not report.exists():
        return RunResult(index, name, "FAILED", elapsed, "no report written")
    return RunResult(index, name, "ok", elapsed)


def cmd_run(args) -> int:
    app = Path(args.app)
    if not app.exists():
        print(f"engine binary not found: {app}\n"
              f"build it with: cmake --build --preset macos-release", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_patterns()
    if args.patterns:
        wanted = parse_index_spec(args.patterns, len(entries))
        entries = [e for e in entries if e[0] in wanted]

    todo = []
    skipped = 0
    for index, name, _side, _tradeable in entries:
        if args.resume and readable_report(report_path(out_dir, index, name)):
            skipped += 1
            continue
        todo.append((index, name))

    # Longest-first: the busted patterns re-scan a parent detector across a
    # 40-bar window, so they are still the slowest even after the pivot reuse.
    # Starting them first stops them straggling past an otherwise idle pool.
    todo.sort(key=lambda t: (not t[1].startswith("busted_"), t[0]))

    print(f"{len(todo)} runs ({skipped} already done) -> {out_dir}  "
          f"jobs={args.jobs} cash={args.cash} data={Path(args.data).stem}")
    started = time.time()
    results: list[RunResult] = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run_one, app, out_dir, i, n, args): (i, n)
                   for i, n in todo}
        for done in futures.as_completed(pending):
            r = done.result()
            results.append(r)
            flag = "" if r.status == "ok" else f"  <-- {r.status}: {r.detail}"
            print(f"  [{len(results):>3}/{len(todo)}] {r.index:>3} {r.name:<44} "
                  f"{r.seconds:6.1f}s{flag}", flush=True)

    bad = [r for r in results if r.status != "ok"]
    print(f"\ndone in {time.time() - started:.0f}s — "
          f"{len(results) - len(bad)} ok, {len(bad)} failed")
    for r in bad:
        print(f"  {r.status} {r.index} {r.name}: {r.detail}")
    return 0


def parse_index_spec(spec: str, total: int) -> set[int]:
    """"0-9,42,100-120" -> the set of indices it names."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), min(int(hi), total - 1) + 1))
        else:
            out.add(int(part))
    return out


def readable_report(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open() as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


# ─── metrics ────────────────────────────────────────────────────────────────


def round_trips(trades: list[dict]) -> list[float]:
    """Realized P&L of every closed round trip, net of fees.

    A port of compute_metrics() in app/src/report.h (the round-trip
    reconstruction at lines 120-180). Per symbol carry a signed position, its
    average entry and the P&L realized so far in the open cycle; a same-side
    fill scales in and blends the average, an opposing fill realizes P&L on the
    closed quantity, and the cycle closes when the position returns to flat. A
    fill that overshoots flat closes the cycle and opens a fresh one with the
    leftover. Each fill's fee is charged to the cycle it belongs to, so the
    numbers here are net of costs exactly as the engine's are.
    """
    positions: dict[str, dict] = {}
    closed: list[float] = []
    for t in trades:
        fill = t["quantity"] if t["side"] == "Buy" else -t["quantity"]
        pos = positions.setdefault(t["symbol"],
                                   {"qty": 0.0, "avg": 0.0, "realized": 0.0})
        pos["realized"] -= t["fee"]

        if pos["qty"] == 0.0:
            pos["qty"] = fill
            pos["avg"] = t["price"]
            continue

        if (pos["qty"] > 0.0) == (fill > 0.0):
            prev, add = abs(pos["qty"]), abs(fill)
            pos["avg"] = (pos["avg"] * prev + t["price"] * add) / (prev + add)
            pos["qty"] += fill
            continue

        closing = min(abs(fill), abs(pos["qty"]))
        pos["realized"] += ((t["price"] - pos["avg"]) if pos["qty"] > 0.0
                            else (pos["avg"] - t["price"])) * closing
        pos["qty"] += -closing if pos["qty"] > 0.0 else closing

        if pos["qty"] == 0.0:
            closed.append(pos["realized"])
            pos["realized"] = 0.0
            pos["avg"] = 0.0
            leftover = abs(fill) - closing
            if leftover > 0.0:
                pos["qty"] = leftover if fill > 0.0 else -leftover
                pos["avg"] = t["price"]
    return closed


@dataclass
class Row:
    index: int
    name: str
    side: str
    tradeable: bool
    status: str = "ok"
    closed: int = 0
    win_pct: Optional[float] = None
    return_pct: float = 0.0
    max_dd_pct: float = 0.0
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    expectancy: Optional[float] = None
    orders: int = 0
    fees: float = 0.0
    note: str = ""


def row_for(index, name, side, tradeable, path: Path) -> Row:
    row = Row(index, name, side, tradeable)
    if not readable_report(path):
        row.status = "MISSING"
        row.note = "no report"
        return row
    with path.open() as f:
        rep = json.load(f)
    m = rep["metrics"]
    row.closed = m["closed_trades"]
    row.win_pct = m.get("win_rate_pct")
    row.return_pct = m["return_pct"]
    row.max_dd_pct = m["max_drawdown_pct"]
    row.orders = m["orders_placed"]
    row.fees = m["total_fees"]

    pnl = round_trips(rep["trades"])
    # The derived statistics are only trustworthy if the replay agrees with the
    # engine about what a closed trade is — check, don't assume.
    wins = [x for x in pnl if x > 0.0]
    losses = [x for x in pnl if x <= 0.0]
    if len(pnl) != m["closed_trades"] or len(wins) != m["winning_trades"]:
        row.status = "REPLAY-MISMATCH"
        row.note = (f"replay {len(pnl)}/{len(wins)} vs engine "
                    f"{m['closed_trades']}/{m['winning_trades']}")
        return row

    if wins:
        row.avg_win = sum(wins) / len(wins)
    if losses:
        row.avg_loss = sum(losses) / len(losses)
    gross_win, gross_loss = sum(wins), -sum(losses)
    if gross_loss > 0.0:
        row.profit_factor = gross_win / gross_loss
    elif gross_win > 0.0:
        row.profit_factor = float("inf")
    if pnl:
        row.expectancy = sum(pnl) / len(pnl)

    if not tradeable:
        row.note = "identify-only"
    elif row.orders == 0:
        row.note = "never fired"
    elif row.closed == 0:
        row.note = "never confirmed"
    return row


def fmt(value, spec, dash="-"):
    return dash if value is None else format(value, spec)


def cmd_table(args) -> int:
    out_dir = Path(args.dir)
    rows = [row_for(i, n, s, t, report_path(out_dir, i, n))
            for i, n, s, t in load_patterns()]

    keys = {"return": lambda r: -r.return_pct,
            "win": lambda r: -(r.win_pct or -1),
            "trades": lambda r: -r.closed,
            "pf": lambda r: -(r.profit_factor or -1)}
    traded = [r for r in rows if r.status == "ok" and r.closed > 0]
    rest = [r for r in rows if not (r.status == "ok" and r.closed > 0)]
    traded.sort(key=keys[args.sort])
    rest.sort(key=lambda r: r.index)
    ordered = traded + (rest if not args.traded_only else [])
    if args.top:
        ordered = ordered[:args.top]

    print(f"| # | pattern | side | closed | win% | return% | avg win | avg loss "
          f"| PF | maxDD% | note |")
    print("|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for r in ordered:
        note = r.note if r.status == "ok" else r.status + (f" ({r.note})" if r.note else "")
        print(f"| {r.index} | `{r.name}` | {r.side} | {r.closed} | "
              f"{fmt(r.win_pct, '.0f')} | {r.return_pct:+.2f} | "
              f"{fmt(r.avg_win, '+.0f')} | {fmt(r.avg_loss, '+.0f')} | "
              f"{fmt(r.profit_factor, '.2f')} | {r.max_dd_pct:.1f} | {note} |")

    ok = [r for r in rows if r.status == "ok"]
    print(f"\n{len(rows)} patterns · {len(traded)} traded · "
          f"{sum(1 for r in ok if r.note == 'never fired')} never fired · "
          f"{sum(1 for r in ok if r.note == 'never confirmed')} never confirmed · "
          f"{sum(1 for r in ok if r.note == 'identify-only')} identify-only · "
          f"{sum(1 for r in rows if r.status != 'ok')} problem rows")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "pattern", "side", "status", "closed", "win_pct",
                        "return_pct", "max_dd_pct", "avg_win", "avg_loss",
                        "profit_factor", "expectancy", "orders", "fees", "note"])
            for r in sorted(rows, key=lambda r: r.index):
                w.writerow([r.index, r.name, r.side, r.status, r.closed, r.win_pct,
                            r.return_pct, r.max_dd_pct, r.avg_win, r.avg_loss,
                            r.profit_factor, r.expectancy, r.orders, r.fees, r.note])
        print(f"csv -> {args.csv}")
    return 0 if all(r.status in ("ok",) for r in rows) else 1


# ─── cli ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="backtest every pattern")
    r.add_argument("--data", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--cash", type=float, default=100000.0)
    r.add_argument("--start", default="")
    r.add_argument("--end", default="")
    r.add_argument("--symbols", default="")
    r.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 4) - 2))
    r.add_argument("--timeout", type=float, default=7200.0)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--patterns", default="", help='index spec, e.g. "0-9,131"')
    r.add_argument("--app", default=str(DEFAULT_APP))
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("table", help="render the league table")
    t.add_argument("dir")
    t.add_argument("--sort", choices=["return", "win", "trades", "pf"], default="return")
    t.add_argument("--top", type=int, default=0)
    t.add_argument("--traded-only", action="store_true")
    t.add_argument("--csv", default="")
    t.set_defaults(func=cmd_table)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
