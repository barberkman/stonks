"""Compare every chart pattern's backtest against buy-and-hold of its market.

`pattern_sweep.py run` produces one engine report per pattern per dataset. This
driver reads those reports for two or more markets, computes the market's own
return over the identical window straight from the price panel, and renders the
comparison as a table (stdout markdown + CSV) and a figure.

Usage (from the project root, with the app venv for pandas/pyarrow/matplotlib):

    app/python/.venv/bin/python tools/pattern_compare.py \
        --market ndx100 app/data/us_ndx100_2020_1d.parquet app/reports/pattern-sweep/ndx100 \
        --market bist   app/data/bist_1d.parquet          app/reports/pattern-sweep/bist \
        --out-csv app/reports/pattern-sweep/comparison.csv \
        --out-png app/reports/pattern-sweep/comparison.png

The benchmark is an equal-weight, daily-rebalanced index of every symbol in the
panel — the same panel the strategy scans, so the two numbers answer the same
question over the same bars. A symbol contributes from its first quoted close to
its last, and no longer: a stale price carried past a delisting would dilute the
index toward zero return, and a symbol that lists mid-window must not be credited
with the move it missed. Days a listed symbol does not print are carried flat.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pattern_sweep import load_patterns, report_path, row_for  # noqa: E402

# dataviz reference palette, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"      # categorical slot 1 — the patterns
MARKET = "#e34948"      # the diverging pair's warm pole — the benchmark


# ─── the benchmark ──────────────────────────────────────────────────────────


@dataclass
class Market:
    name: str
    data: Path
    reports: Path
    return_pct: float = 0.0
    max_dd_pct: float = 0.0
    passive_pct: float = 0.0
    median_symbol_pct: float = 0.0
    symbols: int = 0
    bars: int = 0
    start: str = ""
    end: str = ""


def benchmark(path: Path) -> dict:
    """Equal-weight daily-rebalanced buy-and-hold of the whole panel."""
    df = pd.read_parquet(path, columns=["timestamp", "symbol", "close"])
    close = df.pivot_table(index="timestamp", columns="symbol", values="close",
                           aggfunc="last").sort_index()

    # A symbol is in the index only between its first and last quoted close.
    listed = close.notna()
    live = listed.cummax() & listed[::-1].cummax()[::-1]
    held = close.ffill().where(live)

    rets = held.pct_change(fill_method=None).where(live)
    daily = rets.mean(axis=1, skipna=True).fillna(0.0)
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0

    first = held.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    last = held.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    per_symbol = (last / first - 1.0) * 100.0
    # The do-nothing benchmark: equal money into whatever was already listed on
    # day one, held to the end. Daily rebalancing is a strategy of its own and
    # on a 616-name small-cap panel it earns a large diversification bonus, so
    # the passive number is reported beside it rather than folded into it.
    at_start = listed.iloc[0]
    passive = per_symbol[at_start[at_start].index].mean()

    ts = pd.to_datetime(close.index, unit="ms")
    return {"return_pct": float(equity.iloc[-1] - 1.0) * 100.0,
            "max_dd_pct": float(-drawdown.min()) * 100.0,
            "passive_pct": float(passive),
            "median_symbol_pct": float(per_symbol.median()),
            "symbols": int(close.shape[1]),
            "bars": int(len(df)),
            "start": str(ts.min().date()),
            "end": str(ts.max().date()),
            "equity": pd.Series(equity.values, index=ts)}


# ─── joining the sweeps ─────────────────────────────────────────────────────


def sweep_rows(reports: Path) -> dict:
    """index -> Row for every pattern whose report is in `reports`."""
    return {i: row_for(i, n, s, t, report_path(reports, i, n))
            for i, n, s, t in load_patterns()}


def build_frame(markets: list[Market], sweeps: dict) -> pd.DataFrame:
    meta = load_patterns()
    out = []
    for index, name, side, tradeable in meta:
        rec = {"index": index, "pattern": name, "side": side,
               "tradeable": tradeable}
        for m in markets:
            r = sweeps[m.name][index]
            rec[f"{m.name}_status"] = r.status
            rec[f"{m.name}_closed"] = r.closed
            rec[f"{m.name}_win_pct"] = r.win_pct
            rec[f"{m.name}_return_pct"] = r.return_pct if r.status == "ok" else np.nan
            rec[f"{m.name}_max_dd_pct"] = r.max_dd_pct if r.status == "ok" else np.nan
            rec[f"{m.name}_excess_pct"] = (r.return_pct - m.return_pct
                                           if r.status == "ok" else np.nan)
            rec[f"{m.name}_profit_factor"] = r.profit_factor
            rec[f"{m.name}_note"] = r.note
        out.append(rec)
    return pd.DataFrame(out)


def traded(frame: pd.DataFrame, m: Market) -> pd.DataFrame:
    return frame[(frame[f"{m.name}_status"] == "ok")
                 & (frame[f"{m.name}_closed"] > 0)]


# ─── the figure ─────────────────────────────────────────────────────────────


def render(markets: list[Market], frame: pd.DataFrame, out_png: Path, top: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "grid.color": GRID,
    })

    rows = len(markets)
    head_in = 1.0                      # inches reserved for the figure heading
    fig_h = 4.4 * rows + head_in
    fig, axes = plt.subplots(rows, 2, figsize=(15.5, fig_h),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    axes = np.atleast_2d(axes)

    for r, m in enumerate(markets):
        sub = traded(frame, m).sort_values(f"{m.name}_return_pct", ascending=False)
        vals = sub[f"{m.name}_return_pct"].to_numpy()
        beat = int((vals > m.return_pct).sum())

        # ── left: where the whole library lands against the market ──────────
        ax = axes[r][0]
        ax.hist(vals, bins=32, color=SERIES, edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(m.return_pct, color=MARKET, linewidth=2.0, zorder=5)
        ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
        ax.annotate(f"market  {m.return_pct:+,.0f}%",
                    xy=(m.return_pct, ax.get_ylim()[1]),
                    xytext=(6, -10), textcoords="offset points",
                    color=MARKET, fontsize=10, fontweight="bold",
                    ha="left", va="top")
        ax.set_title(f"{m.name} — {len(sub)} patterns traded, {beat} beat the market",
                     fontsize=12, fontweight="bold", color=INK, loc="left", pad=10)
        ax.set_xlabel("total return over the backtest (%)", fontsize=10)
        ax.set_ylabel("patterns", fontsize=10)

        # ── right: the leaders, named ───────────────────────────────────────
        ax = axes[r][1]
        head = sub.head(top).iloc[::-1]
        y = np.arange(len(head))
        ax.barh(y, head[f"{m.name}_return_pct"], height=0.72, color=SERIES,
                edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(m.return_pct, color=MARKET, linewidth=2.0, zorder=5)
        ax.axvline(0.0, color=AXIS, linewidth=1.0, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(head["pattern"], fontsize=8.5, color=INK_2)
        # Headroom above the top bar so the market label sits clear of its
        # value label rather than on top of it.
        ax.set_ylim(-0.8, len(head) + 0.7)
        for i, (v, n) in enumerate(zip(head[f"{m.name}_return_pct"],
                                       head[f"{m.name}_closed"])):
            ax.annotate(f"{v:+,.0f}%  ({n} trades)", xy=(v, i),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
        ax.annotate(f"market  {m.return_pct:+,.0f}%",
                    xy=(m.return_pct, len(head) + 0.3),
                    xytext=(6, 0), textcoords="offset points",
                    color=MARKET, fontsize=10, fontweight="bold",
                    ha="left", va="center")
        ax.set_title(f"{m.name} — top {len(head)} by return", fontsize=12,
                     fontweight="bold", color=INK, loc="left", pad=10)
        ax.set_xlabel("total return over the backtest (%)", fontsize=10)
        ax.margins(x=0.18)

    for ax in axes.ravel():
        ax.grid(axis="x", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=9)

    span = markets[0]
    fig.text(0.008, 1.0 - 0.38 / fig_h,
             "Bulkowski chart patterns vs. buy-and-hold, "
             f"{span.start} to {span.end}",
             fontsize=15, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.008, 1.0 - 0.70 / fig_h,
             "one engine run per pattern · equal-weight daily-rebalanced panel "
             "as the market · red rule is the market's own return",
             fontsize=10, color=MUTED, ha="left", va="center")
    fig.tight_layout(rect=(0, 0, 1, 1.0 - head_in / fig_h))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    print(f"figure -> {out_png}")


# ─── the table ──────────────────────────────────────────────────────────────


def fmt(v, spec, dash="-"):
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def print_table(markets: list[Market], frame: pd.DataFrame, top: int, sort_by: str):
    key = f"{sort_by}_excess_pct"
    ranked = frame.sort_values(key, ascending=False).head(top)

    head = ["#", "pattern", "side"]
    for m in markets:
        head += [f"{m.name} ret%", f"{m.name} vs mkt", f"{m.name} trades",
                 f"{m.name} win%", f"{m.name} maxDD%"]
    print("| " + " | ".join(head) + " |")
    print("|---|---|---|" + "--:|" * (len(markets) * 5))
    for _, r in ranked.iterrows():
        cells = [str(r["index"]), f"`{r['pattern']}`", r["side"]]
        for m in markets:
            cells += [fmt(r[f"{m.name}_return_pct"], "+.1f"),
                      fmt(r[f"{m.name}_excess_pct"], "+.1f"),
                      str(int(r[f"{m.name}_closed"])),
                      fmt(r[f"{m.name}_win_pct"], ".0f"),
                      fmt(r[f"{m.name}_max_dd_pct"], ".1f")]
        print("| " + " | ".join(cells) + " |")

    print()
    print("| market | window | symbols | market return% | market maxDD% | "
          "buy-and-hold% | median symbol% | patterns traded | beat market |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in markets:
        sub = traded(frame, m)
        beat = int((sub[f"{m.name}_return_pct"] > m.return_pct).sum())
        print(f"| {m.name} | {m.start} → {m.end} | {m.symbols} | "
              f"{m.return_pct:+,.1f} | {m.max_dd_pct:.1f} | "
              f"{m.passive_pct:+,.1f} | {m.median_symbol_pct:+,.1f} | {len(sub)} | "
              f"{beat} ({beat / max(len(sub), 1) * 100:.0f}%) |")


def write_csv(markets: list[Market], frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    meta = path.with_name(path.stem + "-markets.csv")
    with meta.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["market", "data", "start", "end", "symbols", "bars",
                    "return_pct", "max_dd_pct", "passive_pct",
                    "median_symbol_pct"])
        for m in markets:
            w.writerow([m.name, m.data, m.start, m.end, m.symbols, m.bars,
                        m.return_pct, m.max_dd_pct, m.passive_pct,
                        m.median_symbol_pct])
    print(f"csv -> {path}\ncsv -> {meta}")


# ─── cli ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", action="append", nargs=3, required=True,
                    metavar=("NAME", "PARQUET", "REPORT_DIR"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sort-by", default="", help="market name to rank the table by")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-png", default="")
    args = ap.parse_args()

    markets = [Market(n, Path(d), Path(r)) for n, d, r in args.market]
    for m in markets:
        b = benchmark(m.data)
        m.return_pct = b["return_pct"]
        m.max_dd_pct = b["max_dd_pct"]
        m.passive_pct = b["passive_pct"]
        m.median_symbol_pct = b["median_symbol_pct"]
        m.symbols, m.bars = b["symbols"], b["bars"]
        m.start, m.end = b["start"], b["end"]

    sweeps = {m.name: sweep_rows(m.reports) for m in markets}
    frame = build_frame(markets, sweeps)

    print_table(markets, frame, args.top, args.sort_by or markets[0].name)
    if args.out_csv:
        write_csv(markets, frame, Path(args.out_csv))
    if args.out_png:
        render(markets, frame, Path(args.out_png), args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
