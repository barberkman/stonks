"""Single-date alert screen — the port's answer to bist's `configured_alerts`.

Trains the three heads through a report date, scores that date's liquid universe,
and prints bist's table:

    Symbol    n    h5 sigma    h5 exp%   h10 sigma   h10 exp%     h5 dn   composite

`n` counts the up configs whose sigma clears SIGMA_THRESHOLD, `composite` is the
mean of the two up sigmas, and rows are sorted by composite descending — all
bist's, so the two outputs can be read side by side.

**This is a screen, not a backtest.** It trains on data through the very date it
scores, exactly as bist's does, so nothing here is evidence that the rules make
money. `app/python/algo_trade.py`'s `AlgoTradeStrategy` is the backtest, and it
refuses to trade any bar its artifact saw.

Two things not to expect. The numbers will not equal bist's for the same date:
bist runs on a different feed (its `open` is a daily VWAP, its `volume` is lira
turnover, its history starts in 2016), so the code matches and the inputs do not —
`tools/bist_parity.py` is what measures the code half. And the down column is an
overlay: bist displays it and never gates on it, so a high `h5 dn` next to a high
up sigma marks an ambiguous setup rather than a rejected one.

Run from the project root:

    app/python/.venv/bin/python tools/bist_alerts.py
    app/python/.venv/bin/python tools/bist_alerts.py --report-date 2026-06-30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "app" / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app" / "python"))

import algo_trade as A  # noqa: E402

# bist's SIGMA_THRESHOLD: a symbol reaches the table if any up config clears it.
# A pragmatic "above noise" cutoff that keeps the output readable, not a rule.
SIGMA_THRESHOLD = 1.0

log = logging.getLogger("bist_alerts")


def screen(frame, report_date=None, *, train_start=A.TRAIN_START_DATE,
           rounds=None):
    """Fit through `report_date` and score it. Returns (table, model, date)."""
    panel = A._panel(frame)
    index = panel["close"].index
    if report_date is None:
        end = len(index)
    else:
        end = int(np.searchsorted(index.to_numpy(),
                                  A._as_index_value(report_date, index),
                                  side="right"))
    if end <= 0:
        raise SystemExit(f"--report-date {report_date} precedes every bar")
    scored_date = index[end - 1]

    params = dict(A.XGB_PARAMS)
    if rounds is not None:
        params["n_estimators"] = rounds
    model = A.ManipulationModel(params=params, train_start=train_start)
    model.train(panel, train_end=scored_date)

    # Score off the whole truncated panel rather than a trailing window, so `obv`
    # keeps the running level the fit was built on. The strategy cannot do this —
    # it only ever holds LOOKBACK bars — which is why it tracks obv across ticks
    # and hands it to `signal` instead.
    sig = model.signal(A._truncate(panel, end))
    return build_table(sig, model.heads), model, scored_date


def build_table(sig, heads):
    """bist's `_build_sigma_filter_table`: rows above the threshold, ranked."""
    up = [h for h in heads if h.direction == "up"]
    dn = [h for h in heads if h.direction == "dn"]

    scored = sig.loc[sig[[h.name for h in up]].notna().all(axis=1)]
    above = scored.loc[(scored[[h.name for h in up]] > SIGMA_THRESHOLD).any(axis=1)]
    if above.empty:
        return above.assign(n=[], composite=[])

    out = pd.DataFrame(index=above.index)
    out["n"] = (above[[h.name for h in up]] > SIGMA_THRESHOLD).sum(axis=1)
    for head in up:
        out[head.name] = above[head.name]
        out[f"{head.name}_pct"] = above[f"{head.name}_pct"]
    for head in dn:
        out[head.name] = above[head.name]
    out["composite"] = above[[h.name for h in up]].mean(axis=1)
    # Ties break on symbol so two runs of the same fit print the same order.
    return out.sort_values(["composite", "n"], ascending=False, kind="mergesort")


def render(table, model, scored_date, universe):
    heads = {h.name: h for h in model.heads}
    up = [h.name for h in model.heads if h.direction == "up"]
    dn = [h.name for h in model.heads if h.direction == "dn"]

    lines = ["BIST extreme-up alert screen (stonks port of bist configured_alerts)",
             "=" * 78,
             f"Report date:         {pd.Timestamp(scored_date).date()}",
             f"Liquidity universe:  {len(model.liquid)} / {universe} symbols "
             f"(median turnover >= p{model.min_turnover_percentile:.0%})",
             f"Training window:     {model.train_start} through the report date",
             "",
             "NOT A BACKTEST: the fit saw every bar up to and including the date",
             "it scores, exactly as bist's screen does. Use AlgoTradeStrategy for",
             "an out-of-sample result.",
             ""]

    columns = []
    for name in up:
        columns += [(name, f"h{heads[name].horizon} sigma"),
                    (f"{name}_pct", f"h{heads[name].horizon} exp%")]
    columns += [(name, f"h{heads[name].horizon} dn") for name in dn]
    columns += [("composite", "composite")]

    width = max(11, max(len(label) for _, label in columns))
    lines.append(f"-- Symbols with up sigma > {SIGMA_THRESHOLD:.1f} in any config "
                 f"(n = configs above it; sorted by composite) --")
    if table.empty:
        lines.append(f"   (nothing cleared up sigma > {SIGMA_THRESHOLD:.1f})")
    else:
        lines.append(f"   {'Symbol':<8} {'n':>2}  "
                     + "  ".join(f"{label:>{width}}" for _, label in columns))
        for symbol, row in table.iterrows():
            cells = []
            for key, _ in columns:
                value = row.get(key)
                cells.append(f"{'.':>{width}}" if pd.isna(value)
                             else f"{value:>+{width}.2f}")
            lines.append(f"   {symbol:<8} {int(row['n']):>2}  " + "  ".join(cells))

    lines += ["",
              "-- Caveats, bist's own --",
              "* exp% = (e^(sigma * daily_vol * sqrt H) - 1) * 100, the forward",
              "  EXCESS return in percent. NOT a calibrated forecast; rank by",
              "  sigma and treat exp% as a magnitude check.",
              "* Up sigma is the model's belief in a clean +move (no -10% dip on",
              "  the way); down sigma is the unconditional belief in a -move.",
              "* High up sigma AND high down sigma on one name is a spec conflict.",
              "  Read it as ambiguous; nothing in the pipeline rejects it.",
              "* A triage queue, not a trading signal."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="app/data/bist_1d.parquet")
    parser.add_argument("--report-date", default=None,
                        help="date to score; defaults to the last bar in the feed")
    parser.add_argument("--train-start", default=A.TRAIN_START_DATE)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the report here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(args.data)
    log.info("%s: %d rows, %d symbols, %s to %s", args.data, len(frame),
             frame["symbol"].nunique(), frame["timestamp"].min(),
             frame["timestamp"].max())

    table, model, scored_date = screen(
        frame, args.report_date, train_start=args.train_start, rounds=args.rounds)
    report = render(table, model, scored_date, frame["symbol"].nunique())
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
