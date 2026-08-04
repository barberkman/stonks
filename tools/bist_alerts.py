"""Single-date alert screen — the port's answer to bist's `configured_alerts`.

Trains through a report date, scores that date's liquid universe, and prints the
slice the strategy would actually enter:

    Symbol         score      exp%

Rows are the day's top `--top-pct` by score, best first — the same
cross-sectional cut `AlgoTradeStrategy._rank` takes, so the screen shows
candidates rather than an arbitrary "above noise" set. bist's `n` and `composite`
columns are gone with the three heads they counted.

Names trading below their exit moving average never appear: `signal` refuses to
score them, because the fit never saw such a row. See `algo_trade._labels`.

**This is a screen, not a backtest.** It trains on data through the very date it
scores, exactly as bist's does, so nothing here is evidence that the rules make
money. `app/python/algo_trade.py`'s `AlgoTradeStrategy` is the backtest, and it
refuses to trade any bar its artifact saw.

One thing not to expect: the numbers will not equal bist's for the same date.
bist runs on a different feed (its `open` is a daily VWAP, its `volume` is lira
turnover, its history starts in 2016), so the code matches and the inputs do not
— `tools/bist_parity.py` is what measures the code half.

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

# Replaces bist's SIGMA_THRESHOLD. That was an absolute cutoff on sigma, and the
# model no longer emits one — a score is a log return, whose scale moves with the
# fit. The percentile the strategy trades is both meaningful and stable.
SCREEN_TOP_PCT = A.ENTRY_TOP_PCT

log = logging.getLogger("bist_alerts")


def screen(frame, report_date=None, *, train_start=A.TRAIN_START_DATE,
           rounds=None, top_pct=SCREEN_TOP_PCT):
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
    return build_table(sig, top_pct), model, scored_date


def build_table(sig, top_pct=SCREEN_TOP_PCT):
    """The day's top `top_pct` of the scored universe, best score first.

    The same cut `AlgoTradeStrategy._rank` takes, minus the book: a symbol the
    model declined to score drops out rather than sorting to one end, and ties
    break on symbol so two runs of the same fit print the same order.
    """
    scored = sig[A.SCORE].dropna()
    if scored.empty:
        return sig.iloc[:0][[A.SCORE, A.EXPECTED_PCT]]

    keep = max(1, int(len(scored) * top_pct / 100.0))
    order = sorted(scored.index, key=lambda s: (-scored[s], s))[:keep]
    return sig.loc[order, [A.SCORE, A.EXPECTED_PCT]]


def render(table, model, scored_date, universe, top_pct=SCREEN_TOP_PCT):
    lines = ["BIST alert screen (stonks port of bist configured_alerts)",
             "=" * 78,
             f"Report date:         {pd.Timestamp(scored_date).date()}",
             f"Liquidity universe:  {len(model.liquid)} / {universe} symbols "
             f"(median turnover >= p{model.min_turnover_percentile:.0%})",
             f"Training window:     {model.train_start} through the report date",
             f"Trade:               enter above MA{model.exit_ma}, exit on the "
             f"first close below it, {model.max_hold}-bar cap",
             "",
             "NOT A BACKTEST: the fit saw every bar up to and including the date",
             "it scores, exactly as bist's screen does. Use AlgoTradeStrategy for",
             "an out-of-sample result.",
             ""]

    columns = [(A.SCORE, "score"), (A.EXPECTED_PCT, "exp%")]
    width = 11
    lines.append(f"-- The day's top {top_pct:g}% by score --")
    if table.empty:
        lines.append("   (nothing scored)")
    else:
        lines.append(f"   {'Symbol':<8}  "
                     + "  ".join(f"{label:>{width}}" for _, label in columns))
        for symbol, row in table.iterrows():
            cells = [f"{'.':>{width}}" if pd.isna(row.get(key))
                     else f"{row[key]:>+{width}.4f}" for key, _ in columns]
            lines.append(f"   {symbol:<8}  " + "  ".join(cells))

    lines += ["",
              "-- Caveats --",
              "* score is the predicted natural-log return of one trade; exp% is",
              "  that as a percent. NOT a calibrated forecast — rank by score and",
              "  treat exp% as a magnitude check.",
              "* Names below their exit moving average are absent by construction,",
              "  not by rejection: the model has no answer for them.",
              "* A triage queue, not a trading signal."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="app/data/bist_1d.parquet")
    parser.add_argument("--report-date", default=None,
                        help="date to score; defaults to the last bar in the feed")
    parser.add_argument("--train-start", default=A.TRAIN_START_DATE)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--top-pct", type=float, default=SCREEN_TOP_PCT,
                        help="share of the scored universe to list")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the report here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(args.data)
    log.info("%s: %d rows, %d symbols, %s to %s", args.data, len(frame),
             frame["symbol"].nunique(), frame["timestamp"].min(),
             frame["timestamp"].max())

    table, model, scored_date = screen(
        frame, args.report_date, train_start=args.train_start,
        rounds=args.rounds, top_pct=args.top_pct)
    report = render(table, model, scored_date, frame["symbol"].nunique(),
                    args.top_pct)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
