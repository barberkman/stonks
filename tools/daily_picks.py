#!/usr/bin/env python3
"""Refresh the BIST feed, fit AlgoTrade on all of it, and print today's picks.

Standalone and idempotent: every run downloads whatever bars are new, rebuilds
the cleaned parquet, fits a fresh `ManipulationModel` on the entire history, and
scores the newest bar. There is no cached artifact and no train/test split —
this is the live-trading configuration, where withholding data buys nothing.

    python3 tools/daily_picks.py                 # refresh, fit, print top 5%
    python3 tools/daily_picks.py --no-refresh    # skip the download
    python3 tools/daily_picks.py --top-pct 1     # a tighter slice
    python3 tools/daily_picks.py --csv picks.csv # also write the table
    python3 tools/daily_picks.py --min-sigma 1   # only names above 1 sigma

Output mirrors `workspace.ipynb`'s trades frame, so a pick reads against the
blotter without translating. `score` is the notebook's number — a predicted
*sigma* — and a pick is simply a trade whose entry has not happened yet, i.e.
the notebook's `outcome == "open"` row, with the fills, hold and result NaT/NaN
until it closes. `log_return` and `expected_pct` carry the model's own output
alongside.

Two columns mirror the notebook's newer ones. `logret` is the trade's realised
log(exit/entry) and so is NaN on every pick, exactly as `result_pct` and `sigma`
are — it is there so a pick and a closed blotter row have the same shape. Do not
read it as `log_return`, three columns to its right, which is the model's
*predicted* log return and is the one number here that is never NaN. `naive` is
workspace.ipynb's no-fit reference: the cross-sectional average of a calm rank
(`realized_vol_60`, inverted) and an extended rank (`distance_from_ma_20`),
taken over the same names the gate ranks. A pick the score likes and `naive`
does not is the interesting row — that is where the fit is claiming to add
something over the tilt it is mostly made of.

Note what that costs. `algo_trade` fits the plain log return and deliberately
does NOT normalise: measured out of sample, dividing by `vol * sqrt(bars_held)`
lifts the top-2% slice to only +4.9% mean against +8.9% for the plain log return
(algo_trade.py:1383). Ranking on sigma therefore reorders the cross-section
relative to `AlgoTradeStrategy` — a quiet name with a small predicted move
outranks a loud one with a bigger move. Same model, different selection. Pass
`--rank log-return` for the strategy's own ordering.

Either way the score is a *ranking*, not a calibrated forecast. The out-of-sample
evidence in tools/workspace.ipynb is that it orders "will this close green" (win
rate climbs 25% -> 43% across its deciles) and does not order magnitude.

The pipeline, and why it is three separate stages rather than one download:

  1. ~/dataservice fetches per-symbol pkls from TwelveData, incrementally from
     wherever each symbol left off. When that cache is missing it is rebuilt
     from this repo's own `bist_1d_raw.parquet` rather than re-downloaded, so
     the fetch stays proportional to the days since that file was written. Only
     with neither does it fall back to a full paged re-fetch from 2020-01-01,
     which is rate-limited to 50 requests/minute and announces itself first.
  2. `combine_pkl` streams those into one long-format parquet.
  3. `clean_bist` repairs the seven defect classes that feed carries — weekend
     bars, phantom holiday sessions, x100 spike-and-revert prints, unapplied
     corporate actions. Skipping it puts feed damage straight into the features,
     and the +/-10% daily band means most of that damage is not price action.

Stage 1 needs ~/dataservice's own venv (requests, python-dotenv) and its .env
key; stages 2-3 and the fit run in this repo's interpreter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASERVICE = Path.home() / "dataservice"
DS_PYTHON = DATASERVICE / ".venv" / "bin" / "python"
DS_DATA = DATASERVICE / "bist" / "data"
DS_PARQUET = DATASERVICE / "bist" / "parquet" / "bist_1d.parquet"
DS_TZ = "Europe/Istanbul"      # dataservice/bist/update_recent.py: TZ

RAW_OUT = PROJECT_ROOT / "app" / "data" / "bist_1d_raw.parquet"
CLEAN_OUT = PROJECT_ROOT / "app" / "data" / "bist_1d.parquet"

sys.path.insert(0, str(PROJECT_ROOT / "app" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))


def run(argv, cwd, what):
    """Run a subprocess, streaming its output, and fail loudly."""
    print(f"\n=== {what} ===\n$ {' '.join(str(a) for a in argv)}", flush=True)
    result = subprocess.run([str(a) for a in argv], cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"{what} failed (exit {result.returncode})")


def seed_cache():
    """Rebuild dataservice's per-symbol pkls from the local raw parquet.

    `update_recent` is incremental against a pkl cache and has no other way to
    know where a symbol left off, so an empty cache means every symbol gets paged
    back to 2020-01-01. But this repo already holds that history: the raw parquet
    is what `combine_pkl` wrote out of those same pkls, so it can be pivoted back
    and the fetch reduced to the handful of sessions since.

    Seeded from `bist_1d_raw.parquet` and never the cleaned file — the cleaner
    drops rows and rewrites prices across unadjusted corporate actions, and
    feeding that back would make the cache disagree with the vendor about what
    was printed. Only `1day` is restored; the intraday intervals stay absent and
    would still cost a full fetch.
    """
    print(f"\n=== seed cache ===\n{DS_DATA} is empty; rebuilding it from "
          f"{RAW_OUT.relative_to(PROJECT_ROOT)} instead of re-downloading.")
    raw = pd.read_parquet(RAW_OUT)
    DS_DATA.mkdir(parents=True, exist_ok=True)

    written = []
    for symbol, part in raw.groupby("symbol", sort=True):
        # The shape `_values_to_df` produces: tz-aware index named `datetime`,
        # OHLC float64, volume int64, sorted. `update_recent` reads `.index.max()`
        # off this to pick its start date, so the tz has to be right.
        frame = part.drop(columns="symbol").rename(
            columns={"timestamp": "datetime"}).set_index("datetime").sort_index()
        frame.index = frame.index.tz_localize(DS_TZ)
        frame["volume"] = frame["volume"].astype("int64")
        path = DS_DATA / f"{symbol}.pkl"
        pd.to_pickle({"1day": frame[["open", "high", "low", "close", "volume"]],
                      "5min": None, "15min": None, "meta": {}}, path)
        written.append(path)

    print(f"seeded {len(written)} symbols through "
          f"{pd.Timestamp(raw['timestamp'].max()).date()}")
    return written


def refresh():
    """Pull new bars, combine to parquet, clean, and write both feeds."""
    if not DS_PYTHON.exists():
        raise SystemExit(
            f"{DS_PYTHON} not found — dataservice's venv is what carries the\n"
            f"TwelveData client and the API key. Create it, or pass --no-refresh\n"
            f"to fit on the existing {CLEAN_OUT.name}.")

    cached = sorted(p for p in DS_DATA.glob("*.pkl") if not p.name.startswith("_"))
    if not cached and RAW_OUT.exists():
        cached = seed_cache()

    if cached:
        print(f"{len(cached)} cached symbols — fetching only new bars.")
        run([DS_PYTHON, "-m", "bist.update_recent", "--interval", "1day"],
            DATASERVICE, "incremental download")
    else:
        # The pkl cache is what makes a daily run cheap. Without it every symbol
        # has to be paged back to 2020-01-01 under a 50/min limiter.
        print(f"No pkl cache in {DS_DATA} and no {RAW_OUT.name} to seed from —\n"
              f"this is a FULL fetch of ~630 symbols from 2020-01-01, rate-limited\n"
              f"to 50 requests/minute. Expect tens of minutes. Later runs are\n"
              f"incremental.", flush=True)
        run([DS_PYTHON, "-m", "bist.fetch_twelvedata"],
            DATASERVICE, "full download")

    run([DS_PYTHON, "-m", "bist.combine_pkl", "--interval", "1day",
         "--format", "parquet"], DATASERVICE, "combine to parquet")

    if not DS_PARQUET.exists():
        raise SystemExit(f"expected {DS_PARQUET} after combine; it is missing")

    print("\n=== clean ===")
    from clean_bist import _summary, clean

    raw = pd.read_parquet(DS_PARQUET)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_OUT, index=False)
    cleaned, audit = clean(raw)
    cleaned.to_parquet(CLEAN_OUT, index=False)
    print(_summary(audit))
    print(f"\n{len(raw):,} raw rows -> {len(cleaned):,} clean "
          f"({len(raw) - len(cleaned):,} dropped)")
    print(f"wrote {RAW_OUT.relative_to(PROJECT_ROOT)}")
    print(f"wrote {CLEAN_OUT.relative_to(PROJECT_ROOT)}")


def _cell(value):
    """One markdown cell, matching how pandas prints the same value."""
    if value is pd.NaT:
        return "NaT"
    if pd.isna(value):
        return "NaN"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def markdown_table(frame):
    """Render a frame as a GitHub-flavoured markdown table.

    Hand-rolled rather than `DataFrame.to_markdown`, which needs `tabulate` — a
    venv dependency this script would otherwise not have, for one output format.
    Columns are padded to a common width so the text is also readable unrendered.
    """
    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    body = [[_cell(idx)] + [_cell(v) for v in row]
            for idx, row in zip(frame.index, frame.to_numpy())]
    width = [max(len(row[i]) for row in [header, *body])
             for i in range(len(header))]

    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, width)) + " |"

    return "\n".join([line(header),
                      "|" + "|".join("-" * (w + 2) for w in width) + "|",
                      *(line(row) for row in body)])


NAIVE_FEATURES = ("realized_vol_60", "distance_from_ma_20")


def naive_rank(frame):
    """workspace.ipynb's no-fit reference, over one bar's cross-section.

    The average of two percentile ranks — calm (`realized_vol_60`, inverted so
    the quietest name scores 1.0) and extended (`distance_from_ma_20`, the
    ATR-normalised gap above the 20-bar average). The notebook groups by date
    before ranking; a pick run holds a single date, so that groupby collapses to
    a rank over the whole frame.

    A name missing either feature ranks NaN rather than being imputed to the
    middle of the field: `naive` is a yardstick for the score, and a yardstick
    that invents a value for a name it cannot measure is worse than one that
    says so.
    """
    return (frame["realized_vol_60"].rank(pct=True, ascending=False)
            + frame["distance_from_ma_20"].rank(pct=True)) / 2.0


def top_slice(scored, pct):
    """The highest-scoring `pct` percent of one cross-section, at least one name.

    Mirrors `AlgoTradeStrategy`'s gate: `max(1, int(n * pct / 100))`. The floor
    means every percentage below 100/n resolves to the same single pick, so a
    very small `--top-pct` is not a tighter gate, just the same one.
    """
    take = max(1, int(len(scored) * pct / 100.0))
    return scored.nlargest(take, "score"), take


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--no-refresh", action="store_true",
                   help=f"fit on the existing {CLEAN_OUT.name}, download nothing")
    p.add_argument("--top-pct", type=float, default=5.0,
                   help="percent of the cross-section to print (default 5.0). "
                        "Floored at one name, so anything under 100/n is the "
                        "same single pick — see `top_slice`.")
    p.add_argument("--data", type=Path, default=CLEAN_OUT,
                   help="feed to fit and score (default: the cleaned parquet)")
    p.add_argument("--rank", choices=["sigma", "log-return"], default="sigma",
                   help="what `score` holds and the gate ranks on. 'sigma' is "
                        "workspace.ipynb's unit (default); 'log-return' is the "
                        "raw model output AlgoTradeStrategy ranks on.")
    p.add_argument("--min-sigma", type=float,
                   help="drop names whose score_sigma is below this before the "
                        "percentile gate. workspace.ipynb's scale, so 1.0 here "
                        "means what 1.0 means in its blotter.")
    p.add_argument("--markdown", action="store_true",
                   help="also print the picks as a markdown table")
    p.add_argument("--csv", type=Path, help="also write the picks table here")
    args = p.parse_args()

    if not args.no_refresh:
        refresh()
    elif not args.data.exists():
        raise SystemExit(f"{args.data} not found and --no-refresh was passed")

    # The sigma conversion lives in `algo_trade` because `AlgoTradeStrategy`
    # gates on it too — one implementation, so the script and the strategy
    # cannot drift into disagreeing about what a sigma is.
    from algo_trade import (BACKTEST_TRAIN_START, EXIT_MA, EXPECTED_PCT,
                            FEATURE_NAMES, LIMIT_HIT, MAX_HOLD, SCORE,
                            SIGMA_HOLD, SIGMA_VOL_LEN, STOP_PCT,
                            ManipulationModel, _features, _panel, _to_sigma,
                            _trailing_vol)

    frame = pd.read_parquet(args.data)
    panel = _panel(frame)
    index = panel["close"].index
    as_of = pd.Timestamp(index[-1])

    print("\n=== fit ===")
    print(f"{len(frame):,} rows, {frame['symbol'].nunique()} symbols, "
          f"{pd.Timestamp(index[0]).date()} -> {as_of.date()}")

    stale = (pd.Timestamp.now().normalize() - as_of.normalize()).days
    if stale > 4:
        print(f"WARNING: newest bar is {stale} days old. These are picks for "
              f"{as_of.date()}, not for today.")

    # No `train_end`: the embargo exists to keep a backtest honest, and there is
    # nothing to hold out when the question is "what do I buy at the next open".
    # `train_start` is the backtest's full-history value rather than the module
    # default of 2024-01-01, because the trade label already drops roughly half
    # the panel to its entry gate.
    model = ManipulationModel(train_start=BACKTEST_TRAIN_START)
    model.train(panel)
    print(f"trained through {as_of.date()}, "
          f"best_iteration={model.best_iteration}, "
          f"{len(model.liquid)} symbols in the liquid universe")

    # `signal` scores the panel's last bar. A full panel carries its own history,
    # so no ScoringState is needed — that exists for windowed live scoring.
    sig = model.signal(panel)
    scored = sig[np.isfinite(sig[SCORE])].copy()
    scored.index.name = "symbol"
    scored = scored.reset_index()

    close = panel["close"].iloc[-1]
    prev = panel["close"].iloc[-2]
    sma = panel["close"].tail(EXIT_MA * 3).apply(
        lambda col: col.dropna().tail(EXIT_MA).mean())
    scored["close"] = scored["symbol"].map(close)
    scored["vs_ma20_pct"] = 100.0 * (scored["close"] / scored["symbol"].map(sma) - 1.0)
    scored["day_pct"] = 100.0 * (scored["close"] / scored["symbol"].map(prev) - 1.0)
    scored["stop"] = scored["close"] * (1.0 - STOP_PCT / 100.0)
    scored["limit_up"] = scored["day_pct"] >= 100.0 * LIMIT_HIT

    # `naive`, workspace.ipynb's no-fit reference: the average of a calm rank
    # and an extended rank across the day's cross-section. The notebook's
    # per-date groupby collapses to a plain rank here because a pick run scores
    # exactly one bar, and it is taken over `scored` — the same names the gate
    # ranks — before any sigma floor or gate narrows the field, so a name's
    # naive rank does not move when the score's filters do.
    #
    # The two columns are read out of `_features` rather than rebuilt from
    # `vol60` and `vs_ma20_pct`, which are close but not the same numbers:
    # `distance_from_ma_20` is ATR-normalised, `vs_ma20_pct` is not. Paying for
    # one more feature pass beats a naive column that quietly means something
    # different here than it does in the notebook.
    last_features = _features(panel)[-1]                    # (N, 72)
    for name in NAIVE_FEATURES:
        column = dict(zip(panel["close"].columns,
                          last_features[:, FEATURE_NAMES.index(name)]))
        scored[name] = scored["symbol"].map(column)
    scored["naive"] = naive_rank(scored)

    # `score` is workspace.ipynb's number: a predicted *sigma*. The model itself
    # emits a log return, so the conversion happens here and the result takes the
    # name, because the gate ranks on whatever is called `score` and a table whose
    # rank order disagreed with its score column would be unreadable.
    #
    # `AlgoTradeStrategy` ranks on the raw model output unless its `sigma_gate`
    # is on, in which case it ranks on exactly this number. Dividing by each
    # name's own vol reorders the cross-section: a quiet name with a small
    # predicted move outranks a loud one with a bigger move.
    #
    # `_trailing_vol` wants each name's traded closes, newest last — which is
    # what dropping the halted cells out of a panel column leaves.
    vol = _trailing_vol({s: panel["close"][s].dropna().to_numpy()
                         for s in panel["close"].columns})
    scored["vol60"] = scored["symbol"].map(vol)
    scored["log_return"] = scored[SCORE]
    scored["sigma_score"] = _to_sigma(scored["log_return"], scored["vol60"])
    if args.rank == "sigma":
        scored[SCORE] = scored["sigma_score"]
        # A name with too little history for a 60-bar vol has no sigma, so it
        # cannot be ranked on one. Dropping beats ranking it on a NaN.
        scored = scored[np.isfinite(scored[SCORE])]

    if args.min_sigma is not None:
        before = len(scored)
        scored = scored[scored["sigma_score"] >= args.min_sigma]
        print(f"\nsigma floor {args.min_sigma:g}: {len(scored)} of {before} "
              f"scored names survive, and the gate is taken from those.")
        if scored.empty:
            raise SystemExit("nothing clears the sigma floor — no picks today.")

    # The nested ladder workspace.ipynb reports, so `gate` here means what it
    # means there: the narrowest slice that still holds the name.
    ladder = sorted({0.1, 0.5, 1.0, 2.0, args.top_pct})
    gate_of = {}
    for pct in reversed(ladder):
        for symbol in top_slice(scored, pct)[0]["symbol"]:
            gate_of[symbol] = f"top {pct:g}%"

    picks, take = top_slice(scored, args.top_pct)
    picks = picks.sort_values(SCORE, ascending=False).reset_index(drop=True)
    picks.index += 1
    picks.index.name = "rank"

    # The blotter's shape. A pick is a trade whose entry has not happened yet, so
    # it is exactly the notebook's `outcome == "open"` row: the fills, the hold
    # and the result are all unknown, and saying so with NaT/NaN is the same
    # answer the notebook gives for a trade that has not resolved.
    picks["signal_date"] = as_of
    picks["entry_date"] = pd.NaT
    picks["exit_date"] = pd.NaT
    picks["bars_held"] = np.nan
    picks["result_pct"] = np.nan
    # The realised triple, all unknown until the trade closes. `logret` is the
    # notebook's realised log(exit/entry) — NOT `log_return`, which is further
    # right and is the model's *prediction*. They are a column apart and one is
    # always NaN here, which is the whole difference between them.
    picks["logret"] = np.nan
    picks["sigma"] = np.nan
    picks["gate"] = picks["symbol"].map(gate_of)
    picks["outcome"] = "open"

    print(f"\n=== picks: signal bar {as_of.date()}, "
          f"top {args.top_pct:g}% of {len(scored)} scored names = {take} ===")
    print(f"enter at the NEXT open | stop {STOP_PCT:g}% under the close above | "
          f"exit on first close < MA{EXIT_MA}, cap {MAX_HOLD} bars")
    basis = (f"predicted sigma over vol60*sqrt({SIGMA_HOLD}), "
             f"workspace.ipynb's unit" if args.rank == "sigma"
             else "predicted log return, AlgoTradeStrategy's own basis")
    print(f"score = {basis} | result_pct/logret/sigma are the realised trio and "
          f"are NaN until the trade closes | naive is the no-fit rank the score "
          f"has to beat | log_return is the model's prediction\n")

    columns = ["symbol", "signal_date", "entry_date", "exit_date", "bars_held",
               SCORE, "naive", "result_pct", "logret", "sigma", "gate",
               "limit_up", "outcome", "close", "stop", "log_return",
               EXPECTED_PCT, "vs_ma20_pct", "day_pct", "vol60"]
    table = picks[columns].round(4)
    if args.markdown:
        print(markdown_table(table))
    else:
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(table.to_string())

    locked = picks["limit_up"].sum()
    if locked:
        print(f"\n{locked} of {len(picks)} closed at the +{100 * LIMIT_HIT:.0f}% "
              f"band. `AlgoTradeStrategy` skips those — a locked book would not "
              f"have filled the next open.")

    if args.csv:
        picks[columns].to_csv(args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
