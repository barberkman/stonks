"""Parity harness: algo_trade's reimplementation vs bist's own code, same input.

`app/python/algo_trade.py` reimplements the 72-column feature pipeline from
/Users/macmini-1/bist so it can run inside the C++ engine one bar at a time.
This tool checks the reimplementation against the original by running BOTH on
the same OHLCV frame and diffing every column.

Features only. The labels used to be diffed here too, against bist's
`synthetic_labels`, but `algo_trade._labels` no longer computes a fixed-horizon
drawdown-gated sigma — it simulates the trade the strategy holds, which has no
counterpart in bist to diff against. The feature half is untouched and is what
keeps the port honest.

Why not just compare against bist's shipped `data_artifacts/features.parquet`:
that was built on bist's own feed, which is different data — its `open` is
Is Yatirim's daily VWAP (`HGDG_AOF`), its `volume` is lira turnover rather than
share count, and it starts in 2016 rather than 2020. Diffing against it
conflates data differences with code differences. Feeding bist's code our
parquet separates the two, and code differences are the only ones we can fix.

The 7 `intra_*` features need 15-minute bars that this engine's feed does not
carry, so the intraday block is disabled here (see `_disable_intraday`). bist's
own feature selection then drops those columns for being all-NaN, leaving
exactly the 72 that `algo_trade.FEATURE_NAMES` declares — which is where that
number comes from.

Run from the project root:

    app/python/.venv/bin/python tools/bist_parity.py
    app/python/.venv/bin/python tools/bist_parity.py --tail 900 --rows raw
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BIST_ROOT = Path("/Users/macmini-1/bist")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

for path in (BIST_ROOT, PROJECT_ROOT / "app" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _disable_intraday():
    """Make bist's intraday block return its all-NaN guard frame.

    `intraday_features.compute` unconditionally loads a 1 GB 15-minute pickle
    and there is no flag to skip it, but it already handles "no bars match this
    symbol set" by returning all-NaN (intraday_features.py:90-94). Handing it an
    empty frame takes that branch. `_load_15m` is `lru_cache`d, so the cache is
    cleared in case something already populated it.
    """
    from bist_manipulation.features import intraday_features

    intraday_features._load_15m.cache_clear()
    empty = pd.DataFrame(
        {c: pd.Series(dtype="float64") for c in
         ("timestamp", "symbol", "open", "high", "low", "close", "volume")})
    intraday_features._load_15m = lambda: empty
    return intraday_features.FEATURE_COLUMNS


def bist_reference(df):
    """Run bist's real pipeline on `df` and return (features, preprocessed).

    `df` carries this engine's parquet columns; bist wants `date` instead of
    `timestamp`. Everything after that is bist's own code path, in the order
    `scripts/run_feature_pipeline.py` calls it.
    """
    from bist_manipulation.data import loader, preprocessor
    from bist_manipulation.features import pipeline

    intra = _disable_intraday()

    raw = df.rename(columns={"timestamp": "date"})
    raw = loader.normalize_schema(raw)
    raw = loader.validate(raw)
    raw = loader.duplicate_price_columns(raw)
    pp = preprocessor.preprocess(raw)

    features = pipeline.build_features(pp)
    # run_feature_pipeline.py:51-55 — appended after the cross-sectional block,
    # which is why close_adj lands last in feature_cols and volume does not.
    carry = [c for c in ("volume", "close_adj") if c not in features.columns]
    features = features.merge(pp[["symbol", "date", *carry]],
                              on=["symbol", "date"], how="left")
    return features, pp, intra


def bist_feature_cols(features, intra):
    """bist's own feature selection, as every training script performs it.

    numeric minus identifiers minus `volume`, then dropped unless the column has
    more than 1000 finite values (train_model.py:63-73). The all-NaN intraday
    columns fail that test, which is what reduces 79 to 72.
    """
    from bist_manipulation.features.pipeline import IDENTIFIER_COLUMNS

    reserved = set(IDENTIFIER_COLUMNS) | {"volume"}
    numeric = [c for c in features.columns
               if c not in reserved and np.issubdtype(features[c].dtype, np.number)]
    kept = [c for c in numeric if features[c].notna().sum() > 1000]
    dropped = [c for c in numeric if c not in kept]
    assert set(dropped) == set(intra), (
        f"expected exactly the intraday block to drop out, got {dropped}")
    return kept


def port_features(df):
    """`algo_trade._features` as a long frame keyed (date, symbol)."""
    import algo_trade as A

    panel = A._panel(df)
    stacked = A._features(panel)
    index = panel["close"].index
    columns = panel["close"].columns
    out = {}
    for j, name in enumerate(A.FEATURE_NAMES):
        out[name] = pd.DataFrame(
            stacked[:, :, j], index=index, columns=columns
        ).stack(future_stack=True).rename(name)
    frame = pd.concat(out.values(), axis=1)
    frame.index.names = ["date", "symbol"]
    return frame.reset_index()


# A column passes when every shared cell is within numpy's isclose band. rtol
# alone is useless on features that legitimately sit near zero — a cum_return of
# 1e-12 differing by 1e-15 is a 0.1% relative error and complete noise — and atol
# alone is useless on `obv`, which runs to 1e10. `n_off` counts the cells that
# clear neither, which is the number that actually matters.
RTOL = 1e-6
ATOL = 1e-9


def compare(port, ref, names, label):
    """Per-column diff table between two long frames keyed (date, symbol)."""
    merged = port.merge(ref, on=["date", "symbol"], suffixes=("_p", "_b"),
                        how="inner")
    print(f"\n=== {label}: {len(merged)} shared (date, symbol) rows "
          f"[port {len(port)}, bist {len(ref)}] ===")
    rows = []
    for name in names:
        a = merged[f"{name}_p"].to_numpy(dtype=float)
        b = merged[f"{name}_b"].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            diff = np.abs(a[both] - b[both])
            tol = ATOL + RTOL * np.abs(b[both])
            n_off = int((diff > tol).sum())
            max_abs = float(diff.max())
            worst = int(np.argmax(diff / tol))
            max_excess = float((diff / tol).max())
        else:
            n_off = 0
            max_abs = max_excess = np.nan
            worst = -1
        rows.append({"feature": name, "n_both": int(both.sum()),
                     "nan_only_port": int((np.isfinite(b) & ~np.isfinite(a)).sum()),
                     "nan_only_bist": int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                     "n_off": n_off, "max_abs": max_abs,
                     "x_tol": max_excess,
                     "at": (f"{b[both][worst]:.4g}" if worst >= 0 else "")})
    table = pd.DataFrame(rows).set_index("feature")
    table = table.sort_values(["n_off", "x_tol"], ascending=False)
    pd.set_option("display.width", 220)
    print(table.to_string(float_format=lambda v: f"{v:.6g}"))

    clean = table.loc[(table.n_off == 0)
                      & (table.nan_only_port == 0) & (table.nan_only_bist == 0)]
    exact = table.loc[(table.max_abs == 0)
                      & (table.nan_only_port == 0) & (table.nan_only_bist == 0)]
    print(f"\n{len(clean)}/{len(table)} columns agree within "
          f"atol={ATOL:g} + rtol={RTOL:g}, with identical NaN patterns "
          f"({len(exact)} of them bit-exact)")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="app/data/bist_1d.parquet")
    parser.add_argument("--tail", type=int, default=0,
                        help="only the last N dates (0 = the whole panel)")
    parser.add_argument("--rows", choices=("preprocessed", "raw"),
                        default="preprocessed",
                        help="feed the port bist's surviving rows (default) or "
                             "the parquet as-is; 'raw' measures what bist's row "
                             "drops are worth on their own")
    args = parser.parse_args()

    if not BIST_ROOT.exists():
        raise SystemExit(f"{BIST_ROOT} not found — this tool needs the bist repo")

    df = pd.read_parquet(args.data)
    if args.tail:
        dates = np.sort(df["timestamp"].unique())[-args.tail:]
        df = df.loc[df["timestamp"].isin(dates)]
    print(f"{args.data}: {len(df)} rows, {df['symbol'].nunique()} symbols, "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")

    features, pp, intra = bist_reference(df)
    names = bist_feature_cols(features, intra)
    print(f"bist feature_cols: {len(names)} columns")

    import algo_trade as A
    if tuple(names) != A.FEATURE_NAMES:
        extra = [c for c in names if c not in A.FEATURE_NAMES]
        missing = [c for c in A.FEATURE_NAMES if c not in names]
        print(f"!! FEATURE_NAMES mismatch — bist-only {extra}, port-only {missing}")
        if [c for c in names if c in A.FEATURE_NAMES] != [
                c for c in A.FEATURE_NAMES if c in names]:
            print("!! shared columns are in a DIFFERENT ORDER; a fitted booster "
                  "trained on one layout is meaningless against the other")
        names = [c for c in names if c in A.FEATURE_NAMES]
    else:
        print("FEATURE_NAMES matches bist's feature_cols exactly, in order")

    # Feed the port the same bars bist kept, so the diff is about feature code
    # rather than about which rows survived.
    port_input = (pp[["date", "symbol", "open", "high", "low", "close", "volume"]]
                  .rename(columns={"date": "timestamp"})
                  if args.rows == "preprocessed" else df)
    dropped = len(df) - len(pp)
    print(f"bist dropped {dropped} of {len(df)} rows in preprocessing "
          f"({100.0 * dropped / len(df):.3f}%); port fed the "
          f"{'surviving' if args.rows == 'preprocessed' else 'raw'} set")

    ref = features[["date", "symbol", *names]]
    compare(port_features(port_input), ref, names, "features")


if __name__ == "__main__":
    main()
