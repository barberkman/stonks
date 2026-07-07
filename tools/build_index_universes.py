"""Point-in-time S&P 500 / Nasdaq-100 filters over app/data/us_1d.parquet.

Filters the flat, all-symbols us_1d.parquet down to two survivorship-bias-free
universes: for each (symbol, date) row, keeps it only if that symbol was
actually a member of the index on that specific trading day. A row is not
kept just because the symbol is a member *today* — that would silently drop
every company later removed from the index (acquired, delisted, demoted) and
bias any backtest toward today's winners.

Membership sources (both point-in-time, both MIT-licensed):
  - S&P 500: fja05680/sp500 on GitHub — one CSV row per snapshot date,
    reliable from 1996-01-02 onward.
  - Nasdaq-100: jmccarrell/n100tickers on GitHub — one YAML file per year,
    each with a Jan-1 baseline plus dated add/remove events, reliable from
    2015-01-01 onward.

Between snapshot dates, membership is forward-filled (the sources only record
dates where membership *changed*). Ticker matching against us_1d.parquet is
exact-string only — renames/spinoffs (e.g. FB -> META) are not aliased; any
membership ticker with zero matching rows in the source parquet is reported,
not silently dropped.

Usage (from the project root, with the app venv for pandas/pyarrow/pyyaml):

    app/python/.venv/bin/python tools/build_index_universes.py \
        app/data/us_1d.parquet app/data/us_sp500_1d.parquet app/data/us_nasdaq100_1d.parquet
"""

import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SP500_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
NASDAQ100_YAML_URL = (
    "https://raw.githubusercontent.com/jmccarrell/n100tickers/main/"
    "src/nasdaq_100_ticker_history/n100-ticker-changes-{year}.yaml"
)
NASDAQ100_FIRST_YEAR = 2015


class _TickerYamlLoader(yaml.SafeLoader):
    """SafeLoader with the YAML 1.1 bool resolver disabled.

    Bare words like ON, OFF, YES, NO are real tickers in these files (e.g.
    ON Semiconductor) and must not be coerced to Python booleans.
    """


_TickerYamlLoader.yaml_implicit_resolvers = {
    k: [r for r in v if r[0] != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}

SP500_START = pd.Timestamp("1996-01-02", tz="UTC")
NASDAQ100_START = pd.Timestamp("2015-01-01", tz="UTC")


def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def sp500_snapshots():
    """Return sorted [(pd.Timestamp, frozenset(tickers)), ...]."""
    text = fetch(SP500_CSV_URL)
    snapshots = []
    for row in csv.DictReader(io.StringIO(text)):
        d = pd.Timestamp(row["date"], tz="UTC")
        tickers = frozenset(t.strip() for t in row["tickers"].split(",") if t.strip())
        snapshots.append((d, tickers))
    snapshots.sort(key=lambda p: p[0])
    return snapshots


def nasdaq100_snapshots():
    """Return sorted [(pd.Timestamp, frozenset(tickers)), ...]."""
    current_year = datetime.now(timezone.utc).year
    snapshots = []
    for year in range(NASDAQ100_FIRST_YEAR, current_year + 1):
        try:
            text = fetch(NASDAQ100_YAML_URL.format(year=year))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        doc = yaml.load(text, Loader=_TickerYamlLoader)
        members = set(doc["tickers_on_Jan_1"])
        snapshots.append((pd.Timestamp(date(year, 1, 1), tz="UTC"), frozenset(members)))
        for change_date, delta in sorted((doc.get("changes") or {}).items()):
            members -= set(delta.get("difference") or [])
            members |= set(delta.get("union") or [])
            snapshots.append((pd.Timestamp(change_date, tz="UTC"), frozenset(members)))
    snapshots.sort(key=lambda p: p[0])
    return snapshots


def snapshots_to_long_table(snapshots):
    """[(date, frozenset)] -> DataFrame[snapshot_date, ticker], one row per member."""
    rows = [(d, t) for d, members in snapshots for t in members]
    return pd.DataFrame(rows, columns=["snapshot_date", "ticker"])


def filter_point_in_time(price_df, snapshots, start, label):
    ts_dtype = price_df["timestamp"].dtype
    long_membership = snapshots_to_long_table(snapshots)
    long_membership["snapshot_date"] = long_membership["snapshot_date"].astype(ts_dtype)

    trade_dates = pd.DataFrame({"timestamp": price_df["timestamp"].unique()}).sort_values("timestamp")
    snapshot_dates = pd.DataFrame({"snapshot_date": sorted(d for d, _ in snapshots)})
    snapshot_dates["snapshot_date"] = snapshot_dates["snapshot_date"].astype(ts_dtype)
    asof = pd.merge_asof(
        trade_dates, snapshot_dates, left_on="timestamp", right_on="snapshot_date", direction="backward"
    )

    merged = price_df.merge(asof, on="timestamp", how="left")
    merged = merged[merged["timestamp"] >= start]
    result = merged.merge(
        long_membership, left_on=["snapshot_date", "symbol"], right_on=["snapshot_date", "ticker"], how="inner"
    )
    result = result[price_df.columns.tolist()]

    universe_tickers = {t for _, members in snapshots for t in members}
    price_symbols = set(price_df["symbol"].unique())
    missing = sorted(universe_tickers - price_symbols)
    if missing:
        print(f"[{label}] {len(missing)} membership tickers have no matching rows in the source parquet:")
        print(f"[{label}]   {', '.join(missing)}")

    per_date_counts = result.groupby("timestamp")["symbol"].nunique()
    p5, p95 = per_date_counts.quantile([0.05, 0.95])
    print(
        f"[{label}] {result.shape[0]:,} rows, {result['symbol'].nunique():,} distinct symbols, "
        f"{result['timestamp'].min()} .. {result['timestamp'].max()}, "
        f"members/day: median={per_date_counts.median():.0f} p5={p5:.0f} p95={p95:.0f} "
        f"(min/max can be skewed by stray weekend rows already present in the source parquet)"
    )
    return result


def write_like(df, schema, path):
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)
    print(f"wrote {path} ({table.num_rows:,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_parquet")
    ap.add_argument("sp500_out")
    ap.add_argument("nasdaq100_out")
    args = ap.parse_args()

    schema = pq.read_schema(args.source_parquet)
    price_df = pq.read_table(args.source_parquet).to_pandas()

    sp500 = filter_point_in_time(price_df, sp500_snapshots(), SP500_START, "sp500")
    write_like(sp500, schema, args.sp500_out)

    nasdaq100 = filter_point_in_time(price_df, nasdaq100_snapshots(), NASDAQ100_START, "nasdaq100")
    write_like(nasdaq100, schema, args.nasdaq100_out)


if __name__ == "__main__":
    sys.exit(main())
