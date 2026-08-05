"""Repair the defects in app/data/bist_1d.parquet and write a clean feed.

The raw BIST parquet carries bars that BIST's own rules make impossible. Borsa
Istanbul runs a +/-10% daily price limit on equities, so from one close to the
next a share cannot gap more than 10%, and within a session `high / low` cannot
exceed 1.1 / 0.9 = 1.222. The raw file has 4,131 bars beyond +/-10%, 69 beyond
+100%, and single bars that move x100 and come straight back. None of that is
price action; all of it is feed damage, and a pattern scan or a cross-sectional
strategy will happily trade it.

Seven defect classes, each with its own pass below:

  1. weekend bars              15 bars on 4 Sat/Sun dates in Sep-Oct 2020
  2. zero-volume bars          11,002 bars that never traded, including ten
                               whole phantom sessions on Turkish market
                               holidays (2025-10-29 Republic Day carries 421
                               of them) and 2020-03-11, where 334 of 386 bars
                               are flat, volume-less, and quote the raw
                               unadjusted price before reverting the next day
  3. impossible intraday range 31 bars whose high or low is off by a factor of
                               10 or 100 (ATATP 2023-10-27 high 5795 against a
                               close of 57; ISCTR 2023-04-06 low 0.40 against
                               an open of 4.86)
  4. spike-and-revert prints   ~150 bars that jump and undo it on the next bar,
                               including eight x100 prints (KENT 179.50 ->
                               17950 -> 179.60; TKFEN, DAGHL and TRHOL all on
                               2025-01-10, which looks like the vendor emitting
                               kurus for a day)
  5. unadjusted corporate      269 permanent price cliffs across 177 symbols
     actions                   where a bonus issue or split was never applied
                               backwards (CCOLA 2024-08-01, 846.00 -> 78.27)
  6. corrupt volume            BIZIM 2021-03-29 at 1.29e14 shares, i.e. 1.9
                               quadrillion lira of turnover on a market whose
                               entire daily turnover is ~1e11
  7. duplicated symbols        the same company under two tickers. Some are
                               byte-identical (IDEAS/SKYLP share 1,605 of
                               1,641 bars). Others are the same series under
                               two different adjustment conventions: TRHOL is
                               DAGHL with an extra 0.36 factor on the early
                               history, so the two twins disagree about
                               whether the split happened.

Order matters. Zero-volume bars go first because 2020-03-11's flat unadjusted
prints would otherwise read as corporate actions. Spike-and-revert goes before
the corporate-action pass for the same reason: a x100 print that comes back is
not a split. Duplicate collapse goes last, once both twins have been adjusted
onto the same footing, so the choice between them is no longer arbitrary.

Nothing here is silent. Every bar dropped, clamped, adjusted or collapsed is
counted and the individual events are written to a JSON audit next to the
output, because the corporate-action pass is the one that infers intent from
data and its 269 decisions deserve to be reviewable.

Run from the project root:

    app/python/.venv/bin/python tools/clean_bist.py
    app/python/.venv/bin/python tools/clean_bist.py --dry-run
    app/python/.venv/bin/python tools/clean_bist.py app/data/bist_1d.parquet \\
        -o app/data/bist_1d_clean.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# BIST's daily price limit for equities. Everything below is calibrated off it.
PRICE_LIMIT = 0.10

# high / low ceiling implied by the limit is 1.1 / 0.9 = 1.222. The threshold
# sits well above it so that sessions where the limit is lifted (first days of
# a listing, resumption after a suspension) are left alone; the 31 bars this
# actually catches are all off by a factor of 10 or 100.
MAX_RANGE = 1.5

# A bar is a bad print if it moves this far and the next bar undoes it. The
# tolerance is on the sum of the two log returns: a perfect round trip sums to
# zero, and 0.35 leaves room for the real move that happened underneath.
REVERT_MOVE = 0.25
REVERT_TOL = 0.35

# A corporate action shows up as an overnight gap that the price limit forbids.
# 20% is double the largest legal gap, and it is measured open-against-previous-
# close so the day's own return survives the adjustment untouched.
ACTION_GAP = 0.20

# ...and the session that follows the gap has to be one the limit allows. Open
# at limit-down and close at limit-up is 1.1 / 0.9 - 1 = 22.2%, which is the
# widest legal open-to-close move and shows up in the data as a hard ceiling on
# the day BIST reopened after the February 2023 earthquake. A day that breaches
# it is a session with the limit lifted, not a repriced one, and adjusting it
# would scale away a genuine crash.
ACTION_INTRADAY = 0.25

# A gap this large is not a corporate action, it is two different series
# stitched together -- pre-listing filler in front of an IPO. ALFAS trades at
# 0.118 on a constant 3,500 shares a day until 2022-12-22 and at 330 after it.
# History before such a break is truncated rather than scaled onto the new one.
DISCONTINUITY = 20.0

# Volume this far above the symbol's own median is a typo, not a session.
VOLUME_OUTLIER = 1000.0

# Twins are found from the tail of the return series: the same company under
# two tickers has, by definition, the same returns. The agreement has to be
# near-total rather than total, because the twins disagree on the handful of
# days where one of them got a corporate action the other did not. DUP_MOVERS
# keeps two dormant tickers, whose returns are zero every day, from matching
# each other on nothing.
DUP_WINDOW = 250
DUP_MIN_BARS = 240
DUP_AGREEMENT = 0.95
DUP_MOVERS = 50

OHLC = ["open", "high", "low", "close"]


def _drop(df, mask, audit, key, sample_cols=("timestamp", "symbol", "close", "volume")):
    """Drop `mask` rows, recording the count and a sample in the audit."""
    n = int(mask.sum())
    audit[key] = {"bars": n, "symbols": int(df.loc[mask, "symbol"].nunique())}
    if n:
        sample = df.loc[mask, list(sample_cols)].head(20)
        audit[key]["sample"] = json.loads(sample.to_json(orient="records", date_format="iso"))
    return df.loc[~mask].copy()


def drop_weekends(df, audit):
    return _drop(df, df.timestamp.dt.dayofweek >= 5, audit, "weekend_bars")


def drop_untraded(df, audit):
    """Bars with no volume did not trade, so they cannot be backtested.

    This is what removes the phantom holiday sessions and 2020-03-11 wholesale.
    The dates are recorded because a session that vanishes entirely is worth
    seeing in the audit.
    """
    mask = df.volume <= 0
    dates = sorted(df.loc[mask, "timestamp"].dt.date.astype(str).value_counts().head(15).items(),
                   key=lambda kv: -kv[1])
    out = _drop(df, mask, audit, "untraded_bars")
    audit["untraded_bars"]["worst_dates"] = [{"date": d, "bars": n} for d, n in dates]
    emptied = sorted(set(df.timestamp.dt.date) - set(out.timestamp.dt.date))
    audit["untraded_bars"]["sessions_emptied"] = [str(d) for d in emptied]
    return out


def clamp_ranges(df, audit):
    """Rebuild high/low from open/close when the range is physically impossible.

    Usually only one wick is damaged and it is off by a power of ten, so open
    and close are the two fields worth trusting: clamping to their envelope
    keeps the bar and its return, where dropping it would punch a hole in the
    middle of a series over a one-field typo.

    When the envelope is itself impossible the whole bar is a mixed-scale print
    and there is nothing left to trust -- ALFAS's listing day opens at 0.118 in
    the old scale and closes at 330.30 in the new one -- so those are dropped.
    """
    mask = df.high / df.low > MAX_RANGE
    n = int(mask.sum())
    audit["clamped_ranges"] = {"bars": n, "symbols": int(df.loc[mask, "symbol"].nunique())}
    if n:
        before = df.loc[mask, ["timestamp", "symbol", "open", "high", "low", "close"]]
        audit["clamped_ranges"]["sample"] = json.loads(
            before.head(40).to_json(orient="records", date_format="iso"))
        hi = df.loc[mask, ["open", "close"]].max(axis=1)
        lo = df.loc[mask, ["open", "close"]].min(axis=1)
        df.loc[mask, "high"] = hi
        df.loc[mask, "low"] = lo
    return _drop(df, df.high / df.low > MAX_RANGE, audit, "unsalvageable_bars",
                 sample_cols=("timestamp", "symbol", "open", "high", "low", "close"))


def drop_reverting_spikes(df, audit):
    """Drop bars whose move is undone by the very next bar.

    Under a +/-10% limit no genuine two-session sequence can move 25% and give
    it all back, so a round trip that large is a bad print. This has to run
    before the corporate-action pass, which would otherwise read the x100
    prints as splits and rescale the whole history behind them.
    """
    g = df.groupby("symbol", sort=False).close
    prev, nxt = g.shift(1), g.shift(-1)
    r = np.log(df.close / prev)
    r_next = np.log(nxt / df.close)
    mask = (r.abs() > np.log1p(REVERT_MOVE)) & ((r + r_next).abs() < REVERT_TOL)
    mask = mask.fillna(False)
    sample = df.loc[mask].assign(prev_close=prev[mask], next_close=nxt[mask])
    audit["reverting_spikes"] = {
        "bars": int(mask.sum()),
        "symbols": int(df.loc[mask, "symbol"].nunique()),
        "sample": json.loads(sample[["timestamp", "symbol", "prev_close", "close", "next_close"]]
                             .head(40).to_json(orient="records", date_format="iso")),
    }
    return df.loc[~mask].copy()


def _symbol_actions(sym):
    """Indices and factors of the corporate actions in one symbol's bars."""
    prev_close = sym.close.shift(1)
    gap = np.log(sym.open / prev_close)
    intraday = np.log(sym.close / sym.open)
    hit = (gap.abs() > np.log1p(ACTION_GAP)) & (intraday.abs() < np.log1p(ACTION_INTRADAY))
    return [(i, float(sym.open.iloc[i] / prev_close.iloc[i]))
            for i in np.flatnonzero(hit.fillna(False).to_numpy())]


def adjust_corporate_actions(df, audit):
    """Back-adjust splits and bonus issues, truncate series discontinuities.

    The factor is `open / previous close`: the first print after the action
    against the last print before it. Every earlier bar is multiplied by it and
    its volume divided by it, which is what makes the return series continuous
    while leaving the action day's own open-to-close return alone.

    A factor beyond DISCONTINUITY is not an action any company declares -- it
    is filler history in front of a listing -- so those bars are dropped rather
    than scaled, which would otherwise manufacture a plausible-looking price
    series for a period when the stock did not trade.
    """
    events, truncations, keep, adjusted_syms = [], [], [], 0
    for symbol, sym in df.groupby("symbol", sort=False):
        sym = sym.copy()
        actions = _symbol_actions(sym)

        breaks = [i for i, f in actions if f > DISCONTINUITY or f < 1.0 / DISCONTINUITY]
        if breaks:
            cut = breaks[-1]
            truncations.append({
                "symbol": symbol,
                "listed": sym.timestamp.iloc[cut].date().isoformat(),
                "dropped_bars": int(cut),
                "factor": round(dict(actions)[cut], 2),
            })
            sym = sym.iloc[cut:]
            actions = [(i - cut, f) for i, f in actions if i > cut]

        if actions:
            adjusted_syms += 1
            f = np.ones(len(sym))
            for i, factor in actions:
                f[i] = factor
                events.append({
                    "symbol": symbol,
                    "date": sym.timestamp.iloc[i].date().isoformat(),
                    "factor": round(factor, 6),
                    "prev_close": round(float(sym.close.iloc[i - 1]), 4),
                    "open": round(float(sym.open.iloc[i]), 4),
                })
            # multiplier for bar k is the product of every factor after k
            mult = np.append(np.cumprod(f[::-1])[::-1][1:], 1.0)
            sym[OHLC] = sym[OHLC].to_numpy() * mult[:, None]
            sym["volume"] = sym.volume.to_numpy() / mult
        keep.append(sym)

    per_symbol = pd.Series([e["symbol"] for e in events]).value_counts()
    audit["corporate_actions"] = {
        "events": len(events),
        "symbols": adjusted_syms,
        "per_symbol": per_symbol.to_dict(),
        "splits_and_bonus_issues": sorted(events, key=lambda e: e["factor"]),
    }
    audit["truncated_prelisting"] = {
        "symbols": len(truncations),
        "bars": sum(t["dropped_bars"] for t in truncations),
        "detail": truncations,
    }
    return pd.concat(keep, ignore_index=True) if keep else df


def fix_volume_outliers(df, audit):
    """Replace impossible volumes with the symbol's median.

    The price on these bars is fine, only the share count is corrupt, so the
    bar is repaired rather than dropped.
    """
    med = df.groupby("symbol", sort=False).volume.transform("median")
    mask = (med > 0) & (df.volume > VOLUME_OUTLIER * med)
    n = int(mask.sum())
    audit["volume_outliers"] = {"bars": n, "symbols": int(df.loc[mask, "symbol"].nunique())}
    if n:
        audit["volume_outliers"]["detail"] = json.loads(
            df.loc[mask, ["timestamp", "symbol", "close", "volume"]]
              .assign(replaced_with=med[mask].round(0))
              .to_json(orient="records", date_format="iso"))
        df.loc[mask, "volume"] = med[mask]
    return df


def collapse_duplicates(df, audit, repairs):
    """Keep one ticker per underlying company.

    Twins are found from the last DUP_WINDOW sessions: two tickers on the same
    company post the same return every day, so the pair is scored on the share
    of days its two return series agree to the last decimal. That share has to
    be near one rather than exactly one -- twins still disagree on the few days
    where the corporate-action pass repriced one of them and not the other --
    and BLUME/METUR, which were identical for 466 bars years ago and have gone
    their own way since, score far below the threshold and are left alone.
    Untraded days were dropped upstream and are carried forward here so that a
    twin quoted on a day the other was not still lines up.

    The survivor is the one that needed the fewest repairs -- it was the
    better-maintained series to begin with, which for DAGHL/TRHOL is the twin
    that already had the split applied -- then the longer one, then
    alphabetical order so the choice is reproducible.
    """
    sessions = np.sort(df.timestamp.unique())[-DUP_WINDOW:]
    wide = (df[df.timestamp.isin(sessions)]
            .pivot(index="timestamp", columns="symbol", values="close")
            .ffill())
    r = np.log(wide / wide.shift(1)).to_numpy()[1:]
    symbols = list(wide.columns)
    live = (~np.isnan(r)).sum(axis=0) >= DUP_MIN_BARS
    live &= (np.abs(np.nan_to_num(r)) > 1e-9).sum(axis=0) >= DUP_MOVERS

    # Union-find over pairs that agree often enough to be the same company.
    parent = {s: s for s in symbols}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for i, symbol in enumerate(symbols):
        if not live[i]:
            continue
        both = ~np.isnan(r[:, [i]]) & ~np.isnan(r)
        agree = (np.abs(r[:, [i]] - r) < 1e-6) & both
        score = np.divide(agree.sum(axis=0), both.sum(axis=0),
                          out=np.zeros(len(symbols)), where=both.sum(axis=0) > 0)
        for j in np.flatnonzero(live & (score >= DUP_AGREEMENT)):
            if j != i:
                parent[find(symbols[j])] = find(symbol)

    members_of = {}
    for s in symbols:
        members_of.setdefault(find(s), []).append(s)

    bars = df.symbol.value_counts()
    dropped, groups = [], []
    for members in members_of.values():
        if len(members) < 2:
            continue
        winner = min(members, key=lambda s: (repairs.get(s, 0), -bars.get(s, 0), s))
        losers = sorted(s for s in members if s != winner)
        groups.append({"kept": winner, "dropped": losers,
                       "repairs": {s: repairs.get(s, 0) for s in sorted(members)}})
        dropped.extend(losers)

    audit["duplicate_symbols"] = {
        "groups": len(groups),
        "symbols_dropped": len(dropped),
        "detail": sorted(groups, key=lambda g: g["kept"]),
    }
    return df[~df.symbol.isin(dropped)].copy()


def clean(raw):
    """Run every pass. Returns (clean frame, audit dict)."""
    audit = {}
    df = raw.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    audit["input"] = {
        "bars": len(df),
        "symbols": int(df.symbol.nunique()),
        "first": str(df.timestamp.min().date()),
        "last": str(df.timestamp.max().date()),
        "duplicate_rows": int(df.duplicated(["symbol", "timestamp"]).sum()),
    }
    df = df.drop_duplicates(["symbol", "timestamp"], keep="first")
    raw_bars = df.symbol.value_counts()

    df = drop_weekends(df, audit)
    df = drop_untraded(df, audit)
    df = clamp_ranges(df, audit)
    df = drop_reverting_spikes(df, audit)

    df = adjust_corporate_actions(df, audit)
    # How much repair each symbol needed, for the duplicate tie-break below:
    # bars this run had to throw away, plus corporate actions it had to infer.
    lost = (raw_bars - df.symbol.value_counts()).fillna(raw_bars).astype(int)
    repairs = (lost + pd.Series(audit["corporate_actions"]["per_symbol"])
               .reindex(lost.index).fillna(0).astype(int)).to_dict()

    df = fix_volume_outliers(df, audit)
    df = collapse_duplicates(df, audit, repairs)

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    audit["output"] = {
        "bars": len(df),
        "symbols": int(df.symbol.nunique()),
        "bars_removed": audit["input"]["bars"] - len(df),
        "max_abs_daily_return": round(float(
            df.groupby("symbol", sort=False).close.pct_change().abs().max()), 4),
        "max_intraday_range": round(float((df.high / df.low).max()), 4),
    }
    return df, audit


def _summary(audit):
    o, i = audit["output"], audit["input"]
    return "\n".join([
        f"in   {i['bars']:>8,} bars  {i['symbols']:>4} symbols  {i['first']}..{i['last']}",
        f"  weekend bars                {audit['weekend_bars']['bars']:>7,}",
        f"  untraded (zero-volume)      {audit['untraded_bars']['bars']:>7,}"
        f"   ({len(audit['untraded_bars']['sessions_emptied'])} sessions emptied)",
        f"  impossible ranges clamped   {audit['clamped_ranges']['bars']:>7,}"
        f"   ({audit['unsalvageable_bars']['bars']} dropped outright)",
        f"  spike-and-revert prints     {audit['reverting_spikes']['bars']:>7,}",
        f"  corporate actions adjusted  {audit['corporate_actions']['events']:>7,}"
        f"   ({audit['corporate_actions']['symbols']} symbols)",
        f"  pre-listing bars truncated  {audit['truncated_prelisting']['bars']:>7,}"
        f"   ({audit['truncated_prelisting']['symbols']} symbols)",
        f"  volumes repaired            {audit['volume_outliers']['bars']:>7,}",
        f"  duplicate tickers dropped   {audit['duplicate_symbols']['symbols_dropped']:>7,}"
        f"   ({audit['duplicate_symbols']['groups']} groups)",
        f"out  {o['bars']:>8,} bars  {o['symbols']:>4} symbols"
        f"  (-{o['bars_removed']:,})",
        f"     max |daily return| {o['max_abs_daily_return']:.4f}"
        f"   max high/low {o['max_intraday_range']:.4f}",
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("parquet", nargs="?", type=Path,
                   default=PROJECT_ROOT / "app" / "data" / "bist_1d.parquet")
    p.add_argument("-o", "--output", type=Path,
                   help="default: <input>_clean.parquet next to the input")
    p.add_argument("--audit", type=Path, help="default: <output>.audit.json")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = p.parse_args()

    out = args.output or args.parquet.with_name(args.parquet.stem + "_clean.parquet")
    audit_path = args.audit or out.with_suffix(".audit.json")

    df, audit = clean(pd.read_parquet(args.parquet))
    print(_summary(audit))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    df.to_parquet(out, index=False)
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"\nwrote {out}\n      {audit_path}")


if __name__ == "__main__":
    main()
