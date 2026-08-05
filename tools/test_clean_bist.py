"""Tests for tools/clean_bist.py, one per defect class it claims to repair.

The fixture is a synthetic BIST-shaped feed: a handful of symbols on a random
walk that stays inside the +/-10% price limit, with one defect injected per
symbol so the passes can be asserted in isolation. It is deliberately longer
than DUP_WINDOW, because the duplicate detector only looks at the tail.

The cleaner is imported rather than driven as a subprocess -- `clean()` returns
the frame and the audit together, which is the whole contract -- with one
subprocess test to pin the CLI's output paths.

Run from the project root with the app-local venv (pandas/pyarrow needed):

    app/python/.venv/bin/pytest tools/test_clean_bist.py -q
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import clean_bist as cb  # noqa: E402

TOOL = Path(__file__).with_name("clean_bist.py")
SESSIONS = 320
SPLIT_AT = 200


def _walk(seed, n=SESSIONS, start=100.0):
    """A price path that never breaches the daily limit."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.03, n)).clip(-2, 2))
    return close


def _bars(symbol, close, dates):
    """OHLCV around a close path, with a wick on both sides."""
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "volume": np.full(len(close), 1_000_000.0),
    })


@pytest.fixture(scope="module")
def feed():
    """A raw feed carrying one of each defect. Returns (frame, dates)."""
    dates = pd.bdate_range("2024-01-01", periods=SESSIONS)
    frames = {s: _bars(s, _walk(i), dates) for i, s in enumerate(
        ["CLEAN", "WEEKEND", "UNTRADED", "WICK", "MIXED", "SPIKE",
         "SPLIT", "PRELIST", "VOLUME", "TWINA", "TWINB", "LONER"])}

    # A twin is the same company under a second ticker: same returns, and here
    # also the unadjusted scale that the corporate-action pass has to reconcile.
    frames["TWINB"] = frames["TWINA"].assign(symbol="TWINB")
    frames["TWINB"][cb.OHLC] *= 0.36

    # 1. a bar dated on a Saturday
    extra = frames["WEEKEND"].iloc[[10]].copy()
    extra["timestamp"] = pd.Timestamp("2024-01-13")
    frames["WEEKEND"] = pd.concat([frames["WEEKEND"], extra])

    # 2. a session the symbol did not trade, quoted flat at the previous close
    frames["UNTRADED"].loc[20, ["open", "high", "low", "close"]] = \
        frames["UNTRADED"].loc[19, "close"]
    frames["UNTRADED"].loc[20, "volume"] = 0.0

    # 3. a high that is off by a factor of 100
    frames["WICK"].loc[30, "high"] *= 100

    # 4. a bar whose open and close are on different scales
    frames["MIXED"].loc[40, "open"] /= 100
    frames["MIXED"].loc[40, "low"] /= 100

    # 5. a x100 print that comes back on the next bar
    frames["SPIKE"].loc[50, cb.OHLC] *= 100

    # 6. an unadjusted bonus issue: everything from SPLIT_AT on is repriced
    frames["SPLIT"].loc[SPLIT_AT:, cb.OHLC] /= 10
    frames["SPLIT"].loc[SPLIT_AT:, "volume"] *= 10

    # 7. filler history in front of a listing, on a scale of its own
    frames["PRELIST"].loc[:SPLIT_AT - 1, cb.OHLC] /= 3000

    # 8. a volume typo
    frames["VOLUME"].loc[60, "volume"] = 1e14

    raw = pd.concat(frames.values(), ignore_index=True)
    return raw.sort_values(["symbol", "timestamp"]).reset_index(drop=True), dates


@pytest.fixture(scope="module")
def cleaned(feed):
    df, audit = cb.clean(feed[0])
    return df.set_index(["symbol", "timestamp"]).sort_index(), audit


def _closes(cleaned, symbol):
    return cleaned[0].loc[symbol, "close"]


def test_weekend_bars_dropped(cleaned):
    df, audit = cleaned
    assert audit["weekend_bars"]["bars"] == 1
    assert (df.index.get_level_values("timestamp").dayofweek < 5).all()


def test_untraded_bars_dropped(cleaned, feed):
    df, audit = cleaned
    assert audit["untraded_bars"]["bars"] == 1
    assert (df.volume > 0).all()
    assert feed[1][20] not in df.loc["UNTRADED"].index


def test_impossible_wick_clamped_not_dropped(cleaned, feed):
    """The bar survives with its return intact; only the wick is rebuilt.

    Two bars enter the pass -- WICK's and MIXED's -- because the mixed-scale
    bar has an impossible range too. It is clamped like any other and only then
    found to be beyond saving, so it is counted here as well as below.
    """
    df, audit = cleaned
    assert audit["clamped_ranges"]["bars"] == 2
    bar = df.loc[("WICK", feed[1][30])]
    assert bar.high == max(bar.open, bar.close)
    assert (df.high / df.low <= cb.MAX_RANGE).all()


def test_mixed_scale_bar_dropped(cleaned, feed):
    """Nothing in the bar is trustworthy, so clamping cannot save it."""
    df, audit = cleaned
    assert audit["unsalvageable_bars"]["bars"] == 1
    assert feed[1][40] not in df.loc["MIXED"].index


def test_reverting_spike_dropped(cleaned, feed):
    df, audit = cleaned
    assert audit["reverting_spikes"]["bars"] == 1
    assert feed[1][50] not in df.loc["SPIKE"].index


def test_corporate_action_back_adjusted(cleaned, feed):
    """History before the action is scaled onto the post-action price."""
    df, audit = cleaned
    events = [e for e in audit["corporate_actions"]["splits_and_bonus_issues"]
              if e["symbol"] == "SPLIT"]
    assert len(events) == 1
    assert events[0]["factor"] == pytest.approx(0.1, rel=1e-3)
    assert events[0]["date"] == feed[1][SPLIT_AT].date().isoformat()

    # the whole series now sits on the post-action scale, with no cliff
    expect = _walk(6) / 10
    assert df.loc["SPLIT", "close"].to_numpy() == pytest.approx(expect, rel=1e-9)
    assert df.loc["SPLIT", "volume"].to_numpy() == pytest.approx(1e7, rel=1e-9)


def test_prelisting_history_truncated_not_scaled(cleaned, feed):
    """A 3000x break is a different series, so the filler goes rather than
    being scaled up into a plausible-looking price history."""
    df, audit = cleaned
    trunc = {t["symbol"]: t for t in audit["truncated_prelisting"]["detail"]}
    assert trunc["PRELIST"]["dropped_bars"] == SPLIT_AT
    assert trunc["PRELIST"]["listed"] == feed[1][SPLIT_AT].date().isoformat()
    assert len(df.loc["PRELIST"]) == SESSIONS - SPLIT_AT
    # the surviving bars keep their own prices, untouched
    assert df.loc["PRELIST", "close"].to_numpy() == pytest.approx(_walk(7)[SPLIT_AT:])


def test_volume_outlier_replaced_price_untouched(cleaned, feed):
    df, audit = cleaned
    assert audit["volume_outliers"]["bars"] == 1
    bar = df.loc[("VOLUME", feed[1][60])]
    assert bar.volume == pytest.approx(1_000_000.0)
    assert bar.close == pytest.approx(_walk(8)[60])


def test_duplicate_ticker_collapsed(cleaned):
    """One company, one ticker -- and the twin that needed less repair wins."""
    df, audit = cleaned
    groups = audit["duplicate_symbols"]["detail"]
    assert len(groups) == 1
    assert groups[0]["kept"] == "TWINA"
    assert groups[0]["dropped"] == ["TWINB"]
    assert "TWINB" not in df.index.get_level_values("symbol")


def test_unrelated_symbols_survive(cleaned):
    """LONER moves on its own, so it must not be swept up as a twin, and CLEAN
    has no defect to repair so it must come through bit for bit."""
    df, _ = cleaned
    symbols = set(df.index.get_level_values("symbol"))
    assert {"CLEAN", "LONER", "TWINA"} <= symbols
    assert df.loc["CLEAN", "close"].to_numpy() == pytest.approx(_walk(0))
    assert len(df.loc["CLEAN"]) == SESSIONS


def test_output_respects_the_price_limit(cleaned):
    """The point of the exercise: no bar the exchange could not have printed."""
    df, _ = cleaned
    df = df.reset_index()
    r = df.groupby("symbol").close.pct_change().dropna()
    assert r.abs().max() < 0.25
    assert (df.high >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df.low <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df[cb.OHLC] > 0).all().all()


def test_schema_and_ordering_preserved(cleaned, feed):
    df, _ = cleaned
    raw = feed[0]
    out = df.reset_index()[raw.columns]
    assert out.dtypes.equals(raw.dtypes)
    assert out.equals(out.sort_values(["symbol", "timestamp"]))


def test_cli_writes_parquet_and_audit(tmp_path, feed):
    src = tmp_path / "bars.parquet"
    feed[0].to_parquet(src, index=False)
    run = subprocess.run([sys.executable, str(TOOL), str(src)],
                         capture_output=True, text=True, check=True)

    out = tmp_path / "bars_clean.parquet"
    audit = tmp_path / "bars_clean.audit.json"
    assert out.exists() and audit.exists()
    assert "corporate actions adjusted" in run.stdout
    assert json.loads(audit.read_text())["output"]["bars"] == len(pd.read_parquet(out))


def test_dry_run_writes_nothing(tmp_path, feed):
    src = tmp_path / "bars.parquet"
    feed[0].to_parquet(src, index=False)
    run = subprocess.run([sys.executable, str(TOOL), str(src), "--dry-run"],
                         capture_output=True, text=True, check=True)
    assert "nothing written" in run.stdout
    assert not (tmp_path / "bars_clean.parquet").exists()
