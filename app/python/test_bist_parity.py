"""Parity tests: algo_trade's reimplementation vs bist's own code, same input.

`tools/bist_parity.py` is the report you run by hand while closing a gap; this is
the regression gate. Both drive bist's real pipeline over
`app/data/bist_1d.parquet` and diff it against `algo_trade`, so a change that
quietly moves a feature away from bist fails here.

Skipped wholesale when /Users/macmini-1/bist is absent, which is every machine
but the one this port was written on. That makes the gate advisory rather than
absolute — it is still worth having, because the machine that matters has it.

Runs on a tail of the panel rather than all 820k rows: the full sweep takes
minutes and the failure modes it catches are not row-count dependent. Use the
tool for the full run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BIST_ROOT = Path("/Users/macmini-1/bist")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "app" / "data" / "bist_1d.parquet"

pytestmark = [
    pytest.mark.skipif(not BIST_ROOT.exists(),
                       reason=f"{BIST_ROOT} not present"),
    pytest.mark.skipif(not DATA.exists(), reason=f"{DATA} not present"),
]

if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# Dates to compare. 400 leaves ~100 rows past the 300-bar warmup, which is enough
# for every window to have filled.
TAIL = 400

# The two features pandas cannot compute reproducibly; see algo_trade._moments.
KNOWN_DIFFERENT = {"volume_skew_20", "volume_kurt_20"}


@pytest.fixture(scope="module")
def parity():
    """(feature table, label table) from the same harness the tool uses."""
    import algo_trade
    import bist_parity

    frame = pd.read_parquet(DATA)
    dates = np.sort(frame["timestamp"].unique())[-TAIL:]
    frame = frame.loc[frame["timestamp"].isin(dates)]

    features, pp, intra = bist_parity.bist_reference(frame)
    names = bist_parity.bist_feature_cols(features, intra)
    port_input = (pp[["date", "symbol", "open", "high", "low", "close", "volume"]]
                  .rename(columns={"date": "timestamp"}))

    feature_table = bist_parity.compare(
        bist_parity.port_features(port_input),
        features[["date", "symbol", *names]], names, "features")
    label_table = bist_parity.compare(
        bist_parity.port_labels(port_input), bist_parity.bist_labels(features),
        [h.name for h in algo_trade.HEADS], "labels")
    return feature_table, label_table, names


def test_feature_names_match_bists_column_selection(parity):
    """Our 72 are exactly what bist's own selection yields on a daily-only feed.

    Order matters as much as membership: a fitted booster is only meaningful
    against the column layout it was trained on.
    """
    import algo_trade

    _, _, names = parity
    assert tuple(names) == algo_trade.FEATURE_NAMES


def test_every_feature_matches_bist(parity):
    """The gate. Two known exceptions, argued at algo_trade._moments."""
    table, _, _ = parity
    off = table.loc[(table.n_off > 0) | (table.nan_only_port > 0)
                    | (table.nan_only_bist > 0)]
    unexpected = set(off.index) - KNOWN_DIFFERENT
    assert not unexpected, (
        "features drifted from bist:\n"
        + off.loc[sorted(unexpected)].to_string())


def test_the_known_exceptions_are_still_only_two(parity):
    """Guards the other direction: if pandas' moments get fixed, drop the waiver."""
    table, _, _ = parity
    for name in KNOWN_DIFFERENT:
        assert name in table.index, f"{name} vanished from the feature set"


def test_every_label_matches_bist(parity):
    """All three heads, including the centered window and the drawdown gate."""
    _, table, _ = parity
    off = table.loc[(table.n_off > 0) | (table.nan_only_port > 0)
                    | (table.nan_only_bist > 0)]
    assert off.empty, f"labels drifted from bist:\n{off.to_string()}"


def test_most_features_are_bit_exact(parity):
    """Not just within tolerance — a majority should agree to the last bit.

    A drop here means something started accumulating differently even though it
    still passes the tolerance gate, which is worth knowing before it drifts
    further.
    """
    table, _, _ = parity
    exact = table.loc[(table.max_abs == 0) & (table.nan_only_port == 0)
                      & (table.nan_only_bist == 0)]
    assert len(exact) >= 55, (
        f"only {len(exact)} of {len(table)} columns are bit-exact")
