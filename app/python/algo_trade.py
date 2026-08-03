"""AlgoTrade — the bist manipulation model as a stonks strategy.

`ManipulationModel` is three XGBoost regressors on a drawdown-gated
excess-return target, ported from /Users/macmini-1/bist by way of
/Users/macmini-1/trade_algo. Despite the name it is not a manipulation
classifier: it predicts a sigma-multiple of forward market-excess return and
zeroes the cases you would have been stopped out of first, which makes it a
swing-long alpha ranker.

Client surface is two methods:

    model = ManipulationModel().train(pd.read_parquet("app/data/bist_1d.parquet"),
                                      train_end="2024-12-31")
    sigmas = model.signal(ctx.history(ManipulationModel.LOOKBACK))

plus `save`/`load`, because the strategy loads a pre-trained artifact rather
than training inside the engine.

The invariant everything rests on is that `train` and `signal` run the *same*
feature code, and that code is window-bounded: for any i >= LOOKBACK - 1,

    _features(panel)[i] == _features(panel[i - LOOKBACK + 1 : i + 1])[-1]

exactly, at the float32 precision `_design_matrix` hands the trees. That identity
is why a model fit on a whole panel can be scored one bar at a time without the
two disagreeing. `obv` is the sole exception: bist accumulates it from a symbol's
first bar and the trees split on its level, so a window rebases it. The strategy
tracks it across ticks and passes it to `signal` instead — which is also why the
run has to start at the beginning of the feed.

Parity with bist is checked mechanically, not asserted. `tools/bist_parity.py`
drives *bist's own code* over this engine's parquet and diffs it against this
module column by column; the current state is 70 of 72 features and all 3 labels
agreeing to float tolerance, 60 of them bit-exact.

Deviations from bist, all forced:

 1. 72 features, not 79. bist's 7 intraday features aggregate 15-minute bars and
    this engine's feed is daily only, so they would be all-NaN — at which point
    bist's own `notna().sum() > 1000` column filter drops them, leaving exactly
    these 72. Unlike bist all three heads share one feature set, because bist's
    DN_FEATURE_EXCLUDE is entirely intraday.
 2. `volume_skew_20` and `volume_kurt_20` are recomputed per window rather than
    accumulated incrementally as pandas does. Not a preference: pandas' rolling
    moments are not window-invariant (measured at up to 6.7e-8 for kurt), so
    matching bist here would break the contract above. See `_moments`. These are
    the only two features the parity harness reports as differing.
 3. The engine's feed is not bist's. bist's `open` is Is Yatirim's daily VWAP
    (`HGDG_AOF`), its `volume` is lira turnover rather than share count, and it
    starts in 2016 rather than 2020. So the *code* matches and the *numbers* do
    not — a sigma printed here is not comparable to a sigma in a bist report, and
    the parity harness exists precisely to keep that distinction measurable.

Things this port reproduces that it would not have chosen, because they are what
the shipped model was fit on: the `+ EPSILON` guard on every dispersion
denominator (`_div`), filling the design matrix with 0.0 so the trees never learn
a missing-value direction (`_design_matrix`), the centered label window that
scores a move already under way (`_labels`), and no corporate-action handling at
all (`_returns`). Each is argued at its own docstring.

Requires xgboost in the venv (`app/python/.venv/bin/pip install xgboost`; on
macOS also `brew install libomp`). A failed import makes this module invisible
to strategy discovery, so check it before wondering where the strategy went.

Train the artifact from the project root:

    app/python/.venv/bin/python app/python/algo_trade.py --train-end 2024-12-31
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import stonks
from stonks import OrderSide

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy tunables — everything `AlgoTradeStrategy` decides with, in one place.
# The class attributes at the bottom of this module bind to these, so `params`
# and the GUI's per-run overrides are unaffected: edit here to move the default,
# override in the GUI to try one run.
#
# Everything else in this module belongs to the *model*, not the strategy.
# Changing a feature, a label or a training constant invalidates the saved
# artifact and needs a retrain; changing anything below does not.
# ---------------------------------------------------------------------------

# --- Entry -----------------------------------------------------------------
# Take the day's top slice of the model's own cross-sectional ranking, rather
# than bist's absolute ALPHA5 / ALPHA10 sigmas.
#
# Measured, and the difference is the whole strategy. The heads are a *ranker* —
# the out-of-sample decile lift on up_h5 is clean and monotone across the full
# score range — and an absolute threshold throws that away to ask a question the
# model cannot answer, "is 2.8 big?", whose meaning drifts with every retrain.
# Worse, a threshold that extreme only trips on the most explosive bars, and on
# this feed those are the ones that closed at the +10% band: 77 of the 92 signals
# bist's gate produced over 2025-2026, which are fills a real book would not have
# given.
#
# Walk-forward over four cutoffs (2024-06-30, 2024-12-31, 2025-06-30,
# 2025-12-31), each fit evaluated only on bars after itself, mean return per
# trade with limit-locked bars refused:
#
#            abs 2.8/1.8   top 0.5%   top 1%   top 2%
#   mean        -6.14%      +2.35%    +2.33%   +2.15%
#   worst       -7.38%      -1.76%    -0.38%   +0.66%
#
# The absolute gate is negative at every cutoff — that much is robust. Among the
# slices the means are indistinguishable, so the choice is about variance, and
# 2% is the only one positive in all four periods. 0.5% won the window it was
# first measured on (+5.3%) and lost the most recent one, on 78 trades; taking it
# would be fitting the slice width to a single sample.
ENTRY_TOP_PCT = 2.0
# 1 refuses names that closed at the +10% band. bist's backtest disables this
# (LU_OVERRIDE), and reproducing bist was the point while the gate was bist's.
# It is on now because the percentile gate makes it affordable: refusing locked
# names *improves* the top-0.5% slice (+5.3% against +4.5% including them), which
# says buying an exhausted locked move is bad on its own terms and not merely an
# execution nuisance. See `AlgoTradeStrategy._limit_locked`.
SKIP_LIMIT_LOCKED = 1

# --- Exit ------------------------------------------------------------------
# bist's SL_PCT, kept verbatim. The label is drawdown-gated at -10%, so this is
# what makes the strategy trade what the model was taught to find, and it is the
# only exit that can act on a gap.
STOP_PCT = 10.0
# Replaces bist's +30% TP_PCT: a held name leaves when its close finishes below
# its EXIT_MA-bar simple moving average of traded closes.
EXIT_MA = 20

# --- Sizing ----------------------------------------------------------------
# Names held at once, and the sizing that follows from it. bist sizes at 5% of
# available balance with no cap, which was survivable only because its absolute
# gate fired 92 times in 18 months. A percentile gate fires every session — top
# 0.5% of ~630 scored names is roughly three a day — so an uncapped book would
# exhaust cash within a few entries and the rest would be broker rejections,
# leaving the strategy holding whatever happened to rank first that morning.
# Sizing at equity/MAX_POSITIONS instead reaches full deployment exactly at the
# cap and cannot overcommit before it.
MAX_POSITIONS = 10
CASH_BUFFER = 0.02   # held back from entries for fees and overnight gaps


EPS = 1e-9

# Bars of history needed to produce one row of features. The binding constraint
# is obv_slope_20: a 20-bar slope over a 260-bar rolling sum reaches back 279.
LOOKBACK = 300

NO_EXTREME = 9999.0    # bist's sentinel for "no past extreme on record"

# bist's LIMIT_MOVE_THRESHOLD. BIST equities trade inside a +/-10% daily band,
# so a bar whose close-to-close move reaches this printed at the band. Measured
# on app/data/bist_1d.parquet the |return| histogram spikes hard across
# 0.095-0.100 (39,476 bars) and collapses immediately past it, in every year
# from 2020 to 2026 — there is no +/-20% regime in this feed.
LIMIT_HIT = 0.095
# bist's OHLC_ORDER_TOLERANCE: the source data carries float artifacts (a low a
# shade above the close), and strict comparison rejects ~7% of clean rows.
OHLC_ORDER_TOLERANCE = 1e-4

# bist/config.py::XGBOOST_PARAMS verbatim, minus the two dispatch keys bist's own
# `XGBoostDetector.fit` strips before handing them to xgboost
# (`label_forward_buffer`, `val_fraction`). Kept in the sklearn spelling because
# bist fits through `xgb.XGBRegressor`, and the wrapper is not just sugar: it
# derives `base_score` from the training labels and stores `best_iteration` on the
# estimator, both of which the native `xgb.train` path would resolve differently.
#
# `n_jobs: 1` is load-bearing, not conservative. With `tree_method="approx"` the
# histogram reduction order depends on the thread count, so a multi-threaded fit
# is not reproducible against bist's. bist's own comment gives a second reason:
# on macOS, xgboost 3.x and torch in one process trip a libomp mutex unless both
# the tree method and the thread count are pinned.
#
# bist's config records the Huber-vs-squared-error A/B that picked this objective
# (K=10 top-K loss rate 30.9% vs 43.1%) — read it before changing it.
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:pseudohubererror",
    "eval_metric": "mphe",
    "huber_slope": 1.0,
    "tree_method": "approx",
    "early_stopping_rounds": 30,
    "random_state": 0,
    "n_jobs": 1,
}
# sklearn-name -> native-name, for the keys where the two APIs disagree. bist
# fits through `xgb.XGBRegressor`; this port uses `xgb.train`, because the sklearn
# wrapper cannot even be constructed without scikit-learn installed and the
# strategy runs inside the engine's embedded interpreter. Verified equivalent: on
# a 40k x 72 fit that ran 169 boosting rounds before early stopping, both paths
# returned the same `best_iteration` and bit-identical predictions. Since xgboost
# 2.0 the intercept is fitted inside the booster for both APIs, which is the part
# that used to differ.
XGB_NATIVE_NAMES = {
    "learning_rate": "eta",
    "reg_lambda": "lambda",
    "reg_alpha": "alpha",
    "random_state": "seed",
    "n_jobs": "nthread",
}
# Consumed by `xgb.train`'s signature rather than passed in the params dict.
XGB_FIT_KEYS = ("n_estimators", "early_stopping_rounds")

VAL_FRACTION = 0.15      # bist's val_fraction, chronological
DEFAULT_DAILY_VOL = 0.03  # bist's fallback when excess_return_vol_60 is NaN
MIN_HISTORY = 60          # bist's MIN_HISTORY_DAYS
FEATURE_COVERAGE = 0.8    # a row needs this fraction of its features finite

# bist's TRAIN_START_DATE. Its daily features span the full history but the
# production screen's fit is restricted to 2024 onward, because that is where its
# 15-minute coverage becomes dense enough for the 7 intraday features to be
# non-NaN. Those features do not exist in this feed at all, so the cutoff buys
# nothing here — it is kept because training window length is the second-largest
# driver of predicted-sigma scale after the label, and a fit on 2020-2024 is a
# different model.
TRAIN_START_DATE = "2024-01-01"

# ...and bist's backtest deliberately does NOT use it: walk_forward_static.py:65
# reassigns TRAIN_START_DATE to 2000-01-01 before fitting, i.e. full history. That
# pairing is load-bearing, because the strategy's entry gate is an absolute sigma.
# Fit from 2024 only, this model's h5 predictions reach 3.088 at the 99.9th
# percentile, so bist's `h5 >= 3.0 AND h10 >= 2.0` conjunction almost never fires —
# measured at 2 trades across 18 out-of-sample months. Full history widens the
# output, which is what makes those thresholds mean what bist intended.
BACKTEST_TRAIN_START = "2000-01-01"

# bist's MIN_TURNOVER_PERCENTILE: symbols below this percentile of the
# cross-sectional median-turnover distribution are not scored.
MIN_TURNOVER_PERCENTILE = 0.40

# The 72 features, in the order `_features` stacks them. Explicit because column
# order is part of a fitted booster's contract: a saved model is only meaningful
# against this exact layout. Names are unchanged from bist so the two can be
# compared column by column.
FEATURE_NAMES = (
    "volume_zscore_5", "volume_ratio_5", "volume_zscore_20",
    "volume_ratio_20", "volume_zscore_60", "volume_ratio_60",
    "log_volume_ratio_20", "volume_pct_rank_60", "volume_acceleration",
    "volume_spike_count_20", "volume_cv_20", "volume_skew_20",
    "volume_kurt_20", "amihud_illiquidity_20", "vol_return_corr_20", "obv",
    "obv_slope_20", "turnover", "turnover_zscore_20", "turnover_ratio_20",
    "vwap_deviation_20", "log_return_1d", "cum_return_3", "cum_return_5",
    "cum_return_10", "cum_return_20", "body_ratio", "upper_shadow_ratio",
    "lower_shadow_ratio", "close_location_value", "overnight_gap",
    "intraday_return", "overnight_to_intraday_ratio_20",
    "body_ratio_mean_10", "body_ratio_std_10", "body_ratio_mean_20",
    "body_ratio_std_20", "bullish_candle_ratio_20", "shadow_asymmetry_20",
    "distance_from_ma_20", "distance_from_ma_50", "return_autocorr_30",
    "consecutive_up_days", "consecutive_down_days", "efficiency_ratio_20",
    "realized_vol_5", "realized_vol_20", "realized_vol_60",
    "parkinson_vol_20", "garman_klass_vol_20", "rogers_satchell_vol_20",
    "yang_zhang_vol_20", "vol_ratio_short_long", "vol_of_vol_20", "atr_14",
    "atr_expansion", "parkinson_to_close_vol_ratio_20",
    "corwin_schultz_spread_20", "roll_spread_20", "hl_range_ratio_20",
    "kyle_lambda_20", "volume_zscore_rank", "return_rank",
    "volatility_rank", "market_log_return_1d", "excess_return_vol_60",
    "market_realized_vol_60", "market_cum_return_60",
    "market_vol_ratio_short_long", "days_since_past_extreme",
    "move_concentration", "close_adj",
)
VOL_COLUMN = FEATURE_NAMES.index("excess_return_vol_60")
OBV_COLUMN = FEATURE_NAMES.index("obv")
DAYS_SINCE_COLUMN = FEATURE_NAMES.index("days_since_past_extreme")

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Head:
    """One (target, horizon) configuration. All three share the regressor."""

    name: str
    horizon: int
    direction: str              # "up" | "dn"
    max_drawdown: float | None  # None = unconditional, no drawdown gate


HEADS = (
    Head("up_h5", 5, "up", -0.10),
    Head("up_h10", 10, "up", -0.10),
    Head("dn_h5", 5, "dn", None),
)


# ---------------------------------------------------------------------------
# Rolling helpers. Every one takes a (T, N) frame and reduces down the time
# axis, so bist's per-symbol groupby-transform becomes a single call over all N
# columns. Nothing here looks forward.
# ---------------------------------------------------------------------------

def _std(x, w, mp):
    """Rolling std, with the residue on a flat window snapped to exact zero.

    A window whose values are all identical has zero dispersion. pandas
    computes rolling variance incrementally, so instead of 0.0 it can leave
    ~1e-6 of the level behind — and whether it does depends on how many windows
    were processed before it. That makes every z-score built on top
    path-dependent, which `_ratio`'s `den > 0` test then amplifies into a
    NaN-or-not disagreement between batch and windowed scoring.

    max and min carry no accumulation, so comparing them detects a flat window
    exactly. BIST halts and limit-locks constantly, so this is common, not a
    corner case.
    """
    s = x.rolling(w, min_periods=mp).std()
    r = x.rolling(w, min_periods=mp)
    return s.where(r.max() != r.min(), 0.0)


def _wsum(x, w, mp, chunk=256):
    """Rolling sum computed fresh from each window rather than incrementally.

    The same failure as `_std`, one level down. pandas advances a rolling sum by
    adding the entering value and subtracting the leaving one, so the answer
    carries the cancellation history of everything before it: a window of
    returns summing to ~0 lands on 5e-17 or on 0.0 depending on how much panel
    preceded it. That is far below any tolerance worth caring about until a tree
    split sits at zero, at which point the row goes down the other branch.

    Summing the window itself is order-independent given the same contents,
    which is exactly what the windowing contract needs.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    padded = np.vstack([np.full((w - 1, N), np.nan), a])
    out = np.full((T, N), np.nan)
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            padded[lo:hi + w - 1], w, axis=0)
        n = np.isfinite(v).sum(axis=2)
        out[lo:hi] = np.where(n >= mp, np.nansum(v, axis=2), np.nan)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _var(x, w, mp):
    """Rolling variance, flat windows snapped to zero. See `_std`."""
    v = x.rolling(w, min_periods=mp).var()
    r = x.rolling(w, min_periods=mp)
    return v.where(r.max() != r.min(), 0.0)


def _moments(x, w, mp, chunk=256):
    """Exact rolling (skew, kurt), matching pandas' bias-corrected definitions.

    pandas computes these incrementally, which fails on volume: the series spans
    zero to 1e14, so once a symbol's big days age out of the window the running
    power sums have lost the precision to describe what is left. A dead name
    whose window is nineteen zeros and a single one-lot trade has a real skew of
    4.47, and the incremental path returns NaN for it — but only if enough
    history preceded it, which makes it path-dependent as well as wrong.

    Recomputing each window from scratch over a strided view is exact and
    order-independent. Chunked over time because the (T, N, w) intermediates
    would otherwise run to hundreds of MB.

    This is the one place the port knowingly does not match bist bit-for-bit, and
    it is forced rather than preferred. Measured on a volume-like column with a
    dead stretch: computing pandas' rolling skew/kurt over a full column and over
    just its trailing 300-bar window disagrees by up to 3.9e-11 (skew) and 6.7e-8
    (kurt), so pandas is not window-invariant, while this function's batch and
    windowed answers are identical to the bit. Adopting pandas here would break
    the batch-equals-window contract the whole port rests on — a model fit on a
    panel would no longer be the model that scores a bar. On the 820k-row parity
    run these two columns differ from bist on 0.4% and 0.1% of cells; every other
    feature agrees.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    padded = np.vstack([np.full((w - 1, N), np.nan), a])
    skew = np.full((T, N), np.nan)
    kurt = np.full((T, N), np.nan)

    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            padded[lo:hi + w - 1], w, axis=0)              # (hi-lo, N, w)
        n = np.isfinite(v).sum(axis=2).astype(float)
        # An all-NaN window is a symbol that was not listed yet; nanmean warns
        # and returns NaN, which is the answer we want. `ok` drops it either way.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = v - np.nanmean(v, axis=2, keepdims=True)
            m2 = np.nanmean(d ** 2, axis=2)
            m3 = np.nanmean(d ** 3, axis=2)
            m4 = np.nanmean(d ** 4, axis=2)

        with np.errstate(invalid="ignore", divide="ignore"):
            ok = (n >= mp) & (n >= 4) & (m2 > 0)
            g1 = m3 / m2 ** 1.5
            skew[lo:hi] = np.where(ok, np.sqrt(n * (n - 1)) / (n - 2) * g1, np.nan)
            g2 = m4 / m2 ** 2 - 3.0
            kurt[lo:hi] = np.where(
                ok, ((n + 1) * g2 + 6.0) * (n - 1) / ((n - 2) * (n - 3)), np.nan)

    return (pd.DataFrame(skew, index=x.index, columns=x.columns),
            pd.DataFrame(kurt, index=x.index, columns=x.columns))


def _div(num, den):
    """num / (den + EPS) — bist's guard on every dispersion denominator.

    Every denominator this is used on is a dispersion measure: a rolling
    standard deviation, an intraday range, a sum of absolute moves. bist adds
    EPSILON to all of them, and this reproduces that verbatim.

    Be clear about what it costs. When the dispersion is genuinely zero — a
    halted or limit-locked name whose price has not moved for twenty bars —
    `x / 1e-9` is not a large z-score, it is an undefined ratio rendered as
    1e10, and BIST halts often enough for that to be common rather than a
    corner case. Worse, it is not reproducible: whether a collapsed std lands on
    exactly 0.0 or on 1.2e-06 depends on floating-point accumulation order, so
    the same bar can score differently depending on how much history preceded
    it. An earlier version of this port returned NaN here instead, which is both
    the honest answer and a stable one, and which XGBoost handles natively.

    It is reproduced anyway because the fitted trees split on those 1e10 values,
    so a model that does not produce them is a different model. bist's four
    genuine `.where()`-to-NaN sites (limit masking, a negative Corwin-Schultz
    alpha, a non-negative Roll autocovariance, and `safe_hl`) stay as `.where()`.
    """
    return num / (den + EPS)


def _slope(x, w, mp, chunk=256):
    """Rolling OLS slope on a window-LOCAL time index, matching bist's kernel.

    bist runs `rolling().apply(raw=True)` with `t = 0..m-1` demeaned inside each
    window (volume_features.py:61-73). Two things follow that a closed form over
    the global row number gets wrong:

     1. It is not window-invariant in floating point. `t` centred on row 400 and
        the same `t` centred on row 120 give answers differing in the last bits,
        because the closed form's `Stt - St^2/n` cancels two large numbers. Over
        obv, whose magnitude runs to 1e10, that reached 1e-8 relative — enough to
        break the batch-equals-window contract the whole port rests on.
     2. A NaN anywhere in the window makes bist's plain `.sum()` NaN, where the
        fillna-and-count form quietly produces a slope from the rest.

    Note the demeaning of `v` is algebraically a no-op — `sum(t_demeaned) == 0` —
    so it only matters for propagating NaN, which is why it is kept.

    Rows before the window fills use `m = i + 1` values, as pandas hands bist a
    short array there. `m` depends on the row and not the symbol, so those few
    rows are peeled off and the bulk stays fully vectorised.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    out = np.full((T, N), np.nan)

    def kernel(values, m):
        """values: (rows, N, m) -> slope per (row, symbol)."""
        t = np.arange(m, dtype=float)
        t -= t.mean()
        denom = float((t * t).sum())
        if m < 2 or m < mp or denom == 0.0:
            return np.full(values.shape[:2], np.nan)
        return (values * t).sum(axis=2) / denom

    # Leading rows, where the window is not yet full.
    for i in range(min(w - 1, T)):
        out[i] = kernel(a[np.newaxis, : i + 1].transpose(0, 2, 1), i + 1)[0]

    # The bulk: every row from w-1 on has a full window.
    for lo in range(max(w - 1, 0), T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            a[lo - w + 1:hi], w, axis=0)                    # (hi-lo, N, w)
        out[lo:hi] = kernel(v, w)

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _biased_corr(x, y, w, mp):
    """bist's hand-rolled correlation: (E[xy] - E[x]E[y]) / (sx*sy + EPSILON).

    Not Pearson, and deliberately so. The numerator is a *biased* covariance —
    means of products, dividing by n — while the denominator's standard
    deviations come from pandas with ddof=1, dividing by n-1. The result is off
    from the real correlation by a factor of (n-1)/n and can exceed 1.

    Reproduced literally rather than fixed: this is what `vol_return_corr_20`
    (volume_features.py:182) and `kyle_lambda_20` (microstructure.py:106) were
    fit on. `return_autocorr_30` does NOT use this — bist computes that one with
    pandas' own `rolling().corr()`, so it takes a different path.
    """
    mx = x.rolling(w, min_periods=mp).mean()
    my = y.rolling(w, min_periods=mp).mean()
    mxy = (x * y).rolling(w, min_periods=mp).mean()
    return _div(mxy - mx * my, _std(x, w, mp) * _std(y, w, mp))


def _run_length(flag):
    """(T, N) bool -> length of the True run ending at each row.

    One pass down the time axis carrying a per-symbol counter, which is as much
    state as a causal feature may hold.
    """
    out = np.zeros(flag.shape)
    run = np.zeros(flag.shape[1])
    for i in range(len(flag)):
        run = np.where(flag[i], run + 1.0, 0.0)
        out[i] = run
    return out


def _extreme_trigger(excess, vol60):
    """Bars where a symbol's 100-bar idiosyncratic run first clears 3 sigma.

    The 0->1 transition, not "the condition still holds" — a spike stays inside the
    rolling window for ~100 bars after the fact, and bist counts from the edge.
    Both arguments are packed; the result is too.

    Factored out of `_cross_sectional_features` so that the windowed scorer can
    advance `days_since_past_extreme` across ticks off the same definition rather
    than a second copy of it.
    """
    past = _wsum(excess, 100, 50)
    hit = (past > 3.0 * vol60 * np.sqrt(100.0)).fillna(False)
    return hit & ~hit.shift(1, fill_value=False)


def _bars_since(trigger):
    """(T, N) bool -> bars since the last True, `NO_EXTREME` before the first.

    bist's `cumsum(trigger)` group id plus a `cumcount()` within the group, with
    its 9999 sentinel wherever the group id is still 0. The count is unbounded:
    a symbol whose last extreme was 900 bars ago reads 900, not a capped value.

    NaN carries "no trigger yet" because `nan + 1` stays nan, which is what makes
    the sentinel fall out of the same expression as the counter.
    """
    out = np.full(trigger.shape, NO_EXTREME)
    since = np.full(trigger.shape[1], np.nan)
    for i in range(len(trigger)):
        since = np.where(trigger[i], 0.0, since + 1.0)
        out[i] = np.where(np.isfinite(since), since, NO_EXTREME)
    return out


def _move_concentration(abs_ret, thr):
    """Peers on the same date whose |return| clears this row's own threshold.

    Per bist: for row i, #{j != i on the same date : |r_j| > 2 * vol_i}. The
    threshold varies per row, so there is no single per-date count to
    precompute — sort the date's |returns| once and binary-search each row's own
    threshold into it. The self-subtraction at the end matters: a symbol that
    clears its own bar must not count itself.
    """
    out = np.full(abs_ret.shape, np.nan)
    for i in range(len(abs_ret)):
        a, t = abs_ret[i], thr[i]
        known = np.isfinite(t)
        if not known.any():
            continue
        srt = np.sort(a[np.isfinite(a)])
        count = srt.size - np.searchsorted(srt, t[known], side="right")
        own = a[known]
        count = count - (np.isfinite(own) & (own > t[known]))
        out[i, known] = np.maximum(count, 0)
    return out


# ---------------------------------------------------------------------------
# Panel plumbing. Both entry points converge on {field: (T, N) DataFrame},
# indexed by timestamp and columned by symbol.
# ---------------------------------------------------------------------------

def _returns(C):
    """Bar-to-bar close returns, exactly as bist computes them.

    No corporate-action handling: bist's `close_adj` is a verbatim copy of its
    raw close (`data/loader.py:156-168`), so a bonus issue prints as a 2800x
    "return" there and it prints as one here. An earlier version of this port
    booked any move outside the price band as 0%, which was a defensible data
    fix but not what the model was fit on — see the module docstring.
    """
    return C / C.shift(1) - 1.0


def _drop_unusable(df):
    """bist's preprocessing row drops, in bist's order.

    Ports `data/loader.validate` + `data/preprocessor._drop_stale_rows` +
    `_drop_zero_volume_with_price`. Order is load-bearing only in that all three
    run before any rolling window sees the frame; dropping rather than masking is
    what makes a halt shorten the window instead of poisoning it.

    Left out deliberately: bist *raises* on duplicate (symbol, date) and on
    negative volume. Those are assertions about a vendor file, and the engine's
    feed has already been through the parquet writer, so a raise here would only
    fire on a corrupt window at inference time.
    """
    O, H, L, C = (df[f] for f in ("open", "high", "low", "close"))
    # A vendor halt row is stamped O==H==L==0 with a carried-forward close.
    stale = (O == 0) & (H == 0) & (L == 0)
    # Zero volume with a real price is a vendor anomaly, not a session.
    empty = (df["volume"] == 0) & (O > 0)
    # OHLC ordering, with bist's tolerance for float noise in the source.
    tol = OHLC_ORDER_TOLERANCE
    disordered = ((H + tol < L) | (H + tol < O) | (H + tol < C)
                  | (L - tol > O) | (L - tol > C))
    # bist checks ordering only on rows that are not already stale.
    return df.loc[~(stale | empty | (disordered & ~stale))]


def _panel(df):
    """Long-format bars -> {field: (T, N) DataFrame} plus `ret`.

    `df` carries the columns app/data/bist_1d.parquet does: timestamp, symbol,
    open, high, low, close, volume. Missing (timestamp, symbol) cells — a late
    listing, a halt mid-window, a row bist would have dropped — land as NaN,
    which every feature already expects.
    """
    df = _drop_unusable(df)
    panel = {f: df.pivot(index="timestamp", columns="symbol", values=f).sort_index()
             for f in FIELDS}
    panel["ret"] = _returns(panel["close"])
    return panel


def _panel_from_window(window):
    """`ctx.history(n)` -> the same panel shape.

    The window is a ragged long view: rows contiguous and ascending per symbol,
    each symbol's slice ending at the current timestamp, and only symbols that
    printed at that timestamp present at all. Everything past the current bar is
    absent by construction, so the engine's structural lookahead guarantee
    carries into the model without the model having to promise anything.
    """
    df = pd.DataFrame({
        "symbol": list(window.symbol),
        "timestamp": np.asarray(window.timestamp),
        "open": np.asarray(window.open, dtype=float),
        "high": np.asarray(window.high, dtype=float),
        "low": np.asarray(window.low, dtype=float),
        "close": np.asarray(window.close, dtype=float),
        "volume": np.asarray(window.volume, dtype=float),
    })
    return _panel(df)


def _truncate(panel, end):
    """Panel restricted to its first `end` rows."""
    return {k: v.iloc[:end] for k, v in panel.items()}


def _as_index_value(when, index):
    """A date expressed in the panel index's own dtype.

    The panel is indexed by whatever the caller's frame carried: epoch
    milliseconds when the engine built the window, datetime64 when pandas read
    the parquet. Both compare fine against `searchsorted` once converted.
    """
    stamp = pd.Timestamp(when)
    if np.issubdtype(index.to_numpy().dtype, np.integer):
        return stamp.value // 1_000_000
    return np.datetime64(stamp)


def _design_matrix(clipped):
    """Winsorized features -> the array that actually reaches the trees.

    bist's `X = frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)`, run
    identically before every fit and every predict. Two things in it matter:

     1. **The fill.** bist has to fill, because its IsolationForest and
        autoencoder cannot take NaN — but the consequence is that the shipped
        trees have never seen a missing value and never learned a default
        direction for one. Leaving NaN in place, which XGBoost handles natively
        and which is the honest encoding for an undefined z-score on a halted
        name, trains a materially different model.
     2. **The narrowing to float32.** It happens here and not in `_features`,
        because the winsorize quantiles and the cross-sectional ranks are computed
        on the float64 frame. It also happens to be what makes the batch-equals-
        window contract exact rather than approximate: in float64 the two paths
        agree only to ~1e-14 relative, since pandas' rolling mean accumulates
        incrementally and a window starts somewhere else. float32 rounds that away
        at the last step before the trees, which is the precision that counts.
    """
    return np.nan_to_num(clipped, nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


def _liquid_universe(panel, min_percentile):
    """Symbols at or above `min_percentile` of the median-turnover distribution.

    bist's `_liquid_universe`, which gates which names get scored at all. Two
    deliberate differences from bist, both in the direction of less information:

     1. bist takes the median over its entire features frame, which on a
        backtest date includes turnover that has not happened yet. This takes it
        over whatever panel it is handed, so a model trained to `train_end` sees
        only training-period turnover.
     2. bist's turnover is the `turnover` *feature*, `V * (H+L+C)/3`. Same here.
    """
    typical = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    median = (panel["volume"] * typical).median(axis=0)
    cut = median.quantile(min_percentile)
    return set(median.index[median >= cut])


# ---------------------------------------------------------------------------
# Packed view. bist computes every per-symbol feature with
# `groupby("symbol").rolling(...)` over a frame that has already had halt rows
# *dropped*, so a 20-bar window covers a symbol's last 20 **traded** bars.
#
# This port works on a (date, symbol) pivot instead, because that is what lets
# one rolling call serve all ~600 symbols. In a pivot a halted day is a NaN cell
# that still occupies a window slot, which silently makes every window shorter
# than bist's for any symbol that has ever missed a session — measured at 4.7%
# of cells on app/data/bist_1d.parquet, which is enough to perturb the majority
# of 20-bar windows.
#
# The fix is to compact each column to its traded bars before rolling and scatter
# the results back afterwards. Rolling down a packed column is then exactly
# bist's per-symbol rolling, and `.shift(1)` is the previous *traded* bar rather
# than the previous calendar row.
# ---------------------------------------------------------------------------

def _pack_order(printed):
    """Gather indices that lift each column's traded bars to the top.

    `argsort` on the negated mask is stable, so finite rows keep their relative
    order and every hole is pushed below them. The same order is reused for every
    field so the packed views stay row-aligned with each other.
    """
    return np.argsort(~printed.to_numpy(), axis=0, kind="stable")


def _pack(frame, order):
    """(T, N) dated -> (T, N) packed, positional index.

    Slots past a symbol's bar count gather from rows that were NaN to begin with,
    so they stay NaN and the trailing region is inert.
    """
    packed = np.take_along_axis(frame.to_numpy(dtype=float), order, axis=0)
    return pd.DataFrame(packed, columns=frame.columns)


def _unpack(frame, order, index, printed):
    """(T, N) packed -> (T, N) dated, restoring `index`.

    `printed` masks the result: a few features (`volume_spike_count_20`, the
    candle ratios) produce 0.0 rather than NaN on an empty window, and scattering
    those into a date the symbol never traded would invent a bar.
    """
    out = np.full(frame.shape, np.nan)
    np.put_along_axis(out, order, frame.to_numpy(dtype=float), axis=0)
    return pd.DataFrame(out, index=index, columns=frame.columns).where(printed)


def _real_slots(printed):
    """(T, N) bool — packed slot i of a column is a real bar iff i < that count.

    Needed by anything that reads *forward*. Packing is top-aligned, so a column
    ends in dead slots, and `rolling(w, min_periods=mp)` with mp < w will happily
    compute a value inside that dead region from the few real bars above it. A
    backward-looking feature never notices, but the label's `shift(-h)` pulls
    those values back onto real rows, where bist has NaN because its per-symbol
    group genuinely ended. Mask before shifting, not after.
    """
    counts = printed.to_numpy().sum(axis=0)
    return pd.DataFrame(np.arange(len(printed))[:, None] < counts[None, :],
                        columns=printed.columns)


def _packed_panel(panel):
    """Everything the per-symbol feature groups need, in packed form.

    A NaN close means the symbol printed no bar — `_drop_unusable` has already
    removed the rows bist drops, so what is left is exactly bist's surviving set.
    """
    printed = panel["close"].notna()
    order = _pack_order(printed)
    packed = {f: _pack(panel[f], order) for f in FIELDS}
    packed["ret"] = packed["close"] / packed["close"].shift(1) - 1.0
    packed["log_ret"] = np.log(packed["close"] / packed["close"].shift(1))
    packed["limit"] = _limit_hit(packed["ret"], packed["high"],
                                 packed["low"], packed["volume"])
    return packed, order, printed


# ---------------------------------------------------------------------------
# Feature groups. One function per bist module, same feature names, same
# windows and min_periods. Insertion order must match FEATURE_NAMES.
# ---------------------------------------------------------------------------

def _limit_hit(ret, H, L, V):
    """Bars that printed at the price band or locked with no range.

    Port of `data/preprocessor.add_limit_hit`. Load-bearing, not cosmetic: it
    masks the candle-geometry, Parkinson / Garman-Klass / Rogers-Satchell and
    Corwin-Schultz features, all of which divide by an intraday range that
    collapses on a limit day. Roughly 5.3% of bars in this feed qualify.

    A NaN return compares False, which is bist's `.fillna(False)`.
    """
    locked = H.notna() & L.notna() & (H == L) & (V > 0)
    return (ret.abs() >= LIMIT_HIT) | locked


def _true_range(H, L, prev_C):
    """max(|H-L|, |H-C_prev|, |L-C_prev|), skipping NaN as bist's concat().max() does.

    The NaN-skipping matters on a symbol's first bar, where `prev_C` is unknown:
    bist's `pd.concat([...], axis=1).max(axis=1)` returns |H-L| there, whereas an
    elementwise `np.maximum` would propagate NaN and cost the first 14 bars of
    every symbol's ATR.
    """
    parts = [(H - L).abs(), (H - prev_C).abs(), (L - prev_C).abs()]
    stacked = np.stack([p.to_numpy(dtype=float) for p in parts])
    with warnings.catch_warnings():
        # An all-NaN cell is a bar that did not print; NaN is the answer we want.
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmax(stacked, axis=0)
    return pd.DataFrame(out, index=H.index, columns=H.columns)


def _volume_features(out, C, H, L, V, log_ret):
    """Port of bist/features/volume_features.py (21)."""
    abs_ret = log_ret.abs()

    for w in (5, 20, 60):
        mp = max(2, w // 2)
        mean = V.rolling(w, min_periods=mp).mean()
        std = _std(V, w, mp)
        out[f"volume_zscore_{w}"] = _div(V - mean, std)
        out[f"volume_ratio_{w}"] = _div(V, mean)

    ma5 = V.rolling(5, min_periods=3).mean()
    ma20 = V.rolling(20, min_periods=10).mean()
    std20 = _std(V, 20, 10)

    out["log_volume_ratio_20"] = np.log(_div(V + EPS, ma20))
    out["volume_pct_rank_60"] = V.rolling(60, min_periods=30).rank(pct=True)
    out["volume_acceleration"] = _div(ma5, ma20)
    # astype(float) on a comparison against a NaN mean gives False -> 0.0, so a
    # warmup bar counts as "not a spike" rather than unknown. bist's behaviour.
    out["volume_spike_count_20"] = (
        (V > 2.0 * ma20).astype(float).rolling(20, min_periods=10).sum())
    out["volume_cv_20"] = _div(std20, ma20.abs())
    out["volume_skew_20"], out["volume_kurt_20"] = _moments(V, 20, 10)
    out["amihud_illiquidity_20"] = (
        _div(abs_ret, V).rolling(20, min_periods=10).mean())
    out["vol_return_corr_20"] = _biased_corr(V, abs_ret, 20, 10)

    # bist accumulates OBV from the symbol's first bar, and the fitted trees split
    # on its absolute level, so a bounded proxy is a different feature. It is a
    # running sum, so the value depends on where the symbol's history starts:
    # scoring must therefore see the symbol from its first bar in the feed, which
    # is why the strategy needs the run to begin at the start of the data rather
    # than `LOOKBACK` bars before the signal.
    signed = np.sign(log_ret.fillna(0.0)) * V
    obv = signed.cumsum()
    out["obv"] = obv
    # Unaffected by the above: the slope kernel demeans its window, so the
    # additive history constant cancels and 21 bars are enough.
    out["obv_slope_20"] = _slope(obv, 20, 10)

    typical = (H + L + C) / 3.0
    turnover = V * typical
    to_mean = turnover.rolling(20, min_periods=10).mean()
    to_std = _std(turnover, 20, 10)
    out["turnover"] = turnover
    out["turnover_zscore_20"] = _div(turnover - to_mean, to_std)
    out["turnover_ratio_20"] = _div(turnover, to_mean)

    vwap = _div((typical * V).rolling(20, min_periods=10).sum(),
                V.rolling(20, min_periods=10).sum())
    out["vwap_deviation_20"] = _div(C - vwap, _std(typical, 20, 10))


def _price_features(out, O, H, L, C, prev_C, log_ret, limit):
    """Port of bist/features/price_features.py (24)."""
    out["log_return_1d"] = log_ret
    for h in (3, 5, 10, 20):
        out[f"cum_return_{h}"] = _wsum(log_ret, h, max(2, h // 2))

    hl = H - L
    safe_hl = hl.where(~limit & (hl > 0))
    body = C - O
    upper = H - np.maximum(O, C)
    lower = np.minimum(O, C) - L

    out["body_ratio"] = body / safe_hl
    out["upper_shadow_ratio"] = upper / safe_hl
    out["lower_shadow_ratio"] = lower / safe_hl
    out["close_location_value"] = ((C - L) - (H - C)) / safe_hl
    out["overnight_gap"] = _div(O - prev_C, prev_C)
    out["intraday_return"] = _div(C - O, O)
    out["overnight_to_intraday_ratio_20"] = _div(
        out["overnight_gap"].abs().rolling(20, min_periods=10).sum(),
        out["intraday_return"].abs().rolling(20, min_periods=10).sum())

    for w in (10, 20):
        mp = max(3, w // 2)
        out[f"body_ratio_mean_{w}"] = out["body_ratio"].rolling(w, min_periods=mp).mean()
        out[f"body_ratio_std_{w}"] = _std(out["body_ratio"], w, mp)

    out["bullish_candle_ratio_20"] = (
        (C > O).astype(float).rolling(20, min_periods=10).mean())
    out["shadow_asymmetry_20"] = _div(
        upper.where(~limit).rolling(20, min_periods=10).sum(),
        lower.where(~limit).rolling(20, min_periods=10).sum())

    tr = _true_range(H, L, prev_C)
    atr14 = tr.rolling(14, min_periods=7).mean()
    for w in (20, 50):
        ma = C.rolling(w, min_periods=max(5, w // 4)).mean()
        out[f"distance_from_ma_{w}"] = _div(C - ma, atr14)

    # pandas' own rolling correlation, which is what bist uses here — unlike
    # vol_return_corr_20, this one is a real Pearson with consistent ddof.
    out["return_autocorr_30"] = log_ret.rolling(30, min_periods=20).corr(
        log_ret.shift(1))
    out["consecutive_up_days"] = pd.DataFrame(
        _run_length((log_ret > 0).to_numpy()), index=C.index, columns=C.columns)
    out["consecutive_down_days"] = pd.DataFrame(
        _run_length((log_ret < 0).to_numpy()), index=C.index, columns=C.columns)

    net = C - C.shift(20)
    out["efficiency_ratio_20"] = _div(
        net.abs(), C.diff().abs().rolling(20, min_periods=10).sum())


def _volatility_features(out, O, H, L, C, prev_C, log_ret, limit):
    """Port of bist/features/volatility_features.py (12)."""
    ln2 = np.log(2.0)
    ok = ~limit

    for w in (5, 20, 60):
        out[f"realized_vol_{w}"] = _std(log_ret, w, max(3, w // 2))

    pk = (np.log(H / L) ** 2 / (4.0 * ln2)).where(ok)
    out["parkinson_vol_20"] = np.sqrt(
        pk.rolling(20, min_periods=10).mean().clip(lower=0.0))

    gk = (0.5 * np.log(H / L) ** 2 - (2.0 * ln2 - 1.0) * np.log(C / O) ** 2).where(ok)
    out["garman_klass_vol_20"] = np.sqrt(
        gk.rolling(20, min_periods=10).mean().clip(lower=0.0))

    rs = (np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)).where(ok)
    out["rogers_satchell_vol_20"] = np.sqrt(
        rs.rolling(20, min_periods=10).mean().clip(lower=0.0))

    # Yang-Zhang: overnight var + k * open-to-close var + (1-k) * Rogers-Satchell
    on_var = _var(np.log(O / prev_C), 20, 10)
    oc_var = _var(np.log(C / O).where(ok), 20, 10)
    rs_mean = rs.rolling(20, min_periods=10).mean()
    k = 0.34 / (1.34 + 21.0 / 19.0)
    out["yang_zhang_vol_20"] = np.sqrt(
        (on_var + k * oc_var + (1.0 - k) * rs_mean).clip(lower=0.0))

    out["vol_ratio_short_long"] = _div(out["realized_vol_5"], out["realized_vol_60"])
    out["vol_of_vol_20"] = _std(out["realized_vol_5"], 20, 10)

    tr = _true_range(H, L, prev_C)
    out["atr_14"] = tr.rolling(14, min_periods=7).mean()
    out["atr_expansion"] = _div(tr.rolling(5, min_periods=3).mean(),
                                tr.rolling(20, min_periods=10).mean())
    out["parkinson_to_close_vol_ratio_20"] = _div(
        out["parkinson_vol_20"], out["realized_vol_20"])


def _microstructure_features(out, H, L, C, prev_C, V, log_ret, limit):
    """Port of bist/features/microstructure.py (4)."""
    # Corwin-Schultz (2012). Negative alpha implies a negative spread; the
    # estimator is known to misfire when a jump swamps the transaction-cost
    # signal, so those days drop out rather than being clamped to zero.
    prev_H, prev_L = H.shift(1), L.shift(1)
    beta = np.log(H / L) ** 2 + np.log(prev_H / prev_L) ** 2
    gamma = np.log(np.maximum(H, prev_H) / np.minimum(L, prev_L)) ** 2
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.where(~(limit | limit.shift(1, fill_value=False)))
    spread = spread.where(spread > 0)
    out["corwin_schultz_spread_20"] = spread.rolling(20, min_periods=10).mean()

    # Roll (1984): S = 2*sqrt(-cov(dp_t, dp_{t-1})). Positive autocovariance is
    # inconsistent with the model, so those windows are NaN rather than forced.
    #
    # pandas' own rolling covariance, ddof=1 — bist calls `.rolling().cov()` here
    # (microstructure.py:71) rather than the biased means-of-products identity it
    # uses for vol_return_corr_20 and kyle_lambda_20. The two differ by (n-1)/n,
    # which is enough to flip the sign test on a window whose autocovariance sits
    # near zero and so to change which bars are NaN.
    dp = C.diff()
    n, mp = 20, 10
    cov = dp.rolling(n, min_periods=mp).cov(dp.shift(1))
    out["roll_spread_20"] = 2.0 * np.sqrt(-cov.where(cov < 0))

    out["hl_range_ratio_20"] = (
        _div(H - L, C).where(~limit).rolling(20, min_periods=10).mean())

    # Kyle (1985) lambda as the rolling slope of |return| on volume,
    # cov(|r|, V) / var(V). Same biased-covariance-over-ddof=1-variance mix as
    # vol_return_corr_20; see _biased_corr.
    abs_ret = log_ret.abs()
    mv = V.rolling(n, min_periods=mp).mean()
    mr = abs_ret.rolling(n, min_periods=mp).mean()
    mvr = (V * abs_ret).rolling(n, min_periods=mp).mean()
    out["kyle_lambda_20"] = _div(mvr - mv * mr, _var(V, n, mp))


def _cross_sectional_features(out, C, log_ret, order, printed):
    """Port of bist/features/cross_sectional.py (10).

    These rank each symbol against its peers on the same date, so they break
    per-symbol isolation by design: adding a symbol changes every other symbol's
    rank for that date. Per-symbol causality is untouched — a rank on date t
    reads only date t.

    Takes the *dated* view, unlike the four per-symbol groups. `order` and
    `printed` are threaded through for the two features that are per-symbol
    rollings of a date-derived series and therefore have to pack again.

    Note the consequence at inference: `ctx.history` only returns symbols that
    printed this tick, so the peer set is narrower than the training panel's.
    With ~600 BIST names printing daily the market mean is a fine estimate, but
    it is not the identical computation.
    """
    # Plain float64 ranks, which is what bist ranks. An earlier version rounded
    # to float32 first so that two symbols differing only in the last bits could
    # not swap places depending on accumulation order — cheap insurance worth
    # 1/N of a rank. bist does not do it, and it moved ranks enough to show up in
    # the parity harness, so it is gone.
    def _rank(frame):
        return frame.rank(axis=1, pct=True)

    out["volume_zscore_rank"] = _rank(out["volume_zscore_20"])
    out["return_rank"] = _rank(out["log_return_1d"])
    out["volatility_rank"] = _rank(out["realized_vol_20"])

    # Equal-weighted mean log return: the standard market/idiosyncratic split.
    market = log_ret.mean(axis=1)
    out["market_log_return_1d"] = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], C.shape[1], axis=1),
        index=C.index, columns=C.columns)

    # `excess` is date-derived (it subtracts the market mean) but its rolling std
    # is per-symbol, so it has to go back through the packed view.
    excess = _pack(log_ret.sub(market, axis=0), order)
    vol60 = _std(excess, 60, 30)
    out["excess_return_vol_60"] = _unpack(vol60, order, C.index, printed)

    for name, series in (
        ("market_realized_vol_60", _std(market.to_frame(), 60, 30).iloc[:, 0]),
        ("market_cum_return_60", _wsum(market.to_frame(), 60, 30).iloc[:, 0]),
        ("market_vol_ratio_short_long",
         _div(_std(market.to_frame(), 20, 10).iloc[:, 0],
              _std(market.to_frame(), 60, 30).iloc[:, 0])),
    ):
        out[name] = pd.DataFrame(
            np.repeat(series.to_numpy()[:, None], C.shape[1], axis=1),
            index=C.index, columns=C.columns)

    # Past-only mirror of the forward label: has this symbol had an
    # idiosyncratic 3-sigma run over the last 100 bars? Counts from the 0->1
    # transition, not from "the condition still holds" — a spike stays inside
    # the rolling window for ~100 days after the fact.
    trigger = _extreme_trigger(excess, vol60)
    out["days_since_past_extreme"] = _unpack(
        pd.DataFrame(_bars_since(trigger.to_numpy()), columns=excess.columns),
        order, C.index, printed)
    # Handed back so a windowed caller can carry the counter across bars; see
    # `ScoringState`. Only the last row is ever needed, but the frame is cheap.
    out["_trigger"] = _unpack(trigger.astype(float), order, C.index, printed)

    out["move_concentration"] = pd.DataFrame(
        _move_concentration(log_ret.abs().to_numpy(),
                            (2.0 * out["realized_vol_20"]).to_numpy()),
        index=C.index, columns=C.columns)


def _features(panel):
    """(T, N, 72) float64 — `_features_and_carry` without the state hook."""
    return _features_and_carry(panel)[0]


def _features_and_carry(panel):
    """(T, N, 72) float64 aligned to `panel`'s rows, plus this bar's carry inputs.

    Causal, and window-bounded: for any i >= LOOKBACK - 1,

        _features(panel)[i] == _features(_window(panel, i))[-1]

    That identity is what lets `train` run on a whole panel while `signal` runs
    on 300 bars. Every feature is therefore computable from LOOKBACK rows.

    Two views are in play. The four per-symbol groups run on the packed view, so
    their windows cover traded bars the way bist's do; the cross-sectional group
    runs on the dated view, because ranks and the market mean are per-date
    aggregates. `excess_return_vol_60` and `days_since_past_extreme` need both —
    they are per-symbol rollings of a date-derived series — so they pack again
    internally.

    float64 throughout: bist keeps its feature frame in float64 and narrows to
    float32 only when it builds the model's design matrix, so narrowing here
    would move the winsorize bounds and the cross-sectional ranks.
    """
    packed, order, printed = _packed_panel(panel)
    O, H, L, C, V = (packed[f] for f in FIELDS)
    prev_C = C.shift(1)
    log_ret = packed["log_ret"]
    limit = packed["limit"]

    out = {}
    _volume_features(out, C, H, L, V, log_ret)
    _price_features(out, O, H, L, C, prev_C, log_ret, limit)
    _volatility_features(out, O, H, L, C, prev_C, log_ret, limit)
    _microstructure_features(out, H, L, C, prev_C, V, log_ret, limit)

    index = panel["close"].index
    out = {k: _unpack(v, order, index, printed) for k, v in out.items()}
    _cross_sectional_features(out, panel["close"], _unpack(log_ret, order, index,
                                                           printed),
                              order, printed)
    trigger = out.pop("_trigger")
    out["close_adj"] = panel["close"]

    if tuple(out) != FEATURE_NAMES:
        raise RuntimeError(
            "feature layout drifted from FEATURE_NAMES; a fitted booster is "
            "only valid against the order it was trained on")

    stacked = np.stack(
        [np.asarray(out[k], dtype=float) for k in FEATURE_NAMES], axis=-1)

    # What a windowed caller needs to advance the two carried features by one bar,
    # both read off the *packed* view so "the previous bar" is the previous bar the
    # symbol actually traded rather than the previous calendar row. Getting that
    # wrong drops the increment for any symbol that was halted yesterday.
    counts = printed.to_numpy().sum(axis=0)
    signed = (np.sign(packed["log_ret"].fillna(0.0))
              * packed["volume"]).to_numpy()
    columns = np.arange(signed.shape[1])
    live = counts > 0
    step = np.full(signed.shape[1], np.nan)
    step[live] = signed[counts[live] - 1, columns[live]]
    carry = {
        "obv_step": pd.Series(step, index=panel["close"].columns),
        "trigger": trigger.iloc[-1],
    }
    return np.where(np.isfinite(stacked), stacked, np.nan), carry


def _labels(panel, head):
    """(T, N) target for one head. Reads forward — training only.

    Forward-only port of bist/labels/synthetic_labels.py:

        sigma = cum excess log return over [T+1, T+H] / (excess_vol_60 * sqrt H)
        up:     zeroed where the forward window drew down past max_drawdown
        dn:     sign-flipped, unconditional

    bist takes max(forward, centered) where centered reaches back before T. That
    lets a row score high off a move already under way, which an entry at T+1
    cannot capture, so the centered term is dropped here.

    The drawdown gate is what makes the up target tradable rather than a pure
    move detector: it zeroes the spikes you would have been stopped out of
    before they arrived.
    """
    h = head.horizon
    printed = panel["close"].notna()
    order = _pack_order(printed)
    index = panel["close"].index

    C = _pack(panel["close"], order)
    log_ret = np.log(C / C.shift(1))
    # The market mean is a per-date aggregate, so it is taken on the dated view
    # and the result packed back for the per-symbol rollings.
    dated_ret = _unpack(log_ret, order, index, printed)
    market = dated_ret.mean(axis=1)
    excess = _pack(dated_ret.sub(market, axis=0), order)
    denom = _std(excess, 60, 30) * np.sqrt(h)

    mp = min(max(2, h // 2), h)
    real = _real_slots(printed)
    roll = _wsum(excess, h, mp).where(real)
    sign = 1.0 if head.direction == "up" else -1.0
    # Plain division: unlike the features, bist adds no epsilon to the label's
    # denominator (synthetic_labels.py:111). A zero excess-vol therefore yields
    # +/-inf rather than a huge finite number, and the training filters drop it.
    fwd = sign * roll.shift(-h) / denom              # [T+1, T+h]

    if h > 1:
        # bist scores a row on the better of the forward window and a window
        # centred on it, which reaches back h//2 bars before the signal bar. A
        # next-bar entry cannot capture a move that has already started, so this
        # term inflates the label relative to what is tradable and bist's own
        # precision figures inherit that. It is reproduced because it is what the
        # shipped model was fit on — dropping it shrinks the label's spread by
        # about a quarter at q90 and the fit shrinks with it.
        center = sign * roll.shift(-(h // 2)) / denom  # [T-h+1+h//2, T+h//2]
        both_nan = fwd.isna() & center.isna()
        # max() skips NaN, so whichever window resolved wins outright.
        sigma = pd.concat([fwd.stack(future_stack=True),
                           center.stack(future_stack=True)],
                          axis=1).max(axis=1).unstack().where(~both_nan)
    else:
        sigma = fwd

    if head.max_drawdown is None:
        return _unpack(sigma, order, index, printed).to_numpy(dtype=float)

    fwd_min = C.rolling(h, min_periods=mp).min().where(real).shift(-h)
    # bist's `valid = close > 0` guard, not an epsilon.
    drawdown = ((fwd_min - C) / C).where(C > 0)
    loss = drawdown < float(head.max_drawdown)
    # A row whose drawdown is unresolved is not "clean", it is unknown.
    gated = sigma.where(~loss, 0.0).where(sigma.notna() & drawdown.notna())
    return _unpack(gated, order, index, printed).to_numpy(dtype=float)


class ScoringState:
    """Cross-tick carry for the two features a bounded window cannot reproduce.

    bist computes both from a symbol's first bar ever: `obv` is a running signed-
    volume sum, and `days_since_past_extreme` counts bars since the last 3-sigma
    trigger with a 9999 sentinel before the first one. A 300-bar window rebases
    the first and mis-reports the second as "never" whenever the last trigger is
    older than the window — measured at 131 of 625 symbols on one date, which is
    not a rounding error.

    The fix is to seed from the first window that is at least as long as the whole
    history so far, then advance one bar at a time. `ManipulationModel.signal`
    owns both operations, so the feature definitions stay in one place and the
    caller only has to hold this object and pass it every bar.

    The contract the caller must keep: **call `signal` on every bar, in order,
    from the start of the feed.** Skipping bars silently corrupts both values, and
    nothing downstream can detect it. That is why `AlgoTradeStrategy` scores even
    the in-sample bars it will not trade.
    """

    def __init__(self):
        self.obv = {}          # symbol -> running signed-volume sum
        self.days_since = {}   # symbol -> bars since the last trigger, or None
        self.bars = 0          # bars folded in, so the first call can seed

    def seed(self, obv, days_since):
        self.obv.update(obv)
        self.days_since.update(days_since)

    def advance(self, symbols, signed_volume, triggered):
        """Fold one bar in: add signed volume, and step or reset the counter."""
        for i, symbol in enumerate(symbols):
            if np.isfinite(signed_volume[i]):
                self.obv[symbol] = self.obv.get(symbol, 0.0) + signed_volume[i]
            if triggered[i]:
                self.days_since[symbol] = 0.0
            else:
                current = self.days_since.get(symbol)
                self.days_since[symbol] = None if current is None else current + 1.0
        self.bars += 1


class ManipulationModel:
    """XGBoost regressors on bist's drawdown-gated extreme-move targets.

    Data in, alpha out; knows nothing about the engine, the portfolio, or the
    training schedule. `train` needs a panel's worth of history, `signal` needs
    LOOKBACK bars, and both go through the same feature code — which is what
    makes them the same model.
    """

    LOOKBACK = LOOKBACK

    def __init__(self, heads=HEADS, *, k_sigma=3.0, params=None,
                 winsorize=(0.005, 0.995), min_history=MIN_HISTORY,
                 train_start=TRAIN_START_DATE,
                 min_turnover_percentile=MIN_TURNOVER_PERCENTILE):
        self.heads = tuple(heads)
        self.k_sigma = float(k_sigma)
        self.params = dict(params or XGB_PARAMS)
        self.winsorize = winsorize
        self.min_history = int(min_history)
        self.train_start = train_start
        self.min_turnover_percentile = float(min_turnover_percentile)
        self.models = {}
        self.best_iteration = {}
        self.train_end = None   # last timestamp the fit was allowed to see, ms
        self.liquid = None      # symbols above the turnover cut, or None for all
        self._bounds = None     # (lower, upper), each (72,)

    # -- train --------------------------------------------------------------

    def train(self, df, *, train_end=None):
        """Fit every head on bars up to and including `train_end`.

        `df` is long-format pandas with app/data/bist_1d.parquet's columns.
        `train_end` is a date; bars after it are dropped *before* labels are
        built, so a label whose forward window would reach past the cutoff
        resolves to NaN and is dropped. The embargo is structural, not a
        convention the caller has to remember.
        """
        panel = df if isinstance(df, dict) else _panel(df)
        index = panel["close"].index
        end = len(index)
        if train_end is not None:
            end = int(np.searchsorted(index.to_numpy(),
                                      _as_index_value(train_end, index),
                                      side="right"))
        if end <= 0:
            raise ValueError(f"train_end={train_end} precedes every bar")

        panel = _truncate(panel, end)
        index = panel["close"].index
        last = index[-1]
        self.train_end = (int(last) if isinstance(last, (int, np.integer))
                          else int(pd.Timestamp(last).value // 1_000_000))

        self.liquid = _liquid_universe(panel, self.min_turnover_percentile)
        F = _features(panel)

        # bist's training row filters, in bist's order. `symbol_row_index` is a
        # cumcount over surviving rows, so counting printed bars reproduces it.
        listed = np.isfinite(panel["close"].to_numpy(dtype=float))
        age = np.cumsum(listed, axis=0)
        # int(), not the float product: bist's `max(1, int(0.8 * n))` truncates,
        # so the bar for 72 features is 57 finite, not 58.
        required = max(1, int(FEATURE_COVERAGE * len(FEATURE_NAMES)))
        eligible = (np.isfinite(F).sum(axis=2) >= required) & (age >= self.min_history)
        if self.train_start is not None:
            start = np.searchsorted(index.to_numpy(), _as_index_value(
                self.train_start, index), side="left")
            eligible[:start] = False
        if not eligible.any():
            raise ValueError(
                f"no rows survive the training filters (train_start="
                f"{self.train_start}, train_end={train_end})")

        # Per-date winsorize bounds, fit on the ELIGIBLE rows only — bist fits
        # them after its filters, so an excluded young symbol cannot move the
        # 0.5/99.5 cut of its date.
        masked = np.where(eligible[:, :, None], F, np.nan)
        with warnings.catch_warnings():
            # A feature that is all-NaN on some date has no quantile; clipping
            # against NaN leaves the column untouched, which is what bist does.
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanquantile(masked, self.winsorize[0], axis=1)   # (T, 72)
            hi = np.nanquantile(masked, self.winsorize[1], axis=1)
        # bist forward-fills per-date bounds to any date outside the fit range, so
        # a date past the training window always inherits the last fitted date's.
        # Every date we score is past it, so that one row is all `signal` needs.
        fitted = np.flatnonzero(eligible.any(axis=1))
        self._bounds = (lo[fitted[-1]], hi[fitted[-1]])
        X = np.clip(F, lo[:, None, :], hi[:, None, :])

        X = _design_matrix(X)
        row = np.arange(end)[:, None]

        for head in self.heads:
            y = _labels(panel, head)
            # The last `horizon` dates cannot have a resolved forward window. The
            # label is NaN there anyway; the filter states the embargo explicitly.
            keep = eligible & np.isfinite(y) & (row < end - head.horizon)
            if keep.sum() < 1000:
                raise RuntimeError(
                    f"{head.name}: only {int(keep.sum())} usable training rows")

            # Boolean masking walks row-major, so samples come out in date order
            # and the chronological split for early stopping is a slice.
            Xh = X[keep]
            yh = y[keep].astype(np.float32)
            split = int(len(Xh) * (1.0 - VAL_FRACTION))
            names = list(FEATURE_NAMES)
            booster = xgb.train(
                self._native_params(),
                xgb.DMatrix(Xh[:split], label=yh[:split], feature_names=names),
                num_boost_round=int(self.params["n_estimators"]),
                evals=[(xgb.DMatrix(Xh[split:], label=yh[split:],
                                    feature_names=names), "val")],
                early_stopping_rounds=int(self.params["early_stopping_rounds"]),
                verbose_eval=False,
            )
            self.models[head.name] = booster
            self.best_iteration[head.name] = int(booster.best_iteration)

            resolved = np.flatnonzero(keep.any(axis=1))
            scored = self._predict(head, X[eligible])
            log.info(
                "%s: %d rows, newest label row %d reads through %d (< end=%d), "
                "above %.1f sigma %.3f%%, best_iter=%d, "
                "pred q99=%.3f q999=%.3f max=%.3f",
                head.name, len(Xh), resolved[-1], resolved[-1] + head.horizon,
                end, self.k_sigma, 100.0 * float((yh > self.k_sigma).mean()),
                self.best_iteration[head.name],
                float(np.quantile(scored, 0.99)),
                float(np.quantile(scored, 0.999)), float(scored.max()),
            )
        return self

    def _native_params(self):
        """`self.params` in `xgb.train`'s spelling. See XGB_NATIVE_NAMES."""
        return {XGB_NATIVE_NAMES.get(k, k): v for k, v in self.params.items()
                if k not in XGB_FIT_KEYS}

    def _predict(self, head, rows):
        """Raw predicted sigma for a (rows, 72) design matrix.

        `best_iteration` is passed explicitly rather than left to the booster: it
        decides how many trees score a bar, and a JSON round-trip is not something
        scoring should depend on.
        """
        matrix = xgb.DMatrix(rows, feature_names=list(FEATURE_NAMES))
        return self.models[head.name].predict(
            matrix, iteration_range=(0, self.best_iteration[head.name] + 1))

    # -- score --------------------------------------------------------------

    def signal(self, window, state=None):
        """Per-head alpha for the current bar, one row per printing symbol.

        Returns a DataFrame indexed by symbol with two columns per head:

            <head>      raw predicted sigma; bist ranks on this and nothing else
            <head>_pct  bist's expected_excess_pct, (exp(sigma*vol*sqrt H)-1)*100
                        — the forward *excess* return in percent, which unlike
                        raw sigma is comparable across horizons

        bist's caveat, verbatim: "NOT a calibrated forecast; rank by sigma, treat
        exp% as a magnitude check."

        Rows below `min_history`, with too few finite features, or outside the
        liquid universe score NaN rather than a number derived mostly from
        missing inputs.

        Pass a `ScoringState` when scoring a *window* rather than a whole panel.
        `obv` and `days_since_past_extreme` both accumulate from a symbol's first
        bar, so a window rebases them; the state carries the true values across
        bars and this method advances it. Scoring a full panel (the trainer, the
        screen) needs no state — the panel already holds the history. Everything
        else, `obv_slope_20` included, is window-bounded.
        """
        if not self.models:
            raise RuntimeError("train() or load() before signal()")

        panel = window if isinstance(window, dict) else _panel_from_window(window)
        F, carry = _features_and_carry(panel)
        symbols = panel["close"].columns

        lo, hi = self._bounds
        last = F[-1].copy()                                 # (N, 72)
        if state is not None:
            last = self._carry(state, last, carry, symbols)
        X = _design_matrix(np.clip(last, lo, hi))

        # bist reads the *unwinsorized* excess_return_vol_60 here, and falls
        # back to a flat 3% daily vol where it is missing.
        vol = last[:, VOL_COLUMN]
        vol = np.where(np.isfinite(vol), vol, DEFAULT_DAILY_VOL)

        listed = np.isfinite(panel["close"].to_numpy(dtype=float)).sum(axis=0)
        required = max(1, int(FEATURE_COVERAGE * last.shape[1]))
        usable = ((np.isfinite(last).sum(axis=1) >= required)
                  & (listed >= self.min_history))
        if self.liquid is not None:
            usable &= np.array([s in self.liquid for s in symbols])

        out = pd.DataFrame(index=symbols)
        for head in self.heads:
            sigma = np.where(usable, self._predict(head, X), np.nan)
            out[head.name] = sigma
            out[f"{head.name}_pct"] = np.expm1(
                sigma * vol * np.sqrt(head.horizon)) * 100.0

        return out

    def _carry(self, state, last, carry, symbols):
        """Advance `state` by this bar and substitute the carried feature values.

        On the first call the window is the entire history so far, so the window's
        own answers are correct and become the seed. After that the window has
        slid off the front and only the carry is right.

        The counter's sentinel and the seed's sentinel are the same 9999, so
        `None` is used internally for "no trigger on record" — otherwise the
        sentinel would start incrementing like a real count.
        """
        last = last.copy()
        signed = carry["obv_step"].to_numpy()
        fired = np.nan_to_num(carry["trigger"].to_numpy(dtype=float),
                              nan=0.0) > 0.0

        if state.bars == 0:
            seen = last[:, DAYS_SINCE_COLUMN]
            state.seed(
                {s: float(last[i, OBV_COLUMN]) for i, s in enumerate(symbols)
                 if np.isfinite(last[i, OBV_COLUMN])},
                {s: (None if not np.isfinite(seen[i]) or seen[i] >= NO_EXTREME
                     else float(seen[i]))
                 for i, s in enumerate(symbols)})
            state.bars += 1
        else:
            state.advance(symbols, signed, fired)

        for i, symbol in enumerate(symbols):
            obv = state.obv.get(symbol)
            last[i, OBV_COLUMN] = np.nan if obv is None else obv
            since = state.days_since.get(symbol, None)
            last[i, DAYS_SINCE_COLUMN] = NO_EXTREME if since is None else since
        return last

    # -- persistence --------------------------------------------------------

    def save(self, directory):
        """Same layout as bist's configured_alerts cache, one file per head."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save_model(str(d / f"{name}.json"))
        np.savez(d / "bounds.npz", lower=self._bounds[0], upper=self._bounds[1])
        (d / "meta.json").write_text(json.dumps({
            "feature_names": list(FEATURE_NAMES),
            "heads": [vars(h) for h in self.heads],
            "k_sigma": self.k_sigma,
            "params": self.params,
            "winsorize": list(self.winsorize),
            "min_history": self.min_history,
            "train_start": self.train_start,
            "min_turnover_percentile": self.min_turnover_percentile,
            # best_iteration is carried explicitly rather than read back off the
            # booster: it decides how many trees `signal` uses, and relying on a
            # JSON round-trip to preserve it would make scoring depend on an
            # xgboost implementation detail.
            "best_iteration": self.best_iteration,
            "train_end": self.train_end,
            "liquid": sorted(self.liquid) if self.liquid is not None else None,
        }, indent=2))

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no trained model at {d}. Train one first:\n"
                f"  app/python/.venv/bin/python app/python/algo_trade.py "
                f"--out {d}")
        meta = json.loads(meta_path.read_text())
        if tuple(meta["feature_names"]) != FEATURE_NAMES:
            raise RuntimeError("saved model was fit on a different feature set")

        model = cls(heads=tuple(Head(**h) for h in meta["heads"]),
                    k_sigma=meta["k_sigma"], params=meta["params"],
                    winsorize=tuple(meta["winsorize"]),
                    min_history=meta["min_history"],
                    train_start=meta["train_start"],
                    min_turnover_percentile=meta["min_turnover_percentile"])
        model.best_iteration = dict(meta["best_iteration"])
        model.train_end = meta["train_end"]
        model.liquid = None if meta["liquid"] is None else set(meta["liquid"])
        bounds = np.load(d / "bounds.npz")
        model._bounds = (bounds["lower"], bounds["upper"])
        for head in model.heads:
            booster = xgb.Booster()
            booster.load_model(str(d / f"{head.name}.json"))
            model.models[head.name] = booster
        return model


# ---------------------------------------------------------------------------
# Qullamaggie momentum swing — the pine's long breakout, as a second model.
#
# Port of Setup 1 ("BO") from app/pines/qullamaggie_momentum_swing.pine (v15).
# Nothing here touches ManipulationModel; the two share only the panel-packing
# helpers above.
#
# The pine's breakout is a resting buy-stop parked at a pivot high, armed while a
# flag is intact and good for `order_bars` bars. That level is knowable *before*
# price trades through it, which is the whole reason this model exists: a caller
# can park a real stop order in advance instead of reacting to the break.
#
# Deliberately NOT reusing `_panel` / `_packed_panel`: those apply bist's row
# drops (`_drop_unusable`) and compute `_limit_hit` off BIST's price band, none of
# which belongs in a momentum-breakout model. The row drops would shift the base
# window and the order timer relative to the pine.
# ---------------------------------------------------------------------------

def _qm_sma(x, n, chunk=256):
    """Rolling mean, window-invariant and correctly rounded — pine's `ta.sma`.

    Not `x.rolling(n).mean()`, and not `_wsum(x, n, n) / n` either. The reason is
    the one `_wsum`'s docstring gives, only sharper here. ManipulationModel
    survives pandas' incremental accumulation because `_design_matrix` narrows to
    float32 at the last step and rounds the drift away. This model emits booleans:
    `close > sma20` and `sma10 > sma20` are strict comparisons that flip
    discretely, so a last-bit difference becomes a *different armed order*.

    And the ties are not hypothetical. On BIST dailies, where a 20-bar mean of
    cent-quoted prices lands exactly on a cent often enough to matter, the true
    value routinely equals the price being compared against — at which point the
    answer is decided entirely by summation error. Measured over ~9,000 arm
    events on 40 symbols, pandas' running sum and numpy's pairwise sum each
    landed ~4e-16 off the true mean on three bars apiece, and each of those six
    flipped a gate.

    So the window is summed with Neumaier compensation: each window is summed
    independently in a fixed order (window-invariant, unlike a running sum) and
    the result is correctly rounded, so an exact tie compares equal instead of
    resolving on noise. The loop runs over the window length, not the panel.

    Requiring the whole window to be finite reproduces `ta.sma`'s na-until-n.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    padded = np.vstack([np.full((n - 1, N), np.nan), a])
    out = np.full((T, N), np.nan)

    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        v = np.lib.stride_tricks.sliding_window_view(
            padded[lo:hi + n - 1], n, axis=0)               # (hi-lo, N, n)
        total = np.zeros(v.shape[:2])
        comp = np.zeros(v.shape[:2])
        for k in range(n):
            term = np.nan_to_num(v[..., k])
            moved = total + term
            # the lost low-order bits go to `comp`, whichever operand dominates
            comp += np.where(np.abs(total) >= np.abs(term),
                             (total - moved) + term, (term - moved) + total)
            total = moved
        out[lo:hi] = np.where(np.isfinite(v).all(axis=2),
                              (total + comp) / float(n), np.nan)

    return pd.DataFrame(out, index=x.index, columns=x.columns)


def _qm_extremes(H, L, w, chunk=128):
    """(base_high, since_peak, pull_low) over a w-bar window. Pine's tie=recent.

    Three pine series in one pass, because they share a window:

        baseHigh   = ta.highest(high, w)
        sincePk    = -ta.highestbars(high, w)          # 0 = the peak is this bar
        pullLow    = ta.lowest(low, math.max(sincePk, 1))

    The third is the awkward one — its window *length* depends on the second, so
    it is a per-row variable-length reduction. But it is always a suffix of the
    same w-bar window, so reversing the strided view once (index 0 = the current
    bar) turns it into a cumulative minimum indexed at `sincePk - 1`. That window
    spans [i - sincePk + 1, i], which correctly excludes the peak bar itself.

    `argmax` returns the *first* maximal index, and on the reversed view that is
    the most recent — TradingView's `highestbars` tie-break is undocumented, and
    most-recent is what `tools/qm_pine_ref.py` uses by default, so a diff against
    it is apples to apples. The choice is not cosmetic: ties on highs are common
    at round numbers, and the other rule moves both the age gate and the depth
    gate, in opposite directions, on exactly the bars that decide a signal.

    Chunked like `_wsum` and `_moments`; smaller chunk because the cumulative
    minimum materialises a second (chunk, N, w) array.
    """
    h = H.to_numpy(dtype=float)
    l = L.to_numpy(dtype=float)
    T, N = h.shape
    pad = np.full((w - 1, N), np.nan)
    ph = np.vstack([pad, h])
    pl = np.vstack([pad, l])

    base_high = np.full((T, N), np.nan)
    since_peak = np.zeros((T, N))
    pull_low = np.full((T, N), np.nan)

    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        vh = np.lib.stride_tricks.sliding_window_view(
            ph[lo:hi + w - 1], w, axis=0)[..., ::-1]     # (hi-lo, N, w), newest first
        vl = np.lib.stride_tricks.sliding_window_view(
            pl[lo:hi + w - 1], w, axis=0)[..., ::-1]
        # pine's ta.* are na until the window is full, and on the packed view a
        # short column's trailing slots are NaN, so this also kills dead region.
        full = np.isfinite(vh).all(axis=2)
        idx = np.argmax(np.nan_to_num(vh, nan=-np.inf), axis=2)
        cum_min = np.minimum.accumulate(np.nan_to_num(vl, nan=np.inf), axis=2)
        suffix = np.take_along_axis(
            cum_min, np.maximum(idx - 1, 0)[..., None], axis=2)[..., 0]

        base_high[lo:hi] = np.where(full, np.max(
            np.nan_to_num(vh, nan=-np.inf), axis=2), np.nan)
        since_peak[lo:hi] = np.where(full, idx, np.nan)
        pull_low[lo:hi] = np.where(full, suffix, np.nan)

    frame = lambda a: pd.DataFrame(a, index=H.index, columns=H.columns)
    return frame(base_high), frame(since_peak), frame(pull_low)


def _qm_register(bo_setup, entry_level, order_bars):
    """(armed, level, bars_left) — pine's boArmed / boOrderPx / boArmedBar.

    The pine is stateful here, but only nominally: all three variables are
    assigned in exactly one place (pine:217-220), unconditionally, on every bar
    `boSetup` holds. So the arm is a bounded-window function after all — at bar i
    it is live iff some j <= i had a setup with i - j <= order_bars, carrying
    `entry_level[j]` for the *largest* such j. That is why this model needs no
    `ScoringState` analogue.

    Two boundaries worth stating, both easy to get backwards:

     * Expiry is pine's `bar_index - boArmedBar > boOrderBars` (pine:221), so the
       order is live on order_bars + 1 bars and `bars_left == 0` is still
       fillable. It is not "cancelled with zero bars left".
     * Re-arming overwrites the level downward as happily as upward: `baseHigh`
       falls when the old peak slides out of the base window, and the pine keeps
       no memory of the higher level.
    """
    setup = bo_setup.to_numpy()
    level = entry_level.to_numpy(dtype=float)
    rows = np.arange(len(setup))[:, None]

    last = np.maximum.accumulate(np.where(setup, rows, -1), axis=0)
    age = rows - last
    armed = (last >= 0) & (age <= order_bars)
    live = np.take_along_axis(level, np.maximum(last, 0), axis=0)

    frame = lambda a: pd.DataFrame(a, index=bo_setup.index,
                                   columns=bo_setup.columns)
    return (frame(armed),
            frame(np.where(armed, live, np.nan)),
            frame(np.where(armed, order_bars - age, np.nan)))


class QullamaggieSwingModel:
    """The pine's long momentum breakout, emitted before it triggers.

    Port of Setup 1 from `app/pines/qullamaggie_momentum_swing.pine` (v15). Only
    that setup: no opening-range breakout, no short breakdown, no episodic pivot,
    no parabolic short, and none of the pine's trade management.

    Same client surface as `ManipulationModel` minus everything training-related,
    because a rule model has nothing to fit:

        model = QullamaggieSwingModel()
        candidates = model.signal(ctx.history(QullamaggieSwingModel.LOOKBACK))

    `signal` returns one row per *live resting order* — the point of the port. A
    caller reads `level` and parks a buy-stop there; `bars_left` says how long the
    pine would leave it sitting.

    Three things a consumer has to know:

     1. **Position gating is dropped.** The pine's `boSetup` (pine:171) carries
        `not inLong and not inShort`, so its arm state is a function of the
        position's whole lifetime — unbounded, and invisible to a windowed model.
        Everything here is computed as if flat. `bo_setup` is emitted per bar
        precisely so a caller that wants pine fidelity can run the three-line
        register from pine:217-223 against its own book. The `armed` / `level` /
        `bars_left` columns are a convenience that is exact only while nothing is
        held; they cannot be corrected by ANDing your own flat state onto them,
        because the timer's *origin* is position-gated too.
     2. **`stop` and `target` are planning numbers.** The pine computes both from
        the realised fill, `max(open, level)`, and optionally tightens the stop to
        the entry bar's low (pine:361-365). Neither is knowable while the order is
        still resting, so these are anchored to `level`. A gap through the level
        moves the real stop.
     3. **Volume never gates.** Pine v15 made the break-bar volume check
        information-only — `boFill` (pine:348) has no volume term and the tag is
        applied after the fill (pine:376-378). `vol_ratio` and `vol_meets` are
        reported and nothing is filtered on them. Note `tools/qm_pine_ref.py`
        still ANDs volume into its fill: that file targets v12, so diff against it
        with `--use-bo-vol false`.

    Unlike `ManipulationModel` this model carries no cross-tick state, so a caller
    may score any bar without having scored the ones before it.
    """

    # Deep enough for one row of the deepest series plus a full order life. At
    # defaults the binding constraint is avgVol50[1] (51 bars), not the 40-bar
    # base — and only because the informational volume ratio needs it:
    #   max(sma20=20, mom_len+1=25, base_max_len=40, avgVol50[1]=51) + 10 = 61
    # A maxed base (pine caps it at 80) needs 90. The headroom is slack: this is
    # a count of *dates*, while the register runs on traded bars, so a symbol with
    # missed sessions has fewer bars than the window has rows. `__init__` checks
    # the configured params actually fit rather than trusting this comment.
    LOOKBACK = 120

    def __init__(self, *, min_price=5.0, min_avg_vol=0.0, adr_len=20,
                 min_adr=0.1, mom_len=24, min_gain=0.5, require_mas=True,
                 require_ma_stack=False, base_max_len=40, min_base_bars=3,
                 max_depth=40.0, use_vol_dry=False, vol_dry_ratio=1.0,
                 entry_buffer_bps=5.0, order_bars=10, bo_vol_mult=1.3,
                 adr_stop_mult=1.0, target_rr=2.0):
        # Pine's inputs, same defaults, with one deviation: the entry buffer is
        # in basis points rather than `bufTicks * syminfo.mintick`. There is no
        # mintick in the engine and a tick is not scale-free across a 4-lira and
        # a 400-lira name; `qmmomentumswing.py` made the same substitution.
        self.min_price = float(min_price)
        self.min_avg_vol = float(min_avg_vol)
        self.adr_len = int(adr_len)
        self.min_adr = float(min_adr)
        self.mom_len = int(mom_len)
        self.min_gain = float(min_gain)
        self.require_mas = bool(require_mas)
        self.require_ma_stack = bool(require_ma_stack)
        self.base_max_len = int(base_max_len)
        self.min_base_bars = int(min_base_bars)
        self.max_depth = float(max_depth)
        self.use_vol_dry = bool(use_vol_dry)
        self.vol_dry_ratio = float(vol_dry_ratio)
        self.entry_buffer_bps = float(entry_buffer_bps)
        self.order_bars = int(order_bars)
        self.bo_vol_mult = float(bo_vol_mult)
        self.adr_stop_mult = float(adr_stop_mult)
        self.target_rr = float(target_rr)

        if self.min_base_bars < 2:
            # pine's minval (pine:67), and load-bearing: min_base_bars >= 2 with
            # the most-recent tie-break is what makes high[j] < entry_level[j] on
            # an arming bar, so an order can never arm and fill on the same bar.
            raise ValueError("min_base_bars must be at least 2")
        if self.depth() > self.LOOKBACK:
            raise ValueError(
                f"these params need {self.depth()} bars but LOOKBACK is "
                f"{self.LOOKBACK}; lower base_max_len/mom_len/adr_len/order_bars")

    def depth(self):
        """Bars of history one scored row needs, including a full order life."""
        series = [20, 51, self.adr_len, self.mom_len + 1, self.base_max_len]
        if self.require_ma_stack:
            series.append(50)
        if self.use_vol_dry:
            series.append(50)
        return max(series) + self.order_bars

    # -- panel --------------------------------------------------------------

    @staticmethod
    def _panel(source):
        """Bars -> {field: (dates, symbols)}. Accepts a window, frame or panel.

        Not `_panel`: that runs bist's `_drop_unusable`, and dropping rows here
        would shift both the base window and the order timer off the pine's.
        """
        if isinstance(source, dict):
            return source
        frame = source if isinstance(source, pd.DataFrame) else pd.DataFrame({
            "symbol": list(source.symbol),
            "timestamp": np.asarray(source.timestamp),
            **{f: np.asarray(getattr(source, f), dtype=float) for f in FIELDS},
        })
        return {f: frame.pivot(index="timestamp", columns="symbol", values=f)
                .sort_index() for f in FIELDS}

    # -- signal -------------------------------------------------------------

    def setups(self, window):
        """Per-bar, per-symbol frames for the whole window.

        Returns a dict of (dates x symbols) DataFrames:

            bo_setup     pine's boSetup, position gating removed
            entry_level  the level an arm on that bar would carry
            armed        is a resting order live on this bar
            level        that order's price
            bars_left    bars before it expires; 0 is still fillable
            stop         planning stop, anchored to `level` — see caveat 2
            target       planning target at `target_rr` R
            adr_pct      pine's adrPct, the stop's basis
            vol_ratio    volume / avgVol50[1], informational
            vol_meets    vol_ratio >= bo_vol_mult, informational

        Everything runs on the *packed* view, so a window covers a symbol's last
        N **traded** bars the way pine's `bar_index` counts chart bars. On a dated
        panel a symbol that missed three sessions would expire its order three
        calendar rows early.
        """
        panel = self._panel(window)
        index = panel["close"].index
        printed = panel["close"].notna()
        order = _pack_order(printed)
        O, H, L, C, V = (_pack(panel[f], order) for f in FIELDS)

        # ── gates (pine:117-142) ──
        sma10 = _qm_sma(C, 10)
        sma20 = _qm_sma(C, 20)
        adr_pct = _qm_sma(100.0 * (H / L - 1.0), self.adr_len)
        avg_vol20 = _qm_sma(V, 20)
        avg_vol50 = _qm_sma(V, 50)
        gain_pct = 100.0 * (C / C.shift(self.mom_len) - 1.0)

        liq_ok = (C >= self.min_price) & (avg_vol20 >= self.min_avg_vol)
        adr_ok = adr_pct >= self.min_adr
        # A comparison against NaN is False in pandas, which is pine's `na >= x`.
        ma_ok_up = ((C > sma20) & (sma10 > sma20)) if self.require_mas else True
        if self.require_ma_stack:
            sma50 = _qm_sma(C, 50)
            stack_up_ok = sma50.notna() & (sma10 > sma20) & (sma20 > sma50)
        else:
            stack_up_ok = True
        vol_dry_ok = ((_qm_sma(V, 5) < self.vol_dry_ratio * avg_vol50)
                      if self.use_vol_dry else True)
        universe_up = (liq_ok & adr_ok
                       & (gain_pct >= self.min_gain) & ma_ok_up & stack_up_ok)

        # ── base geometry and the armed level (pine:165-171) ──
        base_high, since_peak, pull_low = _qm_extremes(H, L, self.base_max_len)
        retrace_pct = 100.0 * (base_high - pull_low) / base_high
        flag_up_ok = ((since_peak >= self.min_base_bars)
                      & (retrace_pct <= self.max_depth) & vol_dry_ok)
        entry_level = base_high * (1.0 + self.entry_buffer_bps / 10_000.0)
        bo_setup = universe_up & flag_up_ok & base_high.notna()

        armed, level, bars_left = _qm_register(bo_setup, entry_level,
                                               self.order_bars)

        # The pine's stop is ADR-based off the realised fill and then floored at
        # 0.1% of it (pine:362-364); anchored to `level` here, because the fill is
        # unknown while the order rests. `use_lod_stop` cannot apply for the same
        # reason. Computed on the packed view like everything else — on the dated
        # panel a single missed session NaNs the whole ADR window.
        stop = np.minimum(level * (1.0 - self.adr_stop_mult * adr_pct / 100.0),
                          level * 0.999)
        target = level + self.target_rr * (level - stop)

        # ── informational volume, never a gate (pine:146, v15) ──
        prev = avg_vol50.shift(1)
        vol_ratio = (V / prev).where(prev > 0)
        vol_meets = (vol_ratio >= self.bo_vol_mult).fillna(False)

        out = {
            "bo_setup": bo_setup, "entry_level": entry_level.where(bo_setup),
            "armed": armed, "level": level, "bars_left": bars_left,
            "stop": stop, "target": target,
            "adr_pct": adr_pct, "vol_ratio": vol_ratio, "vol_meets": vol_meets,
        }
        return {k: _unpack(v.astype(float), order, index, printed)
                for k, v in out.items()}

    def signal(self, window):
        """Live resting orders on the window's last bar, one row each.

        Columns: symbol, setup, side, action, level, stop, target, bars_left,
        bo_setup, vol_ratio, vol_meets. Empty (with those columns) when nothing
        is armed, so a caller can iterate unconditionally.
        """
        frames = self.setups(window)
        last = {k: v.iloc[-1].to_numpy(dtype=float) for k, v in frames.items()}
        live = np.nan_to_num(last["armed"]) > 0.0

        def flag(key):
            return np.nan_to_num(last[key])[live] > 0.0

        return pd.DataFrame({
            "symbol": np.asarray(frames["armed"].columns)[live],
            "setup": "BO",
            "side": "long",
            "action": "arm",
            "level": last["level"][live],
            "stop": last["stop"][live],
            "target": last["target"][live],
            "bars_left": last["bars_left"][live],
            "bo_setup": flag("bo_setup"),
            "vol_ratio": last["vol_ratio"][live],
            "vol_meets": flag("vol_meets"),
        }, columns=["symbol", "setup", "side", "action", "level", "stop",
                    "target", "bars_left", "bo_setup", "vol_ratio",
                    "vol_meets"]).reset_index(drop=True)


class AlgoTradeStrategy(stonks.Strategy):
    """Long swings on the model's two up heads, exited on a trend break.

    Entry takes the day's top `top_pct` of the model's own cross-sectional
    ranking, ordered by the mean of the two up sigmas, and fills at the next open.

    That is a deliberate break from bist, which gates on absolute sigmas
    (ALPHA5 3.0 / ALPHA10 2.0). The heads are a ranker, so an absolute threshold
    discards the ranking and asks an unanswerable question whose meaning moves
    with every retrain; and set high enough to be selective it fires almost only
    on bars that closed at the price band, which cannot be bought. `ENTRY_TOP_PCT`
    carries the measurements.

    The exit is the one place this deliberately parts from bist. bist rests a -10%
    stop and a +30% target and holds until one leg fills; the stop is kept
    verbatim and the target is replaced by `exit_ma` — a held name leaves when its
    close finishes below its 20-bar simple moving average.

    Keeping the stop is not cosmetic. The label this model was fit on is gated at
    -10%: a name that gained 40% after first dipping 12% is labelled zero, so the
    stop is what makes the strategy trade the thing the model was taught to find.
    It is also the only exit that can act on a gap, since the moving-average rule
    is a bar-close decision that cannot fill until the next open. The +30% target
    is the half of the pair with nothing behind it — the label encodes a drawdown
    floor and no ceiling at all — and it is what capped the winners.

    What remains bist's:

     * The -10% stop, verbatim, for the reason above.
     * The down head is computed and plotted but **not** gated on. bist only ever
       displays it as an avoidance overlay, so a `dn` veto would be this port's
       invention rather than a port of anything.
     * The composite the ranking sorts on — the mean of the two up sigmas.

    What is not:

     * The percentile gate and `skip_limit_locked = 1`, above.
     * `max_positions` with sizing at equity/max_positions. bist commits 5% of
       available balance per name and caps nothing, which is survivable only
       because its gate fires ~92 times in 18 months; a percentile gate fires
       every session and would exhaust cash in a handful of entries.

    The model is pre-trained to a dated artifact and frozen, so the strategy
    refuses to trade any bar the fit was allowed to see — though it still *scores*
    them, because the carried state has to see every bar.

    Caveats worth carrying into the results:

     1. **Start the run at the beginning of the data, not near `train_end`.**
        `obv` and `days_since_past_extreme` accumulate from a symbol's first bar,
        so `ScoringState` seeds them from the first window that covers the whole
        history and advances them one bar at a time thereafter. The engine's
        `--start` truncates the feed at load time, so a late start silently
        reseeds both against a shorter history and scores bars on values the fit
        never saw. Nothing can detect this from inside the strategy — the feed
        simply begins where it begins.
     2. `signal` runs on every tick from the moment the window fills, whether or
        not a slot is free, because `ScoringState` has to see every bar in order.
        Building 72 features over a 300 x ~600 window is far and away the run's
        dominant cost, and there is no way to skip it.
     3. Entry size is `equity / max_positions`, further capped by free cash less
        `cash_buffer`. A market order sized on today's close and filled at
        tomorrow's open can still overdraw on a gap, and the broker rejects —
        never queues — an order it cannot fund at fill time.
     4. **The gate's shape decides how much of the result is real.** bist's
        absolute gate produced 92 signals over 2025-2026 and 77 of them closed at
        the +10% band, which the strategy would have bought at the next open:
        fills a real order book would not have given. That is why the gate is now
        a percentile and `skip_limit_locked` is on. The exposure is reduced, not
        eliminated — a top-slice name can still be locked, and it is simply
        skipped when it is.
     5. **The return is a handful of trades.** On the engine's own 2025-2026 run
        — 221 round trips, 34.8% winners, mean +5.1%, median -4.7% — the five
        best contribute about 150% of the summed P&L, so the remaining 216 lose
        money together. One penny stock at +577% is over half of it. The gate
        width barely moves this: it was 160% at top 0.5% and 148% at 2%. Any
        expectation drawn from the headline return is an expectation about
        catching one of those, and nothing in the walk-forward says they recur on
        schedule.
     6. Two limits the engine cannot model here: the broker fills **fractional
        quantities**, and starting cash is 1000 against names priced into the
        thousands, so fractional sizing is doing real work. Under whole-lot
        constraints much of this book is not buyable at this account size.
        Separately, a stop is not a floor — the worst round trip was -48%, filled
        through the -10% stop on a gap.
     7. The liquid universe comes from the artifact and is computed from training
        turnover only. bist takes the median over its whole frame, future
        included; see `_liquid_universe`.
     8. A moving-average exit fills one bar after it is decided, where the stop
        fills intrabar on the bar it is breached. The rule gives back more on a
        sharp reversal than the fixed target it replaced.
     9. An exiting name holds its place in `book` until the position is actually
        flat, so it cannot be re-entered on the bar its own sale is in flight.
        The sale settles at the next open, and funding an entry out of proceeds
        that have not arrived would have the broker reject it outright — it never
        queues an order it cannot fund.
    """

    artifact = "app/python/artifacts/algotrade"
    # Bound to the tunables block at the top of this module — change them there.
    top_pct = ENTRY_TOP_PCT
    stop_pct = STOP_PCT
    exit_ma = EXIT_MA
    max_positions = MAX_POSITIONS
    cash_buffer = CASH_BUFFER
    skip_limit_locked = SKIP_LIMIT_LOCKED

    params = {
        "top_pct": stonks.Param(
            "enter the day's top slice of the model's cross-sectional ranking",
            unit="%"),
        "stop_pct": stonks.Param("protective stop below the entry bar's close", unit="%"),
        "exit_ma": stonks.Param(
            "close below this simple moving average exits the position", unit="bars"),
        "max_positions": stonks.Param("names held at once"),
        "cash_buffer": stonks.Param(
            "fraction of cash held back from entries for fees and gaps"),
        "skip_limit_locked": stonks.Param(
            "1 to refuse names that closed at the +10% band; bist's backtest "
            "uses 0 and every one of its entries is such a name"),
    }

    indicators = {
        "up_h5": stonks.Indicator("predicted 5-day excess move, sigma", color="#4c9f70"),
        "up_h10": stonks.Indicator("predicted 10-day excess move, sigma", color="#3d7ea6"),
        "dn_h5": stonks.Indicator("predicted 5-day downside, sigma", color="#b5534a"),
    }

    def on_start(self, ctx):
        self.model = ManipulationModel.load(self.artifact)
        # symbols we believe we hold; a filled stop closes them behind our back
        self.book = set()
        # exit order placed, position not yet flat — kept so the rule does not
        # re-send a market sell on the bar between placing it and its fill
        self.exiting = set()
        # obv and days_since_past_extreme, carried bar to bar; see ScoringState
        self.state = ScoringState()
        # this tick's close-to-close return per symbol, for the limit-up gate
        self._ret = {}

    def on_tick(self, ctx):
        w = ctx.history(LOOKBACK)
        if len(w) == 0:
            return
        # `ScoringState` must see every bar in order, so scoring starts as soon as
        # the window is full and continues through the in-sample stretch. Skipping
        # the in-sample bars would leave obv and the extreme counter seeded at
        # train_end instead of at the symbol's first bar.
        if np.unique(w.timestamp).size < LOOKBACK:
            return

        self._ret = self._returns_this_tick(w)
        sig = self.model.signal(w, state=self.state)

        # The artifact saw every bar up to train_end during training; trading
        # them is not a backtest result.
        now = int(np.max(w.timestamp))
        if self.model.train_end is not None and now <= self.model.train_end:
            return

        # Positions the broker still reports: a filled stop or a settled exit
        # drops out here.
        self.book = {s for s in self.book if ctx.position(s) is not None}
        self.exiting &= self.book
        # Exits run before the entry block, which returns early on any bar where
        # nothing qualifies — and that is most bars.
        self._exit_below_ma(ctx, w)

        free = self.max_positions - len(self.book)
        if free <= 0:
            return
        picks = self._rank(sig).iloc[:free]
        if picks.empty:
            return

        latest = self._closes(w)
        # equity/max_positions reaches full deployment exactly at the cap; the
        # cash leg keeps a bar that wants several names at once from ordering
        # more than it can fund, which the broker would reject outright.
        target = ctx.equity() / self.max_positions
        budget = min(target, ctx.cash() * (1.0 - self.cash_buffer) / len(picks))
        # A budget far below the per-slot target means the cash is still tied up
        # in a position being sold — the moving-average exit places a market sell
        # that only settles at the next open, so on that bar the proceeds are not
        # there yet. Entering anyway buys a token quantity that occupies a slot
        # for the whole trade while contributing nothing; the first engine run
        # produced four such orders, one of them 0.0009 lira. Wait for the sale
        # instead. The threshold is deliberately not a knob: anything under a
        # tenth of a slot is noise on any account size.
        if budget <= 0.0 or not np.isfinite(budget) or budget < 0.1 * target:
            return

        for symbol in picks.index:
            close = latest.get(symbol)
            if close is None or not np.isfinite(close) or close <= 0.0:
                continue
            if self._limit_locked(symbol):
                continue
            quantity = budget / close
            if quantity <= 0.0 or not np.isfinite(quantity):
                continue
            entry = ctx.place_market_order(symbol=symbol, side=OrderSide.Buy,
                                           quantity=quantity)
            # Dormant until the entry fills, then eligible from its fill bar, so
            # the stop protects the entry bar itself. reduce_only keeps an
            # orphaned leg from opening a short, and the engine cancels the leg
            # once the position goes flat however it got there — including when
            # the moving-average exit is what closed it.
            stop = close * (1.0 - self.stop_pct / 100.0)
            ctx.place_stop_order(symbol=symbol, side=OrderSide.Sell,
                                 quantity=quantity, price=stop,
                                 parent=entry, reduce_only=True)
            self.book.add(symbol)
            self._print_entry(now, symbol, picks.loc[symbol], close, quantity,
                              stop)
            for head in ("up_h5", "up_h10", "dn_h5"):
                value = sig.at[symbol, head]
                if np.isfinite(value):
                    ctx.plot(head, symbol, float(value))

    def _print_entry(self, ts, symbol, row, close, quantity, stop):
        """One line per entry: the order, and the signal that produced it.

        The two up sigmas are the numbers the gates were applied to, so the line
        can be read back without re-running anything — the gate is a percentile,
        so the raw sigma is the only record of how strong a pick actually was.
        The percents are the model's `expected_excess_pct` over each head's
        horizon, and bist's caveat travels with them — rank on sigma, treat the
        percent as a magnitude check and not a calibrated forecast. `dn` rides
        along unused; bist only ever displays it.
        """
        composite = (row["up_h5"] + row["up_h10"]) / 2.0
        self._print(ts, symbol, (
            f"enter market @ {close:.4f} | qty {quantity:.6g} | "
            f"SL {stop:.4f} ({-self.stop_pct:+.2f}%) | exit < MA{self.exit_ma} | "
            f"h5 {row['up_h5']:.3f} ({row['up_h5_pct']:+.2f}%) "
            f"h10 {row['up_h10']:.3f} ({row['up_h10_pct']:+.2f}%) "
            f"dn {row['dn_h5']:.3f} | composite {composite:.3f}"))

    @staticmethod
    def _print(ts, symbol, msg):
        when = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")
        print(f"[{when} UTC] {symbol} {msg}", flush=True)

    def _exit_below_ma(self, ctx, w):
        """Sell held names that closed under their moving average.

        The average moves every bar, so this cannot be a resting order the way
        the stop is — it is re-decided each bar and sent as a market order, which
        fills at the next open. The name stays in `book` until the position is
        actually flat; see caveat 7.
        """
        pending = self.book - self.exiting
        if not pending:
            return
        for symbol, tail in self._tail_closes(w, pending, self.exit_ma).items():
            # A name with less than a full window has no average to break.
            if len(tail) < self.exit_ma or not np.isfinite(tail).all():
                continue
            if tail[-1] >= tail.mean():
                continue
            position = ctx.position(symbol)
            if position is None:
                continue
            ctx.place_market_order(symbol=symbol, side=OrderSide.Sell,
                                   quantity=abs(position.quantity),
                                   reduce_only=True)
            self.exiting.add(symbol)

    @staticmethod
    def _segments(w):
        """(starts, ends) — the row span of each symbol in the ragged window.

        Rows are contiguous per symbol and every symbol's slice ends at this
        tick, so the rows stamped `ts[-1]` are exactly the segment ends and the
        preceding boundary is the previous end plus one. `ends` is inclusive.
        """
        ts = np.asarray(w.timestamp)
        ends = np.flatnonzero(ts == ts[-1])
        starts = np.empty_like(ends)
        starts[0] = 0
        starts[1:] = ends[:-1] + 1
        return starts, ends

    @classmethod
    def _returns_this_tick(cls, w):
        """{symbol: close-to-previous-close return} for the symbols printing now.

        Read off the ragged window rather than remembered across ticks: the row
        before a segment end is that symbol's previous bar.
        """
        close = np.asarray(w.close, dtype=float)
        out = {}
        for start, end in zip(*cls._segments(w)):
            if end <= start:
                continue                       # a symbol's very first bar
            prev = close[end - 1]
            if prev > 0.0:
                out[w.symbol[end]] = close[end] / prev - 1.0
        return out

    @classmethod
    def _tail_closes(cls, w, symbols, count):
        """The trailing `count` closes of each of `symbols`, newest last.

        Slices the ragged window directly rather than pivoting it: the exit rule
        needs a short average for the held names only, and a full pivot per bar
        to serve a handful of columns is the expensive way to get it.

        The window holds only bars that printed, so the average is over a
        symbol's last `count` *traded* bars — the same convention the model's
        features use.
        """
        close = np.asarray(w.close, dtype=float)
        out = {}
        for start, end in zip(*cls._segments(w)):
            symbol = w.symbol[end]
            if symbol in symbols:
                out[symbol] = close[max(int(start), int(end) + 1 - count):int(end) + 1]
        return out

    def _limit_locked(self, symbol):
        """Did this name close at the +10% band, making tomorrow's fill fiction?

        Off by default, because bist's backtest overrides it off and this is meant
        to reproduce bist. Be clear about what that costs: of the 14 signals
        bist's gate produces over the 2025-2026 out-of-sample stretch on this
        feed, **all 14** closed at the +10% band on the signal bar. The strategy
        buys at the next open, so every one of them is an entry into a name that
        was limit-locked when the decision was made — a fill a real order book
        would not have given. Set `skip_limit_locked` to 1 to exclude them, and
        expect zero trades.

        That is not a quirk of this port. It follows from what the model is: a
        detector for names about to make an outsized move, whose strongest
        readings land on names that have already started making one.
        """
        if not self.skip_limit_locked:
            return False
        return self._ret.get(symbol, 0.0) >= LIMIT_HIT

    def _rank(self, sig):
        """The day's top `top_pct` of the scored universe, best composite first.

        Indexed by symbol, carrying each head's raw sigma and expected excess
        percent, so a pick can be read back without re-joining against `sig`.

        `composite` is bist's: the mean of the two up sigmas. Ties break on symbol
        so the ordering is deterministic. The down head rides along unused; see
        the class docstring.

        The slice is taken over **every symbol scored this bar**, before the held
        names are removed. Ranking after the removal would quietly widen the gate
        as the book fills — holding nine of ten names would promote the tenth-best
        remaining candidate into the top slice — so the cut has to be measured
        against the whole universe and the book applied afterwards.

        A symbol the model declined to score is NaN and drops out of the ranking
        rather than sorting to one end.
        """
        columns = ["up_h5", "up_h5_pct", "up_h10", "up_h10_pct", "dn_h5"]
        composite = ((sig["up_h5"] + sig["up_h10"]) / 2.0).dropna()
        if composite.empty:
            return sig.iloc[:0][columns]

        keep = max(1, int(len(composite) * self.top_pct / 100.0))
        order = sorted(composite.index, key=lambda s: (-composite[s], s))
        order = [s for s in order[:keep] if s not in self.book]
        return sig.loc[order, columns]

    @classmethod
    def _closes(cls, w):
        """This tick's close per symbol."""
        return {w.symbol[i]: float(w.close[i]) for i in cls._segments(w)[1]}


# ---------------------------------------------------------------------------
# Offline trainer. The strategy loads what this writes.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="app/data/bist_1d.parquet")
    parser.add_argument("--train-end", default="2024-12-31",
                        help="last bar the fit may see; the backtest starts after it")
    parser.add_argument("--out", default=AlgoTradeStrategy.artifact)
    parser.add_argument("--train-start", default=BACKTEST_TRAIN_START,
                        help="first bar the fit may see. Defaults to bist's "
                             "backtest setting (full history), NOT its screen's "
                             "2024-01-01 — see BACKTEST_TRAIN_START")
    parser.add_argument("--rounds", type=int, default=XGB_PARAMS["n_estimators"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = pd.read_parquet(args.data)
    log.info("%s: %d rows, %d symbols, %s to %s", args.data, len(frame),
             frame["symbol"].nunique(), frame["timestamp"].min(),
             frame["timestamp"].max())

    params = {**XGB_PARAMS, "n_estimators": args.rounds}
    model = ManipulationModel(params=params, train_start=args.train_start).train(
        frame, train_end=args.train_end)
    model.save(args.out)
    log.info("wrote %s (train_end=%s)", args.out, args.train_end)


if __name__ == "__main__":
    main()
