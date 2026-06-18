#pragma once

#include "strategies/qm_common.h"

// Setup 2 — episodic pivot (gap bar, long).
//
// A liquid stock gaps up on volume with a strong close — a news/earnings "episodic
// pivot". Pine enters at the close of the gap bar; here we enter at the next open.
// R-managed; the stop tightens to the gap bar's low, and the epWithin gate keeps
// that close-to-low risk inside the ADR stop distance. No same-bar bail (Pine's EP
// entry block has none).
struct QMEpisodicPivotStrategy : qm::QMBase<QMEpisodicPivotStrategy>
{
    static constexpr qm::Dir DIR = qm::Dir::Long;
    static constexpr qm::Mgmt MGMT = qm::Mgmt::R;
    static constexpr bool SAME_BAR_BAIL = false;

    std::optional<qm::EntryPlan> signal(const stonks::core::SeriesView& bars)
    {
        // EP only requires liquidity (not the full trend universe).
        if (!qm::liq_ok(bars, p)) return std::nullopt;

        const double o = bars.open.back();
        const double h = bars.high.back();
        const double l = bars.low.back();
        const double c = bars.close.back();
        const double prev_close = bars.close[bars.size() - 2];
        if (prev_close <= 0.0) return std::nullopt;

        // Gap = this bar's open vs the previous close.
        if (100.0 * (o / prev_close - 1.0) < p.ep_min_gap) return std::nullopt;

        // Volume vs the prior bar's 50-bar average.
        const auto prev_avg50 = qm::sma(bars.volume.first(bars.volume.size() - 1), 50);
        if (!prev_avg50 || bars.volume.back() < p.ep_vol_mult * *prev_avg50) return std::nullopt;

        // Strong close: up bar finishing in the upper half of its range.
        if (p.ep_strong_close && !(c > o && c >= (h + l) / 2.0)) return std::nullopt;

        // epWithin: the close-to-low risk must fit inside the ADR stop distance.
        const auto adr = qm::adr_pct(bars.high, bars.low, p.adr_len);
        const double risk_ps = c - l;
        if (!adr || !(risk_ps > 0.0 && risk_ps <= p.adr_stop_mult * *adr / 100.0 * c))
            return std::nullopt;

        return qm::EntryPlan{ .entry_ref = c };
    }
};
