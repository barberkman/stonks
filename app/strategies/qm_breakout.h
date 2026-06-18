#pragma once

#include "strategies/qm_common.h"

// Setup 1 — momentum breakout (long).
//
// After a strong run a stock digests in a tight base; we buy the break above the
// base's pivot high. Mirrors the Pine "Buy BO" resting buy-stop, adapted to a
// daily close-confirmed break that fills at the next open. Long, R-managed
// (stop / partial at 2R or N bars / breakeven / trailing-MA exit), with a
// same-bar bail if the fill bar immediately closes back below the stop.
struct QMBreakoutStrategy : qm::QMBase<QMBreakoutStrategy>
{
    static constexpr qm::Dir DIR = qm::Dir::Long;
    static constexpr qm::Mgmt MGMT = qm::Mgmt::R;
    static constexpr bool SAME_BAR_BAIL = true;

    std::optional<qm::EntryPlan> signal(const stonks::core::SeriesView& bars)
    {
        // Gate on the momentum universe (liquid, volatile, trending up), then look
        // for a valid base whose pivot the current bar has just broken.
        if (!qm::universe_long(bars, p)) return std::nullopt;
        return qm::flag_base_long(bars, p);
    }
};
