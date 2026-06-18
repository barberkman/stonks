#pragma once

#include "strategies/qm_common.h"

// Setup 1c — short breakout (breakdown).
//
// Mirror of the long breakout: in a downtrend a stock bounces into a base, and we
// short the break below the base's pivot low. Mirrors the Pine "Short BO" resting
// sell-stop, adapted to a daily close-confirmed break that fills at the next open.
// Short, with the same R-based management as longs (cover on stop / partial / BE /
// trailing-MA), and a same-bar bail if the fill bar snaps back above the stop.
struct QMShortBreakoutStrategy : qm::QMBase<QMShortBreakoutStrategy>
{
    static constexpr qm::Dir DIR = qm::Dir::Short;
    static constexpr qm::Mgmt MGMT = qm::Mgmt::R;
    static constexpr bool SAME_BAR_BAIL = true;

    std::optional<qm::EntryPlan> signal(const stonks::core::SeriesView& bars)
    {
        if (!qm::universe_short(bars, p)) return std::nullopt;
        return qm::flag_base_short(bars, p);
    }
};
