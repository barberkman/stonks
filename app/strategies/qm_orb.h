#pragma once

#include <algorithm>
#include <cstdint>

#include "strategies/qm_common.h"

// Setup 1b — opening range breakout (intraday, long).
//
// Buy the break above the high of the first N bars of a session. A session is one
// calendar day (UTC); the opening range is the high of its first `orb_bars` bars.
//
// NOTE: this is intraday-only. On daily data every bar is its own calendar day, so
// a session never has more than one bar and bars_into_session never reaches
// orb_bars — the strategy emits nothing by construction. It fires correctly once
// intraday bars are fed (e.g. the 5-minute dataset). Long, R-managed, same-bar bail.
struct QMORBStrategy : qm::QMBase<QMORBStrategy>
{
    static constexpr qm::Dir DIR = qm::Dir::Long;
    static constexpr qm::Mgmt MGMT = qm::Mgmt::R;
    static constexpr bool SAME_BAR_BAIL = true;

    static constexpr std::int64_t MS_PER_DAY = 86'400'000;

    std::optional<qm::EntryPlan> signal(const stonks::core::SeriesView& bars)
    {
        if (!qm::universe_long(bars, p)) return std::nullopt;

        const auto& ts = bars.timestamp;
        const std::int64_t cur_day = ts.back() / MS_PER_DAY;

        // Walk back over the contiguous trailing bars sharing the current calendar
        // day — that run is the current session (bars are time-ordered).
        std::size_t start = ts.size() - 1;
        while (start > 0 && ts[start - 1] / MS_PER_DAY == cur_day) --start;

        const int bars_into_session = static_cast<int>(ts.size() - 1 - start);
        if (bars_into_session < p.orb_bars) return std::nullopt;  // still inside the opening range

        // Opening range high = highest high of the session's first orb_bars bars.
        double or_high = bars.high[start];
        for (int k = 0; k < p.orb_bars; ++k) or_high = std::max(or_high, bars.high[start + k]);

        const double pivot = or_high + p.buf_ticks * qm::MINTICK;
        const bool broke = p.wait_close ? bars.close.back() >= pivot : bars.high.back() >= pivot;
        if (!broke) return std::nullopt;
        return qm::EntryPlan{ .entry_ref = pivot };
    }
};
