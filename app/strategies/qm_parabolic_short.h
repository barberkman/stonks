#pragma once

#include <algorithm>

#include "strategies/qm_common.h"

// Setup 3 — parabolic short (first red bar).
//
// After a steep parabolic run-up of consecutive up-closes, short the first red bar
// (the reversal). Pine enters at that bar's close; here we enter at the next open.
// Distinct (Para) management: cover on a stop above the recent highs, on any green
// close, or after a max holding period — no partial, no trailing MA.
struct QMParabolicShortStrategy : qm::QMBase<QMParabolicShortStrategy>
{
    static constexpr qm::Dir DIR = qm::Dir::Short;
    static constexpr qm::Mgmt MGMT = qm::Mgmt::Para;
    static constexpr bool SAME_BAR_BAIL = false;

    std::optional<qm::EntryPlan> signal(const stonks::core::SeriesView& bars)
    {
        if (!qm::liq_ok(bars, p)) return std::nullopt;

        const auto& close = bars.close;
        const int need = std::max({ p.ps_lookback, p.ps_streak + 1, p.ps_stop_lb }) + 1;
        if (std::ssize(close) < need) return std::nullopt;

        // Parabolic run-up: range expansion over the lookback window.
        const auto hi = qm::highest(bars.high, p.ps_lookback);
        const auto lo = qm::lowest(bars.low, p.ps_lookback);
        if (!hi || !lo || *lo <= 0.0) return std::nullopt;
        if (100.0 * (*hi / *lo - 1.0) < p.ps_min_gain) return std::nullopt;

        // Count consecutive up-closes ending at the prior bar (Pine's upStk[1]).
        int up_streak = 0;
        for (std::size_t i = close.size() - 2; i >= 1 && close[i] > close[i - 1]; --i) ++up_streak;

        // The current bar must be the first red one after a long-enough run.
        if (!(close.back() < close[close.size() - 2] && up_streak >= p.ps_streak))
            return std::nullopt;

        const auto ps_stop = qm::highest(bars.high, p.ps_stop_lb);
        if (!ps_stop) return std::nullopt;
        return qm::EntryPlan{
            .entry_ref = close.back(),
            .explicit_stop = *ps_stop + p.buf_ticks * qm::MINTICK,
            .use_target = false,
        };
    }
};
