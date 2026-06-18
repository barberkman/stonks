#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <optional>
#include <ostream>
#include <span>

#include "stonks/core/types.h"

// External backtest reporter. The engine keeps the run history (trades, orders,
// equity curve) and exposes it via accessors; this turns that plain data into
// derived metrics and prints them. Pure free functions over plain data — no
// engine/template dependency — so the metrics are directly unit-testable.
namespace stonks::app {

// Everything the reporter needs, assembled by the caller after a run. The spans
// view storage owned by the engine/broker and must outlive the call.
struct ReportInput
{
    core::Balance starting_cash;
    std::size_t bars_processed;
    std::span<const core::Trade> trades;
    std::span<const core::Order> orders;
    std::span<const core::EquityPoint> equity_curve;
    core::Balance ending_cash;
    core::Balance ending_equity;
    std::chrono::nanoseconds elapsed;
};

struct ReportMetrics
{
    std::size_t bars_processed;
    std::optional<core::Timestamp> first_ts;
    std::optional<core::Timestamp> last_ts;
    std::size_t trade_count;
    std::size_t orders_placed;
    core::Balance notional;
    core::Balance starting_cash;
    core::Balance ending_cash;
    core::Balance ending_equity;
    std::optional<double> return_pct;   // nullopt when starting cash is 0
    double max_drawdown_pct;
    std::chrono::nanoseconds elapsed;
};

inline ReportMetrics compute_metrics(const ReportInput& in)
{
    core::Balance notional = 0.0;
    for (const auto& t : in.trades) { notional += t.quantity * t.price; }

    // Peak-to-trough of the mark-to-market equity curve.
    double max_drawdown_pct = 0.0;
    core::Balance peak = in.starting_cash;
    for (const auto& p : in.equity_curve) {
        if (p.equity > peak) { peak = p.equity; }
        if (peak > 0.0) {
            const double dd = (peak - p.equity) / peak * 100.0;
            if (dd > max_drawdown_pct) { max_drawdown_pct = dd; }
        }
    }

    std::optional<core::Timestamp> first_ts;
    std::optional<core::Timestamp> last_ts;
    if (!in.equity_curve.empty()) {
        first_ts = in.equity_curve.front().timestamp;
        last_ts = in.equity_curve.back().timestamp;
    }

    std::optional<double> return_pct;
    if (in.starting_cash != 0.0) {
        return_pct = (in.ending_equity - in.starting_cash) / in.starting_cash * 100.0;
    }

    return ReportMetrics{
        in.bars_processed,
        first_ts,
        last_ts,
        in.trades.size(),
        in.orders.size(),
        notional,
        in.starting_cash,
        in.ending_cash,
        in.ending_equity,
        return_pct,
        max_drawdown_pct,
        in.elapsed,
    };
}

inline void print_report(std::ostream& os, const ReportMetrics& m)
{
    std::ios old_state{ nullptr };
    old_state.copyfmt(os);
    os << std::fixed << std::setprecision(2);

    os << "=== Backtest report ===\n";
    os << "Bars processed:  " << m.bars_processed << '\n';
    if (m.first_ts) {
        os << "Time range:      " << *m.first_ts << " -> " << *m.last_ts << '\n';
    }
    os << "Orders placed:   " << m.orders_placed << '\n';
    os << "Trades:          " << m.trade_count << '\n';
    if (m.trade_count != 0) {
        os << "Notional traded: " << m.notional << '\n';
    }
    os << "Starting cash:   " << m.starting_cash << '\n';
    os << "Ending cash:     " << m.ending_cash << '\n';
    os << "Ending equity:   " << m.ending_equity << '\n';
    if (m.return_pct) {
        os << "Return:          " << *m.return_pct << " %\n";
    }
    os << "Max drawdown:    " << m.max_drawdown_pct << " %\n";

    const double total_ms = std::chrono::duration<double, std::milli>{ m.elapsed }.count();
    os << "Elapsed:         " << total_ms << " ms\n";
    if (m.bars_processed > 0) {
        const double per_bar_us =
            std::chrono::duration<double, std::micro>{ m.elapsed }.count()
            / static_cast<double>(m.bars_processed);
        os << "  per bar:       " << per_bar_us << " us\n";
    }

    os.copyfmt(old_state);
}

} // namespace stonks::app
