#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <optional>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "stonks/core/log.h"
#include "stonks/core/types.h"

// External backtest reporter. The engine keeps the run history (trades, orders,
// equity curve) and exposes it via accessors; this turns that plain data into
// derived metrics and prints them. Pure free functions over plain data — no
// engine/template dependency — so the metrics are directly unit-testable.
namespace stonks::app {

// Everything the reporter needs, assembled by the caller after a run. Owns its
// trade/order/equity-curve data, so it is safe to store and outlive the inputs.
struct ReportInput
{
    core::Balance starting_cash;
    std::size_t bars_processed;
    std::vector<core::Trade> trades;
    std::vector<core::Order> orders;
    std::vector<core::EquityPoint> equity_curve;
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
    std::size_t closed_trades;            // round-trip positions reconstructed from fills
    std::size_t winning_trades;           // closed trades with realized P&L > 0
    std::optional<double> win_rate_pct;   // nullopt when nothing closed
    core::Balance starting_cash;
    core::Balance ending_cash;
    core::Balance ending_equity;
    std::optional<double> return_pct;   // nullopt when starting cash is 0
    double max_drawdown_pct;
    std::chrono::nanoseconds elapsed;
};

inline ReportMetrics compute_metrics(const ReportInput& in)
{
    STONKS_LOG("report", "compute_metrics: bars={} trades={} orders={} equity_points={} starting_cash={:.4f}",
        in.bars_processed, in.trades.size(), in.orders.size(),
        in.equity_curve.size(), in.starting_cash);

    core::Balance notional = 0.0;
    for (const auto& t : in.trades) { notional += t.quantity * t.price; }

    // Win rate over *closed* round-trip positions reconstructed from the fill
    // stream. Per symbol we carry a signed position (+long/-short), its
    // average entry, and the realized P&L accumulated over the current open
    // cycle. A fill in the same direction scales in (updates the average); an
    // opposing fill realizes P&L on the closed quantity (long: exit-entry,
    // short: entry-exit). The cycle is counted as one closed trade when the
    // position returns to flat (a win when its realized P&L > 0), and any
    // overshoot opens a fresh position. Positions still open at run end
    // contribute no closed trade.
    struct OpenPosition
    {
        core::Quantity qty = 0.0;
        core::Price avg_price = 0.0;
        core::Balance realized = 0.0;
    };
    std::unordered_map<core::Symbol, OpenPosition> positions;
    std::size_t closed_trades = 0;
    std::size_t winning_trades = 0;
    for (const auto& t : in.trades) {
        const double fill = (t.side == core::OrderSide::Buy ? t.quantity : -t.quantity);
        auto& pos = positions[t.symbol];

        if (pos.qty == 0.0) {
            pos.qty = fill;
            pos.avg_price = t.price;
            continue;
        }

        if ((pos.qty > 0.0) == (fill > 0.0)) {
            // Scaling in: blend the average entry.
            const double prev = std::abs(pos.qty);
            const double add = std::abs(fill);
            pos.avg_price = (pos.avg_price * prev + t.price * add) / (prev + add);
            pos.qty += fill;
            continue;
        }

        // Opposing fill: close against the open position.
        const double closing = std::min(std::abs(fill), std::abs(pos.qty));
        pos.realized += (pos.qty > 0.0 ? (t.price - pos.avg_price) : (pos.avg_price - t.price)) * closing;
        pos.qty += (pos.qty > 0.0 ? -closing : closing);

        if (pos.qty == 0.0) {
            ++closed_trades;
            if (pos.realized > 0.0) { ++winning_trades; }
            pos.realized = 0.0;
            pos.avg_price = 0.0;

            const double leftover = std::abs(fill) - closing;
            if (leftover > 0.0) {   // fill flips the position past flat
                pos.qty = (fill > 0.0 ? leftover : -leftover);
                pos.avg_price = t.price;
            }
        }
    }

    std::optional<double> win_rate_pct;
    if (closed_trades != 0) {
        win_rate_pct = static_cast<double>(winning_trades) / static_cast<double>(closed_trades) * 100.0;
    } else {
        STONKS_LOG("report", "no closed round trips -> win rate suppressed");
    }

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
    } else {
        STONKS_LOG("report", "equity_curve empty -> time range suppressed");
    }

    std::optional<double> return_pct;
    if (in.starting_cash != 0.0) {
        return_pct = (in.ending_equity - in.starting_cash) / in.starting_cash * 100.0;
    } else {
        STONKS_LOG("report", "starting_cash==0 -> return % suppressed");
    }

    STONKS_LOG("report",
        "metrics: notional={:.4f} max_dd={:.4f}% return={:.4f}% win_rate={:.4f}% ({}/{}) ending_cash={:.4f} ending_equity={:.4f}",
        notional, max_drawdown_pct, return_pct.value_or(0.0), win_rate_pct.value_or(0.0),
        winning_trades, closed_trades, in.ending_cash, in.ending_equity);

    return ReportMetrics{
        in.bars_processed,
        first_ts,
        last_ts,
        in.trades.size(),
        in.orders.size(),
        notional,
        closed_trades,
        winning_trades,
        win_rate_pct,
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
    os << "Notional traded: " << m.notional << '\n';
    os << "Win rate:        " << m.win_rate_pct.value_or(0.0) << " % ("
       << m.winning_trades << "/" << m.closed_trades << ")\n";
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
