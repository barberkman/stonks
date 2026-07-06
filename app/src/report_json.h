#pragma once

#include <chrono>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "report.h"
#include "stonks/core/types.h"

// JSON serialization for the backtest report. Mirrors the text reporter in
// report.h but emits the derived metrics plus the full raw materials (every
// trade, order, and equity-curve point) so a run can be archived and consumed
// programmatically. The JSON dependency is confined to this header and the app
// target — stonks_core stays free of it — so the to_json overloads for core
// types are declared here (reopening stonks::core for ADL).
namespace stonks::core {

inline const char* to_string(OrderSide s)
{
    return s == OrderSide::Buy ? "Buy" : "Sell";
}

inline const char* to_string(OrderType t)
{
    switch (t) {
        case OrderType::Market: return "Market";
        case OrderType::Limit: return "Limit";
        case OrderType::Stop: return "Stop";
    }
    return "Unknown";
}

inline const char* to_string(OrderStatus s)
{
    switch (s) {
        case OrderStatus::Open: return "Open";
        case OrderStatus::Filled: return "Filled";
        case OrderStatus::Rejected: return "Rejected";
        case OrderStatus::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

inline const char* to_string(TimeInForce)
{
    return "GTC";
}

// ISO-8601 string, matching the operator<< used by the text report.
inline std::string to_iso8601(Timestamp ts)
{
    std::ostringstream os;
    os << ts;
    return os.str();
}

inline void to_json(nlohmann::json& j, const Trade& t)
{
    j = nlohmann::json{
        { "id", t.id },
        { "order_id", t.order_id },
        { "timestamp", to_iso8601(t.timestamp) },
        { "symbol", t.symbol },
        { "side", to_string(t.side) },
        { "quantity", t.quantity },
        { "price", t.price },
        { "liquidation", t.liquidation },
        { "fee", t.fee },
    };
}

inline void to_json(nlohmann::json& j, const Order& o)
{
    j = nlohmann::json{
        { "id", o.id },
        { "parent_id", o.parent_id ? nlohmann::json(*o.parent_id) : nlohmann::json(nullptr) },
        { "timestamp", to_iso8601(o.timestamp) },
        { "symbol", o.symbol },
        { "side", to_string(o.side) },
        { "type", to_string(o.type) },
        { "status", to_string(o.status) },
        { "price", o.price ? nlohmann::json(*o.price) : nlohmann::json(nullptr) },
        { "quantity", o.quantity },
        { "time_in_force", to_string(o.time_in_force) },
        { "leverage", o.leverage },
    };
}

inline void to_json(nlohmann::json& j, const EquityPoint& p)
{
    j = nlohmann::json{
        { "timestamp", to_iso8601(p.timestamp) },
        { "equity", p.equity },
    };
}

} // namespace stonks::core

namespace stonks::broker {

// The run's broker knobs, so an archived report is reproducible/auditable on
// its own (an external verifier reads these instead of guessing). Lives in
// stonks::broker so nlohmann finds it via ADL.
inline void to_json(nlohmann::json& j, const BrokerConfig& c)
{
    j = nlohmann::json{
        { "fill_policy", c.fill_policy == IntrabarFillPolicy::Conservative
                             ? "Conservative" : "Optimistic" },
        { "maker_fee_bps", c.maker_fee_bps },
        { "taker_fee_bps", c.taker_fee_bps },
        { "fee_per_fill", c.fee_per_fill },
        { "maintenance_margin_rate", c.maintenance_margin_rate },
        { "isolated_loss_cap", c.isolated_loss_cap },
        { "flat_epsilon", c.flat_epsilon },
        { "min_equity", c.min_equity },
        { "min_notional", c.min_notional },
    };
}

} // namespace stonks::broker

namespace stonks::app {

inline void to_json(nlohmann::json& j, const ReportMetrics& m)
{
    const double elapsed_ms = std::chrono::duration<double, std::milli>{ m.elapsed }.count();
    j = nlohmann::json{
        { "bars_processed", m.bars_processed },
        { "first_ts", m.first_ts ? nlohmann::json(core::to_iso8601(*m.first_ts)) : nlohmann::json(nullptr) },
        { "last_ts", m.last_ts ? nlohmann::json(core::to_iso8601(*m.last_ts)) : nlohmann::json(nullptr) },
        { "trade_count", m.trade_count },
        { "orders_placed", m.orders_placed },
        { "notional", m.notional },
        { "total_fees", m.total_fees },
        { "closed_trades", m.closed_trades },
        { "winning_trades", m.winning_trades },
        { "win_rate_pct", m.win_rate_pct ? nlohmann::json(*m.win_rate_pct) : nlohmann::json(nullptr) },
        { "liquidations", m.liquidations },
        { "starting_cash", m.starting_cash },
        { "ending_cash", m.ending_cash },
        { "ending_equity", m.ending_equity },
        { "return_pct", m.return_pct ? nlohmann::json(*m.return_pct) : nlohmann::json(nullptr) },
        { "max_drawdown_pct", m.max_drawdown_pct },
        { "elapsed_ms", elapsed_ms },
    };
}

// Writes the full report — derived metrics plus the raw materials — as JSON.
inline void write_report_json(std::ostream& os, const ReportInput& in, const ReportMetrics& m)
{
    const nlohmann::json j{
        { "metrics", m },
        { "config", in.config },
        { "trades", in.trades },
        { "orders", in.orders },
        { "equity_curve", in.equity_curve },
    };
    os << j.dump(2) << '\n';
}

// Builds a filesystem-portable, per-run report path of the form
// "app/reports/report-YYYYMMDD-HHMMSS.json" from the current wall-clock time
// (no colons). The path is relative to the working directory (the project root
// when the app is run as documented); the caller creates the directory.
inline std::string timestamped_report_path()
{
    using namespace std::chrono;
    const auto now = floor<seconds>(system_clock::now());
    const auto day_point = floor<days>(now);
    const year_month_day ymd{ day_point };
    const hh_mm_ss<seconds> tod{ now - day_point };

    char buf[40];
    std::snprintf(buf, sizeof(buf),
        "report-%04d%02u%02u-%02d%02d%02d.json",
        static_cast<int>(ymd.year()),
        static_cast<unsigned>(ymd.month()),
        static_cast<unsigned>(ymd.day()),
        static_cast<int>(tod.hours().count()),
        static_cast<int>(tod.minutes().count()),
        static_cast<int>(tod.seconds().count()));
    return std::string{ "app/reports/" } + buf;
}

} // namespace stonks::app
