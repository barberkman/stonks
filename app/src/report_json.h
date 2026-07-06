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
        { "reduce_only", o.reduce_only },
    };
}

inline void to_json(nlohmann::json& j, const EquityPoint& p)
{
    j = nlohmann::json{
        { "timestamp", to_iso8601(p.timestamp) },
        { "equity", p.equity },
    };
}

// ── Inverse parsers, for restoring archived reports in a later session ───────

inline OrderSide side_from_string(const std::string& s)
{
    return s == "Buy" ? OrderSide::Buy : OrderSide::Sell;
}

inline OrderType type_from_string(const std::string& s)
{
    if (s == "Market") { return OrderType::Market; }
    if (s == "Stop") { return OrderType::Stop; }
    return OrderType::Limit;
}

inline OrderStatus status_from_string(const std::string& s)
{
    if (s == "Open") { return OrderStatus::Open; }
    if (s == "Filled") { return OrderStatus::Filled; }
    if (s == "Rejected") { return OrderStatus::Rejected; }
    return OrderStatus::Cancelled;
}

// Exact inverse of to_iso8601 / operator<< ("%04d-%02u-%02uT%02d:%02d:%02d.%03dZ").
inline Timestamp from_iso8601(const std::string& s)
{
    int y{}, mo{}, d{}, h{}, mi{}, sec{}, ms{};
    std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d.%dZ", &y, &mo, &d, &h, &mi, &sec, &ms);
    using namespace std::chrono;
    const sys_days day{ year{ y } / mo / d };
    return Timestamp{ time_point_cast<Timestamp::duration>(day)
                      + hours{ h } + minutes{ mi } + seconds{ sec } + milliseconds{ ms } };
}

inline void from_json(const nlohmann::json& j, Trade& t)
{
    t = Trade{
        j.at("id").get<TradeID>(),
        j.at("order_id").get<OrderID>(),
        from_iso8601(j.at("timestamp").get<std::string>()),
        j.at("symbol").get<std::string>(),
        side_from_string(j.at("side").get<std::string>()),
        j.at("quantity").get<double>(),
        j.at("price").get<double>(),
        j.value("liquidation", false),
        j.value("fee", 0.0),   // absent in pre-fee archives
    };
}

inline void from_json(const nlohmann::json& j, Order& o)
{
    o = Order{
        j.at("id").get<OrderID>(),
        j.at("parent_id").is_null() ? std::nullopt
                                    : std::optional<OrderID>{ j.at("parent_id").get<OrderID>() },
        from_iso8601(j.at("timestamp").get<std::string>()),
        j.at("symbol").get<std::string>(),
        side_from_string(j.at("side").get<std::string>()),
        type_from_string(j.at("type").get<std::string>()),
        status_from_string(j.at("status").get<std::string>()),
        j.at("price").is_null() ? std::nullopt
                                : std::optional<Price>{ j.at("price").get<double>() },
        j.at("quantity").get<double>(),
        TimeInForce::GTC,
        j.value("leverage", 1.0),
        j.value("reduce_only", false),   // absent in pre-P6 archives
    };
}

inline void from_json(const nlohmann::json& j, EquityPoint& p)
{
    p = EquityPoint{
        from_iso8601(j.at("timestamp").get<std::string>()),
        j.at("equity").get<double>(),
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

inline void from_json(const nlohmann::json& j, BrokerConfig& c)
{
    c = BrokerConfig{};
    c.fill_policy = j.value("fill_policy", std::string{ "Conservative" }) == "Optimistic"
        ? IntrabarFillPolicy::Optimistic : IntrabarFillPolicy::Conservative;
    c.maker_fee_bps = j.value("maker_fee_bps", 0.0);
    c.taker_fee_bps = j.value("taker_fee_bps", 0.0);
    c.fee_per_fill = j.value("fee_per_fill", 0.0);
    c.maintenance_margin_rate = j.value("maintenance_margin_rate", 0.0);
    c.isolated_loss_cap = j.value("isolated_loss_cap", false);
    c.flat_epsilon = j.value("flat_epsilon", 1e-9);
    c.min_equity = j.value("min_equity", 0.0);
    c.min_notional = j.value("min_notional", 0.0);
}

} // namespace stonks::broker

namespace stonks::app {

// The strategy the run executed and its effective parameter values.
inline void to_json(nlohmann::json& j, const StrategyRunInfo& s)
{
    j = nlohmann::json{
        { "module", s.module },
        { "class", s.cls },
        { "params", s.params },   // std::map serializes natively as an object
    };
}

inline void from_json(const nlohmann::json& j, StrategyRunInfo& s)
{
    s = StrategyRunInfo{
        j.value("module", std::string{}),
        j.value("class", std::string{}),
        j.value("params", std::map<std::string, double>{}),
    };
}

// The data provenance that lets a later session rebuild the candle drill-down.
inline void to_json(nlohmann::json& j, const RunMeta& r)
{
    j = nlohmann::json{
        { "display", r.display },
        { "data_file", r.data_file },
        { "data_key", r.data_key },
        { "start", r.start },
        { "end", r.end },
        { "symbols", r.symbols },
    };
}

inline void from_json(const nlohmann::json& j, RunMeta& r)
{
    r = RunMeta{
        j.value("display", std::string{}),
        j.value("data_file", std::string{}),
        j.value("data_key", std::string{}),
        j.value("start", std::string{}),
        j.value("end", std::string{}),
        j.value("symbols", std::vector<std::string>{}),
    };
}

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
        { "strategy", in.strategy },
        { "run", in.run },
        { "trades", in.trades },
        { "orders", in.orders },
        { "equity_curve", in.equity_curve },
    };
    os << j.dump(2) << '\n';
}

// Rebuild a ReportInput from an archived report, tolerating older schema
// versions (missing fee/reduce_only/strategy/run keys default sensibly).
// Metrics-derived scalars come from the "metrics" block; everything else is
// the raw material the report already carries in full.
inline ReportInput report_input_from_json(const nlohmann::json& j)
{
    ReportInput in{};
    const auto& m = j.at("metrics");
    in.starting_cash = m.at("starting_cash").get<double>();
    in.bars_processed = m.value("bars_processed", std::size_t{ 0 });
    in.ending_cash = m.at("ending_cash").get<double>();
    in.ending_equity = m.value("ending_equity", in.ending_cash);
    in.elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double, std::milli>{ m.value("elapsed_ms", 0.0) });
    in.trades = j.value("trades", std::vector<core::Trade>{});
    in.orders = j.value("orders", std::vector<core::Order>{});
    in.equity_curve = j.value("equity_curve", std::vector<core::EquityPoint>{});
    if (j.contains("config")) { in.config = j.at("config").get<broker::BrokerConfig>(); }
    if (j.contains("strategy")) { in.strategy = j.at("strategy").get<StrategyRunInfo>(); }
    if (j.contains("run")) { in.run = j.at("run").get<RunMeta>(); }
    return in;
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
