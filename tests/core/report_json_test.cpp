// Unit tests for the JSON reporter (report_json.h). write_report_json emits the
// derived metrics plus the full raw materials; these tests serialize hand-built
// inputs, parse the output back with nlohmann::json, and assert on structure,
// enum/optional encoding, and timestamp formatting.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "stonks/core/types.h"

#include "src/report.h"
#include "src/report_json.h"

namespace stonks::app {
namespace {

using core::Balance;
using core::EquityPoint;
using core::Order;
using core::OrderSide;
using core::OrderStatus;
using core::OrderType;
using core::TimeInForce;
using core::Timestamp;
using core::Trade;

EquityPoint eq(std::int64_t ms, Balance equity)
{
    return EquityPoint{ Timestamp::from_millis(ms), equity };
}

Trade trade(std::int64_t ms, OrderSide side, core::Quantity qty, core::Price price,
            bool liquidation = false)
{
    return Trade{
        core::TradeID{ 7 }, core::OrderID{ 3 }, Timestamp::from_millis(ms),
        core::Symbol{ "BTCUSDT" }, side, qty, price, liquidation,
    };
}

nlohmann::json serialize(const ReportInput& in)
{
    std::ostringstream os;
    write_report_json(os, in, compute_metrics(in));
    return nlohmann::json::parse(os.str());   // throws on invalid JSON
}

TEST(ReportJson, EmptyRunSerializesValidJsonWithNullsAndEmptyArrays)
{
    const auto j = serialize(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 100.0,
        .ending_equity = 100.0,
        .elapsed = {},
    });

    EXPECT_TRUE(j.at("trades").empty());
    EXPECT_TRUE(j.at("orders").empty());
    EXPECT_TRUE(j.at("equity_curve").empty());

    const auto& m = j.at("metrics");
    EXPECT_TRUE(m.at("first_ts").is_null());
    EXPECT_TRUE(m.at("last_ts").is_null());
    EXPECT_TRUE(m.at("win_rate_pct").is_null());   // nothing closed
    EXPECT_FALSE(m.at("return_pct").is_null());     // starting_cash != 0
    EXPECT_DOUBLE_EQ(m.at("return_pct").get<double>(), 0.0);
}

TEST(ReportJson, MetricsMirrorComputeMetrics)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 4,
        .trades = { trade(1, OrderSide::Buy, 1.0, 100.0),
                    trade(2, OrderSide::Sell, 1.0, 110.0) },
        .orders = {},
        .equity_curve = { eq(1000, 1'000.0), eq(2000, 1'050.0) },
        .ending_cash = 1'010.0,
        .ending_equity = 1'010.0,
        .elapsed = std::chrono::milliseconds{ 2 },
    };
    const auto expected = compute_metrics(in);

    std::ostringstream os;
    write_report_json(os, in, expected);
    const auto j = nlohmann::json::parse(os.str());
    const auto& m = j.at("metrics");

    EXPECT_EQ(m.at("bars_processed").get<std::size_t>(), expected.bars_processed);
    EXPECT_EQ(m.at("trade_count").get<std::size_t>(), expected.trade_count);
    EXPECT_EQ(m.at("orders_placed").get<std::size_t>(), expected.orders_placed);
    EXPECT_DOUBLE_EQ(m.at("notional").get<double>(), expected.notional);
    EXPECT_EQ(m.at("closed_trades").get<std::size_t>(), expected.closed_trades);
    EXPECT_EQ(m.at("winning_trades").get<std::size_t>(), expected.winning_trades);
    ASSERT_TRUE(expected.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(m.at("win_rate_pct").get<double>(), *expected.win_rate_pct);
    EXPECT_EQ(m.at("liquidations").get<std::size_t>(), expected.liquidations);
    EXPECT_DOUBLE_EQ(m.at("starting_cash").get<double>(), expected.starting_cash);
    EXPECT_DOUBLE_EQ(m.at("ending_cash").get<double>(), expected.ending_cash);
    EXPECT_DOUBLE_EQ(m.at("ending_equity").get<double>(), expected.ending_equity);
    EXPECT_DOUBLE_EQ(m.at("max_drawdown_pct").get<double>(), expected.max_drawdown_pct);
    EXPECT_DOUBLE_EQ(m.at("elapsed_ms").get<double>(), 2.0);
    // ISO timestamps from the equity-curve endpoints.
    EXPECT_EQ(m.at("first_ts").get<std::string>(), "1970-01-01T00:00:01.000Z");
    EXPECT_EQ(m.at("last_ts").get<std::string>(), "1970-01-01T00:00:02.000Z");
}

TEST(ReportJson, TradesSerializeFieldsAndEnums)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 1,
        .trades = { trade(1500, OrderSide::Sell, 2.5, 123.5) },
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto j = serialize(in);
    ASSERT_EQ(j.at("trades").size(), 1u);
    const auto& t = j.at("trades").front();
    EXPECT_EQ(t.at("id").get<std::uint64_t>(), 7u);
    EXPECT_EQ(t.at("order_id").get<std::uint64_t>(), 3u);
    EXPECT_EQ(t.at("timestamp").get<std::string>(), "1970-01-01T00:00:01.500Z");
    EXPECT_EQ(t.at("symbol").get<std::string>(), "BTCUSDT");
    EXPECT_EQ(t.at("side").get<std::string>(), "Sell");
    EXPECT_DOUBLE_EQ(t.at("quantity").get<double>(), 2.5);
    EXPECT_DOUBLE_EQ(t.at("price").get<double>(), 123.5);
    EXPECT_FALSE(t.at("liquidation").get<bool>());
}

TEST(ReportJson, LiquidationTradeAndCountSerialize)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 2,
        .trades = { trade(1000, OrderSide::Buy, 1.0, 100.0),
                    trade(2000, OrderSide::Sell, 1.0, 40.0, true) },
        .orders = {},
        .equity_curve = {},
        .ending_cash = 940.0,
        .ending_equity = 940.0,
        .elapsed = {},
    };
    const auto j = serialize(in);
    ASSERT_EQ(j.at("trades").size(), 2u);
    EXPECT_FALSE(j.at("trades").at(0).at("liquidation").get<bool>());
    EXPECT_TRUE(j.at("trades").at(1).at("liquidation").get<bool>());
    EXPECT_EQ(j.at("metrics").at("liquidations").get<std::size_t>(), 1u);
}

TEST(ReportJson, OrderEnumsAndOptionalsWithValues)
{
    const Order o{
        .id = 42,
        .parent_id = core::OrderID{ 41 },
        .timestamp = Timestamp::from_millis(0),
        .symbol = "ETHUSDT",
        .side = OrderSide::Sell,
        .type = OrderType::Limit,
        .status = OrderStatus::Rejected,
        .price = core::Price{ 2'000.0 },
        .quantity = 1.0,
        .time_in_force = TimeInForce::GTC,
    };
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = { o },
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto j = serialize(in);
    ASSERT_EQ(j.at("orders").size(), 1u);
    const auto& jo = j.at("orders").front();
    EXPECT_EQ(jo.at("parent_id").get<std::uint64_t>(), 41u);
    EXPECT_EQ(jo.at("side").get<std::string>(), "Sell");
    EXPECT_EQ(jo.at("type").get<std::string>(), "Limit");
    EXPECT_EQ(jo.at("status").get<std::string>(), "Rejected");
    EXPECT_DOUBLE_EQ(jo.at("price").get<double>(), 2'000.0);
    EXPECT_EQ(jo.at("time_in_force").get<std::string>(), "GTC");
    // Designated init without .leverage must land on the default 1.0, not 0.0.
    EXPECT_DOUBLE_EQ(jo.at("leverage").get<double>(), 1.0);
}

TEST(ReportJson, StopOrderTypeSerializes)
{
    const Order o{
        .id = 7,
        .parent_id = std::nullopt,
        .timestamp = Timestamp::from_millis(0),
        .symbol = "BTCUSDT",
        .side = OrderSide::Sell,
        .type = OrderType::Stop,
        .status = OrderStatus::Open,
        .price = core::Price{ 95.0 },
        .quantity = 1.0,
        .time_in_force = TimeInForce::GTC,
    };
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = { o },
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto j = serialize(in);
    EXPECT_EQ(j.at("orders").front().at("type").get<std::string>(), "Stop");
}

TEST(ReportJson, OrderLeverageRoundTrips)
{
    const Order o{
        .id = 5,
        .parent_id = std::nullopt,
        .timestamp = Timestamp::from_millis(0),
        .symbol = "BTCUSDT",
        .side = OrderSide::Buy,
        .type = OrderType::Market,
        .status = OrderStatus::Filled,
        .price = std::nullopt,
        .quantity = 2.0,
        .time_in_force = TimeInForce::GTC,
        .leverage = 3.0,
    };
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = { o },
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto jo = serialize(in).at("orders").front();
    EXPECT_DOUBLE_EQ(jo.at("leverage").get<double>(), 3.0);
}

TEST(ReportJson, OrderOptionalsSerializeAsNullWhenEmpty)
{
    const Order o{
        .id = 1,
        .parent_id = std::nullopt,
        .timestamp = Timestamp::from_millis(0),
        .symbol = "SOLUSDT",
        .side = OrderSide::Buy,
        .type = OrderType::Market,
        .status = OrderStatus::Filled,
        .price = std::nullopt,
        .quantity = 3.0,
        .time_in_force = TimeInForce::GTC,
    };
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = { o },
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto jo = serialize(in).at("orders").front();
    EXPECT_TRUE(jo.at("parent_id").is_null());
    EXPECT_TRUE(jo.at("price").is_null());
    EXPECT_EQ(jo.at("type").get<std::string>(), "Market");
    EXPECT_EQ(jo.at("status").get<std::string>(), "Filled");
}

TEST(ReportJson, EquityCurveSerializesTimestampAndEquity)
{
    const ReportInput in{
        .starting_cash = 100.0,
        .bars_processed = 2,
        .trades = {},
        .orders = {},
        .equity_curve = { eq(1000, 100.0), eq(5000, 90.0) },
        .ending_cash = 90.0,
        .ending_equity = 90.0,
        .elapsed = {},
    };
    const auto curve = serialize(in).at("equity_curve");
    ASSERT_EQ(curve.size(), 2u);
    EXPECT_EQ(curve.at(0).at("timestamp").get<std::string>(), "1970-01-01T00:00:01.000Z");
    EXPECT_DOUBLE_EQ(curve.at(0).at("equity").get<double>(), 100.0);
    EXPECT_EQ(curve.at(1).at("timestamp").get<std::string>(), "1970-01-01T00:00:05.000Z");
    EXPECT_DOUBLE_EQ(curve.at(1).at("equity").get<double>(), 90.0);
}

TEST(ReportJson, TimestampedPathHasExpectedShape)
{
    const std::string path = timestamped_report_path();
    EXPECT_TRUE(path.starts_with("app/reports/report-"));
    EXPECT_TRUE(path.ends_with(".json"));
    EXPECT_EQ(path.find(':'), std::string::npos);   // filesystem-portable
    EXPECT_EQ(path.size(), std::string{ "app/reports/report-YYYYMMDD-HHMMSS.json" }.size());
}

} // namespace
} // namespace stonks::app
