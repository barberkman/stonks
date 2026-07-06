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

TEST(ReportJson, TradeFeeTotalFeesAndConfigSerialize)
{
    const Trade t{ 1, 1, Timestamp::from_millis(0), "X", OrderSide::Buy, 1.0, 100.0, false, 0.25 };
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 1,
        .trades = { t },
        .orders = {},
        .equity_curve = {},
        .ending_cash = 999.75,
        .ending_equity = 999.75,
        .elapsed = {},
        .config = broker::BrokerConfig{ .maker_fee_bps = 2.0, .taker_fee_bps = 5.0, .fee_per_fill = 0.1 },
    };
    const auto j = serialize(in);
    EXPECT_DOUBLE_EQ(j.at("trades").front().at("fee").get<double>(), 0.25);
    EXPECT_DOUBLE_EQ(j.at("metrics").at("total_fees").get<double>(), 0.25);
    EXPECT_EQ(j.at("config").at("fill_policy").get<std::string>(), "Conservative");
    EXPECT_DOUBLE_EQ(j.at("config").at("maker_fee_bps").get<double>(), 2.0);
    EXPECT_DOUBLE_EQ(j.at("config").at("taker_fee_bps").get<double>(), 5.0);
    EXPECT_DOUBLE_EQ(j.at("config").at("fee_per_fill").get<double>(), 0.1);
    EXPECT_DOUBLE_EQ(j.at("config").at("flat_epsilon").get<double>(), 1e-9);
    EXPECT_FALSE(j.at("config").at("isolated_loss_cap").get<bool>());
}

TEST(ReportJson, ReportInputRoundTripsThroughJson)
{
    // Serialize a fully-populated input and parse it back: the raw materials
    // (trades / orders / equity curve) must reproduce exactly, and the config /
    // strategy / run blocks field-for-field. This is what lets a later session
    // restore archived runs into the GUI.
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 42,
        .trades = {
            Trade{ 1, 1, Timestamp::from_millis(1'700'000'000'123), "BTCUSDT",
                   OrderSide::Buy, 0.5, 60'273.5, false, 0.25 },
            Trade{ 2, 3, Timestamp::from_millis(1'700'086'400'000), "BTCUSDT",
                   OrderSide::Sell, 0.5, 61'000.0, true, 0.3 },
        },
        .orders = {
            Order{ 1, std::nullopt, Timestamp::from_millis(1'699'913'600'000), "BTCUSDT",
                   OrderSide::Buy, OrderType::Stop, OrderStatus::Filled,
                   core::Price{ 60'273.5 }, 0.5, TimeInForce::GTC, 33.0, false },
            Order{ 2, core::OrderID{ 1 }, Timestamp::from_millis(1'699'913'600'000), "BTCUSDT",
                   OrderSide::Sell, OrderType::Limit, OrderStatus::Cancelled,
                   std::nullopt, 0.5, TimeInForce::GTC, 1.0, true },
        },
        .equity_curve = {
            EquityPoint{ Timestamp::from_millis(1'700'000'000'123), 1'000.0 },
            EquityPoint{ Timestamp::from_millis(1'700'086'400'000), 1'011.5 },
        },
        .ending_cash = 1'011.5,
        .ending_equity = 1'011.5,
        .elapsed = std::chrono::milliseconds{ 1234 },
        .config = broker::BrokerConfig{ .maker_fee_bps = 2.0, .taker_fee_bps = 5.0,
                                        .isolated_loss_cap = true },
        .strategy = StrategyRunInfo{ "qmsignals", "QMSignalsStrategy",
                                     { { "risk_fraction", 0.03 } } },
        .run = RunMeta{ "QM Signals", "app/data/binance_1d.parquet", "binance_1d",
                        "2024-01-01", "2026-01-30", { "BTCUSDT", "ETHUSDT" } },
    };
    const auto j = serialize(in);
    const ReportInput back = report_input_from_json(j);

    EXPECT_EQ(back.trades, in.trades);                 // Trade has operator<=>
    EXPECT_EQ(back.orders, in.orders);
    EXPECT_EQ(back.equity_curve, in.equity_curve);
    EXPECT_DOUBLE_EQ(back.starting_cash, 1'000.0);
    EXPECT_EQ(back.bars_processed, 42u);
    EXPECT_DOUBLE_EQ(back.ending_cash, 1'011.5);
    EXPECT_DOUBLE_EQ(back.ending_equity, 1'011.5);
    const double elapsed_ms = std::chrono::duration<double, std::milli>{ back.elapsed }.count();
    EXPECT_NEAR(elapsed_ms, 1234.0, 1e-6);
    EXPECT_DOUBLE_EQ(back.config.taker_fee_bps, 5.0);
    EXPECT_TRUE(back.config.isolated_loss_cap);
    EXPECT_EQ(back.strategy.module, "qmsignals");
    EXPECT_DOUBLE_EQ(back.strategy.params.at("risk_fraction"), 0.03);
    EXPECT_EQ(back.run.display, "QM Signals");
    EXPECT_EQ(back.run.data_file, "app/data/binance_1d.parquet");
    EXPECT_EQ(back.run.start, "2024-01-01");
    EXPECT_EQ(back.run.symbols, (std::vector<std::string>{ "BTCUSDT", "ETHUSDT" }));
}

TEST(ReportJson, ReportInputFromJsonToleratesOldArchives)
{
    // A pre-feature archive: no fee on trades, no reduce_only on orders, no
    // strategy/run blocks. Parsing must default everything sensibly.
    const nlohmann::json j{
        { "metrics", { { "starting_cash", 500.0 }, { "ending_cash", 510.0 } } },
        { "trades", { { { "id", 1 }, { "order_id", 1 },
                        { "timestamp", "2024-01-01T00:00:00.000Z" }, { "symbol", "X" },
                        { "side", "Buy" }, { "quantity", 1.0 }, { "price", 100.0 },
                        { "liquidation", false } } } },
        { "orders", { { { "id", 1 }, { "parent_id", nullptr },
                        { "timestamp", "2024-01-01T00:00:00.000Z" }, { "symbol", "X" },
                        { "side", "Buy" }, { "type", "Market" }, { "status", "Filled" },
                        { "price", nullptr }, { "quantity", 1.0 },
                        { "time_in_force", "GTC" } } } },
    };
    const ReportInput back = report_input_from_json(j);
    ASSERT_EQ(back.trades.size(), 1u);
    EXPECT_DOUBLE_EQ(back.trades[0].fee, 0.0);
    ASSERT_EQ(back.orders.size(), 1u);
    EXPECT_FALSE(back.orders[0].reduce_only);
    EXPECT_DOUBLE_EQ(back.orders[0].leverage, 1.0);
    EXPECT_DOUBLE_EQ(back.ending_equity, 510.0);       // falls back to ending_cash
    EXPECT_TRUE(back.strategy.module.empty());
    EXPECT_TRUE(back.run.data_file.empty());
}

TEST(ReportJson, IndicatorsRoundTripThroughJson)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 2,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
        .indicator_specs = { IndicatorSpec{ "ema50", "50-bar EMA", "#4f8fe1" },
                             IndicatorSpec{ "adr", "avg daily range", "" } },
        .indicators = { { "BTCUSDT", { { "ema50", {
            core::IndicatorPoint{ Timestamp::from_millis(1000), 100.0 },
            core::IndicatorPoint{ Timestamp::from_millis(2000), 101.5 },
        } } } } },
    };
    const auto j = serialize(in);

    EXPECT_EQ(j.at("indicator_specs")[0].at("name").get<std::string>(), "ema50");
    EXPECT_EQ(j.at("indicator_specs")[0].at("color").get<std::string>(), "#4f8fe1");
    EXPECT_EQ(j.at("indicator_specs")[1].at("color").get<std::string>(), "");
    EXPECT_EQ(j.at("indicators").at("BTCUSDT").at("ema50")[0]
                  .at("timestamp").get<std::string>(),
              "1970-01-01T00:00:01.000Z");

    const ReportInput back = report_input_from_json(j);
    ASSERT_EQ(back.indicator_specs.size(), 2u);
    EXPECT_EQ(back.indicator_specs[0].name, "ema50");
    EXPECT_EQ(back.indicator_specs[0].doc, "50-bar EMA");
    EXPECT_EQ(back.indicator_specs[1].name, "adr");
    const auto& points = back.indicators.at("BTCUSDT").at("ema50");
    ASSERT_EQ(points.size(), 2u);
    EXPECT_EQ(points[0].timestamp, Timestamp::from_millis(1000));
    EXPECT_DOUBLE_EQ(points[1].value, 101.5);
}

TEST(ReportJson, IndicatorsAbsentInOldArchivesDefaultEmpty)
{
    const nlohmann::json j{
        { "metrics", { { "starting_cash", 500.0 }, { "ending_cash", 500.0 } } },
    };
    const ReportInput back = report_input_from_json(j);
    EXPECT_TRUE(back.indicator_specs.empty());
    EXPECT_TRUE(back.indicators.empty());
}

TEST(ReportJson, StrategyKeyIncludesModuleClassAndParams)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
        .strategy = StrategyRunInfo{ "qmsignals", "QMSignalsStrategy",
                                     { { "risk_fraction", 0.03 }, { "cooldown_bars", 3.0 } } },
    };
    const auto j = serialize(in);
    EXPECT_EQ(j.at("strategy").at("module").get<std::string>(), "qmsignals");
    EXPECT_EQ(j.at("strategy").at("class").get<std::string>(), "QMSignalsStrategy");
    EXPECT_DOUBLE_EQ(j.at("strategy").at("params").at("risk_fraction").get<double>(), 0.03);
    EXPECT_DOUBLE_EQ(j.at("strategy").at("params").at("cooldown_bars").get<double>(), 3.0);
}

TEST(ReportJson, StrategyKeyDefaultsToEmptyWhenNotProvided)
{
    const ReportInput in{
        .starting_cash = 1'000.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    };
    const auto j = serialize(in);
    EXPECT_EQ(j.at("strategy").at("module").get<std::string>(), "");
    EXPECT_TRUE(j.at("strategy").at("params").empty());
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
