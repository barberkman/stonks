#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <QVariantList>
#include <QVariantMap>

#include "src/resultmap.h"
#include "stonks/core/types.h"

namespace {

using namespace stonks;
using stonks::core::OrderSide;

// 2,000 synthetic daily candles — well past the old 1,500-candle display cap,
// so any reintroduced downsampling/stride re-indexing fails these tests.
constexpr int kCandles = 2000;
constexpr std::int64_t kDayMs = 86400000;

app::SymbolSeries make_series()
{
    app::SymbolSeries series;
    series.candles.reserve(kCandles);
    for (int i = 0; i < kCandles; ++i) {
        const double base = 100.0 + i * 0.25;
        series.candles.push_back(app::Candle{ base, base + 2.0, base - 2.0, base + 1.0,
                                              1000.0 + i, i * kDayMs });
    }
    return series;
}

core::Trade fill(std::string symbol, OrderSide side, double qty, double price, std::int64_t ms)
{
    static core::TradeID next_id = 0;
    return core::Trade{
        ++next_id, 0, core::Timestamp::from_millis(ms), std::move(symbol), side, qty, price,
    };
}

// One round trip entering at candle 100 and exiting at candle 1900 — indices
// the old stride-based re-mapping (ceil(2000/1500) = 2) would have halved.
QVariantMap build_drill()
{
    std::vector<core::Trade> fills;
    fills.push_back(fill("BTC", OrderSide::Buy, 1.0, 125.0, 100 * kDayMs));
    fills.push_back(fill("BTC", OrderSide::Sell, 1.0, 575.0, 1900 * kDayMs));

    const app::ReportInput input{
        1000.0, kCandles, fills, {}, {}, 1450.0, 1450.0, std::chrono::nanoseconds{ 0 },
    };
    const app::ReportMetrics metrics = app::compute_metrics(input);

    std::map<core::Symbol, app::SymbolSeries> by_symbol;
    by_symbol["BTC"] = make_series();
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data" },
                                                 input, metrics, by_symbol, 252.0);
    return result["perSymbol"].toMap()["BTC"].toMap();
}

} // namespace

TEST(BuildResult, StrategyParamsShippedToResult)
{
    const app::ReportInput input{
        1000.0, 0, {}, {}, {}, 1000.0, 1000.0, std::chrono::nanoseconds{ 0 },
    };
    const app::ReportMetrics metrics = app::compute_metrics(input);
    QVariantMap params;
    params["risk_fraction"] = 0.03;
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data", params },
                                                 input, metrics, {}, 252.0);
    EXPECT_DOUBLE_EQ(result["strategyParams"].toMap()["risk_fraction"].toDouble(), 0.03);
}

TEST(BuildResult, StrategyParamsEmptyWhenRunConfigOmitsThem)
{
    const app::ReportInput input{
        1000.0, 0, {}, {}, {}, 1000.0, 1000.0, std::chrono::nanoseconds{ 0 },
    };
    const app::ReportMetrics metrics = app::compute_metrics(input);
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data" },
                                                 input, metrics, {}, 252.0);
    EXPECT_TRUE(result["strategyParams"].toMap().isEmpty());
}

TEST(BuildResult, CandlesAreColumnarAndFullResolution)
{
    const QVariantMap candles = build_drill()["candles"].toMap();

    for (const char* key : { "t", "o", "h", "l", "c", "v" }) {
        EXPECT_EQ(candles[key].toList().size(), kCandles) << key;
    }

    EXPECT_EQ(candles["t"].toList().first().toLongLong(), 0);
    EXPECT_DOUBLE_EQ(candles["o"].toList().first().toDouble(), 100.0);
    EXPECT_DOUBLE_EQ(candles["h"].toList().first().toDouble(), 102.0);
    EXPECT_DOUBLE_EQ(candles["l"].toList().first().toDouble(), 98.0);
    EXPECT_DOUBLE_EQ(candles["c"].toList().first().toDouble(), 101.0);
    EXPECT_DOUBLE_EQ(candles["v"].toList().first().toDouble(), 1000.0);

    EXPECT_EQ(candles["t"].toList().last().toLongLong(), (kCandles - 1) * kDayMs);
    EXPECT_DOUBLE_EQ(candles["o"].toList().last().toDouble(), 100.0 + (kCandles - 1) * 0.25);
    EXPECT_DOUBLE_EQ(candles["v"].toList().last().toDouble(), 1000.0 + (kCandles - 1));
}

TEST(BuildResult, TradeIndicesAreNaturalCandleIndices)
{
    const QVariantList trades = build_drill()["trades"].toList();
    ASSERT_EQ(trades.size(), 1);

    const QVariantMap trade = trades.first().toMap();
    EXPECT_EQ(trade["entryIdx"].toInt(), 100);
    EXPECT_EQ(trade["exitIdx"].toInt(), 1900);
    EXPECT_EQ(trade["bars"].toInt(), 1800);
}

TEST(BuildResult, SparklineStaysDownsampled)
{
    const QVariantList spark = build_drill()["spark"].toList();
    EXPECT_GT(spark.size(), 0);
    EXPECT_LE(spark.size(), 64);
}

namespace {

// Three flat daily candles plus one plotted series with points at candles 0
// and 2 only — the shape every indicator-overlay test below drills into.
QVariantMap build_indicator_drill(std::vector<app::IndicatorSpec> specs,
                                  core::IndicatorStore store)
{
    app::SymbolSeries series;
    series.candles = {
        app::Candle{ 1, 1, 1, 1, 1, 0 },
        app::Candle{ 1, 1, 1, 1, 1, kDayMs },
        app::Candle{ 1, 1, 1, 1, 1, 2 * kDayMs },
    };

    app::ReportInput input{
        1000.0, 3, {}, {}, {}, 1000.0, 1000.0, std::chrono::nanoseconds{ 0 },
    };
    input.indicator_specs = std::move(specs);
    input.indicators = std::move(store);

    const app::ReportMetrics metrics = app::compute_metrics(input);
    std::map<core::Symbol, app::SymbolSeries> by_symbol;
    by_symbol["BTC"] = series;
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data" },
                                                 input, metrics, by_symbol, 252.0);
    return result["perSymbol"].toMap()["BTC"].toMap();
}

core::IndicatorPoint ipt(std::int64_t ms, double value)
{
    return core::IndicatorPoint{ core::Timestamp::from_millis(ms), value };
}

} // namespace

TEST(BuildResult, IndicatorValuesAlignToCandlesWithNullGaps)
{
    const QVariantList indicators = build_indicator_drill(
        { app::IndicatorSpec{ "ema", "the ema", "#123456" } },
        { { "BTC", { { "ema", { ipt(0, 10.0), ipt(2 * kDayMs, 12.0) } } } } })["indicators"]
        .toList();

    ASSERT_EQ(indicators.size(), 1);
    const QVariantMap entry = indicators.first().toMap();
    EXPECT_EQ(entry["name"].toString(), "ema");
    EXPECT_EQ(entry["doc"].toString(), "the ema");
    EXPECT_EQ(entry["color"].toString(), "#123456");   // author color passes through

    const QVariantList values = entry["values"].toList();
    ASSERT_EQ(values.size(), 3);                       // parallel to the candles
    EXPECT_DOUBLE_EQ(values[0].toDouble(), 10.0);
    EXPECT_FALSE(values[1].isValid());                 // no point at candle 1 -> null gap
    EXPECT_DOUBLE_EQ(values[2].toDouble(), 12.0);
}

TEST(BuildResult, SymbolWithNoIndicatorsGetsEmptyList)
{
    const QVariantMap drill = build_indicator_drill({}, {});
    ASSERT_TRUE(drill.contains("indicators"));
    EXPECT_TRUE(drill["indicators"].toList().isEmpty());
}

TEST(BuildResult, UndeclaredIndicatorGetsPaletteColorAndEmptyDoc)
{
    const QVariantList indicators = build_indicator_drill(
        {},   // nothing declared — e.g. a typo'd name or a C++ strategy
        { { "BTC", { { "mystery", { ipt(0, 5.0) } } } } })["indicators"].toList();

    ASSERT_EQ(indicators.size(), 1);
    const QVariantMap entry = indicators.first().toMap();
    EXPECT_EQ(entry["name"].toString(), "mystery");
    EXPECT_TRUE(entry["doc"].toString().isEmpty());
    EXPECT_EQ(entry["color"].toString(), "#4f8fe1");   // first palette color
}

TEST(BuildResult, DeclaredOrderPreservedOverAlphabeticalStoreOrder)
{
    const QVariantList indicators = build_indicator_drill(
        { app::IndicatorSpec{ "zeta", "", "" }, app::IndicatorSpec{ "alpha", "", "" } },
        { { "BTC", { { "alpha", { ipt(0, 1.0) } }, { "zeta", { ipt(0, 2.0) } } } } })
        ["indicators"].toList();

    ASSERT_EQ(indicators.size(), 2);
    EXPECT_EQ(indicators[0].toMap()["name"].toString(), "zeta");    // declaration order,
    EXPECT_EQ(indicators[1].toMap()["name"].toString(), "alpha");   // not the store's map order
    // Palette colors assigned in declaration order too.
    EXPECT_EQ(indicators[0].toMap()["color"].toString(), "#4f8fe1");
    EXPECT_EQ(indicators[1].toMap()["color"].toString(), "#b98ae8");
}

// The GUI's per-symbol table sorts on these raw numbers rather than parsing the
// display strings ("+$80" / "50%") back to doubles.
TEST(BuildResult, PerSymbolRowsCarryNumericSortKeys)
{
    std::vector<core::Trade> fills;
    fills.push_back(fill("BTC", OrderSide::Buy, 1.0, 100.0, 10 * kDayMs));
    fills.push_back(fill("BTC", OrderSide::Sell, 1.0, 200.0, 20 * kDayMs));   // +100
    fills.push_back(fill("BTC", OrderSide::Buy, 1.0, 100.0, 30 * kDayMs));
    fills.push_back(fill("BTC", OrderSide::Sell, 1.0, 80.0, 40 * kDayMs));    // -20
    fills.push_back(fill("ETH", OrderSide::Buy, 1.0, 100.0, 10 * kDayMs));
    fills.push_back(fill("ETH", OrderSide::Sell, 1.0, 50.0, 20 * kDayMs));    // -50

    const app::ReportInput input{
        1000.0, kCandles, fills, {}, {}, 1030.0, 1030.0, std::chrono::nanoseconds{ 0 },
    };
    const app::ReportMetrics metrics = app::compute_metrics(input);
    std::map<core::Symbol, app::SymbolSeries> by_symbol;
    by_symbol["BTC"] = make_series();
    by_symbol["ETH"] = make_series();
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data" },
                                                 input, metrics, by_symbol, 252.0);

    QVariantMap btc, eth;
    for (const QVariant& v : result["symbols"].toList()) {
        const QVariantMap row = v.toMap();
        if (row["id"].toString() == "BTC") { btc = row; }
        if (row["id"].toString() == "ETH") { eth = row; }
    }
    ASSERT_FALSE(btc.isEmpty());
    ASSERT_FALSE(eth.isEmpty());

    // Numbers, not strings — QML sorts them arithmetically.
    EXPECT_EQ(btc["pnlVal"].userType(), QMetaType::Double);
    EXPECT_EQ(btc["retVal"].userType(), QMetaType::Double);
    EXPECT_EQ(btc["winVal"].userType(), QMetaType::Double);

    EXPECT_DOUBLE_EQ(btc["pnlVal"].toDouble(), 80.0);    // +100 - 20
    EXPECT_DOUBLE_EQ(btc["retVal"].toDouble(), 8.0);     // 80 / 1000 starting cash
    EXPECT_DOUBLE_EQ(btc["winVal"].toDouble(), 50.0);    // 1 of 2 round-trips won
    EXPECT_DOUBLE_EQ(eth["pnlVal"].toDouble(), -50.0);
    EXPECT_DOUBLE_EQ(eth["retVal"].toDouble(), -5.0);
    EXPECT_DOUBLE_EQ(eth["winVal"].toDouble(), 0.0);

    // ...and they agree with the display strings shown in the same row.
    EXPECT_EQ(btc["pnl"].toString(), "+$80");
    EXPECT_EQ(btc["ret"].toString(), "+8.0%");
    EXPECT_EQ(btc["win"].toString(), "50%");
    EXPECT_EQ(eth["pnl"].toString(), "-$50");
    EXPECT_EQ(eth["win"].toString(), "0%");
}

TEST(BuildResult, EquityTimestampsShipped)
{
    constexpr std::int64_t t0 = 1704067200000;   // arbitrary epoch ms
    std::vector<core::EquityPoint> curve;
    for (int i = 0; i < 5; ++i) {
        curve.push_back(core::EquityPoint{ core::Timestamp::from_millis(t0 + i * kDayMs),
                                           1000.0 + i * 10.0 });
    }

    const app::ReportInput input{
        1000.0, 5, {}, {}, curve, 1040.0, 1040.0, std::chrono::nanoseconds{ 0 },
    };
    const app::ReportMetrics metrics = app::compute_metrics(input);
    const std::map<core::Symbol, app::SymbolSeries> by_symbol;
    const QVariantMap result = app::build_result(app::RunConfig{ "1", "Test", "data" },
                                                 input, metrics, by_symbol, 252.0);

    const QVariantList equity = result["equity"].toList();
    const QVariantList times = result["equityT"].toList();
    ASSERT_EQ(times.size(), static_cast<int>(curve.size()));
    EXPECT_EQ(times.size(), equity.size());   // one timestamp per equity point
    for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
        EXPECT_EQ(times[i].toLongLong(), t0 + i * kDayMs) << i;
    }
}
