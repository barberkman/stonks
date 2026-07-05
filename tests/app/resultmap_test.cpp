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
