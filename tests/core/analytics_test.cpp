#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "src/analytics.h"
#include "stonks/core/types.h"

namespace {

using namespace stonks;
using stonks::core::OrderSide;

core::Trade fill(std::string symbol, OrderSide side, double qty, double price, std::int64_t ms)
{
    static core::TradeID next_id = 0;
    return core::Trade{
        ++next_id, 0, core::Timestamp::from_millis(ms), std::move(symbol), side, qty, price,
    };
}

} // namespace

TEST(RoundTrips, SimpleLong)
{
    const std::vector<core::Trade> fills = {
        fill("BTC", OrderSide::Buy, 10.0, 100.0, 1000),
        fill("BTC", OrderSide::Sell, 10.0, 110.0, 2000),
    };
    const auto rts = app::reconstruct_round_trips(fills);
    ASSERT_EQ(rts.size(), 1u);
    EXPECT_EQ(rts[0].symbol, "BTC");
    EXPECT_EQ(rts[0].side, OrderSide::Buy);
    EXPECT_EQ(rts[0].entry_ts, 1000);
    EXPECT_EQ(rts[0].exit_ts, 2000);
    EXPECT_DOUBLE_EQ(rts[0].entry_price, 100.0);
    EXPECT_DOUBLE_EQ(rts[0].exit_price, 110.0);
    EXPECT_DOUBLE_EQ(rts[0].qty, 10.0);
    EXPECT_DOUBLE_EQ(rts[0].realized, 100.0);
}

TEST(RoundTrips, SimpleShort)
{
    const std::vector<core::Trade> fills = {
        fill("BTC", OrderSide::Sell, 10.0, 100.0, 1000),
        fill("BTC", OrderSide::Buy, 10.0, 90.0, 2000),
    };
    const auto rts = app::reconstruct_round_trips(fills);
    ASSERT_EQ(rts.size(), 1u);
    EXPECT_EQ(rts[0].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(rts[0].entry_price, 100.0);
    EXPECT_DOUBLE_EQ(rts[0].exit_price, 90.0);
    EXPECT_DOUBLE_EQ(rts[0].realized, 100.0);   // short profits when price falls
}

TEST(RoundTrips, ScaleInBlendsAverage)
{
    const std::vector<core::Trade> fills = {
        fill("BTC", OrderSide::Buy, 10.0, 100.0, 1000),
        fill("BTC", OrderSide::Buy, 10.0, 200.0, 2000),
        fill("BTC", OrderSide::Sell, 20.0, 180.0, 3000),
    };
    const auto rts = app::reconstruct_round_trips(fills);
    ASSERT_EQ(rts.size(), 1u);
    EXPECT_DOUBLE_EQ(rts[0].entry_price, 150.0);   // (100+200)/2 blended
    EXPECT_DOUBLE_EQ(rts[0].exit_price, 180.0);
    EXPECT_DOUBLE_EQ(rts[0].qty, 20.0);            // peak position
    EXPECT_DOUBLE_EQ(rts[0].realized, 600.0);      // (180-150)*20
}

TEST(RoundTrips, PartialCloses)
{
    const std::vector<core::Trade> fills = {
        fill("BTC", OrderSide::Buy, 10.0, 100.0, 1000),
        fill("BTC", OrderSide::Sell, 5.0, 120.0, 2000),
        fill("BTC", OrderSide::Sell, 5.0, 130.0, 3000),
    };
    const auto rts = app::reconstruct_round_trips(fills);
    ASSERT_EQ(rts.size(), 1u);
    EXPECT_DOUBLE_EQ(rts[0].entry_price, 100.0);
    EXPECT_DOUBLE_EQ(rts[0].exit_price, 125.0);    // volume-weighted (120+130)/2
    EXPECT_DOUBLE_EQ(rts[0].qty, 10.0);            // peak position
    EXPECT_DOUBLE_EQ(rts[0].realized, 250.0);      // (120-100)*5 + (130-100)*5
    EXPECT_EQ(rts[0].exit_ts, 3000);               // flattening fill
}

TEST(RoundTrips, FlipThroughFlatOpensSecondCycle)
{
    const std::vector<core::Trade> fills = {
        fill("BTC", OrderSide::Buy, 10.0, 100.0, 1000),
        fill("BTC", OrderSide::Sell, 15.0, 110.0, 2000),   // closes 10 long, opens 5 short
        fill("BTC", OrderSide::Buy, 5.0, 105.0, 3000),     // closes the short
    };
    const auto rts = app::reconstruct_round_trips(fills);
    ASSERT_EQ(rts.size(), 2u);
    EXPECT_EQ(rts[0].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(rts[0].realized, 100.0);              // (110-100)*10
    EXPECT_EQ(rts[1].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(rts[1].entry_price, 110.0);
    EXPECT_DOUBLE_EQ(rts[1].realized, 25.0);               // (110-105)*5
}

TEST(Sharpe, KnownValueAndDegenerateCases)
{
    // returns: (2-1)/1 = 1.0, (3-2)/2 = 0.5; mean 0.75, sample stdev sqrt(0.125)
    const std::vector<core::EquityPoint> curve = {
        { core::Timestamp::from_millis(0), 1.0 },
        { core::Timestamp::from_millis(1), 2.0 },
        { core::Timestamp::from_millis(2), 3.0 },
    };
    const auto s = app::sharpe_ratio(curve, 1.0);
    ASSERT_TRUE(s.has_value());
    EXPECT_NEAR(*s, 2.12132, 1e-4);

    // Fewer than two returns -> nullopt.
    EXPECT_FALSE(app::sharpe_ratio({ { core::Timestamp::from_millis(0), 1.0 } }, 1.0).has_value());

    // Constant returns -> zero variance -> nullopt.
    const std::vector<core::EquityPoint> geometric = {
        { core::Timestamp::from_millis(0), 1.0 },
        { core::Timestamp::from_millis(1), 2.0 },
        { core::Timestamp::from_millis(2), 4.0 },
    };
    EXPECT_FALSE(app::sharpe_ratio(geometric, 1.0).has_value());
}

TEST(ProfitFactor, MixedWinsAndLosses)
{
    std::vector<app::RoundTrip> trips(4);
    trips[0].realized = 100.0;
    trips[1].realized = -50.0;
    trips[2].realized = 200.0;
    trips[3].realized = -25.0;
    EXPECT_DOUBLE_EQ(app::profit_factor(trips), 4.0);   // 300 / 75

    std::vector<app::RoundTrip> winners(1);
    winners[0].realized = 100.0;
    EXPECT_TRUE(std::isinf(app::profit_factor(winners)));

    std::vector<app::RoundTrip> losers(1);
    losers[0].realized = -100.0;
    EXPECT_DOUBLE_EQ(app::profit_factor(losers), 0.0);
}

TEST(PerSymbol, TalliesPnlAndWins)
{
    std::vector<app::RoundTrip> trips(3);
    trips[0].symbol = "A"; trips[0].realized = 100.0;
    trips[1].symbol = "A"; trips[1].realized = -50.0;
    trips[2].symbol = "B"; trips[2].realized = 200.0;
    const auto by_symbol = app::per_symbol_breakdown(trips);
    ASSERT_EQ(by_symbol.size(), 2u);
    EXPECT_DOUBLE_EQ(by_symbol.at("A").pnl, 50.0);
    EXPECT_EQ(by_symbol.at("A").closed, 2u);
    EXPECT_EQ(by_symbol.at("A").winning, 1u);
    EXPECT_DOUBLE_EQ(by_symbol.at("B").pnl, 200.0);
    EXPECT_EQ(by_symbol.at("B").winning, 1u);
}

TEST(Annotate, IndicesBarsAndExcursionsLong)
{
    app::SymbolSeries series;
    series.candles = {
        { 100, 100, 100, 100, 1, 0 },
        { 100, 105, 98, 103, 1, 1000 },
        { 103, 112, 101, 110, 1, 2000 },
        { 110, 109, 104, 108, 1, 3000 },
        { 108, 108, 108, 108, 1, 4000 },
    };
    app::RoundTrip rt;
    rt.side = OrderSide::Buy;
    rt.entry_ts = 1000;
    rt.exit_ts = 3000;
    rt.entry_price = 100.0;
    rt.exit_price = 108.0;
    rt.qty = 10.0;
    rt.realized = 80.0;

    const auto row = app::annotate(rt, series, 1);
    EXPECT_EQ(row.entry_idx, 1);
    EXPECT_EQ(row.exit_idx, 3);
    EXPECT_EQ(row.bars, 2);
    EXPECT_TRUE(row.is_long);
    EXPECT_NEAR(row.ret_pct, 8.0, 1e-9);     // 80 / (100*10) * 100
    EXPECT_NEAR(row.mfe, 12.0, 1e-9);        // max high 112 -> +12%
    EXPECT_NEAR(row.mae, -2.0, 1e-9);        // min low 98 -> -2%
}

TEST(Annotate, ExcursionsShortAreSignAware)
{
    app::SymbolSeries series;
    series.candles = {
        { 100, 100, 100, 100, 1, 0 },
        { 100, 105, 98, 103, 1, 1000 },
        { 103, 112, 101, 110, 1, 2000 },
        { 110, 109, 104, 108, 1, 3000 },
    };
    app::RoundTrip rt;
    rt.side = OrderSide::Sell;
    rt.entry_ts = 1000;
    rt.exit_ts = 3000;
    rt.entry_price = 100.0;
    rt.exit_price = 95.0;
    rt.qty = 10.0;
    rt.realized = 50.0;

    const auto row = app::annotate(rt, series, 1);
    EXPECT_FALSE(row.is_long);
    EXPECT_NEAR(row.mfe, (100.0 / 98.0 - 1.0) * 100.0, 1e-6);    // lowest low favourable
    EXPECT_NEAR(row.mae, (100.0 / 112.0 - 1.0) * 100.0, 1e-6);   // highest high adverse
}

namespace {

core::IndicatorPoint pt(std::int64_t ms, double value)
{
    return core::IndicatorPoint{ core::Timestamp::from_millis(ms), value };
}

// Flat candles at the given timestamps — align_indicator only reads Candle.t.
std::vector<app::Candle> candles_at(std::initializer_list<std::int64_t> ts)
{
    std::vector<app::Candle> out;
    for (const auto t : ts) { out.push_back(app::Candle{ 1, 1, 1, 1, 1, t }); }
    return out;
}

} // namespace

TEST(AlignIndicator, ExactMatchesTakeValuesAndMissingBarsAreGaps)
{
    const auto aligned = app::align_indicator(
        candles_at({ 1000, 2000, 3000 }), { pt(1000, 10.0), pt(3000, 12.0) });

    ASSERT_EQ(aligned.size(), 3u);
    EXPECT_DOUBLE_EQ(aligned[0].value(), 10.0);
    EXPECT_FALSE(aligned[1].has_value());   // no point at 2000 -> gap
    EXPECT_DOUBLE_EQ(aligned[2].value(), 12.0);
}

TEST(AlignIndicator, RepeatedTimestampResolvesLastCallWins)
{
    const auto aligned = app::align_indicator(
        candles_at({ 1000 }), { pt(1000, 10.0), pt(1000, 11.0) });

    ASSERT_EQ(aligned.size(), 1u);
    EXPECT_DOUBLE_EQ(aligned[0].value(), 11.0);
}

TEST(AlignIndicator, PointWithNoMatchingCandleIsDroppedNotMisassigned)
{
    // 1500 falls between candles: dropped. Later points still land correctly.
    const auto aligned = app::align_indicator(
        candles_at({ 1000, 2000 }), { pt(1500, 99.0), pt(2000, 12.0) });

    ASSERT_EQ(aligned.size(), 2u);
    EXPECT_FALSE(aligned[0].has_value());
    EXPECT_DOUBLE_EQ(aligned[1].value(), 12.0);
}

TEST(AlignIndicator, EmptyPointsProduceAllGaps)
{
    const auto aligned = app::align_indicator(candles_at({ 1000, 2000 }), {});

    ASSERT_EQ(aligned.size(), 2u);
    EXPECT_FALSE(aligned[0].has_value());
    EXPECT_FALSE(aligned[1].has_value());
}
