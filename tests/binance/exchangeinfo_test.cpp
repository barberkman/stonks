// ExchangeInfo filter parsing + rounding/min-notional, from a canned payload.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "stonks/binance/exchangeinfo.h"

namespace stonks::binance {
namespace {

ExchangeInfo make()
{
    const nlohmann::json j = {
        { "symbols", { {
            { "symbol", "BTCUSDT" },
            { "quantityPrecision", 3 },
            { "pricePrecision", 1 },
            { "filters", {
                { { "filterType", "LOT_SIZE" }, { "stepSize", "0.001" } },
                { { "filterType", "PRICE_FILTER" }, { "tickSize", "0.1" } },
                { { "filterType", "MIN_NOTIONAL" }, { "notional", "5" } },
            } },
        } } }
    };
    return ExchangeInfo{ j };
}

TEST(ExchangeInfo, RoundsQuantityDownToStep)
{
    const auto ex = make();
    EXPECT_DOUBLE_EQ(ex.round_qty("BTCUSDT", 1.23456), 1.234);
    EXPECT_DOUBLE_EQ(ex.round_qty("BTCUSDT", 0.0019), 0.001);
    EXPECT_DOUBLE_EQ(ex.round_qty("BTCUSDT", 0.0005), 0.0);   // below one step
}

TEST(ExchangeInfo, RoundsPriceToNearestTick)
{
    const auto ex = make();
    EXPECT_DOUBLE_EQ(ex.round_price("BTCUSDT", 100.04), 100.0);
    EXPECT_DOUBLE_EQ(ex.round_price("BTCUSDT", 100.06), 100.1);
}

TEST(ExchangeInfo, MinNotionalGate)
{
    const auto ex = make();
    EXPECT_FALSE(ex.passes_min_notional("BTCUSDT", 0.001, 100.0));   // 0.10 < 5
    EXPECT_TRUE(ex.passes_min_notional("BTCUSDT", 0.1, 100.0));      // 10 >= 5
}

TEST(ExchangeInfo, UnknownSymbolIsUnconstrained)
{
    const auto ex = make();
    EXPECT_FALSE(ex.has("ETHUSDT"));
    EXPECT_DOUBLE_EQ(ex.round_qty("ETHUSDT", 1.23456789), 1.23456789);
    EXPECT_TRUE(ex.passes_min_notional("ETHUSDT", 0.0001, 1.0));
}

} // namespace
} // namespace stonks::binance
