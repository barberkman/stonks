#include <gtest/gtest.h>

#include <sstream>

#include "stonks/core/types.h"

namespace stonks::core {

TEST(OrderSideTest, BuyAndSellAreDistinct) {
    EXPECT_NE(OrderSide::Buy, OrderSide::Sell);
}

TEST(TimestampTest, FormatsAsIso8601Utc) {
    std::ostringstream oss;
    oss << Timestamp{1700000000123};
    EXPECT_EQ(oss.str(), "2023-11-14T22:13:20.123Z");
}

TEST(TimestampTest, FormatsEpoch) {
    std::ostringstream oss;
    oss << Timestamp{};
    EXPECT_EQ(oss.str(), "1970-01-01T00:00:00.000Z");
}

} // namespace stonks::core
