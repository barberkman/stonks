#include <gtest/gtest.h>

#include "stonks/core/types.h"

namespace stonks::core {

TEST(OrderSideTest, BuyAndSellAreDistinct) {
    EXPECT_NE(OrderSide::Buy, OrderSide::Sell);
}

}
