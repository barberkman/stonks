#include <gtest/gtest.h>

#include "stonks/core/types.h"

namespace stonks::core {

TEST(SideTest, BuyAndSellAreDistinct) {
    EXPECT_NE(Side::Buy, Side::Sell);
}

}
