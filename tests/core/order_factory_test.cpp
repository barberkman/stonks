#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::core {

static_assert(!std::is_default_constructible_v<Order>);
static_assert(!std::is_constructible_v<Order,
    OrderID, Timestamp, Symbol, OrderSide, Price, Quantity, TimeInForce>);
static_assert(std::is_copy_constructible_v<Order>);
static_assert(std::is_move_constructible_v<Order>);

TEST(OrderFactory, StampsClockTimeOnConstruction) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(1000));
    const auto o1 = ctx.make_order(
        Symbol{ "X" }, OrderSide::Buy, Quantity{ 1.0 }, Price{ 100.0 }, TimeInForce::GTC);
    EXPECT_EQ(o1.timestamp, Timestamp::from_millis(1000));

    clock.set(Timestamp::from_millis(5000));
    const auto o2 = ctx.make_order(
        Symbol{ "X" }, OrderSide::Sell, Quantity{ 0.5 }, Price{ 110.0 }, TimeInForce::GTC);
    EXPECT_EQ(o2.timestamp, Timestamp::from_millis(5000));
}

TEST(OrderFactory, AssignsStrictlyIncreasingIds) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    const auto a = ctx.make_order(Symbol{ "X" }, OrderSide::Buy, Quantity{ 1.0 }, Price{ 1.0 }, TimeInForce::GTC);
    const auto b = ctx.make_order(Symbol{ "X" }, OrderSide::Buy, Quantity{ 1.0 }, Price{ 1.0 }, TimeInForce::GTC);
    const auto c = ctx.make_order(Symbol{ "X" }, OrderSide::Sell, Quantity{ 1.0 }, Price{ 1.0 }, TimeInForce::GTC);

    EXPECT_EQ(a.id, 1u);
    EXPECT_EQ(b.id, 2u);
    EXPECT_EQ(c.id, 3u);
}

} // namespace stonks::core
