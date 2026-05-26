#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <type_traits>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::core {

static_assert(!std::is_default_constructible_v<Order>);
static_assert(!std::is_constructible_v<Order,
    OrderID, Timestamp, Symbol, OrderSide, OrderType, std::optional<Price>, Quantity, TimeInForce>);
static_assert(std::is_copy_constructible_v<Order>);
static_assert(std::is_move_constructible_v<Order>);

TEST(OrderFactory, LimitFactoryStampsClockTimeOnConstruction) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(1000));
    const auto o1 = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
        .quantity = Quantity{ 1.0 }, .price = Price{ 100.0 },
    });
    EXPECT_EQ(o1.timestamp, Timestamp::from_millis(1000));

    clock.set(Timestamp::from_millis(5000));
    const auto o2 = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Sell,
        .quantity = Quantity{ 0.5 }, .price = Price{ 110.0 },
    });
    EXPECT_EQ(o2.timestamp, Timestamp::from_millis(5000));
}

TEST(OrderFactory, AssignsStrictlyIncreasingIds) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    const auto a = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
        .quantity = Quantity{ 1.0 }, .price = Price{ 1.0 },
    });
    const auto b = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
        .quantity = Quantity{ 1.0 }, .price = Price{ 1.0 },
    });
    const auto c = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Sell,
        .quantity = Quantity{ 1.0 }, .price = Price{ 1.0 },
    });

    EXPECT_EQ(a.id, 1u);
    EXPECT_EQ(b.id, 2u);
    EXPECT_EQ(c.id, 3u);
}

TEST(OrderFactory, MarketOrderHasNoPrice) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(2000));
    const auto o = ctx.make_market_order({
        .symbol = Symbol{ "X" },
        .side = OrderSide::Buy,
        .quantity = Quantity{ 1.5 },
    });

    EXPECT_EQ(o.type, OrderType::Market);
    EXPECT_FALSE(o.price.has_value());
    EXPECT_EQ(o.symbol, "X");
    EXPECT_EQ(o.side, OrderSide::Buy);
    EXPECT_EQ(o.quantity, Quantity{ 1.5 });
    EXPECT_EQ(o.time_in_force, TimeInForce::GTC);
    EXPECT_EQ(o.timestamp, Timestamp::from_millis(2000));
}

TEST(OrderFactory, LimitOrderCarriesPrice) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    const auto o = ctx.make_limit_order({
        .symbol = Symbol{ "X" },
        .side = OrderSide::Sell,
        .quantity = Quantity{ 2.0 },
        .price = Price{ 99.5 },
    });

    EXPECT_EQ(o.type, OrderType::Limit);
    ASSERT_TRUE(o.price.has_value());
    EXPECT_EQ(*o.price, Price{ 99.5 });
    EXPECT_EQ(o.side, OrderSide::Sell);
}

TEST(OrderFactory, MarketAndLimitShareIDSequence) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    const auto a = ctx.make_market_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
    });
    const auto b = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy,
        .quantity = Quantity{ 1.0 }, .price = Price{ 50.0 },
    });
    const auto c = ctx.make_market_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Sell, .quantity = Quantity{ 1.0 },
    });
    const auto d = ctx.make_limit_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Sell,
        .quantity = Quantity{ 1.0 }, .price = Price{ 51.0 },
    });

    EXPECT_EQ(a.id, 1u);
    EXPECT_EQ(b.id, 2u);
    EXPECT_EQ(c.id, 3u);
    EXPECT_EQ(d.id, 4u);
}

TEST(OrderFactory, MarketFactoryStampsClockTimeOnConstruction) {
    using namespace test;

    StubBroker broker;
    StubFeed feed;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(3000));
    const auto o = ctx.make_market_order({
        .symbol = Symbol{ "X" }, .side = OrderSide::Buy, .quantity = Quantity{ 1.0 },
    });
    EXPECT_EQ(o.timestamp, Timestamp::from_millis(3000));
}

} // namespace stonks::core
