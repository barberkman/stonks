#include <vector>

#include <gtest/gtest.h>

#include "core/test_stubs.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"
#include "stonks/python/contextadapter.h"
#include "stonks/python/icontext.h"

namespace {

using namespace stonks;
using core::test::StubBroker;
using core::test::StubFeed;
using core::test::make_bar;

TEST(ContextAdapter, ForwardsNowCashEquity)
{
    StubFeed feed;
    StubBroker broker;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    python::ContextAdapter<StubBroker, StubFeed> adapter{ ctx };

    clock.set(core::Timestamp::from_millis(12345));

    EXPECT_EQ(adapter.now().value, core::Timestamp::from_millis(12345).value);
    EXPECT_EQ(adapter.cash(), broker.cash());
    EXPECT_EQ(adapter.equity(), broker.equity());
}

TEST(ContextAdapter, KlinesCountForwardsToContext)
{
    StubFeed feed;
    feed.bars = { make_bar(1000, 100.0), make_bar(2000, 101.0), make_bar(3000, 102.0) };
    StubBroker broker;
    core::Clock clock;
    clock.set(core::Timestamp::from_millis(3000));
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    python::ContextAdapter<StubBroker, StubFeed> adapter{ ctx };

    const auto adapted = adapter.klines_count(2);
    const auto direct = ctx.klines(2);

    ASSERT_EQ(adapted.size(), direct.size());
    ASSERT_FALSE(adapted.empty());
    EXPECT_EQ(adapted.back().close, direct.back().close);
}

TEST(ContextAdapter, KlinesRangeForwardsToContext)
{
    StubFeed feed;
    feed.bars = { make_bar(1000, 100.0), make_bar(2000, 101.0), make_bar(3000, 102.0) };
    StubBroker broker;
    core::Clock clock;
    clock.set(core::Timestamp::from_millis(3000));
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    python::ContextAdapter<StubBroker, StubFeed> adapter{ ctx };

    const auto adapted = adapter.klines_range(
        core::Timestamp::from_millis(2000),
        core::Timestamp::from_millis(3000));
    const auto direct = ctx.klines(
        core::Timestamp::from_millis(2000),
        core::Timestamp::from_millis(3000));

    ASSERT_EQ(adapted.size(), direct.size());
    ASSERT_FALSE(adapted.empty());
    EXPECT_EQ(adapted.back().close, direct.back().close);
}

TEST(ContextAdapter, PlaceMarketOrderBuildsAndForwards)
{
    StubFeed feed;
    StubBroker broker;
    std::vector<core::Order> placed;
    broker.placed = &placed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    python::ContextAdapter<StubBroker, StubFeed> adapter{ ctx };

    const bool ok = adapter.place_market_order(core::MarketOrderParams{
        .symbol = "TEST",
        .side = core::OrderSide::Buy,
        .quantity = 1.5,
        .time_in_force = core::TimeInForce::GTC,
    });

    EXPECT_TRUE(ok);
    ASSERT_EQ(placed.size(), 1u);
    EXPECT_EQ(placed[0].symbol, "TEST");
    EXPECT_EQ(placed[0].side, core::OrderSide::Buy);
    EXPECT_EQ(placed[0].quantity, 1.5);
    EXPECT_EQ(placed[0].type, core::OrderType::Market);
    EXPECT_FALSE(placed[0].price.has_value());
}

TEST(ContextAdapter, PlaceLimitOrderBuildsAndForwards)
{
    StubFeed feed;
    StubBroker broker;
    std::vector<core::Order> placed;
    broker.placed = &placed;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    python::ContextAdapter<StubBroker, StubFeed> adapter{ ctx };

    const bool ok = adapter.place_limit_order(core::LimitOrderParams{
        .symbol = "TEST",
        .side = core::OrderSide::Sell,
        .quantity = 2.5,
        .price = 99.5,
        .time_in_force = core::TimeInForce::GTC,
    });

    EXPECT_TRUE(ok);
    ASSERT_EQ(placed.size(), 1u);
    EXPECT_EQ(placed[0].symbol, "TEST");
    EXPECT_EQ(placed[0].side, core::OrderSide::Sell);
    EXPECT_EQ(placed[0].quantity, 2.5);
    EXPECT_EQ(placed[0].type, core::OrderType::Limit);
    ASSERT_TRUE(placed[0].price.has_value());
    EXPECT_EQ(*placed[0].price, 99.5);
}

TEST(ContextAdapter, MakeAdapterDeducesTemplateArgs)
{
    StubFeed feed;
    StubBroker broker;
    core::Clock clock;
    core::Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    std::unique_ptr<python::IContext> adapter = python::make_adapter(ctx);

    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->cash(), broker.cash());
}

} // namespace
