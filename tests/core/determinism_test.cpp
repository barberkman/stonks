#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "stonks/core/engine.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::core {

namespace {

struct OneOrderPerTick
{
    void on_tick(auto& context)
    {
        auto order = context.make_limit_order({
            .symbol = Symbol{ "X" },
            .side = OrderSide::Buy,
            .quantity = Quantity{ 1.0 },
            .price = Price{ 42.0 },
        });
        context.place_order(order);
    }
};

std::vector<Order> run_once()
{
    using namespace test;

    std::vector<Order> captured;
    StubBroker broker;
    broker.placed = &captured;

    StubFeed feed;
    feed.bars = {
        make_bar(1000, 100.0),
        make_bar(2000, 110.0),
        make_bar(3000, 120.0),
        make_bar(4000, 130.0),
    };

    Engine engine{ OneOrderPerTick{}, std::move(feed), std::move(broker) };
    engine.run();
    return captured;
}

} // namespace

TEST(Determinism, IdenticalRunsProduceIdenticalOrderStreams) {
    const auto a = run_once();
    const auto b = run_once();
    ASSERT_EQ(a.size(), 4u);
    EXPECT_EQ(a, b);
}

} // namespace stonks::core
