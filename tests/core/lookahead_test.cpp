#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <utility>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/engine.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::core {

namespace {

struct LookaheadProbe
{
    int* tick_count{};
    bool* saw_future{};

    void on_tick(auto& context)
    {
        const auto now_ts = context.now();
        auto bars = context.klines(Timestamp::from_millis(0),
                                   Timestamp::from_millis(10'000'000'000'000LL));
        for (const auto& b : bars) {
            if (b.timestamp > now_ts) { *saw_future = true; }
        }
        ++(*tick_count);
    }
};

} // namespace

TEST(Lookahead, StrategyNeverSeesFutureBars) {
    using namespace test;

    StubFeed feed;
    feed.bars = {
        make_bar(1000, 100.0),
        make_bar(2000, 110.0),
        make_bar(3000, 120.0),
        make_bar(4000, 130.0),
    };
    StubBroker broker;

    int tick_count = 0;
    bool saw_future = false;

    LookaheadProbe probe;
    probe.tick_count = &tick_count;
    probe.saw_future = &saw_future;

    Engine engine{ std::move(probe), std::move(feed), std::move(broker) };
    engine.run();

    EXPECT_EQ(tick_count, 4);
    EXPECT_FALSE(saw_future);
}

TEST(Lookahead, ContextClampsExplicitlyFutureEndTime) {
    using namespace test;

    StubFeed feed;
    feed.bars = {
        make_bar(1000, 100.0),
        make_bar(2000, 110.0),
        make_bar(3000, 120.0),
    };
    StubBroker broker;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(1500));
    auto bars = ctx.klines(Timestamp::from_millis(0),
                           Timestamp::from_millis(99'999'999'999LL));
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_EQ(bars[0].timestamp, Timestamp::from_millis(1000));
}

TEST(Lookahead, CountOverloadIsResolutionAware) {
    using namespace test;

    StubFeed feed;
    feed.res = std::chrono::milliseconds{ 1000 };
    feed.bars = {
        make_bar(1000, 100.0),
        make_bar(2000, 110.0),
        make_bar(3000, 120.0),
        make_bar(4000, 130.0),
        make_bar(5000, 140.0),
    };
    StubBroker broker;
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };

    clock.set(Timestamp::from_millis(5000));
    auto bars = ctx.klines(3);  // [now - 3*unit, now] = [2000, 5000], inclusive
    ASSERT_EQ(bars.size(), 4u);
    EXPECT_EQ(bars.front().timestamp, Timestamp::from_millis(2000));
    EXPECT_EQ(bars.back().timestamp, Timestamp::from_millis(5000));
}

} // namespace stonks::core
