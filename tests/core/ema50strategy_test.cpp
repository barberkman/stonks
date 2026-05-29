#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "strategies/ema50strategy.h"

namespace stonks::core {

namespace {

// Broker double that lets a test pin equity() (which drives 1%-of-equity sizing)
// and records every order the strategy places. The strategy never calls cash(),
// but the Broker concept requires it, so it mirrors equity().
struct RecordingBroker
{
    Balance equity_value{};
    std::vector<Order> placed;
    std::vector<Trade> m_trades;

    Balance cash() const { return equity_value; }
    Balance equity() const { return equity_value; }
    const std::vector<Trade>& trades() const { return m_trades; }
    bool place_order(const Order& o) { placed.push_back(o); return true; }
    void on_tick(const KLine&) {}
};

// Feed double surfacing a single "current" bar. Context::klines(1) returns this
// bar (its timestamp tracks the clock, so it survives the no-lookahead filter),
// and the strategy reads klines(1).back().
struct OneBarFeed
{
    KLine current{};
    Timestamp::duration res{ std::chrono::milliseconds{ 1000 } };

    std::optional<Timestamp> next_timestamp() const { return std::nullopt; }
    void advance() {}
    KLine current_kline() const { return current; }
    std::vector<KLine> klines(Timestamp, Timestamp) const { return { current }; }
    Timestamp::duration resolution() const { return res; }
};

KLine make_bar(std::int64_t ms, const Symbol& symbol, double close)
{
    return KLine{
        Timestamp::from_millis(ms),
        symbol,
        Price{ close },
        Price{ close },
        Price{ close },
        Price{ close },
        Volume{ 1.0 },
    };
}

// Drives EMA50Strategy through a real Context (the only minter of Orders) one
// bar at a time, with the equity in force at each tick.
struct Harness
{
    RecordingBroker broker;
    OneBarFeed feed;
    Clock clock;
    Context<RecordingBroker, OneBarFeed> ctx{ broker, feed, clock };
    EMA50Strategy strat;
    std::int64_t t{};

    void tick(const Symbol& symbol, double close, Balance equity)
    {
        broker.equity_value = equity;
        t += 1000;
        feed.current = make_bar(t, symbol, close);
        clock.set(Timestamp::from_millis(t));
        strat.on_tick(ctx);
    }

    // Feed `count` flat bars for one symbol to accumulate EMA seed samples
    // without crossing (close == EMA never triggers an entry).
    void seed(const Symbol& symbol, int count, double close, Balance equity)
    {
        for (int i = 0; i < count; ++i) { tick(symbol, close, equity); }
    }
};

constexpr Balance EQUITY = 100'000.0;

TEST(EMA50Strategy, NoOrdersWhileSeeding)
{
    Harness h;
    // 49 rising closes: price is climbing, but the EMA needs 50 samples before
    // it exists, so nothing can trade yet.
    for (int i = 0; i < 49; ++i) { h.tick("AAA", 100.0 + i, EQUITY); }
    EXPECT_TRUE(h.broker.placed.empty());
}

TEST(EMA50Strategy, EntrySizedToOnePercentOfEquity)
{
    Harness h;
    h.seed("AAA", 49, 100.0, EQUITY);
    h.tick("AAA", 101.0, EQUITY);  // 50th bar: EMA seeds to SMA 100.02; 101 > it -> enter

    ASSERT_EQ(h.broker.placed.size(), 1u);
    const auto& o = h.broker.placed.front();
    EXPECT_EQ(o.symbol, "AAA");
    EXPECT_EQ(o.side, OrderSide::Buy);
    EXPECT_EQ(o.type, OrderType::Market);
    EXPECT_DOUBLE_EQ(o.quantity, EQUITY * 0.01 / 101.0);
}

TEST(EMA50Strategy, OnePositionPerSymbol)
{
    Harness h;
    h.seed("AAA", 49, 100.0, EQUITY);
    h.tick("AAA", 101.0, EQUITY);  // enter
    h.tick("AAA", 102.0, EQUITY);  // still above EMA, but already long

    // A second above-EMA bar must not stack another position.
    ASSERT_EQ(h.broker.placed.size(), 1u);
    EXPECT_EQ(h.broker.placed.front().side, OrderSide::Buy);
}

TEST(EMA50Strategy, ExitSellsFullPositionThenReenters)
{
    Harness h;
    h.seed("AAA", 49, 100.0, EQUITY);
    h.tick("AAA", 101.0, EQUITY);   // enter
    h.tick("AAA", 1.0, EQUITY);     // far below EMA -> exit
    h.tick("AAA", 1000.0, EQUITY);  // back above EMA -> re-enter

    ASSERT_EQ(h.broker.placed.size(), 3u);
    EXPECT_EQ(h.broker.placed[0].side, OrderSide::Buy);
    EXPECT_EQ(h.broker.placed[1].side, OrderSide::Sell);
    // The exit liquidates exactly the quantity held from the entry.
    EXPECT_DOUBLE_EQ(h.broker.placed[1].quantity, h.broker.placed[0].quantity);
    EXPECT_EQ(h.broker.placed[2].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(h.broker.placed[2].quantity, EQUITY * 0.01 / 1000.0);
}

TEST(EMA50Strategy, ConcurrentPositionsAcrossSymbols)
{
    Harness h;
    // Seed two symbols in parallel, then cross both. 1% sizing leaves equity for
    // both, and the per-symbol state keeps them independent.
    for (int i = 0; i < 49; ++i) {
        h.tick("AAA", 100.0, EQUITY);
        h.tick("BBB", 200.0, EQUITY);
    }
    h.tick("AAA", 101.0, EQUITY);  // AAA enters
    h.tick("BBB", 202.0, EQUITY);  // BBB enters independently (AAA still held)

    ASSERT_EQ(h.broker.placed.size(), 2u);
    EXPECT_EQ(h.broker.placed[0].symbol, "AAA");
    EXPECT_EQ(h.broker.placed[0].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(h.broker.placed[0].quantity, EQUITY * 0.01 / 101.0);
    EXPECT_EQ(h.broker.placed[1].symbol, "BBB");
    EXPECT_EQ(h.broker.placed[1].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(h.broker.placed[1].quantity, EQUITY * 0.01 / 202.0);
}

TEST(EMA50Strategy, SizingTracksCurrentEquity)
{
    Harness h;
    // Identical price paths, different equity in force at each entry. Sizing off
    // a fixed initial capital would make the quantities equal; sizing off live
    // equity makes them differ in proportion to it.
    for (int i = 0; i < 49; ++i) {
        h.tick("AAA", 100.0, EQUITY);
        h.tick("BBB", 100.0, EQUITY);
    }
    h.tick("AAA", 101.0, 100'000.0);  // entry at equity 100k
    h.tick("BBB", 101.0, 300'000.0);  // entry at equity 300k

    ASSERT_EQ(h.broker.placed.size(), 2u);
    EXPECT_DOUBLE_EQ(h.broker.placed[0].quantity, 100'000.0 * 0.01 / 101.0);
    EXPECT_DOUBLE_EQ(h.broker.placed[1].quantity, 300'000.0 * 0.01 / 101.0);
}

} // namespace

} // namespace stonks::core
