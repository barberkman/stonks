#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
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
    std::vector<Order> placed;          // strategy-intent capture, in placement order
    std::unordered_map<TradeID, Trade> m_trades;
    std::unordered_map<OrderID, Order> m_orders;
    OrderID next_id{ 1 };

    Balance cash() const { return equity_value; }
    Balance equity() const { return equity_value; }
    const std::unordered_map<TradeID, Trade>& trades() const { return m_trades; }
    const std::unordered_map<OrderID, Order>& orders() const { return m_orders; }

    OrderID place_order(const MarketOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Market, OrderStatus::Open,
                             std::nullopt, p.quantity, p.time_in_force });
    }
    OrderID place_order(const LimitOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Limit, OrderStatus::Open,
                             p.price, p.quantity, p.time_in_force });
    }
    OrderID place_order(const StopOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Stop, OrderStatus::Open,
                             p.price, p.quantity, p.time_in_force });
    }
    std::optional<Position> position(const Symbol&) const { return std::nullopt; }
    bool cancel_order(OrderID) { return false; }
    void on_tick(const KLine&) {}

private:
    OrderID record(Order o)
    {
        const OrderID id = o.id;
        placed.push_back(o);
        m_orders.try_emplace(id, std::move(o));
        ++next_id;
        return id;
    }
};

// Feed double presenting one timestamp's window. The strategy reads it via
// Context::history(n); set_bars() loads the symbols printing this tick, backed
// by member columns so the SeriesView spans stay valid for the call.
struct WindowFeed
{
    Timestamp::duration res{ std::chrono::milliseconds{ 1000 } };

    std::vector<Symbol> syms;
    std::vector<std::int64_t> ts;
    std::vector<double> open, high, low, close, volume;

    void set_bars(const std::vector<KLine>& bars)
    {
        syms.clear(); ts.clear(); open.clear(); high.clear();
        low.clear(); close.clear(); volume.clear();
        for (const auto& b : bars) {
            syms.push_back(b.symbol);
            ts.push_back(b.timestamp.value.time_since_epoch().count());
            open.push_back(b.open); high.push_back(b.high); low.push_back(b.low);
            close.push_back(b.close); volume.push_back(b.volume);
        }
    }

    std::optional<Timestamp> next_timestamp() const { return std::nullopt; }
    void advance() {}

    std::vector<KLine> current_bars() const
    {
        std::vector<KLine> out;
        for (std::size_t i = 0; i < syms.size(); ++i) {
            out.push_back(KLine{ Timestamp::from_millis(ts[i]), syms[i],
                                 open[i], high[i], low[i], close[i], volume[i] });
        }
        return out;
    }

    MarketWindow window(int count) const
    {
        MarketWindow w;
        const std::size_t cnt = (count <= 0) ? 0u : 1u;   // one bar per symbol this tick
        for (std::size_t i = 0; i < syms.size(); ++i) {
            w.series.push_back(SymbolSeries{
                std::string_view{ syms[i] },
                SeriesView{
                    std::span<const std::int64_t>{ &ts[i], cnt },
                    std::span<const double>{ &open[i], cnt },
                    std::span<const double>{ &high[i], cnt },
                    std::span<const double>{ &low[i], cnt },
                    std::span<const double>{ &close[i], cnt },
                    std::span<const double>{ &volume[i], cnt },
                },
            });
        }
        return w;
    }

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
    WindowFeed feed;
    Clock clock;
    Context<RecordingBroker, WindowFeed> ctx{ broker, feed, clock };
    EMA50Strategy strat;
    std::int64_t t{};

    // One timestamp tick with the given printing symbols (their timestamps are
    // stamped to this tick's clock).
    void tick_at(std::vector<KLine> bars, Balance equity)
    {
        broker.equity_value = equity;
        t += 1000;
        for (auto& b : bars) { b.timestamp = Timestamp::from_millis(t); }
        feed.set_bars(bars);
        clock.set(Timestamp::from_millis(t));
        strat.on_tick(ctx);
    }

    void tick(const Symbol& symbol, double close, Balance equity)
    {
        tick_at({ make_bar(t, symbol, close) }, equity);
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

TEST(EMA50Strategy, ProcessesEverySymbolPrintingInOneTick)
{
    Harness h;
    // AAA and BBB print together at every timestamp; the strategy must handle
    // both within a single on_tick (per-timestamp) call.
    for (int i = 0; i < 49; ++i) {
        h.tick_at({ make_bar(0, "AAA", 100.0), make_bar(0, "BBB", 200.0) }, EQUITY);
    }
    // 50th bar for both: each crosses above its just-seeded EMA in the same tick.
    h.tick_at({ make_bar(0, "AAA", 101.0), make_bar(0, "BBB", 202.0) }, EQUITY);

    ASSERT_EQ(h.broker.placed.size(), 2u);
    std::vector<Symbol> syms{ h.broker.placed[0].symbol, h.broker.placed[1].symbol };
    EXPECT_NE(std::find(syms.begin(), syms.end(), "AAA"), syms.end());
    EXPECT_NE(std::find(syms.begin(), syms.end(), "BBB"), syms.end());
    for (const auto& o : h.broker.placed) { EXPECT_EQ(o.side, OrderSide::Buy); }
}

} // namespace

} // namespace stonks::core
