#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "stonks/core/context.h"
#include "stonks/core/engine.h"
#include "stonks/core/types.h"

#include "test_stubs.h"

namespace stonks::core {

namespace {

struct SymbolCols
{
    std::vector<std::int64_t> ts;
    std::vector<double> close;
};

// One tick's window, captured per symbol so tests can assert after the run.
struct Capture
{
    std::int64_t now_ms;
    std::map<Symbol, SymbolCols> by_symbol;
};

struct RecordingProbe
{
    std::vector<Capture>* log{};
    int n{};

    void on_tick(auto& ctx)
    {
        Capture c;
        c.now_ms = ctx.now().value.time_since_epoch().count();
        for (const auto& s : ctx.history(n).series) {
            SymbolCols cols;
            cols.ts.assign(s.bars.timestamp.begin(), s.bars.timestamp.end());
            cols.close.assign(s.bars.close.begin(), s.bars.close.end());
            c.by_symbol[Symbol{ s.symbol }] = std::move(cols);
        }
        log->push_back(std::move(c));
    }
};

// A and B print at every timestamp; disjoint close ranges (A 100s, B 200s).
std::vector<KLine> two_symbol_bars()
{
    using test::make_bar;
    return {
        make_bar(1000, "A", 100.0), make_bar(1000, "B", 200.0),
        make_bar(2000, "A", 101.0), make_bar(2000, "B", 201.0),
        make_bar(3000, "A", 102.0), make_bar(3000, "B", 202.0),
    };
}

std::vector<Capture> run_capture(int n)
{
    using namespace test;
    std::vector<Capture> log;
    StubFeed feed;
    feed.bars = two_symbol_bars();
    StubBroker broker;
    Engine engine{ RecordingProbe{ &log, n }, std::move(feed), std::move(broker) };
    engine.run();
    return log;
}

} // namespace

TEST(Lookahead, WindowHoldsTodaysPrintersAndNeverTheFuture)
{
    const auto log = run_capture(/*n=*/100);
    ASSERT_EQ(log.size(), 3u);   // one tick per timestamp, not per (symbol,bar)

    for (const auto& c : log) {
        ASSERT_EQ(c.by_symbol.size(), 2u);                 // both symbols printed
        ASSERT_TRUE(c.by_symbol.count("A") && c.by_symbol.count("B"));
        for (const auto& [sym, cols] : c.by_symbol) {
            ASSERT_FALSE(cols.ts.empty());
            for (const auto t : cols.ts) { EXPECT_LE(t, c.now_ms); }   // no future
            EXPECT_EQ(cols.ts.back(), c.now_ms);                       // last is today's
            for (std::size_t i = 1; i < cols.ts.size(); ++i) {
                EXPECT_LT(cols.ts[i - 1], cols.ts[i]);                 // chronological
            }
        }
    }
}

TEST(Lookahead, WindowReturnsPerSymbolHistoryInOrder)
{
    const auto log = run_capture(/*n=*/100);
    const auto& last = log.back();   // now = 3000
    EXPECT_EQ(last.now_ms, 3000);
    EXPECT_EQ(last.by_symbol.at("A").ts, (std::vector<std::int64_t>{ 1000, 2000, 3000 }));
    EXPECT_EQ(last.by_symbol.at("A").close, (std::vector<double>{ 100.0, 101.0, 102.0 }));
    EXPECT_EQ(last.by_symbol.at("B").close, (std::vector<double>{ 200.0, 201.0, 202.0 }));
}

TEST(Lookahead, WindowClampsToCountAndAvailableBars)
{
    const auto log = run_capture(/*n=*/2);
    EXPECT_EQ(log.back().by_symbol.at("A").ts, (std::vector<std::int64_t>{ 2000, 3000 }));
    EXPECT_EQ(log.front().by_symbol.at("A").ts, (std::vector<std::int64_t>{ 1000 }));
}

} // namespace stonks::core
