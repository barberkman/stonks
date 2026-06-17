#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "stonks/core/types.h"
#include "stonks/datafeed/klinefeed.h"

namespace stonks::datafeed {

namespace {

using Row = KLineFeed::Row;

Row row(std::int64_t ms, std::string sym, double close)
{
    return Row{ ms, std::move(sym), close, close, close, close, core::Volume{ 1.0 } };
}

struct Bar
{
    std::int64_t ts;
    core::Symbol symbol;
    double close;
    bool operator==(const Bar&) const = default;
};

// Drive per timestamp: one group of bars per tick.
std::vector<std::vector<Bar>> drive(KLineFeed& feed)
{
    std::vector<std::vector<Bar>> groups;
    while (const auto ts = feed.next_timestamp()) {
        std::vector<Bar> g;
        for (const auto& k : feed.current_bars()) {
            g.push_back(Bar{ k.timestamp.value.time_since_epoch().count(), k.symbol, k.close });
        }
        groups.push_back(std::move(g));
        feed.advance();
    }
    return groups;
}

std::map<core::Symbol, std::vector<double>> closes_by_symbol(const core::MarketWindow& w)
{
    std::map<core::Symbol, std::vector<double>> out;
    for (const auto& s : w.series) {
        out[core::Symbol{ s.symbol }].assign(s.bars.close.begin(), s.bars.close.end());
    }
    return out;
}

// A and B print at every timestamp; disjoint close ranges.
std::vector<Row> two_symbol_rows()
{
    return {
        row(1000, "A", 100.0), row(1000, "B", 200.0),
        row(2000, "A", 101.0), row(2000, "B", 201.0),
        row(3000, "A", 102.0), row(3000, "B", 202.0),
    };
}

} // namespace

TEST(KLineFeed, IteratesOneGroupPerTimestamp)
{
    KLineFeed feed{ two_symbol_rows() };
    const auto groups = drive(feed);

    ASSERT_EQ(groups.size(), 3u);   // 3 distinct timestamps
    EXPECT_EQ(groups[0], (std::vector<Bar>{ { 1000, "A", 100.0 }, { 1000, "B", 200.0 } }));
    EXPECT_EQ(groups[2], (std::vector<Bar>{ { 3000, "A", 102.0 }, { 3000, "B", 202.0 } }));
    EXPECT_EQ(feed.size(), 6u);     // size() still counts rows (bars)
}

TEST(KLineFeed, CurrentBarsResolveInternedSymbols)
{
    KLineFeed feed{ two_symbol_rows() };
    const auto bars = feed.current_bars();
    ASSERT_EQ(bars.size(), 2u);
    EXPECT_EQ(bars[0].symbol, "A");
    EXPECT_EQ(bars[1].symbol, "B");
}

TEST(KLineFeed, WindowGivesEachPrinterItsOwnLastN)
{
    KLineFeed feed{ two_symbol_rows() };
    feed.advance();
    feed.advance();   // -> timestamp 3000

    const auto by_symbol = closes_by_symbol(feed.window(100));
    ASSERT_EQ(by_symbol.size(), 2u);
    EXPECT_EQ(by_symbol.at("A"), (std::vector<double>{ 100.0, 101.0, 102.0 }));
    EXPECT_EQ(by_symbol.at("B"), (std::vector<double>{ 200.0, 201.0, 202.0 }));
}

TEST(KLineFeed, WindowClampsToCountAndToBarsSeen)
{
    KLineFeed feed{ two_symbol_rows() };

    // First timestamp: only one bar available per symbol.
    EXPECT_EQ(closes_by_symbol(feed.window(2)).at("A"), (std::vector<double>{ 100.0 }));

    feed.advance();
    feed.advance();   // -> 3000
    EXPECT_EQ(closes_by_symbol(feed.window(2)).at("A"), (std::vector<double>{ 101.0, 102.0 }));
    EXPECT_EQ(feed.window(0).series.front().bars.size(), 0u);
}

TEST(KLineFeed, WindowNeverExceedsNow)
{
    KLineFeed feed{ two_symbol_rows() };
    while (const auto now = feed.next_timestamp()) {
        const auto now_ms = now->value.time_since_epoch().count();
        const auto w = feed.window(100);
        ASSERT_GT(w.size(), 0u);
        for (const auto& s : w.series) {
            ASSERT_GT(s.bars.size(), 0u);
            for (const auto t : s.bars.timestamp) { EXPECT_LE(t, now_ms); }
            EXPECT_EQ(s.bars.timestamp.back(), now_ms);
        }
        feed.advance();
    }
}

TEST(KLineFeed, SortsEachSymbolBlockChronologicallyEvenIfFileUnordered)
{
    KLineFeed feed{ std::vector<Row>{
        row(3000, "A", 102.0),
        row(1000, "A", 100.0),
        row(2000, "A", 101.0),
    } };

    const auto groups = drive(feed);
    ASSERT_EQ(groups.size(), 3u);
    EXPECT_EQ(groups[0], (std::vector<Bar>{ { 1000, "A", 100.0 } }));
    EXPECT_EQ(groups[1], (std::vector<Bar>{ { 2000, "A", 101.0 } }));
    EXPECT_EQ(groups[2], (std::vector<Bar>{ { 3000, "A", 102.0 } }));
}

TEST(KLineFeed, ConstructionIsDeterministic)
{
    KLineFeed a{ two_symbol_rows() };
    KLineFeed b{ two_symbol_rows() };
    EXPECT_EQ(drive(a), drive(b));
}

} // namespace stonks::datafeed
