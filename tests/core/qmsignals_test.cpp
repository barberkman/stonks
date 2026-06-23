#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "strategies/qmsignals.h"

#include "test_stubs.h"

namespace {

using namespace stonks::core;
using stonks::core::test::StubBroker;
using stonks::core::test::StubFeed;
using Signal = QMSignalsStrategy::Signal;

constexpr std::int64_t DAY = 86'400'000;
constexpr std::int64_t HOUR = 3'600'000;

// ─── Bar builders ─────────────────────────────────────────────────────────────
struct Row { double o, h, l, c, v; };

Row bar(double c, double vol = 1000.0, double spread = 0.01)
{
    return Row{ c, c * (1 + spread), c * (1 - spread), c, vol };
}

// Close `c` with explicit high/low (open defaults to the close).
Row bar_hl(double c, double hi, double lo, double vol = 1000.0)
{
    return Row{ c, hi, lo, c, vol };
}

std::vector<KLine> to_klines(const std::string& sym, const std::vector<Row>& rows,
                             std::int64_t start = 0, std::int64_t step = DAY)
{
    std::vector<KLine> out;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        out.push_back(KLine{ Timestamp::from_millis(start + static_cast<std::int64_t>(i) * step),
                             sym, r.o, r.h, r.l, r.c, r.v });
    }
    return out;
}

std::vector<Row> uptrend(int n, double start = 10.0, double step = 0.8, double spread = 0.01)
{
    std::vector<Row> r;
    for (int i = 0; i < n; ++i) r.push_back(bar(start + i * step, 1000.0, spread));
    return r;
}

std::vector<Row> downtrend(int n, double start = 50.0, double step = 0.8, double spread = 0.01)
{
    std::vector<Row> r;
    for (int i = 0; i < n; ++i) r.push_back(bar(start - i * step, 1000.0, spread));
    return r;
}

std::vector<KLine> intraday(int days = 6, int per_day = 5, double start = 10.0, double step = 0.5)
{
    std::vector<KLine> out;
    int k = 0;
    for (int d = 0; d < days; ++d) {
        for (int b = 0; b < per_day; ++b) {
            const double c = start + k * step;
            const std::int64_t ts = static_cast<std::int64_t>(d) * DAY + static_cast<std::int64_t>(b) * HOUR;
            out.push_back(KLine{ Timestamp::from_millis(ts), "AAA", c, c * 1.01, c * 0.99, c, 1000.0 });
            ++k;
        }
    }
    return out;
}

// Drive the scanner over `bars` through a real Context + StubFeed (no-lookahead,
// per-timestamp windows). It places no orders, so a stub broker is enough. Tests
// build their data so the bar of interest is the final one and assert on the
// signals fired on that last tick via strat.signals(sym).
void drive(QMSignalsStrategy& strat, std::vector<KLine> bars)
{
    StubBroker broker;
    StubFeed feed;
    feed.bars = std::move(bars);
    Clock clock;
    Context<StubBroker, StubFeed> ctx{ broker, feed, clock };
    while (auto ts = feed.next_timestamp()) {
        clock.set(*ts);
        strat.on_tick(ctx);
        feed.advance();
    }
}

const Signal* find(const std::vector<Signal>& sigs, std::string_view setup)
{
    const auto it = std::find_if(sigs.begin(), sigs.end(),
        [&](const Signal& s) { return s.setup == setup; });
    return it == sigs.end() ? nullptr : &*it;
}

std::vector<Signal> run(std::vector<KLine> bars)
{
    QMSignalsStrategy s;
    drive(s, std::move(bars));
    return s.last_signals("AAA");
}

// ════════════════════════════════════════════════════════════════════════════
//  breakout (long)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> breakout_setup(double break_close = 51.0, double break_vol = 2000.0)
{
    auto rows = uptrend(51, 10.0, 0.8);
    for (double c : { 48.0, 47.0, 46.0, 47.0, 48.0, 47.0, 48.0, 48.0 })
        rows.push_back(bar(c, 1000.0, 0.012));
    rows.push_back(Row{ 49.0, break_close + 0.6, 48.6, break_close, break_vol });
    return rows;
}

TEST(QMSignals, BreakoutFiresWithLevels)
{
    const auto* b = find(run(to_klines("AAA", breakout_setup())), "breakout");
    ASSERT_NE(b, nullptr);
    EXPECT_NEAR(b->entry, 50.5, 0.01);            // entry = the pivot
    EXPECT_NEAR(b->stop, 49.2923, 1e-3);
    EXPECT_NEAR(b->sell, 52.9153, 1e-3);
    EXPECT_LT(b->stop, b->entry);
    EXPECT_LT(b->entry, b->sell);
    EXPECT_DOUBLE_EQ(b->sell, b->entry + 2.0 * (b->entry - b->stop));
}

TEST(QMSignals, BreakoutSilentWhenCloseBelowPivot)
{
    EXPECT_EQ(find(run(to_klines("AAA", breakout_setup(49.0))), "breakout"), nullptr);
}

TEST(QMSignals, BreakoutSilentWithoutVolumeExpansion)
{
    EXPECT_EQ(find(run(to_klines("AAA", breakout_setup(51.0, 1000.0))), "breakout"), nullptr);
}

TEST(QMSignals, FlatSeriesProducesNoSignals)
{
    std::vector<Row> flat(60, bar(20.0));
    EXPECT_TRUE(run(to_klines("AAA", flat)).empty());
}

// ════════════════════════════════════════════════════════════════════════════
//  short_breakout (short)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> short_setup(double break_close = 9.5, double break_vol = 2000.0)
{
    auto rows = downtrend(51, 50.0, 0.8);
    for (double c : { 11.5, 12.0, 11.5, 11.0, 11.5, 11.0, 11.5, 11.0 })
        rows.push_back(bar(c, 1000.0, 0.012));
    rows.push_back(Row{ 11.0, 11.0, break_close - 0.4, break_close, break_vol });
    return rows;
}

TEST(QMSignals, ShortBreakoutFiresWithLevels)
{
    const auto* s = find(run(to_klines("AAA", short_setup())), "short_breakout");
    ASSERT_NE(s, nullptr);
    EXPECT_NEAR(s->entry, 9.9, 0.05);             // entry = the base-low pivot
    EXPECT_GT(s->stop, s->entry);                 // stop above entry (short)
    EXPECT_LT(s->sell, s->entry);                 // take-profit below entry
    EXPECT_DOUBLE_EQ(s->sell, s->entry - 2.0 * (s->stop - s->entry));
}

TEST(QMSignals, ShortBreakoutSilentWhenCloseAbovePivot)
{
    EXPECT_EQ(find(run(to_klines("AAA", short_setup(11.0))), "short_breakout"), nullptr);
}

// ════════════════════════════════════════════════════════════════════════════
//  episodic_pivot (long)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> ep_setup(Row gap)
{
    auto rows = uptrend(51, 10.0, 0.2, 0.03);  // wide-range warmup so epWithin has room
    rows.push_back(gap);
    return rows;
}

TEST(QMSignals, EpisodicPivotFiresWithLevels)
{
    const auto* e = find(run(to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 21.4, 21.9, 2000.0 }))),
                         "episodic_pivot");
    ASSERT_NE(e, nullptr);
    EXPECT_DOUBLE_EQ(e->entry, 21.9);             // entry = the gap bar's close
    EXPECT_LT(e->stop, e->entry);
    EXPECT_LT(e->entry, e->sell);
    EXPECT_DOUBLE_EQ(e->sell, e->entry + 2.0 * (e->entry - e->stop));
}

TEST(QMSignals, EpisodicPivotSilentOnSmallGap)
{
    EXPECT_EQ(find(run(to_klines("AAA", ep_setup(Row{ 20.05, 22.0, 21.4, 21.9, 2000.0 }))),
                   "episodic_pivot"), nullptr);
}

TEST(QMSignals, EpisodicPivotSilentWithoutVolume)
{
    EXPECT_EQ(find(run(to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 21.4, 21.9, 1000.0 }))),
                   "episodic_pivot"), nullptr);
}

TEST(QMSignals, EpisodicPivotSilentWhenRiskTooWide)
{
    EXPECT_EQ(find(run(to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 18.0, 21.9, 2000.0 }))),
                   "episodic_pivot"), nullptr);
}

// ════════════════════════════════════════════════════════════════════════════
//  parabolic_short (short)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> parabolic_base()
{
    std::vector<Row> rows(15, bar(20.0));               // flat warmup, no run-up
    for (double c : { 21.0, 23.0, 26.0, 30.0, 35.0 }) rows.push_back(bar(c));  // 5 up-closes
    rows.push_back(bar_hl(33.0, 35.5, 32.5));           // first red bar after the run
    return rows;
}

TEST(QMSignals, ParabolicShortFiresWithLevels)
{
    const auto* p = find(run(to_klines("AAA", parabolic_base())), "parabolic_short");
    ASSERT_NE(p, nullptr);
    EXPECT_DOUBLE_EQ(p->entry, 33.0);             // entry = the first red bar's close
    EXPECT_NEAR(p->stop, 35.5, 1e-9);             // stop = highest high of the last 3 bars
    EXPECT_LT(p->sell, p->entry);                 // synthetic 2R take-profit below entry
    EXPECT_DOUBLE_EQ(p->sell, p->entry - 2.0 * (p->stop - p->entry));
}

TEST(QMSignals, ParabolicShortSilentWithoutRunup)
{
    std::vector<Row> rows(20, bar(20.0));
    rows.push_back(bar(19.5));  // red, but no parabolic run-up
    EXPECT_EQ(find(run(to_klines("AAA", rows)), "parabolic_short"), nullptr);
}

// ════════════════════════════════════════════════════════════════════════════
//  orb (intraday long)
// ════════════════════════════════════════════════════════════════════════════
TEST(QMSignals, ORBFiresOnIntradayBreakout)
{
    const auto* o = find(run(intraday()), "orb");
    ASSERT_NE(o, nullptr);
    EXPECT_LT(o->stop, o->entry);
    EXPECT_LT(o->entry, o->sell);
    EXPECT_DOUBLE_EQ(o->sell, o->entry + 2.0 * (o->entry - o->stop));
}

TEST(QMSignals, ORBSilentOnDailyData)
{
    // Daily bars that pass the universe filter still never form a multi-bar session.
    EXPECT_EQ(find(run(to_klines("AAA", uptrend(60))), "orb"), nullptr);
}

} // namespace
