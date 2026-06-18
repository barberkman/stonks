#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/types.h"

#include "strategies/qm_breakout.h"
#include "strategies/qm_episodic_pivot.h"
#include "strategies/qm_orb.h"
#include "strategies/qm_parabolic_short.h"
#include "strategies/qm_short_breakout.h"

#include "test_stubs.h"

namespace {

using namespace stonks::core;
using stonks::core::test::StubFeed;

constexpr std::int64_t DAY = 86'400'000;
constexpr std::int64_t HOUR = 3'600'000;
constexpr Balance EQUITY = 100'000.0;
constexpr Balance CASH = 100'000.0;

// ─── Bar builders (mirror tests/python/qm_helpers.py) ────────────────────────
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

void extend(std::vector<Row>& a, const std::vector<Row>& b) { a.insert(a.end(), b.begin(), b.end()); }

// Broker double: pins equity()/cash() (which drive risk sizing) and records every
// placed order. on_tick is a no-op, so orders are never filled — exactly like the
// Python FakeContext, the strategy tracks its own position regardless.
struct RecordingBroker
{
    Balance equity_value{};
    Balance cash_value{};
    std::vector<Order> placed;
    std::vector<Trade> m_trades;

    Balance cash() const { return cash_value; }
    Balance equity() const { return equity_value; }
    const std::vector<Trade>& trades() const { return m_trades; }
    const std::vector<Order>& orders() const { return placed; }
    bool place_order(const Order& o) { placed.push_back(o); return true; }
    void on_tick(const KLine&) {}
};

// Drive `strat` over `bars` through a real Context + StubFeed (which reproduces the
// engine's no-lookahead, per-timestamp multi-bar windows), one timestamp per tick.
template <class StrategyT>
std::vector<Order> run(StrategyT& strat, std::vector<KLine> bars,
                       Balance equity = EQUITY, Balance cash = CASH)
{
    RecordingBroker broker;
    broker.equity_value = equity;
    broker.cash_value = cash;
    StubFeed feed;
    feed.bars = std::move(bars);
    Clock clock;
    Context<RecordingBroker, StubFeed> ctx{ broker, feed, clock };
    while (auto ts = feed.next_timestamp()) {
        clock.set(*ts);
        strat.on_tick(ctx);
        feed.advance();
    }
    return broker.placed;
}

int n_side(const std::vector<Order>& os, OrderSide s)
{
    return static_cast<int>(std::count_if(os.begin(), os.end(),
        [&](const Order& o) { return o.side == s; }));
}

std::vector<Order> side_orders(const std::vector<Order>& os, OrderSide s)
{
    std::vector<Order> r;
    std::copy_if(os.begin(), os.end(), std::back_inserter(r),
        [&](const Order& o) { return o.side == s; });
    return r;
}

// ════════════════════════════════════════════════════════════════════════════
//  Shared indicators / sizing
// ════════════════════════════════════════════════════════════════════════════
TEST(QMCommon, SmaAndInsufficientHistory)
{
    std::vector<double> a{ 1, 2, 3, 4 };
    EXPECT_DOUBLE_EQ(*qm::sma(std::span<const double>{ a }, 2), 3.5);
    EXPECT_FALSE(qm::sma(std::span<const double>{ a }, 5).has_value());
}

TEST(QMCommon, EmaOfConstantSeriesIsTheConstant)
{
    std::vector<double> a(30, 5.0);
    EXPECT_DOUBLE_EQ(*qm::ema(std::span<const double>{ a }, 10), 5.0);
}

TEST(QMCommon, HighestLowest)
{
    std::vector<double> a{ 3, 1, 4, 1, 5, 9, 2 };
    EXPECT_DOUBLE_EQ(*qm::highest(std::span<const double>{ a }, 3), 9.0);
    EXPECT_DOUBLE_EQ(*qm::lowest(std::span<const double>{ a }, 3), 2.0);
}

TEST(QMCommon, AdrPctIsMeanBarRange)
{
    std::vector<double> high{ 2, 2, 2 };
    std::vector<double> low{ 1, 1, 1 };
    EXPECT_DOUBLE_EQ(*qm::adr_pct(std::span<const double>{ high }, std::span<const double>{ low }, 3), 100.0);
}

TEST(QMCommon, GainPctOverLookback)
{
    std::vector<double> close{ 100, 105, 110 };
    EXPECT_NEAR(*qm::gain_pct(std::span<const double>{ close }, 2), 10.0, 1e-9);
    EXPECT_FALSE(qm::gain_pct(std::span<const double>{ close }, 5).has_value());
}

TEST(QMCommon, SizeByRiskUsesRiskFraction)
{
    EXPECT_DOUBLE_EQ(qm::size_by_risk(100'000.0, 100'000.0, 100.0, 90.0, qm::Params{}), 50.0);
}

TEST(QMCommon, SizeByRiskCapsAtAvailableCash)
{
    // Tight stop wants 50 shares of risk, but $1000 cash caps the notional.
    EXPECT_DOUBLE_EQ(qm::size_by_risk(1'000.0, 1'000.0, 100.0, 99.9, qm::Params{}),
                     1'000.0 * 0.99 / 100.0);
}

TEST(QMCommon, SizeByRiskZeroWhenNoRisk)
{
    EXPECT_DOUBLE_EQ(qm::size_by_risk(100'000.0, 100'000.0, 100.0, 100.0, qm::Params{}), 0.0);
}

// ════════════════════════════════════════════════════════════════════════════
//  Setup 1 — momentum breakout (long)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> breakout_setup(double break_close = 51.0, double break_vol = 2000.0)
{
    auto rows = uptrend(51, 10.0, 0.8);
    for (double c : { 48.0, 47.0, 46.0, 47.0, 48.0, 47.0, 48.0, 48.0 })
        rows.push_back(bar(c, 1000.0, 0.012));
    rows.push_back(Row{ 49.0, break_close + 0.6, 48.6, break_close, break_vol });
    return rows;
}

TEST(QMBreakout, FiresOnCleanSetup)
{
    QMBreakoutStrategy s;
    const auto o = run(s, to_klines("AAA", breakout_setup()));
    EXPECT_EQ(n_side(o, OrderSide::Buy), 1);
    EXPECT_EQ(n_side(o, OrderSide::Sell), 0);
}

TEST(QMBreakout, RejectsWhenCloseBelowPivot)
{
    QMBreakoutStrategy s;
    EXPECT_TRUE(run(s, to_klines("AAA", breakout_setup(49.0))).empty());
}

TEST(QMBreakout, RejectsWithoutVolumeExpansion)
{
    QMBreakoutStrategy s;
    EXPECT_TRUE(run(s, to_klines("AAA", breakout_setup(51.0, 1000.0))).empty());
}

TEST(QMBreakout, RejectsWhenNotTrending)
{
    QMBreakoutStrategy s;
    std::vector<Row> flat(60, bar(20.0));
    EXPECT_TRUE(run(s, to_klines("AAA", flat)).empty());
}

TEST(QMBreakout, StopsOut)
{
    auto rows = breakout_setup();
    rows.push_back(bar_hl(50.0, 50.5, 49.7));  // fill bar, holds above the stop
    rows.push_back(bar_hl(48.5, 49.5, 48.0));  // low pierces the stop -> exit
    QMBreakoutStrategy s;
    const auto o = run(s, to_klines("AAA", rows));
    EXPECT_EQ(n_side(o, OrderSide::Buy), 1);
    EXPECT_GE(n_side(o, OrderSide::Sell), 1);
    EXPECT_TRUE(s.positions.empty());
}

TEST(QMBreakout, PartialThenExit)
{
    auto rows = breakout_setup();
    rows.push_back(bar_hl(51.5, 52.0, 50.8));  // fill bar
    rows.push_back(bar_hl(53.0, 53.5, 52.0));  // tags 2R target -> partial + breakeven
    for (double c : { 52.0, 51.0, 50.0, 49.0 }) rows.push_back(bar(c));  // fades back -> exit
    QMBreakoutStrategy s;
    const auto o = run(s, to_klines("AAA", rows));
    const auto buys = side_orders(o, OrderSide::Buy);
    const auto sells = side_orders(o, OrderSide::Sell);
    ASSERT_EQ(buys.size(), 1u);
    ASSERT_EQ(sells.size(), 2u);  // partial, then remainder
    EXPECT_DOUBLE_EQ(sells[0].quantity, buys[0].quantity * 0.5);
    EXPECT_TRUE(s.positions.empty());
}

// ════════════════════════════════════════════════════════════════════════════
//  Setup 1c — short breakout (breakdown)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> short_setup(double break_close = 9.5, double break_vol = 2000.0)
{
    auto rows = downtrend(51, 50.0, 0.8);
    for (double c : { 11.5, 12.0, 11.5, 11.0, 11.5, 11.0, 11.5, 11.0 })
        rows.push_back(bar(c, 1000.0, 0.012));
    rows.push_back(Row{ 11.0, 11.0, break_close - 0.4, break_close, break_vol });
    return rows;
}

TEST(QMShortBreakout, FiresOnCleanSetup)
{
    QMShortBreakoutStrategy s;
    const auto o = run(s, to_klines("AAA", short_setup()));
    EXPECT_EQ(n_side(o, OrderSide::Sell), 1);  // sell-to-open
    EXPECT_EQ(n_side(o, OrderSide::Buy), 0);
}

TEST(QMShortBreakout, RejectsWhenCloseAbovePivot)
{
    QMShortBreakoutStrategy s;
    EXPECT_TRUE(run(s, to_klines("AAA", short_setup(11.0))).empty());
}

TEST(QMShortBreakout, CoversOnStop)
{
    auto rows = short_setup();
    rows.push_back(bar_hl(9.6, 10.0, 9.3));    // fill bar, holds below the stop
    rows.push_back(bar_hl(12.0, 13.0, 11.5));  // high pierces the stop -> cover
    QMShortBreakoutStrategy s;
    const auto o = run(s, to_klines("AAA", rows));
    EXPECT_EQ(n_side(o, OrderSide::Sell), 1);
    EXPECT_GE(n_side(o, OrderSide::Buy), 1);
    EXPECT_TRUE(s.positions.empty());
}

// ════════════════════════════════════════════════════════════════════════════
//  Setup 2 — episodic pivot (gap bar, long)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> ep_setup(Row gap)
{
    auto rows = uptrend(51, 10.0, 0.2, 0.03);  // wide-range warmup so epWithin has room
    rows.push_back(gap);
    return rows;
}

TEST(QMEpisodicPivot, FiresOnGapVolumeStrongClose)
{
    QMEpisodicPivotStrategy s;
    const auto o = run(s, to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 21.4, 21.9, 2000.0 })));
    EXPECT_EQ(n_side(o, OrderSide::Buy), 1);
    EXPECT_EQ(n_side(o, OrderSide::Sell), 0);
}

TEST(QMEpisodicPivot, RejectsSmallGap)
{
    QMEpisodicPivotStrategy s;  // open only 0.25% above the prior close of 20
    EXPECT_TRUE(run(s, to_klines("AAA", ep_setup(Row{ 20.05, 22.0, 21.4, 21.9, 2000.0 }))).empty());
}

TEST(QMEpisodicPivot, RejectsWithoutVolume)
{
    QMEpisodicPivotStrategy s;
    EXPECT_TRUE(run(s, to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 21.4, 21.9, 1000.0 }))).empty());
}

TEST(QMEpisodicPivot, RejectsRiskTooWide)
{
    QMEpisodicPivotStrategy s;  // low far below the close -> risk exceeds the ADR stop distance
    EXPECT_TRUE(run(s, to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 18.0, 21.9, 2000.0 }))).empty());
}

// ════════════════════════════════════════════════════════════════════════════
//  Setup 3 — parabolic short (first red bar)
// ════════════════════════════════════════════════════════════════════════════
std::vector<Row> parabolic_base()
{
    std::vector<Row> rows(15, bar(20.0));               // flat warmup, no run-up
    for (double c : { 21.0, 23.0, 26.0, 30.0, 35.0 }) rows.push_back(bar(c));  // 5 up-closes
    rows.push_back(bar_hl(33.0, 35.5, 32.5));           // first red bar after the run
    return rows;
}

TEST(QMParabolicShort, FiresOnFirstRedBar)
{
    QMParabolicShortStrategy s;
    const auto o = run(s, to_klines("AAA", parabolic_base()));
    EXPECT_EQ(n_side(o, OrderSide::Sell), 1);  // sell-to-open
    EXPECT_EQ(n_side(o, OrderSide::Buy), 0);
}

TEST(QMParabolicShort, RejectsWithoutRunup)
{
    QMParabolicShortStrategy s;
    std::vector<Row> rows(20, bar(20.0));
    rows.push_back(bar(19.5));  // red, but no parabolic run-up
    EXPECT_TRUE(run(s, to_klines("AAA", rows)).empty());
}

TEST(QMParabolicShort, CoversOnGreenClose)
{
    auto rows = parabolic_base();
    rows.push_back(bar(32.0));  // fill bar
    rows.push_back(bar(34.0));  // green close -> cover
    QMParabolicShortStrategy s;
    const auto o = run(s, to_klines("AAA", rows));
    EXPECT_EQ(n_side(o, OrderSide::Sell), 1);
    EXPECT_EQ(n_side(o, OrderSide::Buy), 1);
    EXPECT_TRUE(s.positions.empty());
}

TEST(QMParabolicShort, CoversOnTime)
{
    auto rows = parabolic_base();
    for (double c : { 31.0, 30.0, 29.0, 28.0, 27.0, 26.0 }) rows.push_back(bar(c));  // no green/stop
    QMParabolicShortStrategy s;
    const auto o = run(s, to_klines("AAA", rows));
    EXPECT_EQ(n_side(o, OrderSide::Buy), 1);  // covered after ps_max_hold bars
    EXPECT_TRUE(s.positions.empty());
}

// ════════════════════════════════════════════════════════════════════════════
//  Setup 1b — opening range breakout (intraday only)
// ════════════════════════════════════════════════════════════════════════════
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

TEST(QMORB, ProducesNoSignalsOnDailyData)
{
    // Daily bars that DO pass the universe filter — ORB still emits nothing because
    // each day is a one-bar session (the documented no-op).
    QMORBStrategy s;
    EXPECT_TRUE(run(s, to_klines("AAA", uptrend(60))).empty());
}

TEST(QMORB, FiresOnIntradayBreakout)
{
    QMORBStrategy s;
    EXPECT_GE(n_side(run(s, intraday()), OrderSide::Buy), 1);
}

} // namespace
