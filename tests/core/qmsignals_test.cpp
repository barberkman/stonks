#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "stonks/broker/backtestbroker.h"
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

// Broker double with a settable equity (StubBroker's is always 0, which zeroes
// the risk-based quantity) that records every placed order for inspection.
struct SizingBroker
{
    Balance equity_value{ 0.0 };
    std::vector<Order>* placed{ nullptr };
    std::unordered_map<TradeID, Trade> m_trades;
    std::unordered_map<OrderID, Order> m_orders;
    std::unordered_map<Symbol, Position> fake_positions;   // test-settable; position() reads it
    OrderID next_id{ 1 };

    Balance cash() const { return equity_value; }
    Balance equity() const { return equity_value; }
    const std::unordered_map<TradeID, Trade>& trades() const { return m_trades; }
    const std::unordered_map<OrderID, Order>& orders() const { return m_orders; }

    std::optional<Position> position(const Symbol& symbol) const
    {
        const auto it = fake_positions.find(symbol);
        if (it == fake_positions.end()) { return std::nullopt; }
        return it->second;
    }

    bool cancel_order(OrderID id)
    {
        const auto it = m_orders.find(id);
        if (it == m_orders.end() || it->second.status != OrderStatus::Open) { return false; }
        it->second.status = OrderStatus::Cancelled;
        for (auto& [oid, o] : m_orders) {   // cascade to dormant children, like the broker
            if (o.parent_id == id && o.status == OrderStatus::Open) { o.status = OrderStatus::Cancelled; }
        }
        return true;
    }

    OrderID place_order(const MarketOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Market, OrderStatus::Open,
                             std::nullopt, p.quantity, p.time_in_force, p.leverage, p.reduce_only });
    }
    OrderID place_order(const LimitOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Limit, OrderStatus::Open,
                             p.price, p.quantity, p.time_in_force, p.leverage, p.reduce_only });
    }
    OrderID place_order(const StopOrderParams& p, std::optional<OrderID> parent = std::nullopt)
    {
        return record(Order{ next_id, parent, Timestamp{}, p.symbol, p.side,
                             OrderType::Stop, OrderStatus::Open,
                             p.price, p.quantity, p.time_in_force, p.leverage, p.reduce_only });
    }
    void on_tick(const KLine&) {}

private:
    OrderID record(Order o)
    {
        const OrderID id = o.id;
        if (placed) { placed->push_back(o); }
        m_orders.try_emplace(id, std::move(o));
        ++next_id;
        return id;
    }
};

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
    const auto sigs = run(to_klines("AAA", breakout_setup()));
    const auto* b = find(sigs, "breakout");
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
    const auto sigs = run(to_klines("AAA", short_setup()));
    const auto* s = find(sigs, "short_breakout");
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
    const auto sigs = run(to_klines("AAA", ep_setup(Row{ 21.0, 22.0, 21.4, 21.9, 2000.0 })));
    const auto* e = find(sigs, "episodic_pivot");
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
    const auto sigs = run(to_klines("AAA", parabolic_base()));
    const auto* p = find(sigs, "parabolic_short");
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
    const auto sigs = run(intraday());
    const auto* o = find(sigs, "orb");
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

// ════════════════════════════════════════════════════════════════════════════
//  entry_leverage (formulas §9: max isolated leverage, floored below the stop)
// ════════════════════════════════════════════════════════════════════════════
TEST(QMSignalsLeverage, LongExactIntegerStepsDownBelowStop)
{
    QMSignalsStrategy s;
    // Lmax = 100/(100-95) = 20 exactly -> step one below so liquidation stays under the stop.
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 95.0, true), 19.0);
}

TEST(QMSignalsLeverage, LongNonIntegerFloors)
{
    QMSignalsStrategy s;
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 94.0, true), 16.0);   // Lmax = 16.67 -> 16
}

TEST(QMSignalsLeverage, ShortMirrorsLong)
{
    QMSignalsStrategy s;
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 106.0, false), 16.0); // Lmax = 100/6 -> 16
}

TEST(QMSignalsLeverage, CapsAtMaxLeverage)
{
    QMSignalsStrategy s;
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 99.5, true), 125.0);  // Lmax = 200, capped
}

TEST(QMSignalsLeverage, ClampsToOneForWideStop)
{
    QMSignalsStrategy s;
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 40.0, true), 1.0);    // Lmax = 1.67 -> floor 1
}

TEST(QMSignalsLeverage, MaintenanceMarginLowersLeverage)
{
    QMSignalsStrategy s;
    s.maint_margin = 0.004;                                        // denom 5.38 -> 18.59 -> 18
    EXPECT_DOUBLE_EQ(s.entry_leverage(100.0, 95.0, true), 18.0);
}

// Drive the strategy over `bars` against a SizingBroker, invoking `pre_tick`
// before each strategy tick so tests can toggle fake positions mid-run.
template <class PreTick>
std::vector<Order> drive_sized(QMSignalsStrategy& strat, std::vector<KLine> bars,
                               SizingBroker& broker, PreTick&& pre_tick)
{
    std::vector<Order> placed;
    broker.placed = &placed;
    StubFeed feed;
    feed.bars = std::move(bars);
    Clock clock;
    Context<SizingBroker, StubFeed> ctx{ broker, feed, clock };
    int tick = 0;
    while (auto ts = feed.next_timestamp()) {
        clock.set(*ts);
        pre_tick(tick, broker);
        strat.on_tick(ctx);
        feed.advance();
        ++tick;
    }
    return placed;
}

// ════════════════════════════════════════════════════════════════════════════
//  P8 gating: one trade at a time, cooldown, pending-entry replacement
// ════════════════════════════════════════════════════════════════════════════
TEST(QMSignalsGating, DoesNotFireWhilePositionIsOpen)
{
    QMSignalsStrategy strat;
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    broker.fake_positions["AAA"] = Position{ 5.0, 48.0, 1, 10.0 };   // already long
    const auto placed = drive_sized(strat, to_klines("AAA", breakout_setup()), broker,
                                    [](int, SizingBroker&) {});
    EXPECT_TRUE(placed.empty());                      // in a trade -> no new bracket
    EXPECT_FALSE(strat.last_signals("AAA").empty());  // though the scanner did fire
}

TEST(QMSignalsGating, OppositeSideSignalSuppressedNotNettedWhileInATrade)
{
    // A LONG breakout fires while we hold a SHORT: the gate must suppress it
    // rather than let the "entry" silently net out the open position.
    QMSignalsStrategy strat;
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    broker.fake_positions["AAA"] = Position{ -5.0, 55.0, 1, 10.0 };   // short
    const auto placed = drive_sized(strat, to_klines("AAA", breakout_setup()), broker,
                                    [](int, SizingBroker&) {});
    EXPECT_TRUE(placed.empty());
}

TEST(QMSignalsGating, CooldownAfterCloseSuppressesTheSignal)
{
    // Position held from the start through tick 56 (which also gates the
    // fixture's incidental parabolic_short at tick 51), gone at 57 ->
    // cooldown(5) starts there and still has 3 bars left on the breakout
    // signal bar (59): suppressed.
    QMSignalsStrategy strat;
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    const auto placed = drive_sized(strat, to_klines("AAA", breakout_setup()), broker,
        [](int tick, SizingBroker& b) {
            if (tick <= 56) { b.fake_positions["AAA"] = Position{ 5.0, 48.0, 1, 10.0 }; }
            else { b.fake_positions.erase("AAA"); }
        });
    EXPECT_TRUE(placed.empty());
}

TEST(QMSignalsGating, FiresAgainAfterCooldownElapses)
{
    // Same shape, but the close comes early and the cooldown is short: by the
    // breakout signal bar (59) the cooldown has run out and the bracket goes out.
    QMSignalsStrategy strat;
    strat.cooldown_bars = 1;
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    const auto placed = drive_sized(strat, to_klines("AAA", breakout_setup()), broker,
        [](int tick, SizingBroker& b) {
            if (tick <= 51) { b.fake_positions["AAA"] = Position{ 5.0, 48.0, 1, 10.0 }; }
            else { b.fake_positions.erase("AAA"); }
        });
    EXPECT_EQ(placed.size(), 3u);
}

TEST(QMSignalsGating, ReplacesStalePendingEntryOnNewSignal)
{
    QMSignalsStrategy strat;
    strat.min_base_days = 0;                          // let the very next bar re-qualify
    strat.ps_min_gain = 1e9;                          // silence the fixture's tick-51 parabolic_short
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    auto rows = breakout_setup();
    rows.push_back(Row{ 51.2, 52.4, 51.0, 52.0, 2000.0 });   // second breakout bar
    const auto placed = drive_sized(strat, to_klines("AAA", rows), broker,
                                    [](int, SizingBroker&) {});

    ASSERT_EQ(placed.size(), 6u);                     // two full brackets
    const auto& entry1 = placed[0];
    const auto& entry2 = placed[3];
    // The unfilled first bracket was cancelled wholesale before the second went out.
    EXPECT_EQ(broker.m_orders.at(entry1.id).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.m_orders.at(placed[1].id).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.m_orders.at(placed[2].id).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.m_orders.at(entry2.id).status, OrderStatus::Open);
    ASSERT_TRUE(entry1.price.has_value());
    ASSERT_TRUE(entry2.price.has_value());
    EXPECT_NE(*entry1.price, *entry2.price);          // fresh levels, not a duplicate
}

TEST(QMSignalsGating, ReanchorsBracketWhenEntryFillGapsPastThePlan)
{
    // The planned levels come from the signal; the entry stop then gaps and
    // fills above them. The strategy must observe the actual fill via
    // ctx.position() and replace both legs at plan × (fill / planned_entry) —
    // the proportional shift that keeps the stop inside the liquidation price.
    QMSignalsStrategy strat;
    strat.ps_min_gain = 1e9;                       // silence the fixture's tick-51 parabolic_short
    stonks::broker::BacktestBroker broker{ Balance{ 10'000.0 } };
    auto rows = breakout_setup();
    rows.push_back(Row{ 52.0, 52.5, 51.5, 52.2, 1000.0 });   // gaps open above the 50.5 trigger -> fills @52
    rows.push_back(Row{ 52.2, 52.4, 52.0, 52.1, 1000.0 });   // quiet bar: the strategy reacts to the fill
    StubFeed feed;
    feed.bars = to_klines("AAA", rows);
    Clock clock;
    Context<stonks::broker::BacktestBroker, StubFeed> ctx{ broker, feed, clock };
    while (auto ts = feed.next_timestamp()) {
        clock.set(*ts);
        for (const auto& b : feed.current_bars()) { broker.on_tick(b); }
        strat.on_tick(ctx);
        feed.advance();
    }

    // Planned levels, recomputed from the signal fixture alone.
    const auto planned = run(to_klines("AAA", breakout_setup()));
    const auto* b = find(planned, "breakout");
    ASSERT_NE(b, nullptr);
    const double ratio = 52.0 / b->entry;
    ASSERT_GT(ratio, 1.0);

    std::vector<Order> all;
    for (const auto& [id, o] : broker.orders()) { all.push_back(o); }
    std::ranges::sort(all, {}, &Order::id);
    ASSERT_EQ(all.size(), 5u);                     // entry + original legs + re-anchored legs
    const auto& entry = all[0];
    EXPECT_EQ(entry.status, OrderStatus::Filled);
    ASSERT_TRUE(broker.position("AAA").has_value());
    EXPECT_DOUBLE_EQ(broker.position("AAA")->price, 52.0);   // the gapped fill

    EXPECT_EQ(all[1].status, OrderStatus::Cancelled);        // original SL replaced
    EXPECT_EQ(all[2].status, OrderStatus::Cancelled);        // original TP replaced
    const auto& new_sl = all[3];
    const auto& new_tp = all[4];
    EXPECT_EQ(new_sl.type, OrderType::Stop);
    EXPECT_EQ(new_tp.type, OrderType::Limit);
    EXPECT_TRUE(new_sl.reduce_only);
    EXPECT_TRUE(new_tp.reduce_only);
    ASSERT_TRUE(new_sl.parent_id.has_value());
    EXPECT_EQ(*new_sl.parent_id, entry.id);
    ASSERT_TRUE(new_sl.price.has_value());
    ASSERT_TRUE(new_tp.price.has_value());
    EXPECT_NEAR(*new_sl.price, b->stop * ratio, 1e-9);
    EXPECT_NEAR(*new_tp.price, b->sell * ratio, 1e-9);
    EXPECT_EQ(new_sl.status, OrderStatus::Open);             // live protection at the new levels
    EXPECT_EQ(new_tp.status, OrderStatus::Open);
}

TEST(QMSignals, EntryOrderCarriesRiskQuantityAndComputedLeverage)
{
    QMSignalsStrategy strat;
    std::vector<Order> placed;
    SizingBroker broker;
    broker.equity_value = 100'000.0;
    broker.placed = &placed;
    StubFeed feed;
    feed.bars = to_klines("AAA", breakout_setup());
    Clock clock;
    Context<SizingBroker, StubFeed> ctx{ broker, feed, clock };
    while (auto ts = feed.next_timestamp()) {
        clock.set(*ts);
        strat.on_tick(ctx);
        feed.advance();
    }

    // The breakout fires on the final bar; its 3 legs are the last placed orders.
    ASSERT_GE(placed.size(), 3u);
    const auto& entry = placed[placed.size() - 3];
    const auto& stop = placed[placed.size() - 2];
    const auto& tp = placed[placed.size() - 1];

    const auto sigs = strat.last_signals("AAA");
    const auto* b = find(sigs, "breakout");
    ASSERT_NE(b, nullptr);
    ASSERT_EQ(sigs.front().setup, "breakout");   // the long setup that drove these orders

    const double expected_qty = 100'000.0 * strat.risk_fraction / std::abs(b->entry - b->stop);
    const double expected_lev = strat.entry_leverage(b->entry, b->stop, true);
    ASSERT_GT(expected_lev, 1.0);                // the fixture's stop is tight enough to leverage

    // Entry: risk-sized quantity, stop order at the signal price, computed isolated leverage.
    EXPECT_EQ(entry.type, OrderType::Stop);
    EXPECT_EQ(entry.side, OrderSide::Buy);
    ASSERT_TRUE(entry.price.has_value());
    EXPECT_DOUBLE_EQ(*entry.price, b->entry);
    EXPECT_NEAR(entry.quantity, expected_qty, 1e-6);
    EXPECT_DOUBLE_EQ(entry.leverage, expected_lev);

    // Protective legs: the stop-loss is a stop at s.stop, the take-profit a limit
    // at s.sell — same quantity, default 1x leverage, bracketed under the entry.
    EXPECT_EQ(stop.type, OrderType::Stop);
    ASSERT_TRUE(stop.price.has_value());
    EXPECT_DOUBLE_EQ(*stop.price, b->stop);
    EXPECT_EQ(tp.type, OrderType::Limit);
    ASSERT_TRUE(tp.price.has_value());
    EXPECT_DOUBLE_EQ(*tp.price, b->sell);
    EXPECT_NEAR(stop.quantity, expected_qty, 1e-6);
    EXPECT_NEAR(tp.quantity, expected_qty, 1e-6);
    EXPECT_DOUBLE_EQ(stop.leverage, 1.0);
    EXPECT_DOUBLE_EQ(tp.leverage, 1.0);
    EXPECT_FALSE(entry.reduce_only);
    EXPECT_TRUE(stop.reduce_only);                    // orphaned legs may never open
    EXPECT_TRUE(tp.reduce_only);
    ASSERT_TRUE(stop.parent_id.has_value());
    ASSERT_TRUE(tp.parent_id.has_value());
    EXPECT_EQ(*stop.parent_id, entry.id);
    EXPECT_EQ(*tp.parent_id, entry.id);
}

} // namespace
