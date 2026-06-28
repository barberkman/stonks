// Unit tests for BacktestBroker against the SL/TP-on-order model:
//   place_order   -> ENTRY, opens a position on a clear symbol; may carry
//                    stop_loss / take_profit levels the position inherits.
//   close         -> market-close a position at the next bar's open.
//   update_exits  -> retarget the SL/TP on a resting entry or a live position.
// The broker closes a position when a later bar touches its stop (active) or
// target (passive); the stop wins ties. Orders are stamped with the broker's
// m_now (set in on_tick), so the recipe to fill an order is: place it, then
// on_tick a LATER bar.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "stonks/broker/backtestbroker.h"
#include "stonks/core/types.h"

namespace stonks::broker {

using namespace stonks::core;

namespace {

KLine bar(std::int64_t ms, const Symbol& sym, Price o, Price h, Price l, Price c)
{
    return KLine{ Timestamp::from_millis(ms), sym, o, h, l, c, Volume{ 1.0 } };
}
KLine bar(std::int64_t ms, Price o, Price h, Price l, Price c)
{
    return bar(ms, Symbol{ "X" }, o, h, l, c);
}
KLine flat(std::int64_t ms, Price p) { return bar(ms, p, p, p, p); }
KLine flat(std::int64_t ms, const Symbol& sym, Price p) { return bar(ms, sym, p, p, p, p); }

std::vector<Trade> sorted_trades(const BacktestBroker& b)
{
    std::vector<Trade> v;
    for (const auto& [id, t] : b.trades()) { v.push_back(t); }
    std::ranges::sort(v, {}, &Trade::id);
    return v;
}

OrderParams entry_market(const Symbol& sym, OrderSide side, Quantity qty)
{
    return OrderParams{ .symbol = sym, .side = side, .type = OrderType::Market, .quantity = qty };
}
OrderParams limit(const Symbol& sym, OrderSide side, Quantity qty, Price price)
{
    return OrderParams{ .symbol = sym, .side = side, .type = OrderType::Limit, .quantity = qty, .price = price };
}

} // namespace

// --- Construction ------------------------------------------------------------

TEST(BacktestBroker, ConstructorSetsCashAndStartsEmpty)
{
    BacktestBroker broker{ Balance{ 50'000.0 } };
    EXPECT_DOUBLE_EQ(broker.cash(), 50'000.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 50'000.0);
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_TRUE(broker.orders().empty());
    EXPECT_FALSE(broker.position("X").has_value());
}

// --- Placement & validation --------------------------------------------------

TEST(BacktestBroker, PlaceOrderRecordsAndDoesNotMoveCash)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(entry_market("X", OrderSide::Buy, 1.0));
    EXPECT_EQ(broker.orders().size(), 1u);
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);   // placing never moves cash
}

TEST(BacktestBroker, OrderIdsStrictlyIncrease)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto a = broker.place_order(entry_market("X", OrderSide::Buy, 1.0));
    const auto b = broker.place_order(limit("Y", OrderSide::Buy, 1.0, 50.0));
    EXPECT_EQ(a, 1u);
    EXPECT_EQ(b, 2u);
}

TEST(BacktestBroker, RejectsNonPositiveQuantity)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(entry_market("X", OrderSide::Buy, 0.0));
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);
}

TEST(BacktestBroker, RejectsLimitWithoutPositivePrice)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(
        OrderParams{ .symbol = "X", .side = OrderSide::Buy, .type = OrderType::Limit, .quantity = 1.0 });
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);   // no price
}

TEST(BacktestBroker, EntryRejectedWhileSymbolHasAPosition)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));                 // long opened
    const auto id2 = broker.place_order(entry_market("X", OrderSide::Buy, 5.0));
    EXPECT_EQ(broker.orders().at(id2).status, OrderStatus::Rejected);   // one position per symbol
}

// --- Entry fills -------------------------------------------------------------

TEST(BacktestBroker, MarketEntryFillsAtNextOpenAndDebitsCash)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->quantity, 10.0);
    EXPECT_DOUBLE_EQ(pos->price, 100.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 9'000.0);
    EXPECT_EQ(broker.trades().size(), 1u);
}

TEST(BacktestBroker, NoFillOnThePlacementBar)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));                 // m_now -> 1000
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));   // stamped 1000
    broker.on_tick(flat(1000, 100.0));                 // same ts -> gated, no fill
    EXPECT_FALSE(broker.position("X").has_value());
    broker.on_tick(flat(2000, 100.0));                 // later bar -> fills
    EXPECT_TRUE(broker.position("X").has_value());
}

TEST(BacktestBroker, LimitBuyEntryFillsWhenLowReachesLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(limit("X", OrderSide::Buy, 10.0, 100.0));
    broker.on_tick(bar(1000, 105.0, 106.0, 99.0, 104.0));   // low 99 <= 100 -> fill at min(100,105)=100
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.position("X")->price, 100.0);
}

TEST(BacktestBroker, LimitBuyEntryStaysOpenWhenLowAboveLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(limit("X", OrderSide::Buy, 10.0, 100.0));
    broker.on_tick(bar(1000, 105.0, 110.0, 101.0, 108.0)); // low 101 > 100 -> no fill
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
}

TEST(BacktestBroker, InsufficientCashRejectsEntry)
{
    BacktestBroker broker{ Balance{ 500.0 } };
    const auto id = broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));                 // cost 1000 > 500 -> reject
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 500.0);
}

TEST(BacktestBroker, EntryInheritsStopLossAndTakeProfit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.stop_loss = 95.0;
    p.take_profit = 110.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    ASSERT_TRUE(pos->stop_loss.has_value());
    EXPECT_DOUBLE_EQ(*pos->stop_loss, 95.0);
    ASSERT_TRUE(pos->take_profit.has_value());
    EXPECT_DOUBLE_EQ(*pos->take_profit, 110.0);
}

// --- Stop-loss / take-profit exits -------------------------------------------

TEST(BacktestBroker, LongStopLossClosesOnAdverseMove)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.stop_loss = 95.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100, cash 9000
    broker.on_tick(bar(2000, 98.0, 99.0, 94.0, 96.0));       // low 94 <= 95 -> stop at min(95,98)=95
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'950.0);                // 9000 + 1000 + (95-100)*10
}

TEST(BacktestBroker, LongTakeProfitClosesOnFavorableMove)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.take_profit = 110.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100, cash 9000
    broker.on_tick(bar(2000, 105.0, 115.0, 104.0, 108.0));   // high 115 >= 110 -> tp at max(110,105)=110
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);               // 9000 + 1000 + (110-100)*10
}

TEST(BacktestBroker, ShortStopLossClosesOnAdverseMove)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Sell, 10.0);
    p.stop_loss = 105.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // short -10 @100, cash 9000
    broker.on_tick(bar(2000, 102.0, 106.0, 101.0, 104.0));   // high 106 >= 105 -> stop at max(105,102)=105
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'950.0);                // 9000 + 1000 + (100-105)*10
}

TEST(BacktestBroker, GapThroughStopFillsAtTheOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.stop_loss = 95.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100
    broker.on_tick(bar(2000, 90.0, 92.0, 88.0, 91.0));       // gaps below the stop -> fill at min(95,90)=90
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'900.0);                // 9000 + 1000 + (90-100)*10
}

TEST(BacktestBroker, StopWinsWhenBarSpansBothLevels)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.stop_loss = 95.0;
    p.take_profit = 110.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100
    broker.on_tick(bar(2000, 100.0, 115.0, 94.0, 108.0));    // spans both -> pessimistic: stop fills
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'950.0);                // stop at min(95,100)=95 -> (95-100)*10
}

TEST(BacktestBroker, NoExitOnTheEntryFillBar)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = limit("X", OrderSide::Buy, 10.0, 100.0);
    p.take_profit = 105.0;
    broker.place_order(p);
    // The entry's own bar also spans the TP price; the exit must defer to next bar.
    broker.on_tick(bar(1000, 100.0, 106.0, 99.0, 104.0));
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 1u);                   // only the entry filled this bar
    broker.on_tick(bar(2000, 104.0, 107.0, 103.0, 106.0));   // now the TP fills
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 2u);
}

TEST(BacktestBroker, ShortRoundTripRealizesPnlViaTakeProfit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Sell, 10.0);       // short entry
    p.take_profit = 90.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->quantity, -10.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 9'000.0);                // cash-secured short ties up full notional
    broker.on_tick(bar(2000, 91.0, 92.0, 89.0, 90.0));       // low 89 <= 90 -> cover at min(90,91)=90
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);               // 9000 + 1000 + (100-90)*10
}

TEST(BacktestBroker, ExitTradeAttributesToEntryOrder)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.take_profit = 110.0;
    const auto entry = broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));
    broker.on_tick(bar(2000, 109.0, 111.0, 108.0, 110.0));   // tp 110 hit
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].order_id, entry);                    // entry trade
    EXPECT_EQ(trades[1].order_id, entry);                    // exit attributes back to the entry
}

// --- close() -----------------------------------------------------------------

TEST(BacktestBroker, CloseFlattensAtNextBarOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100
    EXPECT_TRUE(broker.close("X"));
    broker.on_tick(bar(2000, 110.0, 112.0, 109.0, 111.0));   // close at the open: 110
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);               // 9000 + 1000 + (110-100)*10
}

TEST(BacktestBroker, CloseIsNoOpWhenFlat)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    EXPECT_FALSE(broker.close("X"));                          // nothing to close
    broker.on_tick(flat(1000, 100.0));
    EXPECT_FALSE(broker.position("X").has_value());
}

// --- update_exits() ----------------------------------------------------------

TEST(BacktestBroker, UpdateExitsOnLivePositionArmsTighterStop)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.stop_loss = 90.0;                                      // loose initial stop
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                       // long 10 @100
    EXPECT_TRUE(broker.update_exits("X", 98.0, std::nullopt)); // tighten the stop
    broker.on_tick(bar(2000, 100.0, 101.0, 97.0, 99.0));     // low 97: misses old 90, hits new 98
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'980.0);                // 9000 + 1000 + (98-100)*10
}

TEST(BacktestBroker, UpdateExitsOnRestingEntryCarriesToFilledPosition)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    auto p = limit("X", OrderSide::Buy, 10.0, 100.0);
    p.take_profit = 110.0;
    broker.place_order(p);                                   // resting entry
    EXPECT_TRUE(broker.update_exits("X", 95.0, 120.0));      // retarget before it fills
    broker.on_tick(bar(1000, 100.0, 101.0, 99.0, 100.0));    // entry fills at 100
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    ASSERT_TRUE(pos->stop_loss.has_value());
    EXPECT_DOUBLE_EQ(*pos->stop_loss, 95.0);
    ASSERT_TRUE(pos->take_profit.has_value());
    EXPECT_DOUBLE_EQ(*pos->take_profit, 120.0);
}

TEST(BacktestBroker, UpdateExitsIsNoOpWhenFlat)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    EXPECT_FALSE(broker.update_exits("X", 95.0, 110.0));
}

// --- Equity, marking, ids, independence -------------------------------------

TEST(BacktestBroker, EquityMarksToLatestClose)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));                 // long 10 @100
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0);       // 9000 + 1000 reserved + 0 upnl
    broker.on_tick(flat(2000, 110.0));                 // mark -> 110
    EXPECT_DOUBLE_EQ(broker.equity(), 10'100.0);       // + (110-100)*10
}

TEST(BacktestBroker, OnlyFillsAdvanceTheTradeIdCounter)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 0.0));    // rejected (qty 0); no trade
    auto p = entry_market("X", OrderSide::Buy, 10.0);
    p.take_profit = 110.0;
    broker.place_order(p);
    broker.on_tick(flat(1000, 100.0));                            // entry fills -> trade 1
    broker.on_tick(bar(2000, 109.0, 111.0, 108.0, 110.0));        // tp closes -> trade 2
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_EQ(trades[0].id, 1u);
    EXPECT_EQ(trades[1].id, 2u);
}

TEST(BacktestBroker, PositionsAreIndependentPerSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("A", OrderSide::Buy, 1.0));
    broker.place_order(entry_market("B", OrderSide::Sell, 1.0));
    broker.on_tick(flat(1000, "A", 100.0));
    broker.on_tick(flat(1000, "B", 200.0));
    ASSERT_TRUE(broker.position("A").has_value());
    ASSERT_TRUE(broker.position("B").has_value());
    EXPECT_GT(broker.position("A")->quantity, 0.0);    // long A
    EXPECT_LT(broker.position("B")->quantity, 0.0);    // short B
}

} // namespace stonks::broker
