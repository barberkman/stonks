// Unit tests for BacktestBroker against the reduce-only model:
//   place_order -> ENTRY, opens a position, only valid on a clear symbol.
//   place_exit  -> reduce-only EXIT (stop-loss / take-profit), only reduces an
//                  existing opposite position; never opens or flips.
// Orders are stamped with the broker's m_now (set in on_tick), so the recipe to
// fill an order is: place it, then on_tick a LATER bar.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
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
OrderParams stop(const Symbol& sym, OrderSide side, Quantity qty, Price price)
{
    return OrderParams{ .symbol = sym, .side = side, .type = OrderType::Stop, .quantity = qty, .price = price };
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
    EXPECT_FALSE(broker.orders().at(id).reduce_only);
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
    EXPECT_EQ(broker.orders().at(id2).status, OrderStatus::Rejected);   // one context per symbol
}

TEST(BacktestBroker, ExitRejectedWithoutAPosition)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_exit(limit("X", OrderSide::Sell, 5.0, 110.0));
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);   // nothing to reduce
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

TEST(BacktestBroker, StopBuyEntryTriggersOnBreakout)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(stop("X", OrderSide::Buy, 10.0, 110.0));
    broker.on_tick(bar(1000, 105.0, 109.0, 104.0, 108.0)); // high 109 < 110 -> no trigger
    EXPECT_FALSE(broker.position("X").has_value());
    broker.on_tick(bar(2000, 108.0, 112.0, 107.0, 111.0)); // high 112 >= 110 -> fill at max(110,108)=110
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.position("X")->price, 110.0);
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

// --- Reduce-only exits -------------------------------------------------------

TEST(BacktestBroker, ExitReducesPositionAndKeepsSiblingOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));                 // long 10 @100, cash 9000
    broker.place_exit(limit("X", OrderSide::Sell, 4.0, 110.0));    // partial TP (id 2)
    const auto far = broker.place_exit(limit("X", OrderSide::Sell, 10.0, 120.0)); // farther TP (id 3)
    broker.on_tick(bar(2000, 109.0, 112.0, 108.0, 111.0));   // 110 hit (sell 4); 120 not hit
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->quantity, 6.0);
    // cash: 9000 + 4*100 (collateral) + (110-100)*4 (pnl) = 9440
    EXPECT_DOUBLE_EQ(broker.cash(), 9'440.0);
    EXPECT_EQ(broker.orders().at(far).status, OrderStatus::Open);  // sibling survives partial exit
}

TEST(BacktestBroker, FullExitClosesAndCancelsSiblingNoReverse)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));
    const auto tp = broker.place_exit(limit("X", OrderSide::Sell, 10.0, 110.0));
    const auto sl = broker.place_exit(stop("X", OrderSide::Sell, 10.0, 95.0));
    broker.on_tick(bar(2000, 105.0, 115.0, 104.0, 108.0)); // TP 110 hit; SL 95 not (low 104)
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);   // OCO
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);          // 9000 + 1000 + 100 pnl

    broker.on_tick(bar(3000, 108.0, 115.0, 104.0, 108.0));  // nothing left to act on
    EXPECT_EQ(broker.trades().size(), 2u);
    EXPECT_FALSE(broker.position("X").has_value());
}

TEST(BacktestBroker, OversizedExitClampsToHeldNoFlip)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));
    broker.place_exit(limit("X", OrderSide::Sell, 25.0, 110.0));   // oversized
    broker.on_tick(bar(2000, 109.0, 115.0, 108.0, 111.0));        // fills min(25,10)=10 -> flat
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 2u);             // entry + one close, no short opened
}

TEST(BacktestBroker, StopLossExitTriggersOnAdverseMove)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));
    broker.on_tick(flat(1000, 100.0));
    broker.place_exit(stop("X", OrderSide::Sell, 10.0, 95.0));
    broker.on_tick(bar(2000, 98.0, 99.0, 94.0, 96.0)); // low 94 <= 95 -> sell-stop at min(95,98)=95 -> flat
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'950.0);          // 9000 + 1000 + (95-100)*10
}

TEST(BacktestBroker, ShortRoundTripRealizesPnl)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(entry_market("X", OrderSide::Sell, 10.0));  // short entry
    broker.on_tick(flat(1000, 100.0));
    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->quantity, -10.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 9'000.0);          // cash-secured short ties up full notional
    broker.place_exit(limit("X", OrderSide::Buy, 10.0, 90.0));     // cover
    broker.on_tick(bar(2000, 91.0, 92.0, 89.0, 90.0)); // buy-limit 90 hit -> cover -> flat
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);         // 9000 + 1000 + (100-90)*10
}

// --- Brackets (pre-placed) ---------------------------------------------------

TEST(BacktestBroker, PreplacedExitsDormantUntilEntryFillsThenArmedNextBar)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(limit("X", OrderSide::Buy, 10.0, 100.0));   // entry
    broker.place_exit(limit("X", OrderSide::Sell, 10.0, 105.0));   // TP at 105
    // The entry's bar also spans the TP price; arming must defer the TP to next bar.
    broker.on_tick(bar(1000, 100.0, 106.0, 99.0, 104.0));
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 1u);             // only the entry filled this bar
    broker.on_tick(bar(2000, 104.0, 107.0, 103.0, 106.0));        // now the TP fills
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 2u);
}

TEST(BacktestBroker, RejectedEntryCancelsItsPreplacedExits)
{
    BacktestBroker broker{ Balance{ 50.0 } };          // too little to afford the entry
    const auto entry = broker.place_order(limit("X", OrderSide::Buy, 10.0, 100.0));
    const auto tp = broker.place_exit(limit("X", OrderSide::Sell, 10.0, 110.0));
    broker.on_tick(bar(1000, 100.0, 101.0, 99.0, 100.0));   // entry triggers but cost 1000 > 50 -> reject
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Cancelled);
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.trades().size(), 0u);
}

// A working (unfilled) entry holds the symbol's context. A later signal's bracket
// is rejected wholesale: its entry collides with the working one, and its exits
// belong to that superseded signal — they must NOT attach to the tick-1000 entry,
// or they could (mis-)close its eventual position at a foreign size/level.
TEST(BacktestBroker, StaleExitsFromALaterBracketAreRejected)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    // Tick 1000: a limit-buy entry that won't fill yet, plus its own bracket exits
    // (same tick as the entry -> accepted).
    broker.on_tick(bar(1000, 100.0, 101.0, 99.0, 100.0));
    const auto entry = broker.place_order(limit("X", OrderSide::Buy, 10.0, 90.0));
    const auto sl = broker.place_exit(stop("X", OrderSide::Sell, 10.0, 85.0));
    const auto tp = broker.place_exit(limit("X", OrderSide::Sell, 10.0, 110.0));
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Open);

    // Tick 2000: entry still working (low 95 > 90). A new signal's bracket arrives:
    // its entry collides (one context per symbol) and its exits are stale.
    broker.on_tick(bar(2000, 100.0, 101.0, 95.0, 100.0));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Open);
    const auto stale_entry = broker.place_order(limit("X", OrderSide::Buy, 5.0, 88.0));
    const auto stale_sl = broker.place_exit(stop("X", OrderSide::Sell, 5.0, 80.0));
    const auto stale_tp = broker.place_exit(limit("X", OrderSide::Sell, 5.0, 120.0));
    EXPECT_EQ(broker.orders().at(stale_entry).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(stale_sl).status, OrderStatus::Rejected);   // would partially close at a foreign stop without the guard
    EXPECT_EQ(broker.orders().at(stale_tp).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);             // original bracket intact
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Open);

    // The entry finally fills; only its own exits exist, so it closes full-size at
    // its own take-profit (110), not the stale 120/80 levels.
    broker.on_tick(bar(3000, 92.0, 93.0, 88.0, 90.0));   // low 88 <= 90 -> fill @ min(90,92)=90
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.position("X")->quantity, 10.0);
    broker.on_tick(bar(4000, 100.0, 112.0, 99.0, 111.0));   // TP 110 hit -> full close
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);        // OCO
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
    broker.place_order(entry_market("X", OrderSide::Buy, 10.0));   // id 1
    broker.place_order(entry_market("X", OrderSide::Buy, 1.0));    // id 2 -> rejected (working entry exists)
    broker.on_tick(flat(1000, 100.0));                 // fill 1 -> trade 1
    broker.place_exit(entry_market("X", OrderSide::Sell, 10.0));   // market exit
    broker.on_tick(flat(2000, 110.0));                 // close -> trade 2
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
