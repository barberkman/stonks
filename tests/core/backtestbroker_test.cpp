// Direct unit tests for BacktestBroker against the new cash-secured,
// one-position-per-symbol model. Orders are stamped with the broker's m_now
// (set in on_tick), so the recipe to fill an order is: place it, then on_tick a
// later bar. To stamp at a specific timestamp T (e.g. for the gate), on_tick a
// bar at T first, then place.

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

} // namespace

// --- Construction ------------------------------------------------------------

TEST(BacktestBroker, ConstructorSetsCashAndStartsEmpty)
{
    BacktestBroker broker{ Balance{ 50'000.0 } };
    EXPECT_DOUBLE_EQ(broker.cash(), 50'000.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 50'000.0);
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_TRUE(broker.orders().empty());
}

// --- Placement & validation --------------------------------------------------

TEST(BacktestBroker, PlaceOrderRecordsAndDoesNotMoveCash)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    EXPECT_EQ(broker.orders().size(), 1u);
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);   // placing never moves cash
}

TEST(BacktestBroker, OrderIdsStrictlyIncreaseAcrossMarketAndLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto a = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto b = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 50.0 });
    const auto c = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    EXPECT_EQ(a, OrderID{ 1 });
    EXPECT_EQ(b, OrderID{ 2 });
    EXPECT_EQ(c, OrderID{ 3 });
}

TEST(BacktestBroker, MarketOrderHasNoPrice_LimitCarriesPrice)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto m = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.5 });
    const auto l = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0, .price = 99.5 });

    EXPECT_EQ(broker.orders().at(m).type, OrderType::Market);
    EXPECT_FALSE(broker.orders().at(m).price.has_value());
    EXPECT_DOUBLE_EQ(broker.orders().at(m).quantity, 1.5);

    EXPECT_EQ(broker.orders().at(l).type, OrderType::Limit);
    ASSERT_TRUE(broker.orders().at(l).price.has_value());
    EXPECT_DOUBLE_EQ(*broker.orders().at(l).price, 99.5);
}

TEST(BacktestBroker, OrderStampedWithCurrentTime)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    // Before any tick, m_now is epoch.
    const auto early = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    EXPECT_EQ(broker.orders().at(early).timestamp, Timestamp{});

    broker.on_tick(flat(5000, 100.0));   // m_now = 5000
    const auto later = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    EXPECT_EQ(broker.orders().at(later).timestamp, Timestamp::from_millis(5000));
}

TEST(BacktestBroker, RejectsNonPositiveQuantityAtPlacement)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto zero = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.0 });
    const auto neg = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = -1.0 });
    EXPECT_EQ(broker.orders().at(zero).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(neg).status, OrderStatus::Rejected);
    // A rejected order never fills.
    broker.on_tick(flat(2000, 100.0));
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BacktestBroker, RejectsLimitWithoutPositivePrice)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto zero = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 0.0 });
    const auto neg = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = -5.0 });
    EXPECT_EQ(broker.orders().at(zero).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(neg).status, OrderStatus::Rejected);
}

// --- Market fills -------------------------------------------------------------

TEST(BacktestBroker, MarketBuyFillsAtNextBarOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));   // fills @ open=110

    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 220.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 112.0);   // marked to close
}

TEST(BacktestBroker, MarketSellWithNoPositionOpensShort)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(flat(2000, 110.0));   // short opens @110

    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 110.0);   // collateral tied up (cash-secured)
    EXPECT_DOUBLE_EQ(broker.equity(), 1'000.0);         // flat-bar fill -> no immediate P&L
}

TEST(BacktestBroker, OrderDoesNotFillOnItsOwnBarTimestamp)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(2000, 100.0));   // m_now = 2000
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });

    broker.on_tick(flat(2000, 110.0));   // same ts: 2000 >= 2000 -> gate, no fill
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);

    broker.on_tick(bar(3000, 120.0, 120.0, 120.0, 120.0));   // later -> fills @120
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 120.0);
}

TEST(BacktestBroker, OrderFillsOnlyAgainstItsOwnSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, Symbol{ "Y" }, 50.0));   // different symbol -> no fill
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
    broker.on_tick(flat(2000, Symbol{ "X" }, 100.0));  // matches -> fills @100
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 100.0);
}

// --- Limit fills --------------------------------------------------------------

TEST(BacktestBroker, LimitBuyFillsWhenLowReachesLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 95.0 });
    broker.on_tick(bar(2000, 100.0, 105.0, 90.0, 100.0));   // low=90 <= 95 -> min(95,100)=95
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 95.0);
}

TEST(BacktestBroker, LimitBuyGapDownFillsAtOpen)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 90.0 });
    broker.on_tick(bar(2000, 80.0, 85.0, 75.0, 82.0));   // open=80 below limit -> min(90,80)=80
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 80.0);
}

TEST(BacktestBroker, LimitBuyStaysOpenWhenLowAboveLimit)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 95.0 });
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));   // low=105 > 95 -> no fill
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);   // lingers, NOT rejected
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);
}

TEST(BacktestBroker, LimitSellFillsWhenHighReachesLimit)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 125.0 });
    broker.on_tick(bar(2000, 100.0, 130.0, 95.0, 120.0));   // high=130 >= 125 -> max(125,100)=125 (short opens)
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 125.0);       // collateral for the short
}

TEST(BacktestBroker, LimitSellStaysOpenWhenHighBelowLimit)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    const auto id = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 125.0 });
    broker.on_tick(bar(2000, 100.0, 120.0, 95.0, 110.0));   // high=120 < 125 -> no fill
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Open);
}

// --- Trade recording ----------------------------------------------------------

TEST(BacktestBroker, FillRecordsOneTradeWithAllFields)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));

    ASSERT_EQ(broker.trades().size(), 1u);
    const auto& t = broker.trades().at(TradeID{ 1 });
    EXPECT_EQ(t.id, TradeID{ 1 });
    EXPECT_EQ(t.order_id, id);
    EXPECT_EQ(t.timestamp, Timestamp::from_millis(2000));
    EXPECT_EQ(t.symbol, "X");
    EXPECT_EQ(t.side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(t.quantity, 2.0);
    EXPECT_DOUBLE_EQ(t.price, 110.0);
}

TEST(BacktestBroker, OnlyFillsAdvanceTheTradeIdCounter)
{
    BacktestBroker broker{ Balance{ 50.0 } };   // too poor to fill the first buy
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));          // cost 100 > 50 -> rejected, no trade
    EXPECT_TRUE(broker.trades().empty());

    // A later affordable order gets trade id 1 (the rejected attempt didn't burn one).
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(3000, 40.0));
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 1u);
    EXPECT_EQ(trades.front().id, TradeID{ 1 });
}

// --- Position model: the rewrite's core --------------------------------------

TEST(BacktestBroker, InsufficientCashRejectsAndDoesNotLinger)
{
    BacktestBroker broker{ Balance{ 50.0 } };
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));   // cost 100 > 50 -> Rejected on this bar
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Rejected);
    EXPECT_DOUBLE_EQ(broker.cash(), 50.0);

    // Key new behavior: it does NOT linger to a later, affordable bar.
    broker.on_tick(flat(3000, 40.0));
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_DOUBLE_EQ(broker.cash(), 50.0);
}

TEST(BacktestBroker, SameSideAddIsRejected)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));   // opens long 1 @100 (m_now=2000)

    const auto add = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(3000, 110.0));   // same-side against existing long -> Rejected
    EXPECT_EQ(broker.orders().at(add).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.trades().size(), 1u);                       // only the first open
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 100.0);          // add never settled
}

TEST(BacktestBroker, PartialCloseShrinksPositionKeepsEntry)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 3.0 });
    broker.on_tick(flat(2000, 100.0));   // long 3 @100

    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(flat(3000, 110.0));   // close 1 of 3 @110

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].quantity, 1.0);
    // cash = 10000 - 300 (open) + (1*100 + 10 pnl) (close 1)
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 300.0 + 110.0);
    // remaining 2 @ entry 100, marked to 110
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 2.0 * 110.0);
}

TEST(BacktestBroker, FullCloseErasesPositionAndRealizesPnl)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(flat(2000, 100.0));   // long 2 @100
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0 });
    broker.on_tick(flat(3000, 120.0));   // close 2 @120 -> +40 pnl, flat

    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 200.0 + 240.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat: equity == cash
}

TEST(BacktestBroker, OversizedCloseClampsToHeldNoFlip)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(flat(2000, 100.0));   // long 2 @100
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 5.0 });
    broker.on_tick(flat(3000, 120.0));   // sell 5 clamps to held 2 @120, no short opens

    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_DOUBLE_EQ(trades[1].quantity, 2.0);          // only the held 2 closed
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 200.0 + 240.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat, not short
}

TEST(BacktestBroker, ShortRoundTripRealizesPnl)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(flat(2000, 110.0));   // short 1 @110, cash=890
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(3000, 90.0));    // cover @90 -> +20 profit

    EXPECT_DOUBLE_EQ(broker.cash(), 1'020.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat
}

TEST(BacktestBroker, IndependentPositionsAcrossSymbols)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "A", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.place_order(MarketOrderParams{ .symbol = "B", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(flat(2000, Symbol{ "A" }, 100.0));   // A fills @100
    broker.on_tick(flat(2000, Symbol{ "B" }, 50.0));    // B fills @50

    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 100.0 - 100.0);
    broker.on_tick(flat(3000, Symbol{ "A" }, 120.0));   // remark A only
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 1.0 * 120.0 + 2.0 * 50.0);
}

// --- Equity marking -----------------------------------------------------------

TEST(BacktestBroker, EquityMarksToLatestClosePerSymbol)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    broker.on_tick(flat(2000, 110.0));   // fill 2 @110
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 110.0);
    broker.on_tick(flat(3000, 130.0));   // remark
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 130.0);
}

TEST(BacktestBroker, StaleMarkWhenHeldSymbolStopsPrinting)
{
    // A held symbol that stops printing keeps contributing at its last-seen close.
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "A", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.place_order(MarketOrderParams{ .symbol = "B", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, Symbol{ "A" }, 100.0));
    broker.on_tick(flat(2000, Symbol{ "B" }, 50.0));
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0);   // 9850 + 100 + 50

    // B never prints again; A keeps printing. B stays frozen at 50.
    broker.on_tick(flat(3000, Symbol{ "A" }, 200.0));
    broker.on_tick(flat(4000, Symbol{ "A" }, 300.0));
    EXPECT_DOUBLE_EQ(broker.equity(), (10'000.0 - 150.0) + 300.0 + 50.0);
}

// --- Chained orders -----------------------------------------------------------

TEST(BacktestBroker, ChainedOrderStaysDormantUntilParentFills)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    // Entry: a limit buy that won't fill until price dips to 90.
    const auto entry = broker.place_order(LimitOrderParams{
        .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 90.0 });
    // Chained exit: a sell limit at 110, dormant until the entry fills.
    const auto exit = broker.place_order(LimitOrderParams{
        .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 110.0 }, entry);

    // Bar reaches 110 (the exit's price) but not 90 (the entry's): the entry lingers,
    // and the dormant exit must NOT fire even though its own price was hit.
    broker.on_tick(bar(2000, 100.0, 130.0, 95.0, 100.0));
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Open);
    EXPECT_EQ(broker.orders().at(exit).status, OrderStatus::Open);   // still dormant

    // Price dips to 90: the entry fills (@85), which activates the exit.
    broker.on_tick(bar(3000, 85.0, 88.0, 80.0, 86.0));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);
    ASSERT_EQ(broker.trades().size(), 1u);

    // Next bar reaches 110: the now-active exit fills and closes the position.
    broker.on_tick(bar(4000, 120.0, 125.0, 115.0, 122.0));
    EXPECT_EQ(broker.orders().at(exit).status, OrderStatus::Filled);
    ASSERT_EQ(broker.trades().size(), 2u);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());   // flat
}

// --- Float-dust flat snap & notional floor -------------------------------------

TEST(BacktestBroker, ThreeWayScaleOutSnapsDustFlatAndAllowsReentry)
{
    // 0.3 - 0.1 - 0.1 - 0.1 leaves ~ -2.8e-17 in doubles; without the snap the
    // symbol reads as positioned forever and same-side entries stay rejected.
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.3 });
    broker.on_tick(flat(1000, 100.0));
    for (std::int64_t ms : { 2000, 3000, 4000 }) {
        broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.1 });
        broker.on_tick(flat(ms, 100.0));
    }
    EXPECT_FALSE(broker.position("X").has_value());        // snapped to exactly flat

    const auto reentry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.5 });
    broker.on_tick(flat(5000, 100.0));
    EXPECT_EQ(broker.orders().at(reentry).status, OrderStatus::Filled);   // not same_side_add
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.position("X")->quantity, 0.5);
}

TEST(BacktestBroker, DustSnapTriggersCancelOnFlatForSiblingLeg)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.3 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.3, .price = 90.0, .reduce_only = true }, entry);
    for (double tp_price : { 105.0, 110.0, 115.0 }) {
        broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.1, .price = tp_price, .reduce_only = true }, entry);
    }
    broker.on_tick(flat(1000, 100.0));                     // entry fills, legs armed
    broker.on_tick(bar(2000, 104.0, 106.0, 103.0, 105.0)); // TP1 -> 0.2 left
    broker.on_tick(bar(3000, 109.0, 111.0, 108.0, 110.0)); // TP2 -> 0.1 left
    broker.on_tick(bar(4000, 114.0, 116.0, 113.0, 115.0)); // TP3 -> dust -> snapped flat
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);   // cancel-on-flat fired
}

TEST(BacktestBroker, GenuineLeftoverAboveEpsilonSurvives)
{
    // The snap is for float dust only: a real 1e-7 remainder (>> the relative
    // 1e-9 default) stays a live position.
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(1000, 100.0));
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.9999999 });
    broker.on_tick(flat(2000, 100.0));
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_NEAR(broker.position("X")->quantity, 1e-7, 1e-12);
}

TEST(BacktestBroker, MinNotionalRejectsTinyOpensButNeverBlocksCloses)
{
    BacktestBroker broker{ Balance{ 10'000.0 }, BrokerConfig{ .min_notional = 50.0 } };
    const auto tiny = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.4 });
    broker.on_tick(flat(1000, 100.0));                     // notional 40 < 50
    EXPECT_EQ(broker.orders().at(tiny).status, OrderStatus::Rejected);

    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));                     // notional 100: fills
    const auto small_close = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.1 });
    broker.on_tick(flat(3000, 100.0));                     // notional 10, but it's a close
    EXPECT_EQ(broker.orders().at(small_close).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.position("X")->quantity, 0.9);
}

// --- Bracket cancellation (cancel-on-reject & cancel-on-flat) -----------------

TEST(BacktestBroker, RejectedEntryCancelsDormantChildren)
{
    BacktestBroker broker{ Balance{ 100.0 } };   // too little to afford the entry

    const auto entry = broker.place_order(
        MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto tp = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 220.0 }, entry);
    const auto sl = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 180.0 }, entry);

    // Entry market-buys at open=200 -> cost 200 > 100 cash -> rejected, brackets cancelled.
    broker.on_tick(flat(1000, 200.0));

    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_DOUBLE_EQ(broker.equity(), 100.0);
}

TEST(BacktestBroker, PartialExitKeepsSiblingAlive)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    const auto entry = broker.place_order(
        MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    const auto tp1 = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 4.0, .price = 110.0 }, entry);
    const auto tp2 = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 6.0, .price = 120.0 }, entry);

    broker.on_tick(flat(1000, 100.0));                     // entry fills @100, both targets armed
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);

    broker.on_tick(bar(2000, 105.0, 112.0, 104.0, 108.0)); // hits 110, not 120
    EXPECT_EQ(broker.orders().at(tp1).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(tp2).status, OrderStatus::Open);   // partial -> sibling survives
    EXPECT_EQ(broker.trades().size(), 2u);

    broker.on_tick(bar(3000, 118.0, 125.0, 117.0, 122.0)); // now hits 120
    EXPECT_EQ(broker.orders().at(tp2).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());      // flat
    EXPECT_DOUBLE_EQ(broker.cash(), 10'160.0);
}

TEST(BacktestBroker, FullExitCancelsRemainingLegAndNoReverse)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    const auto entry = broker.place_order(
        MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    const auto tp1 = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 110.0 }, entry);
    const auto tp2 = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 130.0 }, entry);

    broker.on_tick(flat(1000, 100.0));                     // entry fills @100
    broker.on_tick(bar(2000, 105.0, 115.0, 104.0, 108.0)); // hits 110 -> tp1 closes all 10 -> flat

    EXPECT_EQ(broker.orders().at(tp1).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(tp2).status, OrderStatus::Cancelled);  // cancel-on-flat
    EXPECT_EQ(broker.trades().size(), 2u);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());

    // A later bar reaching tp2's old price must NOT reopen a short.
    broker.on_tick(bar(3000, 128.0, 135.0, 127.0, 132.0));
    EXPECT_EQ(broker.trades().size(), 2u);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());      // still flat, no reverse position
    EXPECT_DOUBLE_EQ(broker.cash(), 10'100.0);
}

TEST(BacktestBroker, RestingEntrySurvivesCancelOnFlat)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    const auto entryA = broker.place_order(
        MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    const auto tpA = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 110.0 }, entryA);
    // Independent re-entry order, NOT part of A's bracket (no parent).
    const auto entryB = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 5.0, .price = 80.0 });

    broker.on_tick(flat(1000, 100.0));                     // entryA fills @100; entryB rests (low 100 > 80)
    broker.on_tick(bar(2000, 105.0, 115.0, 104.0, 108.0)); // tpA closes A -> flat -> cancel-on-flat(A)

    EXPECT_EQ(broker.orders().at(tpA).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(entryB).status, OrderStatus::Open);   // entry survived the sweep

    // And it still works: price dips to 80 -> entryB fills, opening a fresh position.
    broker.on_tick(bar(3000, 79.0, 82.0, 78.0, 81.0));
    EXPECT_EQ(broker.orders().at(entryB).status, OrderStatus::Filled);
}

TEST(BacktestBroker, CancelOnFlatRecursesToGrandchildren)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };

    const auto entry = broker.place_order(
        MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    const auto tp = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 110.0 }, entry);
    // A grandchild hanging off the take-profit (a nested leg).
    const auto grandchild = broker.place_order(
        LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 50.0 }, tp);

    broker.on_tick(flat(1000, 100.0));                     // entry fills @100
    broker.on_tick(bar(2000, 105.0, 115.0, 104.0, 108.0)); // tp closes position -> flat

    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(grandchild).status, OrderStatus::Cancelled);  // reached via recursion
}

} // namespace stonks::broker
