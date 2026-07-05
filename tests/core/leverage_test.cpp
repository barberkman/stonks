// Unit tests for per-position isolated leverage in BacktestBroker: margin is
// notional/leverage, fixed by the order that opens the position; a position is
// force-closed at its bankruptcy price (long entry*(1-1/L), short entry*(1+1/L))
// once a bar's adverse extreme reaches it, and the account halts entirely when
// equity reaches zero. Liquidation checks run after each bar's fill sweep, so a
// resting exit that fills on the crash bar pre-empts its position's liquidation.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
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

// --- Placement & validation ----------------------------------------------------

TEST(BacktestBrokerLeverage, DefaultsLeverageToOneOnBothOrderTypes)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto m = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    const auto l = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 120.0 });
    EXPECT_DOUBLE_EQ(broker.orders().at(m).leverage, 1.0);
    EXPECT_DOUBLE_EQ(broker.orders().at(l).leverage, 1.0);
}

TEST(BacktestBrokerLeverage, RejectsLeverageBelowOneNaNOrInfinite)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto half = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .leverage = 0.5 });
    const auto zero = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .leverage = 0.0 });
    const auto nan = broker.place_order(MarketOrderParams{
        .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0,
        .leverage = std::numeric_limits<double>::quiet_NaN() });
    const auto inf = broker.place_order(LimitOrderParams{
        .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 100.0,
        .leverage = std::numeric_limits<double>::infinity() });   // would mean zero margin
    EXPECT_EQ(broker.orders().at(half).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(zero).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(nan).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(inf).status, OrderStatus::Rejected);
    broker.on_tick(flat(2000, 100.0));   // none of them ever fills
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BacktestBrokerLeverage, AcceptsLeverageOfExactlyOneAndAbove)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto one = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .leverage = 1.0 });
    const auto ten = broker.place_order(LimitOrderParams{ .symbol = "Y", .side = OrderSide::Buy, .quantity = 1.0, .price = 100.0, .leverage = 10.0 });
    EXPECT_EQ(broker.orders().at(one).status, OrderStatus::Open);
    EXPECT_EQ(broker.orders().at(ten).status, OrderStatus::Open);
}

// --- Margin on open -------------------------------------------------------------

TEST(BacktestBrokerLeverage, OpenLongDebitsNotionalOverLeverage)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 4.0, .leverage = 4.0 });
    broker.on_tick(flat(2000, 100.0));   // notional 400 -> margin 100
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 100.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0);   // flat-bar fill: margin back + zero uPnL
}

TEST(BacktestBrokerLeverage, OpenShortDebitsNotionalOverLeverage)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0, .leverage = 5.0 });
    broker.on_tick(flat(2000, 100.0));   // notional 200 -> margin 40
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0 - 40.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 1'000.0);
}

TEST(BacktestBrokerLeverage, LeverageAffordsAPositionCashAloneCouldNot)
{
    BacktestBroker broker{ Balance{ 100.0 } };
    // Notional 400 dwarfs the 100 cash, but margin 400/4 fits it exactly.
    const auto id = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 4.0, .leverage = 4.0 });
    broker.on_tick(flat(2000, 100.0));
    EXPECT_EQ(broker.orders().at(id).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.cash(), 0.0);
}

TEST(BacktestBrokerLeverage, MarginGateStillRejectsAndCancelsBracketChildren)
{
    BacktestBroker broker{ Balance{ 50.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 4.0, .leverage = 4.0 });
    const auto tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 4.0, .price = 150.0 }, entry);
    broker.on_tick(flat(2000, 100.0));   // margin 100 > 50 cash
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Rejected);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Cancelled);
    EXPECT_DOUBLE_EQ(broker.cash(), 50.0);
    EXPECT_TRUE(broker.trades().empty());
}

TEST(BacktestBrokerLeverage, ExplicitLeverageOneMatchesCashSecuredAccounting)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0, .leverage = 1.0 });
    broker.on_tick(bar(2000, 110.0, 115.0, 105.0, 112.0));
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0 - 220.0);
    EXPECT_DOUBLE_EQ(broker.equity(), 10'000.0 - 220.0 + 2.0 * 112.0);
}

// --- Margin on close ------------------------------------------------------------

TEST(BacktestBrokerLeverage, RoundTripCashDeltaIsExactlyPnl)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 5.0, .leverage = 5.0 });
    broker.on_tick(flat(2000, 100.0));   // margin 100 -> cash 900
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 5.0 });
    broker.on_tick(flat(3000, 110.0));   // margin back + 5 * 10 pnl
    EXPECT_DOUBLE_EQ(broker.cash(), 1'050.0);   // leverage amplifies the size, not the math
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(BacktestBrokerLeverage, CloseUsesPositionLeverageNotClosingOrderLeverage)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 4.0, .leverage = 4.0 });
    broker.on_tick(flat(2000, 100.0));   // margin 100 -> cash 900
    // The closing order carries a different leverage; it must be ignored.
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 4.0, .leverage = 10.0 });
    broker.on_tick(flat(3000, 100.0));   // flat close, zero pnl
    EXPECT_DOUBLE_EQ(broker.cash(), 1'000.0);   // the full 100 margin came back, not 400/10
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(BacktestBrokerLeverage, PartialCloseReturnsProportionalMargin)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 4.0, .leverage = 2.0 });
    broker.on_tick(flat(2000, 100.0));   // notional 400 -> margin 200, cash 800
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(flat(3000, 110.0));   // close 1: margin back 1*100/2 = 50, pnl +10
    EXPECT_DOUBLE_EQ(broker.cash(), 800.0 + 50.0 + 10.0);
    // Remaining 3 @100 at 2x: reserved 150, marked to 110 -> upnl +30.
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash() + 150.0 + 30.0);
}

// --- Equity ----------------------------------------------------------------------

TEST(BacktestBrokerLeverage, EquityReservesNotionalOverLeveragePlusUpnl)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 5.0 });
    broker.on_tick(flat(2000, 100.0));   // margin 200 -> cash 9800
    broker.on_tick(flat(3000, 108.0));   // remark (bankruptcy price 80 untouched)
    EXPECT_DOUBLE_EQ(broker.equity(), 9'800.0 + 200.0 + 10.0 * 8.0);
}

// --- Long liquidation --------------------------------------------------------------

TEST(BacktestBrokerLeverage, LongNeverLiquidatesAtLeverageOne)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));
    broker.on_tick(flat(3000, 0.01));   // a 1x long's bankruptcy price is 0 — unreachable
    ASSERT_EQ(broker.trades().size(), 1u);   // just the open, no forced close
    EXPECT_DOUBLE_EQ(broker.equity(), 900.0 + 0.01);
}

TEST(BacktestBrokerLeverage, LongLiquidatesWhenLowBreachesBankruptcyPrice)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    broker.on_tick(flat(2000, 100.0));                     // margin 100 -> cash 900, B = 90
    broker.on_tick(bar(3000, 95.0, 96.0, 89.0, 94.0));     // low 89 <= 90 -> forced close @90
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_EQ(trades[1].side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(trades[1].quantity, 10.0);
    EXPECT_DOUBLE_EQ(trades[1].price, 90.0);               // the bankruptcy price, not the low
    EXPECT_DOUBLE_EQ(broker.cash(), 900.0);                // total loss = exactly the margin
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());      // flat
}

TEST(BacktestBrokerLeverage, LongLiquidationGapThroughFillsAtOpenAndCostsMoreThanMargin)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    broker.on_tick(flat(2000, 100.0));                     // margin 100 -> cash 900, B = 90
    broker.on_tick(bar(3000, 85.0, 88.0, 84.0, 86.0));     // gaps open below B -> fills @85
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_DOUBLE_EQ(trades[1].price, 85.0);
    // pnl -150 exceeds the 100 margin; the extra 50 comes out of cash.
    EXPECT_DOUBLE_EQ(broker.cash(), 850.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(BacktestBrokerLeverage, LiquidationCancelsBracketChildren)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    const auto sl = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 95.0 }, entry);
    const auto tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 120.0 }, entry);
    broker.on_tick(flat(2000, 100.0));                     // entry fills @100, legs armed, B = 90
    // Gap below the stop's limit: a limit sell at 95 can't fill when the whole
    // bar trades under it, so the liquidation is what flattens the position.
    broker.on_tick(bar(3000, 89.0, 89.5, 88.0, 89.0));     // open 89 < B -> liquidated @89
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Cancelled);
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_DOUBLE_EQ(trades[1].price, 89.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 890.0);                // margin 100 + pnl -110
}

TEST(BacktestBrokerLeverage, RestingExitFillingOnTheCrashBarPreemptsLiquidation)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    const auto sl = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 10.0, .price = 95.0 }, entry);
    broker.on_tick(flat(2000, 100.0));                     // entry @100, B = 90
    // The crash bar opens above the stop, so the stop fills in the ordinary
    // sweep (@96) and the low of 89 finds no position left to liquidate.
    broker.on_tick(bar(3000, 96.0, 97.0, 89.0, 91.0));
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_FALSE(trades[1].liquidation);
    EXPECT_DOUBLE_EQ(trades[1].price, 96.0);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    EXPECT_DOUBLE_EQ(broker.cash(), 960.0);                // margin 100 + pnl -40
}

// --- Short liquidation --------------------------------------------------------------

TEST(BacktestBrokerLeverage, ShortLiquidatesAtTwiceEntryEvenAtLeverageOne)
{
    // A cash-secured short posts exactly the entry notional; a doubling of the
    // price consumes all of it. This is the one deliberate behavior change at
    // leverage 1 versus the old hold-through-anything model.
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    broker.on_tick(flat(2000, 110.0));                       // short @110, cash 890, B = 220
    broker.on_tick(bar(3000, 180.0, 225.0, 175.0, 210.0));   // high 225 >= 220 -> forced cover @220
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_EQ(trades[1].side, OrderSide::Buy);
    EXPECT_DOUBLE_EQ(trades[1].price, 220.0);
    EXPECT_DOUBLE_EQ(broker.cash(), 890.0);                  // margin 110 + pnl -110 = 0 back
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(BacktestBrokerLeverage, ShortLiquidationGapThroughFillsAtOpen)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0, .leverage = 4.0 });
    broker.on_tick(flat(2000, 100.0));                       // short 2 @100, margin 50, B = 125
    broker.on_tick(bar(3000, 130.0, 140.0, 128.0, 135.0));   // gaps open above B -> fills @130
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_DOUBLE_EQ(trades[1].price, 130.0);
    // pnl = (100-130)*2 = -60 vs margin 50: the extra 10 comes out of cash.
    EXPECT_DOUBLE_EQ(broker.cash(), 950.0 + 50.0 - 60.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());
}

TEST(BacktestBrokerLeverage, HigherLeverageMovesTheBankruptcyPriceCloserToEntry)
{
    // At 10x a short liquidates on a 10% adverse move; the same bar is harmless at 1x.
    BacktestBroker at10{ Balance{ 1'000.0 } };
    at10.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .leverage = 10.0 });
    at10.on_tick(flat(2000, 100.0));                         // B = 110
    at10.on_tick(bar(3000, 105.0, 112.0, 104.0, 108.0));     // high 112 >= 110 -> liquidated
    ASSERT_EQ(sorted_trades(at10).size(), 2u);
    EXPECT_TRUE(sorted_trades(at10)[1].liquidation);
    EXPECT_DOUBLE_EQ(sorted_trades(at10)[1].price, 110.0);

    BacktestBroker at1{ Balance{ 1'000.0 } };
    at1.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0 });
    at1.on_tick(flat(2000, 100.0));                          // B = 200
    at1.on_tick(bar(3000, 105.0, 112.0, 104.0, 108.0));      // harmless at 1x
    EXPECT_EQ(sorted_trades(at1).size(), 1u);
}

TEST(BacktestBrokerLeverage, PositionOpenedAndLiquidatedOnTheSameBar)
{
    // The whole-bar OHLC approximation: an entry filling at this bar's open can
    // be liquidated by the same bar's adverse extreme.
    BacktestBroker broker{ Balance{ 1'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    broker.on_tick(bar(2000, 100.0, 101.0, 88.0, 95.0));     // fills @100 (B=90), low 88 -> liquidated @90
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    EXPECT_FALSE(trades[0].liquidation);
    EXPECT_TRUE(trades[1].liquidation);
    EXPECT_EQ(trades[0].timestamp, trades[1].timestamp);
    EXPECT_DOUBLE_EQ(broker.cash(), 900.0);   // the position's whole margin, gone in one bar
}

// --- Liquidation trade/order synthesis ---------------------------------------------

TEST(BacktestBrokerLeverage, LiquidationTradeReferencesASyntheticFilledMarketOrder)
{
    BacktestBroker broker{ Balance{ 1'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    broker.on_tick(flat(2000, 100.0));   // B = 90
    broker.on_tick(flat(3000, 90.0));    // low == B -> liquidation triggers on touch
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 2u);
    const auto& liq = trades[1];
    EXPECT_TRUE(liq.liquidation);
    EXPECT_EQ(liq.timestamp, Timestamp::from_millis(3000));

    const auto& synthetic = broker.orders().at(liq.order_id);
    EXPECT_EQ(synthetic.id, entry + 1);                      // same monotonic counter
    EXPECT_EQ(synthetic.type, OrderType::Market);
    EXPECT_EQ(synthetic.status, OrderStatus::Filled);
    EXPECT_EQ(synthetic.side, OrderSide::Sell);
    EXPECT_DOUBLE_EQ(synthetic.quantity, 10.0);
    EXPECT_FALSE(synthetic.price.has_value());
    EXPECT_FALSE(synthetic.parent_id.has_value());
    EXPECT_EQ(synthetic.timestamp, Timestamp::from_millis(3000));

    // Later ids keep increasing past the synthetic one.
    const auto next = broker.place_order(MarketOrderParams{ .symbol = "Y", .side = OrderSide::Buy, .quantity = 1.0 });
    EXPECT_EQ(next, synthetic.id + 1);
}

// --- Account bankruptcy stop ---------------------------------------------------------

TEST(BacktestBrokerLeverage, BankruptcySweepClosesEverythingCancelsOrdersAndHaltsTrading)
{
    BacktestBroker broker{ Balance{ 100.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "A", .side = OrderSide::Buy, .quantity = 50.0, .leverage = 10.0 });
    broker.place_order(MarketOrderParams{ .symbol = "B", .side = OrderSide::Buy, .quantity = 10.0, .leverage = 10.0 });
    const auto resting = broker.place_order(LimitOrderParams{ .symbol = "C", .side = OrderSide::Buy, .quantity = 1.0, .price = 50.0 });

    broker.on_tick(flat(2000, Symbol{ "A" }, 10.0));   // A: margin 50 -> cash 50
    broker.on_tick(flat(2000, Symbol{ "B" }, 10.0));   // B: margin 10 -> cash 40
    broker.on_tick(flat(2000, Symbol{ "C" }, 100.0));  // C's limit rests (low 100 > 50)
    EXPECT_FALSE(broker.bankrupt());

    // A crashes 90% through its bankruptcy price (9): the gap-through fill @1
    // realizes -450 against 50 margin, sinking account equity below zero. The
    // sweep closes B at its stale mark (10) and cancels C's resting order.
    broker.on_tick(bar(3000, Symbol{ "A" }, 1.0, 1.5, 0.8, 1.2));

    EXPECT_TRUE(broker.bankrupt());
    const auto trades = sorted_trades(broker);
    ASSERT_EQ(trades.size(), 4u);                     // A open, B open, A liq, B liq
    EXPECT_TRUE(trades[2].liquidation);
    EXPECT_EQ(trades[2].symbol, "A");
    EXPECT_DOUBLE_EQ(trades[2].price, 1.0);           // gapped through B(9) -> open
    EXPECT_TRUE(trades[3].liquidation);
    EXPECT_EQ(trades[3].symbol, "B");
    EXPECT_DOUBLE_EQ(trades[3].price, 10.0);          // swept at the last mark
    EXPECT_EQ(broker.orders().at(resting).status, OrderStatus::Cancelled);

    // Post-sweep the book is flat and cash equals (negative) equity.
    EXPECT_DOUBLE_EQ(broker.cash(), -350.0);
    EXPECT_DOUBLE_EQ(broker.equity(), broker.cash());

    // A bankrupt account rejects every new order and never moves again.
    const auto post = broker.place_order(MarketOrderParams{ .symbol = "A", .side = OrderSide::Buy, .quantity = 1.0 });
    EXPECT_EQ(broker.orders().at(post).status, OrderStatus::Rejected);
    broker.on_tick(flat(4000, Symbol{ "A" }, 500.0));
    EXPECT_DOUBLE_EQ(broker.cash(), -350.0);
    EXPECT_EQ(broker.trades().size(), 4u);
    EXPECT_TRUE(broker.bankrupt());
}

} // namespace stonks::broker
