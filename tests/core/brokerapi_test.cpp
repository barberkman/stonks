// P6 observability API on BacktestBroker: position query, cancel_order,
// attaching protection to an already-filled parent, and reduce-only semantics
// (a reduce-only order may shrink an opposite-side position but is cancelled —
// never filled — when it would open or add).

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

// --- Position query ------------------------------------------------------------

TEST(BrokerAPI, PositionReflectsLifecycleFlatOpenClosed)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    EXPECT_FALSE(broker.position("X").has_value());   // flat book

    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0, .leverage = 4.0 });
    broker.on_tick(flat(1000, 100.0));

    const auto pos = broker.position("X");
    ASSERT_TRUE(pos.has_value());
    EXPECT_DOUBLE_EQ(pos->quantity, 2.0);
    EXPECT_DOUBLE_EQ(pos->price, 100.0);
    EXPECT_DOUBLE_EQ(pos->leverage, 4.0);
    EXPECT_EQ(pos->entry_id, entry);
    EXPECT_FALSE(broker.position("Y").has_value());   // other symbols unaffected

    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0 });
    broker.on_tick(flat(2000, 105.0));
    EXPECT_FALSE(broker.position("X").has_value());   // closed -> flat again
}

// --- cancel_order ----------------------------------------------------------------

TEST(BrokerAPI, CancelOrderCancelsRestingOrderAndItsDormantChildren)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto entry = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 110.0 });
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0, .reduce_only = true }, entry);

    EXPECT_TRUE(broker.cancel_order(entry));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Cancelled);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Cancelled);

    // A bar that would have filled the entry does nothing: it left the working set.
    broker.on_tick(bar(2000, 109.0, 112.0, 108.0, 111.0));
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_FALSE(broker.position("X").has_value());
}

TEST(BrokerAPI, CancelOrderReturnsFalseForUnknownFilledOrAlreadyCancelled)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    EXPECT_FALSE(broker.cancel_order(OrderID{ 424242 }));   // unknown id

    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(1000, 100.0));
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);
    EXPECT_FALSE(broker.cancel_order(entry));               // already filled

    const auto resting = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 200.0 });
    EXPECT_TRUE(broker.cancel_order(resting));
    EXPECT_FALSE(broker.cancel_order(resting));             // already cancelled
}

TEST(BrokerAPI, CancelOrderRefusedOnceBankrupt)
{
    BacktestBroker broker{ Balance{ 100.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "A", .side = OrderSide::Buy, .quantity = 50.0, .leverage = 10.0 });
    const auto resting = broker.place_order(LimitOrderParams{ .symbol = "C", .side = OrderSide::Buy, .quantity = 1.0, .price = 50.0 });
    broker.on_tick(flat(2000, Symbol{ "A" }, 10.0));                  // margin 50
    broker.on_tick(bar(3000, Symbol{ "A" }, 1.0, 1.5, 0.8, 1.2));     // gap-through -> bankrupt
    ASSERT_TRUE(broker.bankrupt());
    EXPECT_FALSE(broker.cancel_order(resting));   // account halted; everything already swept
}

// --- Attaching protection after the fill -------------------------------------------

TEST(BrokerAPI, ChildAttachedToFilledParentIsAcceptedAndFills)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(1000, 100.0));            // entry fills; no bracket yet
    EXPECT_EQ(broker.orders().at(entry).status, OrderStatus::Filled);

    // Protection added to the live position, parented under the filled entry.
    const auto sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0, .reduce_only = true }, entry);
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Open);

    broker.on_tick(bar(2000, 98.0, 99.0, 94.0, 96.0));   // trigger breached -> exits at 95
    EXPECT_EQ(broker.orders().at(sl).status, OrderStatus::Filled);
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 9'995.0);
}

TEST(BrokerAPI, ChildAttachedToRejectedOrCancelledParentIsRejected)
{
    BacktestBroker broker{ Balance{ 100.0 } };
    const auto poor = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 10.0 });
    broker.on_tick(flat(1000, 100.0));            // cost 1000 > 100 -> rejected
    EXPECT_EQ(broker.orders().at(poor).status, OrderStatus::Rejected);
    const auto onto_rejected = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 90.0 }, poor);
    EXPECT_EQ(broker.orders().at(onto_rejected).status, OrderStatus::Rejected);

    const auto resting = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 0.1, .price = 50.0 });
    ASSERT_TRUE(broker.cancel_order(resting));
    const auto onto_cancelled = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 0.1, .price = 40.0 }, resting);
    EXPECT_EQ(broker.orders().at(onto_cancelled).status, OrderStatus::Rejected);
}

// --- Reduce-only -------------------------------------------------------------------

TEST(BrokerAPI, ReduceOnlyCancelsInsteadOfOpeningWhenFlat)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.on_tick(flat(1000, 100.0));
    const auto orphan = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0, .reduce_only = true });

    broker.on_tick(bar(2000, 98.0, 99.0, 94.0, 96.0));   // trigger breached on a flat book
    EXPECT_EQ(broker.orders().at(orphan).status, OrderStatus::Cancelled);
    EXPECT_TRUE(broker.trades().empty());
    EXPECT_FALSE(broker.position("X").has_value());      // no phantom short
    EXPECT_DOUBLE_EQ(broker.cash(), 10'000.0);
}

TEST(BrokerAPI, ReduceOnlyCancelsInsteadOfAddingSameSide)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(1000, 100.0));                   // long 1 @100

    const auto same_side = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 99.0, .reduce_only = true });
    broker.on_tick(bar(2000, 100.0, 101.0, 98.0, 100.0));   // limit crossable, but same side
    EXPECT_EQ(broker.orders().at(same_side).status, OrderStatus::Cancelled);   // not Rejected, not Filled
    ASSERT_TRUE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.position("X")->quantity, 1.0);   // unchanged
}

TEST(BrokerAPI, ReduceOnlyExitStillClosesNormally)
{
    BacktestBroker broker{ Balance{ 10'000.0 } };
    const auto entry = broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 2.0 });
    const auto tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 2.0, .price = 110.0, .reduce_only = true }, entry);
    broker.on_tick(flat(1000, 100.0));
    broker.on_tick(bar(2000, 108.0, 112.0, 107.0, 111.0));
    EXPECT_EQ(broker.orders().at(tp).status, OrderStatus::Filled);
    EXPECT_FALSE(broker.position("X").has_value());
    EXPECT_DOUBLE_EQ(broker.cash(), 10'020.0);           // -200 + 220
}

TEST(BrokerAPI, OrphanedBracketLegCanNoLongerOpenAPhantomPosition)
{
    // The S2 cascade regression: a root opposite-side "entry" nets out an open
    // position; its own armed reduce-only children must then die quietly
    // instead of opening an unmanaged position.
    BacktestBroker broker{ Balance{ 10'000.0 } };
    broker.place_order(MarketOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0 });
    broker.on_tick(flat(2000, 100.0));                   // long 1 @100

    const auto netting = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Sell, .quantity = 1.0, .price = 95.0 });
    const auto net_sl = broker.place_order(StopOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 99.0, .reduce_only = true }, netting);
    const auto net_tp = broker.place_order(LimitOrderParams{ .symbol = "X", .side = OrderSide::Buy, .quantity = 1.0, .price = 90.0, .reduce_only = true }, netting);

    // The "short entry" triggers and nets the long to flat; the same bar also
    // touches its SL's trigger — which must cancel, not open a phantom long.
    broker.on_tick(bar(3000, 96.0, 99.5, 94.0, 95.0));
    EXPECT_EQ(broker.orders().at(netting).status, OrderStatus::Filled);
    EXPECT_EQ(broker.orders().at(net_sl).status, OrderStatus::Cancelled);
    EXPECT_FALSE(broker.position("X").has_value());

    // The TP's trigger is touched on a later bar: same fate.
    broker.on_tick(bar(4000, 92.0, 93.0, 89.0, 91.0));
    EXPECT_EQ(broker.orders().at(net_tp).status, OrderStatus::Cancelled);
    EXPECT_FALSE(broker.position("X").has_value());
    ASSERT_EQ(sorted_trades(broker).size(), 2u);         // open + net close, nothing else
    EXPECT_DOUBLE_EQ(broker.cash(), 9'995.0);            // 10'000 - 100 + 95
}

} // namespace stonks::broker
