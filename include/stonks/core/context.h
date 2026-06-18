#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/log.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <class BrokerT, class DataFeedT>
class Context
{
    static_assert(Broker<BrokerT>, "BrokerT must satisfy the Broker concept");
    static_assert(DataFeed<DataFeedT>, "DataFeedT must satisfy the DataFeed concept");

public:
    Context(BrokerT& broker, const DataFeedT& dataFeed, const Clock& clock)
    : m_broker{ broker },
      m_dataFeed{ dataFeed },
      m_clock{ clock }
    {}

    Timestamp now() const { return m_clock.now(); }
    Balance cash() const { return m_broker.cash(); }
    Balance equity() const { return m_broker.equity(); }

    // This tick's window: every symbol that printed at the current timestamp,
    // each with its last `count` bars (including today's). No-lookahead by
    // construction; see KLineFeed::window.
    MarketWindow history(int count) const
    {
        MarketWindow w = m_dataFeed.window(count);
        STONKS_LOG("ctx", "history count={} -> series={} ts={}",
            count, w.size(), m_clock.now().value.time_since_epoch().count());
        return w;
    }

    Order make_market_order(MarketOrderParams params)
    {
        Order order{
            OrderID{ m_next_order_id++ },
            m_clock.now(),
            std::move(params.symbol),
            params.side,
            OrderType::Market,
            std::nullopt,
            params.quantity,
            params.time_in_force
        };
        STONKS_LOG("ctx", "make market order id={} sym={} side={} qty={:.6f} ts={}",
            order.id, order.symbol, order.side == OrderSide::Buy ? "Buy" : "Sell",
            order.quantity, order.timestamp.value.time_since_epoch().count());
        return order;
    }

    Order make_limit_order(LimitOrderParams params)
    {
        Order order{
            OrderID{ m_next_order_id++ },
            m_clock.now(),
            std::move(params.symbol),
            params.side,
            OrderType::Limit,
            params.price,
            params.quantity,
            params.time_in_force
        };
        STONKS_LOG("ctx", "make limit order id={} sym={} side={} qty={:.6f} price={:.4f} ts={}",
            order.id, order.symbol, order.side == OrderSide::Buy ? "Buy" : "Sell",
            order.quantity, order.price.value_or(0.0),
            order.timestamp.value.time_since_epoch().count());
        return order;
    }

    bool place_order(const Order& order)
    {
        STONKS_LOG("ctx", "place order id={} sym={} -> broker", order.id, order.symbol);
        return m_broker.place_order(order);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
    std::uint64_t m_next_order_id{ 1 };
};

} // namespace stonks::core
