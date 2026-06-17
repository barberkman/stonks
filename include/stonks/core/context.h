#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/datafeed.h"
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
    MarketWindow history(int count) const { return m_dataFeed.window(count); }

    Order make_market_order(MarketOrderParams params)
    {
        return Order{
            OrderID{ m_next_order_id++ },
            m_clock.now(),
            std::move(params.symbol),
            params.side,
            OrderType::Market,
            std::nullopt,
            params.quantity,
            params.time_in_force
        };
    }

    Order make_limit_order(LimitOrderParams params)
    {
        return Order{
            OrderID{ m_next_order_id++ },
            m_clock.now(),
            std::move(params.symbol),
            params.side,
            OrderType::Limit,
            params.price,
            params.quantity,
            params.time_in_force
        };
    }

    bool place_order(const Order& order)
    {
        return m_broker.place_order(order);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
    std::uint64_t m_next_order_id{ 1 };
};

} // namespace stonks::core
