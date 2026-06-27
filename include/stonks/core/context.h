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
    MarketWindow history(int count) const
    {
        MarketWindow w = m_dataFeed.window(count);
        return w;
    }

    OrderID place_order(const MarketOrderParams& parameters, std::optional<OrderID> parent = std::nullopt)
    {
        return m_broker.place_order(parameters, parent);
    }

    OrderID place_order(const LimitOrderParams& parameters, std::optional<OrderID> parent = std::nullopt)
    {
        return m_broker.place_order(parameters, parent);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
