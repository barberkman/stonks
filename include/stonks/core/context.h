#pragma once

#include <optional>

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

    // The current position on a symbol, or nullopt if flat. Lets a strategy
    // attach exits to (or manage) a position it already holds.
    std::optional<Position> position(const Symbol& symbol) const { return m_broker.position(symbol); }

    // This tick's window: every symbol that printed at the current timestamp,
    // each with its last `count` bars. No-lookahead by construction.
    MarketWindow history(int count) const { return m_dataFeed.window(count); }

    OrderID place_order(const OrderParams& parameters) { return m_broker.place_order(parameters); }  // entry
    OrderID place_exit (const OrderParams& parameters) { return m_broker.place_exit(parameters); }   // reduce-only exit

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
