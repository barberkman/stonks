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

    OrderID place_order(const OrderParams& parameters) { return m_broker.place_order(parameters); }  // entry (may carry SL/TP)

    // Market-close the position on a symbol at the next bar's open; false if flat.
    bool close(const Symbol& symbol) { return m_broker.close(symbol); }

    // Retarget the SL/TP levels on the symbol's resting entry or live position
    // (replaces both; nullopt clears that level). False if there is nothing to update.
    bool update_exits(const Symbol& symbol, std::optional<Price> stop_loss, std::optional<Price> take_profit)
    {
        return m_broker.update_exits(symbol, stop_loss, take_profit);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
