#pragma once

#include <cstdint>
#include <optional>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/log.h"
#include "stonks/core/logfmt.h"
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
        STONKS_LOG("ctx", "ev=place_req type=Market sym={} side={} qty={:.6f} parent={} now={}",
                   parameters.symbol, stonks::log::side_str(parameters.side), parameters.quantity,
                   parent.value_or(0), stonks::log::ts_ms(m_clock.now()));
        return m_broker.place_order(parameters, parent);
    }

    OrderID place_order(const LimitOrderParams& parameters, std::optional<OrderID> parent = std::nullopt)
    {
        STONKS_LOG("ctx", "ev=place_req type=Limit sym={} side={} qty={:.6f} price={:.4f} parent={} now={}",
                   parameters.symbol, stonks::log::side_str(parameters.side), parameters.quantity,
                   parameters.price, parent.value_or(0), stonks::log::ts_ms(m_clock.now()));
        return m_broker.place_order(parameters, parent);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
