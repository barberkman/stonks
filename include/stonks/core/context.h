#pragma once

#include <iostream>
#include <vector>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/types.h"

namespace stonks::core {

template<Broker BrokerT, DataFeed DataFeedT>
class Context
{
public:
    explicit Context(BrokerT& broker, const DataFeedT& dataFeed, const Clock& clock)
    : m_broker{ broker },
    m_dataFeed{ dataFeed },
    m_clock{ clock }
    {}

    Timestamp now() const
    {
        return m_clock.now();
    }

    Balance cash() const
    {
        return m_broker.cash();
    }

    Balance equity() const
    {
        return m_broker.equity();
    }

    std::vector<KLine> klines(int count) const
    {
        return m_dataFeed.klines(count);
    }

    bool place_order(const Order& order)
    {
        return m_broker.place_order(order);
    }

private:
    BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
