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
    explicit Context(const BrokerT& broker, const DataFeedT& dataFeed, const Clock& clock)
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

    std::vector<KLine> kline(int count)
    {
        return m_dataFeed.kline(count);
    }

private:
    const BrokerT& m_broker;
    const DataFeedT& m_dataFeed;
    const Clock& m_clock;
};

} // namespace stonks::core
