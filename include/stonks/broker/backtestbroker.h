#pragma once

#include "stonks/core/types.h"

namespace stonks::broker {

class BacktestBroker
{
public:
    core::Balance cash() const;
    core::Balance equity() const;
    bool place_order(const core::Order& order);
};

} // namespace stonks::broker
