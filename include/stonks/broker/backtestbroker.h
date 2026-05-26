#pragma once

#include <unordered_map>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::broker {

class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash);

    core::Balance cash() const;
    core::Balance equity() const;

    bool place_order(const core::Order& order);
    void on_tick(const core::KLine& bar);

private:
    core::Balance m_cash;
    std::unordered_map<core::Symbol, core::Quantity> m_positions;
    std::unordered_map<core::Symbol, core::Price> m_last_price;
    std::vector<core::Order> m_open_orders;
};

} // namespace stonks::broker
