#pragma once

#include <concepts>

#include "stonks/core/types.h"

namespace stonks::core {

template <class BrokerT>
concept HasCash = requires(const BrokerT& broker) {
    { broker.cash() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept HasEquity = requires(const BrokerT& broker) {
    { broker.equity() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept Broker = std::movable<BrokerT> && HasCash<BrokerT> && HasEquity<BrokerT>;

} // namespace stonks::core
