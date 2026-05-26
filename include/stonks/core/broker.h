#pragma once

#include <concepts>

#include "stonks/core/types.h"

namespace stonks::core {

template <class BrokerT>
concept BrokerHasCash = requires(const BrokerT& broker) {
    { broker.cash() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept BrokerHasEquity = requires(const BrokerT& broker) {
    { broker.equity() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept Broker = std::movable<BrokerT> && BrokerHasCash<BrokerT> && BrokerHasEquity<BrokerT>;

} // namespace stonks::core
