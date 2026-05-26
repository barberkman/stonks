#pragma once

#include <concepts>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core {

template <class ContextT>
concept ContextHasNow = requires(const ContextT& context) {
    { context.now() } -> std::convertible_to<Timestamp>;
};

template <class ContextT>
concept ContextHasCash = requires(const ContextT& context) {
    { context.cash() } -> std::convertible_to<Balance>;
};

template <class ContextT>
concept ContextHasEquity = requires(const ContextT& context) {
    { context.equity() } -> std::convertible_to<Balance>;
};

template <class ContextT>
concept ContextHasKLine = requires(const ContextT& context, int count) {
    { context.kline(count) } -> std::same_as<std::vector<KLine>>;
};

template <class ContextT>
concept Context = std::movable<ContextT> && ContextHasNow<ContextT> && ContextHasCash<ContextT> && ContextHasEquity<ContextT> && ContextHasKLine<ContextT>;

} // namespace stonks::core
