#pragma once

#include <concepts>

#include "stonks/core/context.h"

namespace stonks::core {

template <class StrategyT, class ContextT>
concept StrategyHasOnStart = Context<ContextT> && requires(StrategyT strategy, ContextT& context) { strategy.on_start(context); };

template <class StrategyT, class ContextT>
concept StrategyHasOnStop = Context<ContextT> && requires(StrategyT strategy, ContextT& context) { strategy.on_stop(context); };

template <class StrategyT, class ContextT>
concept StrategyHasOnKLine = Context<ContextT> && requires(StrategyT strategy, ContextT& context) { strategy.on_kline(context); };

template <class StrategyT>
concept Strategy = std::movable<StrategyT>;

} // namespace stonks::core
