#pragma once

#include <concepts>
#include <utility>

#include "stonks/core/types.h"

namespace stonks::core {

class Context;

template <class S>
concept HasOnStart = requires(S strategy, Context& context) { strategy.on_start(context); };

template <class S>
concept HasOnStop = requires(S strategy, Context& context) { strategy.on_stop(context); };

template <class S>
concept HasOnKLine = requires(S strategy, Context& context) { strategy.on_kline(context); };

template <class S>
concept Strategy = std::movable<S> && HasOnKLine<S>;

}