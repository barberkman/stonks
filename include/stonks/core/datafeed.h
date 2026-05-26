#pragma once

#include <concepts>
#include <optional>

#include "stonks/core/types.h"

namespace stonks::core {

template <class DataFeedT>
concept DataFeedHasPeek = requires(const DataFeedT& dataFeed, Timestamp now) {
    { dataFeed.peek(now) } -> std::convertible_to<std::optional<Timestamp>>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT> && DataFeedHasPeek<DataFeedT>;

} // namespace stonks::core
