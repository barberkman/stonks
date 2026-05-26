#pragma once

#include <concepts>
#include <optional>

#include "stonks/core/types.h"

namespace stonks::core {

template <class DataFeedT>
concept HasNextTimestamp = requires(const DataFeedT& dataFeed) {
    { dataFeed.next_timestamp() } -> std::convertible_to<std::optional<Timestamp>>;
};

template <class DataFeedT>
concept HasAdvance = requires(DataFeedT& dataFeed) {
    { dataFeed.advance() } -> std::same_as<void>;
};

template <class DataFeedT>
concept HasResolution = requires(const DataFeedT& dataFeed) {
    { dataFeed.resolution() } -> std::convertible_to<Timestamp::duration>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT>
    && HasNextTimestamp<DataFeedT>
    && HasAdvance<DataFeedT>
    && HasResolution<DataFeedT>;

} // namespace stonks::core
