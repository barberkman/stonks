#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <vector>

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
concept HasCurrentKLine = requires(const DataFeedT& dataFeed) {
    { dataFeed.current_kline() } -> std::convertible_to<KLine>;
};

template <class DataFeedT>
concept HasKLines = requires(const DataFeedT& dataFeed, Timestamp start, Timestamp end) {
    { dataFeed.klines(start, end) } -> std::convertible_to<std::vector<KLine>>;
};

// Optional capability: a feed that knows its total bar count up front. Used by
// the engine to render a percentage/ETA progress bar; not required by DataFeed.
template <class DataFeedT>
concept HasSize = requires(const DataFeedT& dataFeed) {
    { dataFeed.size() } -> std::convertible_to<std::size_t>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT>
    && HasNextTimestamp<DataFeedT>
    && HasAdvance<DataFeedT>
    && HasResolution<DataFeedT>
    && HasCurrentKLine<DataFeedT>
    && HasKLines<DataFeedT>;

} // namespace stonks::core
