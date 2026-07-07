// LiveKlineFeed seeding, no-lookahead (forming candle excluded), tick emission,
// window views, and cancel — all with a controlled clock and fake klines.

#include <gtest/gtest.h>

#include <atomic>
#include <string>

#include <nlohmann/json.hpp>

#include "stonks/binance/liveklinefeed.h"
#include "fake_binance.h"

namespace stonks::binance {
namespace {

using namespace stonks::core;
using json = nlohmann::json;

constexpr std::int64_t kMinute = 60'000;

// A candle whose OHLC all equal its minute index, so tests can identify it.
json candle(std::int64_t open_ms)
{
    const double v = static_cast<double>(open_ms) / kMinute;
    const std::string s = std::to_string(v);
    return json::array({ open_ms, s, s, s, s, "1", open_ms + kMinute });
}

// Responds to seed requests (no startTime -> candles for minutes 1..11) and tick
// requests (startTime -> the single candle at that open time).
Transport klines_transport(test::FakeBinance& fake)
{
    fake.responder = [](const HttpRequest& req, test::FakeBinance&) -> HttpResponse {
        json arr = json::array();
        if (const auto st = test::FakeBinance::param(req, "startTime")) {
            arr.push_back(candle(std::stoll(*st)));
        } else {
            for (std::int64_t open = 1 * kMinute; open <= 11 * kMinute; open += kMinute) {
                arr.push_back(candle(open));
            }
        }
        return HttpResponse{ 200, arr.dump() };
    };
    return fake.transport();
}

TEST(LiveKlineFeed, SeedExcludesFormingCandleAndReportsResolution)
{
    test::FakeBinance fake;
    std::int64_t now = 700'000;   // 11m40s: minute-11 candle (closes at 12m) is still forming
    LiveKlineFeed feed{ test::test_config(), { "BTCUSDT" }, "1m", 20, nullptr,
                        klines_transport(fake), [&now] { return now; } };

    EXPECT_EQ(feed.resolution(), Timestamp::duration{ kMinute });

    // Minutes 1..10 are closed (10 candles); minute 11 is forming and excluded.
    const MarketWindow w = feed.window(100);
    ASSERT_EQ(w.series.size(), 1u);
    EXPECT_EQ(w.series[0].bars.size(), 10u);
}

TEST(LiveKlineFeed, EmitsNextClosedCandleAndRollsItIntoTheWindow)
{
    test::FakeBinance fake;
    std::int64_t now = 700'000;
    LiveKlineFeed feed{ test::test_config(), { "BTCUSDT" }, "1m", 20, nullptr,
                        klines_transport(fake), [&now] { return now; } };

    // Cursor is minute 11 (open 660000); advance the clock past its close + grace.
    now = 11 * kMinute + kMinute + 2'000;

    const auto ts = feed.next_timestamp();
    ASSERT_TRUE(ts.has_value());
    EXPECT_EQ(ts->value.time_since_epoch().count(), 11 * kMinute);

    const auto bars = feed.current_bars();
    ASSERT_EQ(bars.size(), 1u);
    EXPECT_EQ(bars[0].symbol, "BTCUSDT");
    EXPECT_DOUBLE_EQ(bars[0].close, 11.0);

    // Window now ends on minute 11; the last three are minutes 9,10,11.
    const MarketWindow w = feed.window(3);
    ASSERT_EQ(w.series.size(), 1u);
    ASSERT_EQ(w.series[0].bars.size(), 3u);
    EXPECT_DOUBLE_EQ(w.series[0].bars.close[2], 11.0);

    feed.advance();
    EXPECT_TRUE(feed.current_bars().empty());
}

TEST(LiveKlineFeed, CancelEndsTheFeed)
{
    test::FakeBinance fake;
    std::int64_t now = 700'000;   // before the cursor candle closes -> would block
    std::atomic<bool> cancel{ false };
    LiveKlineFeed feed{ test::test_config(), { "BTCUSDT" }, "1m", 20, &cancel,
                        klines_transport(fake), [&now] { return now; } };

    cancel.store(true);
    EXPECT_FALSE(feed.next_timestamp().has_value());   // returns end-of-feed, no sleep
}

} // namespace
} // namespace stonks::binance
