#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <string>

#include "stonks/core/datafeed.h"
#include "stonks/core/progressbar.h"

#include "test_stubs.h"

namespace stonks::core {

namespace {

using namespace std::chrono_literals;

int count_occurrences(const std::string& haystack, const std::string& needle)
{
    int n = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++n;
        pos += needle.size();
    }
    return n;
}

// Optional-capability contract: StubFeed reports a size, a bare feed does not.
struct NoSizeFeed {};
static_assert(HasSize<test::StubFeed>);
static_assert(!HasSize<NoSizeFeed>);

} // namespace

TEST(ProgressBar, KnownTotalHalfwayShowsPercentAndCounts) {
    const auto s = format_progress(100, 50, 1s);
    EXPECT_NE(s.find("50%"), std::string::npos);
    EXPECT_NE(s.find("50/100"), std::string::npos);
}

TEST(ProgressBar, KnownTotalZeroPercentAndCompleteEndpoints) {
    const auto zero = format_progress(100, 0, 1s);
    EXPECT_NE(zero.find("0%"), std::string::npos);
    EXPECT_NE(zero.find("0/100"), std::string::npos);

    const auto full = format_progress(100, 100, 2s);
    EXPECT_NE(full.find("100%"), std::string::npos);
    EXPECT_NE(full.find("100/100"), std::string::npos);
}

TEST(ProgressBar, BarFillIsProportionalToFraction) {
    // 50/100 -> 0.5 * 22 cols = exactly 11 full blocks, no partial.
    const auto half = format_progress(100, 50, 1s);
    EXPECT_EQ(count_occurrences(half, "█"), 11);

    // 1/8 -> 0.125 * 22 = 2.75 cols -> 2 full blocks + a 6/8 partial block.
    const auto eighth = format_progress(8, 1, 1s);
    EXPECT_EQ(count_occurrences(eighth, "█"), 2);
    EXPECT_NE(eighth.find("▊"), std::string::npos);
}

TEST(ProgressBar, RateAndEtaComputedFromElapsed) {
    // 20/100 in 2s -> 10 bars/s, remaining 80 bars at 10/s -> 8s ETA.
    const auto s = format_progress(100, 20, 2s);
    EXPECT_NE(s.find("[00:02<00:08, 10 bars/s]"), std::string::npos);
}

TEST(ProgressBar, ZeroElapsedDoesNotDivideByZero) {
    const auto s = format_progress(100, 50, 0ns);
    EXPECT_NE(s.find("50%"), std::string::npos);
    EXPECT_NE(s.find("[00:00<00:00, 0 bars/s]"), std::string::npos);
}

TEST(ProgressBar, DurationFormatsMinutesAndHours) {
    EXPECT_NE(format_progress(100, 10, 65s).find("01:05"), std::string::npos);
    EXPECT_NE(format_progress(100, 50, 3661s).find("1:01:01"), std::string::npos);
}

TEST(ProgressBar, CountOnlyModeHasNoPercentage) {
    const auto s = format_progress(std::nullopt, 1500, 3s);
    EXPECT_EQ(s.find('%'), std::string::npos);
    EXPECT_NE(s.find("1500 bars"), std::string::npos);
    EXPECT_NE(s.find("500 bars/s"), std::string::npos);
    EXPECT_NE(s.find("00:03"), std::string::npos);
}

TEST(ProgressBar, ZeroTotalRendersComplete) {
    const auto s = format_progress(0, 0, 1s);
    EXPECT_NE(s.find("100%"), std::string::npos);
    EXPECT_NE(s.find("0/0"), std::string::npos);
}

TEST(ProgressBar, UnitIsCustomizable) {
    const auto s = format_progress(100, 50, 1s, "ticks");
    EXPECT_NE(s.find("ticks/s"), std::string::npos);
}

// --- ProgressState snapshot (consumed by a GUI) ------------------------------

TEST(ProgressState, ComputeKnownTotalDerivesPercentRateAndEta) {
    // 20/100 in 2s -> 20%, 10 bars/s, 80 remaining at 10/s -> 8s ETA.
    const auto st = compute_progress(100, 20, 2s);
    ASSERT_TRUE(st.total.has_value());
    EXPECT_EQ(*st.total, 100u);
    EXPECT_EQ(st.current, 20u);
    EXPECT_EQ(st.percent, 20);
    EXPECT_DOUBLE_EQ(st.rate, 10.0);
    EXPECT_EQ(st.eta, 8s);
    EXPECT_EQ(st.elapsed, 2s);
}

TEST(ProgressState, ComputeZeroTotalIsComplete) {
    const auto st = compute_progress(0, 0, 1s);
    EXPECT_EQ(st.percent, 100);
    EXPECT_EQ(st.eta, 0ns);
}

TEST(ProgressState, ComputeUnknownTotalHasNoPercentOrEta) {
    const auto st = compute_progress(std::nullopt, 1500, 3s);
    EXPECT_FALSE(st.total.has_value());
    EXPECT_EQ(st.percent, -1);
    EXPECT_EQ(st.eta, 0ns);
    EXPECT_DOUBLE_EQ(st.rate, 500.0); // 1500 / 3s
    EXPECT_EQ(st.current, 1500u);
}

TEST(ProgressState, DefaultIsZeroedAndUnknown) {
    const ProgressState st;
    EXPECT_FALSE(st.total.has_value());
    EXPECT_EQ(st.current, 0u);
    EXPECT_EQ(st.percent, -1);
}

// update() records current/total in any mode (even when not rendering, as under
// ctest where stderr is not a TTY), so state() is valid for a silent GUI.
TEST(ProgressBar, StateTracksCurrentAndTotalInConsoleMode) {
    ProgressBar bar{ 100, "bars", ProgressOutput::Console };
    bar.update(40);
    const auto st = bar.state();
    ASSERT_TRUE(st.total.has_value());
    EXPECT_EQ(*st.total, 100u);
    EXPECT_EQ(st.current, 40u);
    EXPECT_EQ(st.percent, 40);
}

TEST(ProgressBar, StateTracksCurrentAndTotalInSilentMode) {
    ProgressBar bar{ 100, "bars", ProgressOutput::Silent };
    bar.update(75);
    const auto st = bar.state();
    ASSERT_TRUE(st.total.has_value());
    EXPECT_EQ(*st.total, 100u);
    EXPECT_EQ(st.current, 75u);
    EXPECT_EQ(st.percent, 75);
}

} // namespace stonks::core
