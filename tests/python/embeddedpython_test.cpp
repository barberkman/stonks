#include <optional>

#include <gtest/gtest.h>

#include "stonks/python/embeddedpython.h"

// $STONKS_DEBUGPY gating. The listener itself can't be exercised here — it
// blocks until an editor attaches — so these cover the decision that precedes
// it: when to open a listener at all, and on which port.
namespace {

using stonks::python::EmbeddedPython;

TEST(DebugpyPortTest, UnsetOrEmptyDisablesTheListener)
{
    EXPECT_EQ(EmbeddedPython::debugpy_port(nullptr), std::nullopt);
    EXPECT_EQ(EmbeddedPython::debugpy_port(""), std::nullopt);
}

TEST(DebugpyPortTest, ZeroDisablesTheListener)
{
    // STONKS_DEBUGPY=0 reads as "off" rather than as port 0.
    EXPECT_EQ(EmbeddedPython::debugpy_port("0"), std::nullopt);
}

TEST(DebugpyPortTest, UnprivilegedPortNumberIsTakenLiterally)
{
    EXPECT_EQ(EmbeddedPython::debugpy_port("5678"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("1024"), 1024);
    EXPECT_EQ(EmbeddedPython::debugpy_port("65535"), 65535);
}

TEST(DebugpyPortTest, TruthyNonPortValuesUseTheDefaultPort)
{
    EXPECT_EQ(EmbeddedPython::debugpy_port("1"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("on"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("yes"), 5678);
}

TEST(DebugpyPortTest, OutOfRangeAndMalformedValuesUseTheDefaultPort)
{
    // Privileged, over-max, negative, and trailing-garbage values all fall back
    // rather than handing an unusable port to debugpy.listen().
    EXPECT_EQ(EmbeddedPython::debugpy_port("80"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("70000"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("-1"), 5678);
    EXPECT_EQ(EmbeddedPython::debugpy_port("5678x"), 5678);
}

} // namespace
