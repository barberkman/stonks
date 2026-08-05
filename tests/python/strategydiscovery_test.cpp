// discover_strategies() against the test fixtures directory: single-class
// modules resolve (with their declared param specs), ambiguous modules are
// skipped, and paramless strategies ship an empty spec vector.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "stonks/python/embeddedpython.h"

#include "strategies/strategydiscovery.h"

namespace {

using stonks::app::StrategyInfo;
using stonks::app::discover_strategies;

void ensure_python_setup()
{
    ::setenv("STONKS_VENV", STONKS_VENV_DIR, 0);
    static stonks::python::EmbeddedPython python_handle;
    static const bool paths_added = []() {
        stonks::python::EmbeddedPython::add_sys_path(STONKS_PYTHON_PACKAGE_DIR);
        stonks::python::EmbeddedPython::add_sys_path(STONKS_TEST_FIXTURES_DIR);
        return true;
    }();
    (void)python_handle;
    (void)paths_added;
}

class StrategyDiscoveryTest : public ::testing::Test
{
protected:
    void SetUp() override { ensure_python_setup(); }
};

const StrategyInfo* find(const std::vector<StrategyInfo>& all, const std::string& module)
{
    const auto it = std::ranges::find(all, module, &StrategyInfo::module);
    return it == all.end() ? nullptr : &*it;
}

TEST_F(StrategyDiscoveryTest, ParamDeclaringStrategyExposesSpecs)
{
    const auto all = discover_strategies(STONKS_TEST_FIXTURES_DIR);
    const auto* info = find(all, "paramfixture");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->cls, "ParamFixture");

    ASSERT_EQ(info->params.size(), 4u);          // declaration order preserved
    EXPECT_EQ(info->params[0].name, "risk");
    EXPECT_DOUBLE_EQ(info->params[0].default_value, 0.05);
    EXPECT_EQ(info->params[0].type_name, "float");
    EXPECT_EQ(info->params[0].doc, "risk per trade");
    EXPECT_EQ(info->params[0].unit, "%");
    EXPECT_EQ(info->params[1].name, "lookback");
    EXPECT_DOUBLE_EQ(info->params[1].default_value, 30.0);
    EXPECT_EQ(info->params[1].type_name, "int");
    EXPECT_EQ(info->params[2].name, "use_trend");
    EXPECT_DOUBLE_EQ(info->params[2].default_value, 0.0);   // False -> 0
    EXPECT_EQ(info->params[2].type_name, "bool");
}

// A param with named alternatives still travels as a number — the index — so
// the override transport is unchanged and only the GUI needs the labels.
TEST_F(StrategyDiscoveryTest, ChoiceParamCarriesItsLabelsAndStaysNumeric)
{
    const auto all = discover_strategies(STONKS_TEST_FIXTURES_DIR);
    const auto* info = find(all, "paramfixture");
    ASSERT_NE(info, nullptr);
    ASSERT_EQ(info->params.size(), 4u);

    const stonks::app::ParamSpec& mode = info->params[3];
    EXPECT_EQ(mode.name, "mode");
    EXPECT_EQ(mode.type_name, "int");
    EXPECT_DOUBLE_EQ(mode.default_value, 1.0);   // the index, not the label
    EXPECT_EQ(mode.choices,
              (std::vector<std::string>{ "breakout", "pullback", "reversal" }));
}

// Everything that is not a named selection reports no choices, so the GUI can
// use emptiness alone to decide between a dropdown and a number box.
TEST_F(StrategyDiscoveryTest, OrdinaryParamsHaveNoChoices)
{
    const auto all = discover_strategies(STONKS_TEST_FIXTURES_DIR);
    const auto* info = find(all, "paramfixture");
    ASSERT_NE(info, nullptr);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(info->params[i].choices.empty()) << info->params[i].name;
    }
}

TEST_F(StrategyDiscoveryTest, NonDeclaringStrategyHasEmptyParamsVector)
{
    const auto all = discover_strategies(STONKS_TEST_FIXTURES_DIR);
    const auto* info = find(all, "noparamfixture");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->cls, "NoParamFixture");
    EXPECT_TRUE(info->params.empty());
}

TEST_F(StrategyDiscoveryTest, AmbiguousModuleStillSkipped)
{
    // fixturestrats.py defines many Strategy subclasses: the resolver must
    // keep treating that as ambiguous under the new tuple-returning shape.
    const auto all = discover_strategies(STONKS_TEST_FIXTURES_DIR);
    EXPECT_EQ(find(all, "fixturestrats"), nullptr);
}

TEST_F(StrategyDiscoveryTest, FileAddedAfterAFirstScanIsFoundBySecondScan)
{
    // The GUI re-scans app/python when the setup view opens, so a strategy
    // dropped in while the app runs must appear without a restart. That relies
    // on discover_strategies() holding no cache of its own.
    const auto dir = std::filesystem::temp_directory_path() / "stonks_discovery_rescan";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    stonks::python::EmbeddedPython::add_sys_path(dir.string());

    EXPECT_TRUE(discover_strategies(dir).empty());

    {
        std::ofstream out{ dir / "latefixture.py" };
        out << "import stonks\n\n\n"
            << "class LateFixture(stonks.Strategy):\n"
            << "    params = { \"span\": stonks.Param(\"bars\", \"\") }\n"
            << "    span = 14\n\n"
            << "    def on_tick(self, ctx):\n"
            << "        pass\n";
    }

    const auto all = discover_strategies(dir);
    const auto* info = find(all, "latefixture");
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->cls, "LateFixture");
    ASSERT_EQ(info->params.size(), 1u);
    EXPECT_EQ(info->params[0].name, "span");
    EXPECT_DOUBLE_EQ(info->params[0].default_value, 14.0);
    EXPECT_EQ(info->params[0].type_name, "int");

    std::filesystem::remove_all(dir);
}

} // namespace
