#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace stonks::app {

// One GUI-editable strategy parameter, extracted from stonks.param_specs().
struct ParamSpec
{
    std::string name;
    double default_value{};   // bool as 0/1, matching the override transport
    std::string type_name;    // "float" | "int" | "bool"
    std::string doc;
    std::string unit;
};

// A selectable Python strategy: its file stem (the import module name), the
// strategy class to instantiate, and its declared GUI-editable parameters.
// `display` is what the GUI shows.
struct StrategyInfo
{
    std::string display;   // class name, e.g. "QMLiteralStrategy"
    std::string module;    // file stem, e.g. "qmliteral"
    std::string cls;       // class to instantiate (same as display today)
    std::vector<ParamSpec> params;   // empty when the strategy declares none
};

// Globs `dir` for *.py files and resolves each to its single stonks.Strategy
// subclass via the embedded interpreter (importlib + inspect). Requires an
// initialized interpreter (an EmbeddedPython must be alive); the caller's thread
// must own the GIL is NOT required — this acquires it internally. Files that
// fail to import or contain no (or more than one) strategy class are skipped.
std::vector<StrategyInfo> discover_strategies(const std::filesystem::path& dir = "app/python");

} // namespace stonks::app
