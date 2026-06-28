#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace stonks::app {

// A selectable Python strategy: its file stem (the import module name) and the
// strategy class to instantiate. `display` is what the GUI shows.
struct StrategyInfo
{
    std::string display;   // class name, e.g. "EMA50Strategy"
    std::string module;    // file stem, e.g. "ema50strategy"
    std::string cls;       // class to instantiate (same as display today)
};

// Globs `dir` for *.py files and resolves each to its single stonks.Strategy
// subclass via the embedded interpreter (importlib + inspect). Requires an
// initialized interpreter (an EmbeddedPython must be alive); the caller's thread
// must own the GIL is NOT required — this acquires it internally. Files that
// fail to import or contain no (or more than one) strategy class are skipped.
std::vector<StrategyInfo> discover_strategies(const std::filesystem::path& dir = "app/python");

} // namespace stonks::app
