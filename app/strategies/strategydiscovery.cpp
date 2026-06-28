#include "strategies/strategydiscovery.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace stonks::app {

std::vector<StrategyInfo> discover_strategies(const std::filesystem::path& dir)
{
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    // Helper run inside the embedded interpreter: import a module and return the
    // name of its single stonks.Strategy subclass *defined in that module*, or
    // "" if there is none, more than one, or the import fails. The
    // __module__ == modname guard ignores the imported base class and enums.
    static const char* const kResolver = R"PY(
import importlib, inspect
import stonks

def _resolve(modname):
    try:
        m = importlib.import_module(modname)
    except Exception:
        return ""
    found = ""
    for _name, obj in inspect.getmembers(m, inspect.isclass):
        if (issubclass(obj, stonks.Strategy) and obj is not stonks.Strategy
                and getattr(obj, "__module__", "") == modname):
            if found:
                return ""   # ambiguous: more than one strategy class
            found = obj.__name__
    return found
)PY";

    py::dict ns;
    try {
        py::exec(kResolver, ns);
    } catch (py::error_already_set& e) {
        throw std::runtime_error(std::string{ "strategy discovery helper failed: " } + e.what());
    }
    auto resolve = ns["_resolve"];

    // Candidate module stems from *.py files (skip tests / dunder files).
    std::vector<std::string> stems;
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file()) { continue; }
            const auto& path = entry.path();
            if (path.extension() != ".py") { continue; }
            const std::string stem = path.stem().string();
            if (stem.starts_with("test_") || stem.ends_with("_test")) { continue; }
            if (stem.starts_with("__")) { continue; }
            stems.push_back(stem);
        }
    }
    std::ranges::sort(stems);

    std::vector<StrategyInfo> out;
    for (const auto& stem : stems) {
        std::string cls;
        try {
            cls = resolve(stem).cast<std::string>();
        } catch (py::error_already_set&) {
            continue;
        }
        if (!cls.empty()) {
            out.push_back(StrategyInfo{ cls, stem, cls });
        }
    }
    return out;
}

} // namespace stonks::app
