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

    // Helper run inside the embedded interpreter: import a module and return
    // (class_name, param_specs) for its single stonks.Strategy subclass
    // *defined in that module*, or ("", []) if there is none, more than one,
    // or the import fails. The __module__ == modname guard ignores the
    // imported base class and enums. Spec extraction lives in the framework
    // (stonks.param_specs) so it stays pytest-testable; a broken `params`
    // declaration raises there and drops the strategy like an import failure.
    static const char* const kResolver = R"PY(
import importlib, inspect
import stonks

def _resolve(modname):
    try:
        m = importlib.import_module(modname)
    except Exception:
        return ("", [])
    found = None
    for _name, obj in inspect.getmembers(m, inspect.isclass):
        if (issubclass(obj, stonks.Strategy) and obj is not stonks.Strategy
                and getattr(obj, "__module__", "") == modname):
            if found is not None:
                return ("", [])   # ambiguous: more than one strategy class
            found = obj
    if found is None:
        return ("", [])
    return (found.__name__, stonks.param_specs(found))
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
        std::vector<ParamSpec> params;
        try {
            py::tuple result = resolve(stem).cast<py::tuple>();
            cls = result[0].cast<std::string>();
            if (!cls.empty()) {
                for (auto item : result[1]) {
                    py::dict d = item.cast<py::dict>();
                    const std::string type_name = d["type"].cast<std::string>();
                    double default_value = 0.0;
                    if (type_name == "bool") {
                        default_value = d["default"].cast<bool>() ? 1.0 : 0.0;
                    } else if (type_name == "int") {
                        default_value = static_cast<double>(d["default"].cast<long long>());
                    } else {
                        default_value = d["default"].cast<double>();
                    }
                    // `choices` is newer than this reader's oldest callers, so
                    // a spec without it is still a valid spec.
                    std::vector<std::string> choices;
                    if (d.contains("choices")) {
                        for (const auto& c : d["choices"].cast<py::list>()) {
                            choices.push_back(c.cast<std::string>());
                        }
                    }
                    params.push_back(ParamSpec{
                        d["name"].cast<std::string>(),
                        default_value,
                        type_name,
                        d["doc"].cast<std::string>(),
                        d["unit"].cast<std::string>(),
                        std::move(choices),
                    });
                }
            }
        } catch (py::error_already_set&) {
            continue;
        }
        if (!cls.empty()) {
            out.push_back(StrategyInfo{ cls, stem, cls, std::move(params) });
        }
    }
    return out;
}

} // namespace stonks::app
