#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include "stonks/python/contextadapter.h"
#include "stonks/python/embeddedpython.h"
#include "stonks/python/icontext.h"

// Strategy adapter that loads a user-authored Python class and forwards engine
// callbacks to it. The Python class must define on_tick(ctx); on_start(ctx)
// and on_stop(ctx) are optional.
//
// Construction: ("mypkg.module", "ClassName") imports the module and
// instantiates the class eagerly — import / attribute errors surface here, not
// mid-run. The presence of optional hooks is cached at construction so we
// don't pay a hasattr() lookup on every tick.
//
// `overrides` carries per-run parameter values chosen in the GUI. Each key must
// be declared in the class's `params` dict; the value is coerced to the
// declared default's Python type (bool: nonzero, int: truncated, float: as-is)
// and set on the instance — before on_start, so strategies see final values.
//
// The visibility attribute matches pybind11's hidden visibility for py::object;
// without it GCC warns the wrapper exposes a hidden-visibility type.
struct __attribute__((visibility("hidden"))) PythonStrategy
{
    PythonStrategy(std::string module, std::string cls,
                   std::map<std::string, double> overrides = {})
    : m_module_name{ std::move(module) }, m_class_name{ std::move(cls) }
    {
        pybind11::gil_scoped_acquire gil;
        try {
            auto mod = pybind11::module_::import(m_module_name.c_str());
            m_py_instance = mod.attr(m_class_name.c_str())();
        } catch (pybind11::error_already_set& e) {
            throw std::runtime_error("PythonStrategy(" + m_module_name + ":"
                + m_class_name + ") init failed: " + e.what());
        }
        if (!overrides.empty()) {
            // getattr-with-default so a class with no `params` dict at all
            // still reaches the clear unknown-key error below.
            const pybind11::object declared =
                pybind11::getattr(m_py_instance, "params", pybind11::dict{});
            for (const auto& [name, value] : overrides) {
                if (!declared.contains(name)) {
                    throw std::runtime_error("PythonStrategy(" + m_module_name + ":"
                        + m_class_name + ") override '" + name + "' is not a declared param");
                }
                const pybind11::object current = m_py_instance.attr(name.c_str());
                pybind11::object coerced;
                if (pybind11::isinstance<pybind11::bool_>(current)) {   // before int: bool subclasses int
                    coerced = pybind11::bool_(value != 0.0);
                } else if (pybind11::isinstance<pybind11::int_>(current)) {
                    coerced = pybind11::int_(static_cast<long long>(value));   // truncates toward zero
                } else {
                    coerced = pybind11::float_(value);
                }
                m_py_instance.attr(name.c_str()) = coerced;
            }
        }
        if (!pybind11::hasattr(m_py_instance, "on_tick")) {
            throw std::runtime_error("PythonStrategy(" + m_module_name + ":"
                + m_class_name + ") has no on_tick(ctx) method");
        }
        m_has_on_start = pybind11::hasattr(m_py_instance, "on_start");
        m_has_on_stop = pybind11::hasattr(m_py_instance, "on_stop");
    }

    PythonStrategy(PythonStrategy&&) noexcept = default;
    PythonStrategy& operator=(PythonStrategy&&) noexcept = default;
    PythonStrategy(const PythonStrategy&) = delete;
    PythonStrategy& operator=(const PythonStrategy&) = delete;

    void on_start(auto& context) { if (m_has_on_start) { invoke("on_start", context); } }
    void on_tick(auto& context)  { invoke("on_tick", context); }
    void on_stop(auto& context)  { if (m_has_on_stop) { invoke("on_stop", context); } }

private:
    void invoke(const char* method, auto& context)
    {
        if (!m_adapter) {
            m_adapter = stonks::python::make_adapter(context);
        }
        pybind11::gil_scoped_acquire gil;
        try {
            m_py_instance.attr(method)(pybind11::cast(
                m_adapter.get(), pybind11::return_value_policy::reference));
        } catch (pybind11::error_already_set& e) {
            throw std::runtime_error("PythonStrategy::" + std::string{ method }
                + " (" + m_module_name + ":" + m_class_name + ") raised: " + e.what());
        }
    }

    // App-local defaults so `./app` from the project root finds the venv and
    // strategy module without any env-var setup. Both setenv calls use
    // overwrite=0 so a caller-supplied env var (their own venv, their own
    // strategy dir) wins. Declared first so it runs before EmbeddedPython
    // reads the env vars in its initialize().
    struct DefaultPaths
    {
        DefaultPaths()
        {
            ::setenv("STONKS_VENV", "app/python/.venv", 0);
            ::setenv("STONKS_PYTHONPATH", "app/python", 0);
        }
    };

    // m_defaults must precede m_python — env vars are read during EmbeddedPython
    // initialize(). m_python in turn precedes the py::object members so the
    // interpreter is alive before any PyObject* exists and dies after the
    // refcounts release.
    DefaultPaths m_defaults{};
    stonks::python::EmbeddedPython m_python{};
    std::string m_module_name;
    std::string m_class_name;
    pybind11::object m_py_instance;
    bool m_has_on_start{ false };
    bool m_has_on_stop{ false };
    std::unique_ptr<stonks::python::IContext> m_adapter;
};
