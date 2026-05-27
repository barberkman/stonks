#pragma once

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include <pybind11/embed.h>

namespace stonks::python {

// RAII handle for an embedded CPython interpreter.
//
// Multiple EmbeddedPython instances are safe — only the first one to be
// constructed initializes the interpreter; subsequent ones bump a refcount.
// Py_Finalize is deliberately never called: numpy / sklearn / pytorch do not
// tolerate initialize -> finalize -> initialize cycles in the same process,
// and the trade-off (small leak at process exit) is preferable to breaking
// those packages. Process termination reclaims the interpreter's memory.
//
// On first construction the interpreter is initialized with sys.path
// extended in this order (later inserts take priority):
//   1. current working directory
//   2. site-packages of $STONKS_VENV, if that env var is set
//   3. colon-separated paths from $STONKS_PYTHONPATH
class EmbeddedPython
{
public:
    EmbeddedPython()
    {
        if (s_refcount.fetch_add(1) == 0) {
            initialize();
        }
        m_owns = true;
    }

    EmbeddedPython(EmbeddedPython&& other) noexcept : m_owns{ other.m_owns }
    {
        other.m_owns = false;
    }

    EmbeddedPython& operator=(EmbeddedPython&& other) noexcept
    {
        if (this != &other) {
            release();
            m_owns = other.m_owns;
            other.m_owns = false;
        }
        return *this;
    }

    EmbeddedPython(const EmbeddedPython&) = delete;
    EmbeddedPython& operator=(const EmbeddedPython&) = delete;

    ~EmbeddedPython() { release(); }

    // Test hook: insert a path at the front of sys.path. Requires the
    // interpreter to already be initialized (i.e. at least one EmbeddedPython
    // alive). GIL is acquired internally.
    static void add_sys_path(const std::string& path)
    {
        pybind11::gil_scoped_acquire gil;
        pybind11::module_::import("sys").attr("path").attr("insert")(0, path);
    }

private:
    void release()
    {
        if (m_owns) {
            m_owns = false;
            s_refcount.fetch_sub(1);
            // Intentionally never call Py_Finalize. See class doc.
        }
    }

    static void initialize()
    {
        namespace py = pybind11;
        py::initialize_interpreter();

        auto sys_mod = py::module_::import("sys");
        auto sys_path = sys_mod.attr("path");

        // CWD so users can `import my_strats.foo` when they run from their
        // project directory.
        sys_path.attr("insert")(0, std::filesystem::current_path().string());

        if (const char* venv = std::getenv("STONKS_VENV")) {
            const std::string version = std::to_string(PY_MAJOR_VERSION)
                + "." + std::to_string(PY_MINOR_VERSION);
            const std::string site_packages = std::string{ venv }
                + "/lib/python" + version + "/site-packages";
            // site.addsitedir, not sys.path.insert: it processes the .pth files
            // that editable installs (e.g. `pip install -e`) rely on. Bare
            // sys.path.insert would find the dir but skip the .pth file that
            // wires up the editable package's source location.
            py::module_::import("site").attr("addsitedir")(site_packages);
        }

        if (const char* extra = std::getenv("STONKS_PYTHONPATH")) {
            std::string_view s{ extra };
            while (!s.empty()) {
                const auto pos = s.find(':');
                const std::string_view segment = (pos == std::string_view::npos)
                    ? s : s.substr(0, pos);
                if (!segment.empty()) {
                    sys_path.attr("insert")(0, std::string{ segment });
                }
                if (pos == std::string_view::npos) { break; }
                s.remove_prefix(pos + 1);
            }
        }
    }

    bool m_owns{ false };
    static inline std::atomic<int> s_refcount{ 0 };
};

} // namespace stonks::python
