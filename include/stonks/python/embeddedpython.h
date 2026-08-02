#pragma once

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
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
//
// Setting $STONKS_DEBUGPY additionally opens a debugpy listener and blocks
// until a debugger attaches — see debugpy_port() and attach_debugpy() below.
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

    // Interpret $STONKS_DEBUGPY: nullopt disables the listener, otherwise the
    // TCP port to listen on. A value in the unprivileged port range is taken
    // literally; every other truthy value ("1", "on", "yes") means "enabled,
    // default port", so callers never have to remember the port number.
    // Public only so the mapping is unit-testable.
    static std::optional<int> debugpy_port(const char* value)
    {
        constexpr int default_port = 5678;   // debugpy's own default
        if (value == nullptr) { return std::nullopt; }
        const std::string_view text{ value };
        if (text.empty() || text == "0") { return std::nullopt; }
        int port = 0;
        const auto* const end = value + std::strlen(value);
        const auto [stop, ec] = std::from_chars(value, end, port);
        const bool is_port = ec == std::errc{} && stop == end
            && port >= 1024 && port <= 65535;
        return is_port ? port : default_port;
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

        // Last, so debugpy itself resolves from the venv configured above.
        if (const auto port = debugpy_port(std::getenv("STONKS_DEBUGPY"))) {
            attach_debugpy(*port);
        }
    }

    // Open a debugpy listener and block until an editor attaches. lldb / gdb
    // debug the host process and know nothing about Python source, so a
    // breakpoint in a .py strategy is inert in those sessions; attaching
    // debugpy to this interpreter is what makes Python-level stepping work.
    // Blocking is the point — it holds the run at startup so breakpoints are
    // in place before the first on_tick.
    static void attach_debugpy(int port)
    {
        namespace py = pybind11;
        try {
            auto debugpy = py::module_::import("debugpy");
            // sys.executable is the host binary in an embedded interpreter, but
            // debugpy.listen() spawns `sys.executable -m debugpy.adapter`, so it
            // has to be told which interpreter to use.
            if (const char* venv = std::getenv("STONKS_VENV")) {
                py::dict config;
                config["python"] = std::string{ venv } + "/bin/python";
                debugpy.attr("configure")(config);
            }
            debugpy.attr("listen")(port);
            std::cerr << "[stonks] debugpy listening on 127.0.0.1:" << port
                      << " — waiting for the debugger to attach...\n";
            std::cerr.flush();
            debugpy.attr("wait_for_client")();
            std::cerr << "[stonks] debugger attached.\n";
            std::cerr.flush();
        } catch (py::error_already_set& e) {
            throw std::runtime_error("STONKS_DEBUGPY is set but the debugpy "
                "listener could not start (is debugpy installed in $STONKS_VENV?): "
                + std::string{ e.what() });
        }
    }

    bool m_owns{ false };
    static inline std::atomic<int> s_refcount{ 0 };
};

} // namespace stonks::python
