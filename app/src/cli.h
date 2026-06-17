#pragma once

#include <string_view>

namespace stonks::app {

// Returns true if "--gui" appears among the command-line arguments.
inline bool wants_gui(int argc, const char* const* argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string_view{ argv[i] } == "--gui") {
            return true;
        }
    }
    return false;
}

}  // namespace stonks::app
