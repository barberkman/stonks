cmake_minimum_required(VERSION 3.20)

project(stonks
    VERSION 0.0.1
    DESCRIPTION "Versatile C++20 backtesting and live-trading system"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STONKS_BUILD_TESTS "Build unit tests" ON)
option(STONKS_PYTHON "Build embedded Python support (requires CPython 3.10+ headers; Linux/macOS)" ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Qt 6's FindWrapOpenGL forces "-framework AGL" onto the link line when it can't
# locate AGL, but Apple removed AGL.framework from recent macOS SDKs, which breaks
# linking the Qt app. Point the AGL lookup at OpenGL.framework (still present) so
# Qt links that harmless stand-in instead. No effect on non-Apple platforms.
if(APPLE)
    set(WrapOpenGL_AGL "/System/Library/Frameworks/OpenGL.framework"
        CACHE FILEPATH "AGL shim: OpenGL.framework (AGL removed from modern macOS SDKs)" FORCE)
endif()

include(cmake/Dependencies.cmake)

add_subdirectory(src/core)
add_subdirectory(src/broker)
add_subdirectory(src/datafeed)
if(STONKS_PYTHON)
    add_subdirectory(src/python)
endif()
add_subdirectory(app)

if(STONKS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
