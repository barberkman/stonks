include(GoogleTest)

add_executable(core_tests
    core/types_test.cpp
    core/progressbar_test.cpp
    core/report_test.cpp
    core/report_json_test.cpp
    core/analytics_test.cpp
    core/backtestbroker_test.cpp
    core/stoporder_test.cpp
    core/intrabarfillpolicy_test.cpp
    core/leverage_test.cpp
    core/scenarios_test.cpp
    core/lookahead_test.cpp
    core/determinism_test.cpp
    core/ema50strategy_test.cpp
    core/qmsignals_test.cpp
)

target_include_directories(core_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/app
)

target_link_libraries(core_tests PRIVATE
    stonks_core
    stonks_broker
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

gtest_discover_tests(core_tests)

# Datafeed tests link the real KLineFeed (and thus Arrow), kept separate so the
# stub-only core_tests stay free of that dependency. peekparquet_test writes a
# fixture parquet at runtime, so it links Arrow/Parquet directly.
find_package(Arrow REQUIRED QUIET)
find_package(Parquet REQUIRED QUIET)

add_executable(datafeed_tests
    core/klinefeed_test.cpp
    core/peekparquet_test.cpp
)

target_link_libraries(datafeed_tests PRIVATE
    stonks_core
    stonks_datafeed
    Arrow::arrow_shared
    Parquet::parquet_shared
    GTest::gtest_main
)

# Longer discovery timeout: the first exec loads the full Arrow/Parquet/AWS dylib
# stack, which can exceed the 5s default on macOS (first-run dyld validation).
gtest_discover_tests(datafeed_tests DISCOVERY_TIMEOUT 60)

# App-layer tests: the QVariant result map the QML views consume. The app
# target is an executable, so resultmap.cpp is compiled into the test binary
# directly rather than linked from a library.
add_subdirectory(app)

if(STONKS_PYTHON)
    add_subdirectory(python)
endif()
