add_executable(python_tests
    contextadapter_test.cpp
    pythonstrategy_test.cpp
)

target_include_directories(python_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/app
)

target_link_libraries(python_tests PRIVATE
    stonks_core
    stonks_broker
    stonks_python
    GTest::gtest_main
)

# Fixtures live in tests/python/fixtures; the Python package this build emits
# is in python/. Tests inject both into sys.path so the embedded interpreter
# can `import stonks` and `import fixturestrats`.
target_compile_definitions(python_tests PRIVATE
    STONKS_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
    STONKS_PYTHON_PACKAGE_DIR="${PROJECT_SOURCE_DIR}/python"
    STONKS_VENV_DIR="${PROJECT_SOURCE_DIR}/app/python/.venv"
)

gtest_discover_tests(python_tests)

# Pure-Python strategy tests (pytest + stonks.testing.FakeContext), registered as
# a ctest so `ctest` runs them alongside the C++ suites. Requires the app venv
# (with the editable `stonks` install) created per app/python/README.md.
set(_qm_venv_python "${PROJECT_SOURCE_DIR}/app/python/.venv/bin/python")
if(EXISTS "${_qm_venv_python}")
    add_test(
        NAME qm_python_strategies
        COMMAND "${_qm_venv_python}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
else()
    message(STATUS "app venv not found at ${_qm_venv_python}; skipping qm_python_strategies test")
endif()
