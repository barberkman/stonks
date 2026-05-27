# INTERFACE library consumed by the app + tests when they need to embed CPython.
# Header-only on the C++ side: IContext, ContextAdapter, EmbeddedPython.
add_library(stonks_python INTERFACE)
target_include_directories(stonks_python INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(stonks_python INTERFACE stonks_core pybind11::embed)

# Python extension module: importable from Python as `stonks._core`.
# Output goes into python/stonks/ so `pip install -e python/` picks it up.
pybind11_add_module(_core bindings.cpp)
target_link_libraries(_core PRIVATE stonks_core)
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python/stonks
)
