add_library(stonks_core INTERFACE)

target_include_directories(stonks_core INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(stonks_core INTERFACE cxx_std_20)

option(STONKS_LOG "Enable diagnostic STONKS_LOG logging" OFF)
if(STONKS_LOG)
    target_compile_definitions(stonks_core INTERFACE STONKS_LOG_ENABLED)
endif()
