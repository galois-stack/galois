
add_library(galois_config_target INTERFACE)
target_include_directories(galois_config_target INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(galois_config_target
    INTERFACE prajna_config_target
    INTERFACE nlohmann_json
    # INTERFACE mtensor
    # INTERFACE Boost::variant
    # INTERFACE Boost::optional
    # INTERFACE Boost::fusion
    # INTERFACE Boost::algorithm
    # INTERFACE Boost::dll
    # INTERFACE Boost::process
    # INTERFACE Boost::asio
    # INTERFACE fmt::fmt
    INTERFACE eigen
)

add_subdirectory(ir)

# target_compile_options(galois_config_target
    # INTERFACE "-Werror"
# )

# add_subdirectory(ast)
# add_subdirectory(parser)
