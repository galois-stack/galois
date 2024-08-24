add_library(galois_ir OBJECT
    global_context.cpp
)

target_link_libraries(galois_ir
    PUBLIC galois_config_target
)
