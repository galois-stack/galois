# add_executable(galois_test main.cpp matrix_multiply.cpp)
add_executable(galois_test  main.cpp matrix_multiply_test.cpp)
target_link_libraries(galois_test galois_config_target galois_ir prajna_core gtest_main cpuinfo thpool)
target_compile_options(galois_test PRIVATE -mavx)
