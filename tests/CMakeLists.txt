# add_executable(galois_test main.cpp matrix_multiply.cpp)
add_executable(galois_test
    main.cpp
    gemm_test.cpp
    unary_instrinsic_test.cpp
    sum_test.cpp
    fill_test.cpp
    softmatx_test.cpp
    full_test.cpp
    slice_test.cpp
    padding_test.cpp
    tile_matrix_multiply_test.cpp
    reference_count_test.cpp
    arithmetic_test.cpp
    blas_test.cpp
    operation_count_test.cpp
    matrix_multiply_kernel_test.cpp
    ir_print_test.cpp
    load_binary_test.cpp
    concatenate_test.cpp
    stack_test.cpp
    flatten_test.cpp
    transpose_test.cpp
    sigmoid_test.cpp
    mnist_test.cpp
    relu_test.cpp
    gelu_test.cpp
    copy_test.cpp
    embedding_test.cpp
    normalize_test.cpp
    convolution_test.cpp
    convolution_3d_test.cpp
    indexing_test.cpp
    broadcast_test.cpp
    gather_test.cpp
)

target_link_libraries(galois_test galois_config_target galois_ir prajna_core gtest benchmark thpool cpuinfo libz3)
target_compile_definitions(galois_ir PRIVATE  EIGEN_MAX_ALIGN_BYTES=32)
target_compile_definitions(galois_test PRIVATE  EIGEN_MAX_ALIGN_BYTES=32)
