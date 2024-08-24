add_library(galois_ast INTERFACE)
target_link_libraries(galois_ast
    INTERFACE galois_config_target
    # PRIVATE Boost::multiprecision
)

target_compile_definitions(galois_ast
    # 参阅third_party/boost/libs/mpl/include/boost/mpl/list/list50.hpp
    # 并非任意数字都行, 需要是10,20,30,40,50
    # boost::variant的模板参数受此影响
    INTERFACE BOOST_MPL_LIMIT_LIST_SIZE=50
    INTERFACE BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
)
