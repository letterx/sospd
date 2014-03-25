set(test-sources
    "test-higher-order-energy.cpp"
    "test-submodular-ibfs.cpp"
)

###
### Unit test executable
###

add_executable(unit-test "test-main.cpp" ${test-sources})

target_link_libraries(unit-test sos-opt ${libs} boost_unit_test_framework)

