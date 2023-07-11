cmake_minimum_required(VERSION 3.20)

link_libraries(${PROJECT_NAME}::${PROJECT_NAME})

file(GLOB_RECURSE SRC_REF_GPIS_MAP gpis_map/src/*.cpp)
add_library(ref_gpis_map ${SRC_REF_GPIS_MAP})
target_include_directories(ref_gpis_map PUBLIC gpis_map/include)

add_executable(test_gpis_map test_gpis_map.cpp)
target_link_libraries(test_gpis_map PRIVATE ref_gpis_map)
add_custom_command(TARGET test_gpis_map
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/double ${CMAKE_CURRENT_BINARY_DIR}/double
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/float ${CMAKE_CURRENT_BINARY_DIR}/float)
add_test(NAME test_gpis_map
        COMMAND test_gpis_map)

add_executable(test_log_gpis_map gtest/test_log_gpis_map.cpp)
add_custom_command(TARGET test_log_gpis_map
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/double ${CMAKE_CURRENT_BINARY_DIR}/double
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/float ${CMAKE_CURRENT_BINARY_DIR}/float)
add_test(NAME test_log_gpis_map
        COMMAND test_log_gpis_map)

set(command ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/../python/${PROJECT_NAME}/gp_sdf/main_gpi_sdf.py
        --house-expo-index 1451 # 4
        --skip 20
        --map-resolution 0.05
        --draw-boundary
        --draw-sddf-v2
        --use-log)
add_test(NAME test_gpi_sdf_py
        COMMAND ${command}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../python)
set_tests_properties(test_gpi_sdf_py PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/../python")
