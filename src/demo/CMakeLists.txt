add_executable(demo_sdf_regression sdf_regression.cpp)
target_link_libraries(demo_sdf_regression
        PRIVATE erl_common erl_covariance erl_gaussian_process)
target_compile_definitions(demo_sdf_regression PUBLIC "ERL_USE_PLPLOT")

add_executable(demo_log_sdf_regression log_sdf_regression.cpp)
target_link_libraries(demo_log_sdf_regression
        PRIVATE erl_common erl_covariance erl_gaussian_process erl_sdf_mapping)
target_compile_definitions(demo_log_sdf_regression PUBLIC "ERL_USE_PLPLOT")

add_executable(demo_gp_occ_regression gp_occ_regression.cpp)
target_link_libraries(demo_gp_occ_regression
        PRIVATE erl_common erl_covariance erl_gaussian_process erl_sdf_mapping)
target_compile_definitions(demo_gp_occ_regression PUBLIC "ERL_USE_PLPLOT")

add_executable(demo_variance_estimation variance_estimation.cpp)
target_link_libraries(demo_variance_estimation
        PRIVATE erl_common erl_covariance erl_gaussian_process erl_sdf_mapping)
target_compile_definitions(demo_variance_estimation PUBLIC "ERL_USE_PLPLOT")
