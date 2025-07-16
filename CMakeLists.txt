cmake_minimum_required(VERSION 3.16)

project(erl_gp_sdf
        LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "erl_gp_sdf is a C++ library for SDF estimation.")
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

if (NOT COMMAND erl_project_setup)
    find_package(erl_common REQUIRED)
endif ()
erl_project_setup(
        ERL_PACKAGES
        erl_common erl_covariance erl_geometry erl_gaussian_process
)
# if (NOT TARGET Open3D::Open3D)
#     erl_config_open3d()
# endif ()
erl_setup_ros(
        CATKIN_COMPONENTS
        roscpp std_msgs sensor_msgs geometry_msgs tf2_ros tf2_eigen
        message_generation grid_map_ros grid_map_msgs

        CATKIN_DEPENDS
        roscpp std_msgs sensor_msgs geometry_msgs tf2_ros tf2_eigen
        message_runtime grid_map_ros grid_map_msgs

        LIBRARIES
        ${PROJECT_NAME}

        SRV_FILES
        SdfQuery.srv SaveMap.srv

        MSG_DEPENDENCIES
        std_msgs geometry_msgs
)

####################################################################################################
# BUILD
####################################################################################################

########################
# define the libraries #
########################
file(GLOB SRC_FILES src/*.cpp)
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}>)
erl_target_dependencies(${PROJECT_NAME})
if (ERL_USE_LIBTORCH)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES})
endif ()

# if (ROS1_ACTIVATED)
#     add_executable(sdf_mapping_node src/ros1/sdf_mapping_node.cpp)
#     target_link_libraries(sdf_mapping_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})

#     add_executable(sdf_visualization_node src/ros1/sdf_visualization_node.cpp)
#     target_link_libraries(sdf_visualization_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})
# else ()
# #     if (DEFINED ${PROJECT_NAME}_ERL_PACKAGES)
# #         target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_ERL_PACKAGES})
# #     endif ()
# endif ()
add_subdirectory(src/ros)
add_subdirectory(src/demo)

###################
# Python bindings #
###################
erl_add_pybind_module(
        PYBIND_MODULE_NAME py${PROJECT_NAME}
        PYBIND_SRC_DIR ${${PROJECT_NAME}_PYTHON_BINDING_DIR}
        LIBRARIES ${PROJECT_NAME}
)
erl_add_python_package()

####################################################################################################
# INSTALL
####################################################################################################
erl_install(
        LIBRARIES ${PROJECT_NAME}
        PYBIND_MODULES py${PROJECT_NAME})

####################################################################################################
# TEST
####################################################################################################
erl_add_tests(
        LIBRARIES ${PROJECT_NAME}
)

####################################################################################################
# export variables
####################################################################################################
erl_mark_project_found()
