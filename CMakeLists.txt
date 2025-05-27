cmake_minimum_required(VERSION 3.16)

project(erl_sdf_mapping
        LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "erl_sdf_mapping is a C++ library for SDF estimation.")
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

if (NOT COMMAND erl_project_setup)
    find_package(erl_common REQUIRED)
endif ()
erl_project_setup(
        ${PROJECT_NAME}
        ERL_PACKAGES
        erl_common erl_covariance erl_geometry erl_gaussian_process
        CATKIN_COMPONENTS
        roscpp std_msgs sensor_msgs geometry_msgs tf2_ros tf2_eigen
        message_generation grid_map_ros grid_map_msgs
        CATKIN_DEPENDS
        roscpp std_msgs sensor_msgs geometry_msgs tf2_ros tf2_eigen
        message_runtime grid_map_ros grid_map_msgs
)
erl_setup_ros(
        CATKIN_COMPONENTS ${${PROJECT_NAME}_CATKIN_COMPONENTS}
        SRV_FILES SdfQuery.srv SaveMap.srv
        MSG_DEPENDENCIES std_msgs geometry_msgs
)
erl_catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME} py${PROJECT_NAME}
        CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_DEPENDS}
        DEPENDS ${${PROJECT_NAME}_DEPENDS})   # non-catkin dependencies are collected via erl_find_package

if (NOT TARGET Open3D::Open3D)
    erl_config_open3d()
endif ()
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
if (ROS1_ACTIVATED)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES})

    add_executable(sdf_mapping_node src/ros1/sdf_mapping_node.cpp)
    target_link_libraries(sdf_mapping_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})

    add_executable(sdf_visualization_node src/ros1/sdf_visualization_node.cpp)
    target_link_libraries(sdf_visualization_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})
else ()
    if (DEFINED ${PROJECT_NAME}_ERL_PACKAGES)
        target_link_libraries(${PROJECT_NAME} PUBLIC ${${PROJECT_NAME}_ERL_PACKAGES})
    endif ()
endif ()

###################
# Python bindings #
###################
erl_add_pybind_module(
        PYBIND_MODULE_NAME py${PROJECT_NAME}
        PYBIND_SRC_DIR ${${PROJECT_NAME}_PYTHON_BINDING_DIR}
        PYTHON_PKG_DIR ${${PROJECT_NAME}_PYTHON_DIR}/${PROJECT_NAME}
        LIBRARIES ${PROJECT_NAME}
)
erl_add_python_package(
        PYTHON_PKG_DIR ${${PROJECT_NAME}_PYTHON_DIR}/${PROJECT_NAME}
        DEPENDS_PYTHON_PKGS ${${PROJECT_NAME}_ERL_PACKAGES})

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
erl_mark_project_found(${PROJECT_NAME})
