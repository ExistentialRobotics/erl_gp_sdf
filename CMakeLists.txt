cmake_minimum_required(VERSION 3.20)

project(erl_sdf_mapping
        LANGUAGES CXX
        VERSION 0.1.0
        DESCRIPTION "erl_sdf_mapping is a C++ library for SDF estimation.")
message(STATUS "Configuring ${PROJECT_NAME} ${PROJECT_VERSION}")

if (NOT COMMAND erl_project_setup)
    find_package(erl_common REQUIRED)
endif ()
set(${PROJECT_NAME}_CATKIN_COMPONENTS roscpp std_msgs sensor_msgs tf2_ros tf2_eigen erl_common erl_gaussian_process
        erl_geometry cv_bridge message_generation)
set(${PROJECT_NAME}_CATKIN_DEPENDS roscpp std_msgs sensor_msgs tf2_ros tf2_eigen erl_common erl_gaussian_process
        erl_geometry cv_bridge message_runtime)
erl_project_setup()
erl_setup_ros(
        CATKIN_COMPONENTS ${${PROJECT_NAME}_CATKIN_COMPONENTS}
        SRV_FILES PredictSdf.srv
        MSG_DEPENDENCIES std_msgs
)
erl_catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME} py${PROJECT_NAME}
        CATKIN_DEPENDS ${${PROJECT_NAME}_CATKIN_DEPENDS}
        DEPENDS ${${PROJECT_NAME}_DEPENDS})   # dependencies are collected via erl_find_package

#######################################################################################################################
# BUILD
#######################################################################################################################

########################
# define the libraries #
########################
file(GLOB SRC_FILES src/*.cpp src/gpis/*.cpp)
add_library(${PROJECT_NAME} ${SRC_FILES})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${${PROJECT_NAME}_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${${PROJECT_NAME}_INSTALL_INCLUDE_DIR}>)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC erl_common erl_gaussian_process erl_geometry ${catkin_LIBRARIES})

if (ROS_ACTIVATED)
    add_executable(erl_sdf_mapping_node src/ros/erl_sdf_mapping_node.cpp)
    target_link_libraries(erl_sdf_mapping_node PUBLIC ${PROJECT_NAME})
endif ()

###################
# Python bindings #
###################
erl_add_python_package(
        PYTHON_PKG_DIR ${${PROJECT_NAME}_PYTHON_DIR}/${PROJECT_NAME}
        PYBIND_MODULE_NAME py${PROJECT_NAME}
        DEPENDS_PYTHON_PKGS erl_common erl_gaussian_process erl_geometry)

#######################################################################################################################
# INSTALL
#######################################################################################################################
erl_install(
        LIBRARIES ${PROJECT_NAME}
        PYBIND_MODULES py${PROJECT_NAME})

#######################################################################################################################
# TEST
#######################################################################################################################
erl_add_tests(
        LIBRARIES ${PROJECT_NAME}
)

#######################################################################################################################
# export variables
#######################################################################################################################
erl_mark_project_found()
