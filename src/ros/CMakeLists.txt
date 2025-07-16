if (ROS1_ACTIVATED)
    add_executable(sdf_mapping_node ros1/sdf_mapping_node.cpp)
    target_link_libraries(sdf_mapping_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})

    add_executable(sdf_visualization_node ros1/sdf_visualization_node.cpp)
    target_link_libraries(sdf_visualization_node PUBLIC ${PROJECT_NAME}::${PROJECT_NAME})
elseif (ROS2_ACTIVATED)


endif ()
