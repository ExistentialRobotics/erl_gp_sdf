#!/usr/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROS_WS_DIR=$(dirname $(dirname $(dirname ${SCRIPT_DIR})))
source /opt/ros/noetic/setup.bash
# source /opt/zed_camera_ws/devel/setup.bash
source ${ROS_WS_DIR}/devel/setup.bash

roslaunch erl_sdf_mapping zed_ros_live.launch
