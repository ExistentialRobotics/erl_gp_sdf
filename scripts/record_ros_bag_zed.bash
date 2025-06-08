#!/usr/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $HOME/ros_bag_zed
cd $HOME/ros_bag_zed
rosbag record \
    --all \
    --output-prefix zed
