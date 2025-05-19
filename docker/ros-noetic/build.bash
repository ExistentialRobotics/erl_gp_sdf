#! /usr/bin/bash

set -e
docker build --rm -t erl/ros-noetic:cpu-sdf-mapping . \
  --build-arg BASE_IMAGE=erl/geometry:20.04 $@
