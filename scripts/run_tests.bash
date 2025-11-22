#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BUILD_DIR=${SCRIPT_DIR}/../../../build
BIN_DIR=${BUILD_DIR}/src/erl_gp_sdf

cd ${BUILD_DIR}

function build_target() {
    cd ${BUILD_DIR}
    local target=$1
    if ! make -j8 ${target}; then
        echo "Failed to build target: ${target}"
        exit 1
    fi
}

function run_target() {
    local target=$1
    if [ ! -x "${BIN_DIR}/${target}" ]; then
        echo "${target} does not exist"
        exit 1
    fi
    cd "${BIN_DIR}"
    "./${target}" "${@:2}"
}

function run_test() {
    local target=$1
    build_target ${target}
    run_target $@
}

tests=(
"test_bayesian_hilbert_surface_mapping --gtest_filter=BayesianHilbertSurfaceMapping.2Df --dataset_name gazebo_room_2d --surface_mapping_config_file ${SCRIPT_DIR}/../config/gazebo/bayesian_hilbert_surf_mapping_float.yaml --visualize true --test_io true"
"test_bayesian_hilbert_gp_sdf_mapping_2d --gtest_filter=GpSdfMapping.BayesianHilbert2Df --dataset_name gazebo_room_2d --surface_mapping_config_file ${SCRIPT_DIR}/../config/gazebo/bayesian_hilbert_surf_mapping_float.yaml --sdf_mapping_config_file ${SCRIPT_DIR}/../config/gazebo/bayesian_hilbert_sdf_mapping_float.yaml --visualize true --test_io true --hold true --interactive true"
"test_bayesian_hilbert_surface_mapping --gtest_filter=BayesianHilbertSurfaceMapping.3Df --dataset_name mesh --surface_mapping_config_file ${SCRIPT_DIR}/../config/replica/bayesian_hilbert_surf_mapping_lidar_360_float.yaml --sensor_frame_type erl::geometry::LidarFrame3D<float> --sensor_frame_config_file ${SCRIPT_DIR}/../config/sensors/lidar_frame_3d_360.yaml --test_io true --surf_normal_scale 0.125"
)

for spec in "${tests[@]}"; do
    run_test $spec
done