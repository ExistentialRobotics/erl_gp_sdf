`erl_sdf_mapping`
=================

# Dependencies

- [erl_common](https://github.com/ExistentialRobotics/erl_common)
- [erl_covariance](https://github.com/ExistentialRobotics/erl_covariance)
- [erl_gaussian_process](https://github.com/ExistentialRobotics/erl_gaussian_process)
- [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry)

# Installation

1. Create a source directory in your workspace
2. Clone the repositories into the source directory
    ```bash
    cd /path/to/your/workspace/src
    for repo in erl_common erl_covariance erl_gaussian_process erl_geometry erl_sdf_mapping; do
        git clone --recursive https://github.com/ExistentialRobotics/${repo}.git
    done
    ```
3. Create a top-level CMakeLists.txt in your workspace
    ```cmake
    cmake_minimum_required(VERSION 3.24)
    project(your_project_name)
   
    add_subdirectory(src/erl_common)
    add_subdirectory(src/erl_covariance)
    add_subdirectory(src/erl_geometry)
    add_subdirectory(src/erl_gaussian_process)
    add_subdirectory(src/erl_sdf_mapping)
    ```
4. Build your workspace (Without ROS)
    ```bash
    cd /path/to/your/workspace
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j`nproc`
    ```
5. Or build your workspace (With ROS Noetic)
    ```bash
    source /opt/ros/noetic/setup.bash
    cd /path/to/your/workspace
    catkin build --verbose
    ```

# Usage

## C++

- [Gaussian Process based surface mapping 2D](test/gtest/test_gp_occ_surface_mapping_2d.cpp)
- [Gaussian Process based surface mapping 3D](test/gtest/test_gp_occ_surface_mapping_3d.cpp)
- [Gaussian Process based SDF mapping 2D](test/gtest/test_gp_sdf_mapping_2d.cpp)
- [Gaussian Process based SDF mapping 3D](test/gtest/test_gp_sdf_mapping_3d.cpp)

## Python

- [Gaussian Process based SDF mapping 3D](test/pytest/test_gp_sdf_mapping_3d.py)

# Pretrained GP-SDF Models

- [3D GP-SDF Model](https://drive.google.com/file/d/1K69JHQLg7LuNNc5ZhkY8-frIqXQisSpP/view?usp=sharing) trained
  on [Cow And Lady](https://projects.asl.ethz.ch/datasets/doku.php?id=iros2017), which can be loaded using the
  implementation from
  [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry/blob/main/include/erl_geometry/cow_and_lady.hpp).
- [3D GP-SDF Model](https://drive.google.com/file/d/1fraha9Fm00-3uKDujFBdsTSOJ4ZXsjdp/view?usp=sharing) trained
  on [Replica Hotel](data/replica-hotel-0.ply) with 640x480 simulated depth camera.
- [3D GP-SDF Model](https://drive.google.com/file/d/106SZjY4xzPJWYWYkD4LjINdxmUadlABV/view?usp=sharing) trained
  on [Replica Hotel](data/replica-hotel-0.ply) with simulated 3D LiDAR.
