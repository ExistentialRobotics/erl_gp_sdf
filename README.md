# erl_gp_sdf

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ROS1](https://img.shields.io/badge/ROS1-noetic-blue)](http://wiki.ros.org/)
[![ROS2](https://img.shields.io/badge/ROS2-humble-blue)](https://docs.ros.org/)

![](test/gtest/assets/test_gp_sdf_mapping_cow_and_lady.png)

**A C++ library for Gaussian Process regression on Signed Distance Fields.**

## Features

- **Gaussian Process Regression**: Implements Gaussian Process regression for SDFs.
- **Real-time Mapping**: Supports real-time mapping with SDFs.
- **Accurate SDF Prediction**: Provides accurate SDF predictions using Gaussian Processes.
- **C++ Implementation**: Written in C++ for performance and efficiency.
- **Python Bindings**: Includes Python bindings for easy integration with Python applications.
- **ROS Support**: Compatible with both ROS1 (Noetic) and ROS2 (Humble). 🚪 [erl_gp_sdf_ros](https://github.com/ExistentialRobotics/erl_gp_sdf_ros)

## Getting Started

### Prerequisites

- C++17 compatible compiler
- CMake 3.24 or higher

### Create Workspace

```shell
mkdir -p <your_workspace>/src && \
vcs import --input https://raw.githubusercontent.com/ExistentialRobotics/erl_gp_sdf/refs/head/main/erl_gp_sdf.repos <your_workspace>/src
```

### Dependencies

- [erl_cmake_tools](https://github.com/ExistentialRobotics/erl_cmake_tools)
- [erl_common](https://github.com/ExistentialRobotics/erl_common)
- [erl_covariance](https://github.com/ExistentialRobotics/erl_covariance)
- [erl_gaussian_process](https://github.com/ExistentialRobotics/erl_gaussian_process)
- [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry)

```bash
# Ubuntu 20.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_20.04.bash | bash
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_geometry/refs/heads/main/scripts/setup_ubuntu_20.04.bash | bash
# Ubuntu 22.04, 24.04
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/refs/heads/main/scripts/setup_ubuntu_22.04_24.04.bash | bash
wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_geometry/refs/heads/main/scripts/setup_ubuntu_22.04_24.04.bash | bash
```

### Docker Option

The easiest way to get started is to use the provided [Docker files](https://github.com/ExistentialRobotics/erl_geometry/tree/main/docker), which contains all dependencies.

### Use as a standard CMake package

```bash
cd <your_workspace>
touch CMakeLists.txt
```

Add the following lines to your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(<your_project_name>)
add_subdirectory(src/erl_cmake_tools)
add_subdirectory(src/erl_common)
add_subdirectory(src/erl_covariance)
add_subdirectory(src/erl_geometry)
add_subdirectory(src/erl_gaussian_process)
add_subdirectory(src/erl_gp_sdf)
```

Then run the following commands:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j`nproc`
```

### Use as a ROS Package

```bash
cd <your_workspace>
source /opt/ros/<distro>/setup.bash
# for ROS1
catkin build erl_gp_sdf
source devel/setup.bash
# for ROS2
colcon build --packages-up-to erl_gp_sdf
source install/setup.bash
```
See also 🚪[erl_gp_sdf_ros](https://github.com/ExistentialRobotics/erl_gp_sdf_ros) for additional ROS tools.

### Install As Python Package

- Make sure you have installed all dependencies.
- Make sure you have the correct Python environment activated, `pipenv` is recommended.

```bash
cd <your_workspace>
for package in erl_cmake_tools erl_common erl_covariance erl_geometry erl_gaussian_process erl_gp_sdf; do
    cd src/$package
    pip install . --verbose
    cd ../..
done
```

## Usage

### Without ROS

#### C++

- [Gallery](../test/gtest/README.md)
- [Gaussian Process based surface mapping 2D / 3D](../test/gtest/test_gp_occ_surface_mapping.cpp)
- [Gaussian Process based SDF mapping 2D](../test/gtest/test_gp_sdf_mapping_2d.cpp)
- [Gaussian Process based SDF mapping 3D](../test/gtest/test_gp_sdf_mapping_3d.cpp)

#### Python

- [Gallery](../test/pytest/README.md)
- [Gaussian Process based SDF mapping 3D](../test/pytest/test_gp_sdf_mapping_3d.py)

<!-- TODO: update links to the new location -->
<!-- # Pretrained GP-SDF Models

- [3D GP-SDF Model](https://drive.google.com/file/d/1K69JHQLg7LuNNc5ZhkY8-frIqXQisSpP/view?usp=sharing)
  trained on [Cow And Lady](https://projects.asl.ethz.ch/datasets/doku.php?id=iros2017), which can
  be loaded using the implementation
  from [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry/blob/main/include/erl_geometry/cow_and_lady.hpp).
- [3D GP-SDF Model](https://drive.google.com/file/d/1fraha9Fm00-3uKDujFBdsTSOJ4ZXsjdp/view?usp=sharing)
  trained on [Replica Hotel](data/replica-hotel-0.ply) with 640x480 simulated depth camera.
- [3D GP-SDF Model](https://drive.google.com/file/d/106SZjY4xzPJWYWYkD4LjINdxmUadlABV/view?usp=sharing)
  trained on [Replica Hotel](data/replica-hotel-0.ply) with simulated 3D
  LiDAR ([Velodyne Puck](https://www.amtechs.co.jp/product/VLP-16-Puck.pdf)).
- [3D GP-SDF Model](https://drive.google.com/file/d/135hlITMUeMNLi42VgdIteQmb2YK2m1y5/view?usp=sharing)
  trained on [Replica Hotel](data/replica-hotel-0.ply) with simulated 3D 360 LiDAR.
- [2D GP-SDF Model](https://drive.google.com/file/d/1ET0JUxA8fpUzYNkZXLheApPk3vqrPdiL/view?usp=sharing)
  trained on [UCSD-FAH-2D](data/ucsd_fah_2d.dat).
- [2D GP-SDF Model](https://drive.google.com/file/d/1JEZcFxGaI2ctoL_tiyqtAK-ARvFpHsFg/view?usp=sharing)
  trained on [Gazebo Room 2D](data/gazebo_train.dat) with simulated 2D LiDAR.
- [2D GP-SDF Model](https://drive.google.com/file/d/1hwmpCe2c8NZ6K9RAcrWyFfP1RMW9dr9l/view?usp=sharing)
  trained on [House Expo LiDAR 2D](data/house_expo_room_1451.json) with simulated 2D LiDAR. -->

### 🚪 [With ROS](https://github.com/ExistentialRobotics/erl_gp_sdf_ros)
