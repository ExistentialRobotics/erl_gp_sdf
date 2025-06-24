`erl_gp_sdf`
=================

![](test/gtest/assets/test_gp_sdf_mapping_cow_and_lady.png)

# Dependencies

- [erl_common](https://github.com/ExistentialRobotics/erl_common)
- [erl_covariance](https://github.com/ExistentialRobotics/erl_covariance)
- [erl_gaussian_process](https://github.com/ExistentialRobotics/erl_gaussian_process)
- [erl_geometry](https://github.com/ExistentialRobotics/erl_geometry)

# Setup

1. 📦 Install dependencies:
    ```shell
    # Ubuntu 20.04
    wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_common/main/scripts/setup_ubuntu_20.04.bash | bash
    wget -qO - https://raw.githubusercontent.com/ExistentialRobotics/erl_geometry/main/scripts/setup_ubuntu_20.04.bash | bash
    # Ubuntu 22.04, 24.04
    # Available soon
    ```
2. 🚪 [Without ROS](docs/setup_no_ros.md)
3. 🚪 [With ROS1 Noetic](docs/setup_ros1.md)
4. 🚪 [With ROS2 Humble](docs/setup_ros2.md)

# Build With Docker

Currently, only the Ubuntu 20.04 base images are available. Ubuntu 22.04 and 24.04 will be available
soon.

1. Build base images:
    ```shell
    cd /path/to/erl_common
    cd docker
    ./build.bash
    
    cd /path/to/erl_geometry
    cd docker
    ./build.bash
    ```
2. If you want to build without ROS in the docker, you can use `erl/geometry:20.04` directly.
3. To use a ROS1 image, run the following commands and use the generated image
   `erl/ros-noetic:cpu-sdf-mapping`.
    ```shell
    cd /path/to/erl_gp_sdf
    cd docker/ros-noetic
    ./build.bash
    ```
4. To use a ROS2 image, run:
    ```shell
    # Available soon
    ```

# Usage

1. 🚪 [Without ROS](docs/usage_no_ros.md)
2. 🚪 [With ROS1](docs/usage_ros1.md)

# Known Issues

- The program slows down (GPOCC-SDF on Gazebo 2D slows down by 30%) when `libtorch` is linked to.
