# Setup Without ROS

1. Create a source directory in your workspace
2. Clone the repositories into the source directory
    ```shell
    cd <workspace>/src
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
4. Build
    ```shell
    cd <workspace>
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j`nproc`
    ```
