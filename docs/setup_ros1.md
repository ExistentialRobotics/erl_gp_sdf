Setup With ROS1
====

1. Create a catkin workspace
    ```shell
    mkdir -p <workspace>/src
    cd <workspace>/src
    ```
2. Clone the repositories into the `src` directory
    ```shell
    for repo in erl_common erl_covariance erl_gaussian_process erl_geometry erl_gp_sdf; do
        git clone --recursive https://github.com/ExistentialRobotics/${repo}.git
    done
    ```
3. Build
    ```shell
    cd <workspace>
    source /opt/ros/noetic/setup.bash
    catkin build --verbose
    source devel/setup.bash
    ```
