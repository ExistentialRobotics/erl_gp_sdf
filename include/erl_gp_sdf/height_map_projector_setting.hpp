#pragma once

#include "erl_common/yaml.hpp"
#include "erl_geometry/aabb.hpp"

namespace erl::gp_sdf {

    template<typename Dtype>
    struct HeightMapProjectorSetting : common::Yamlable<HeightMapProjectorSetting<Dtype>> {
        /// Desired output grid resolution for Nav2.
        Dtype target_resolution = 0.1f;
        /// Robot height. If (max_z - min_z) exceeds this in a ground-only cell, the
        /// upper ground layer is treated as overhead and min_z is used as the floor.
        Dtype robot_height = 0.4f;
        /// Maximum traversable step height between neighboring cells.
        Dtype max_step_height = 0.15f;
        /// Minimum Z relative to sensor position for ground band.
        Dtype ground_z_min = -1.0f;
        /// Maximum Z relative to sensor position for ground band.
        Dtype ground_z_max = 0.5f;
        /// Minimum |n_z| for a triangle to be considered ground-like.
        Dtype min_normal_z = 0.9f;
        /// Re-evaluate near-ground BHMs when sensor Z changes by this much.
        Dtype sensor_z_change_threshold = 0.5f;
        /// If true, limit the map to a 2D bounding box.
        bool use_bounding_box = false;
        /// XY bounding box (ignored if use_bounding_box is false).
        geometry::Aabb<Dtype, 2> bounding_box = {};

        ERL_REFLECT_SCHEMA(
            HeightMapProjectorSetting,
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, target_resolution),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, robot_height),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, max_step_height),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, ground_z_min),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, ground_z_max),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, min_normal_z),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, sensor_z_change_threshold),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, use_bounding_box),
            ERL_REFLECT_MEMBER(HeightMapProjectorSetting, bounding_box));
    };

    extern template struct HeightMapProjectorSetting<float>;
    extern template struct HeightMapProjectorSetting<double>;

}  // namespace erl::gp_sdf
