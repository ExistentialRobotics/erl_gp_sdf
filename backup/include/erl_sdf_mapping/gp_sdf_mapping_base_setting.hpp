#pragma once

#include "log_edf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {
    template<typename Dtype>
    struct GpSdfMappingBaseSetting : common::Yamlable<GpSdfMappingBaseSetting<Dtype>> {
        struct TestQuery : common::Yamlable<TestQuery> {
            Dtype max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
            Dtype search_area_half_size = 4.8;
            Dtype softmin_temperature = 10.;
            int num_neighbor_gps = 4;        // number of neighbors used for prediction.
            bool use_smallest = false;       // if true, use the smallest sdf for prediction.
            bool compute_covariance = true;  // if true, compute covariance of prediction.
            bool use_gp_covariance = false;  // if true, compute variance with the GP.

            struct YamlConvertImpl {
                static YAML::Node
                encode(const TestQuery& setting);

                static bool
                decode(const YAML::Node& node, TestQuery& setting);
            };
        };

        uint32_t num_threads = 64;
        Dtype update_hz = 20;                // frequency that Update() is called.
        Dtype gp_sdf_area_scale = 4;         // ratio between GP area and Quadtree cluster area
        Dtype max_valid_gradient_var = 0.1;  // maximum gradient variance qualified for training.
        Dtype invalid_position_var = 2.;     // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
        Dtype offset_distance = 0.04;        // distance to shift for surface data.
        bool use_occ_sign = false;           // if true, use sign from occupancy tree: unknown -> occupied, otherwise, free.
        std::shared_ptr<typename LogEdfGaussianProcess<Dtype>::Setting> edf_gp = std::make_shared<typename LogEdfGaussianProcess<Dtype>::Setting>();
        std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by Test.

        struct YamlConvertImpl {
            static YAML::Node
            encode(const GpSdfMappingBaseSetting& setting);

            static bool
            decode(const YAML::Node& node, GpSdfMappingBaseSetting& setting);
        };
    };

    using GpSdfMappingBaseSetting_d = GpSdfMappingBaseSetting<double>;
    using GpSdfMappingBaseSetting_f = GpSdfMappingBaseSetting<float>;
}  // namespace erl::sdf_mapping

#include "gp_sdf_mapping_base_setting.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting_d::TestQuery> : erl::sdf_mapping::GpSdfMappingBaseSetting_d::TestQuery::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting_f::TestQuery> : erl::sdf_mapping::GpSdfMappingBaseSetting_f::TestQuery::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting_d> : erl::sdf_mapping::GpSdfMappingBaseSetting_d::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting_f> : erl::sdf_mapping::GpSdfMappingBaseSetting_f::YamlConvertImpl {};
