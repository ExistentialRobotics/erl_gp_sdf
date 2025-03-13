#pragma once

#include "sdf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    struct GpSdfMappingSetting : common::Yamlable<GpSdfMappingSetting<Dtype, Dim>> {

        using SdfGp = SdfGaussianProcess<Dtype, Dim>;

        struct TestQuery {
            Dtype max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
            Dtype search_area_half_size = 4.8;
            Dtype softmin_temperature = 10.;
            int num_neighbor_gps = 4;        // number of neighbors used for prediction.
            bool use_smallest = false;       // if true, use the smallest sdf for prediction.
            bool compute_covariance = true;  // if true, compute covariance of prediction.
            bool use_gp_covariance = false;  // if true, compute variance with the GP.
        };

        TestQuery test_query;                        // parameters used by Test.
        uint32_t num_threads = 64;                   // number of threads for testing.
        Dtype update_hz = 20;                        // update frequency in Hz.
        Dtype sensor_noise = 0.01;                   // sensor noise for surface data.
        Dtype gp_sdf_area_scale = 4;                 // ratio between GP area and cluster area
        Dtype max_valid_gradient_var = 0.1;          // maximum gradient variance qualified for training.
        Dtype invalid_position_var = 2.;             // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
        Dtype offset_distance = 0.04;                // distance to shift for surface data.
        bool use_sign_from_surface_mapping = false;  // if true, use sign from surface mapping
        std::shared_ptr<typename SdfGp::Setting> sdf_gp = std::make_shared<typename SdfGp::Setting>();

        struct YamlConvertImpl {
            static YAML::Node
            encode(const GpSdfMappingSetting& setting);

            static bool
            decode(const YAML::Node& node, GpSdfMappingSetting& setting);
        };
    };

    using GpSdfMappingSetting3Dd = GpSdfMappingSetting<double, 3>;
    using GpSdfMappingSetting3Df = GpSdfMappingSetting<float, 3>;
    using GpSdfMappingSetting2Dd = GpSdfMappingSetting<double, 2>;
    using GpSdfMappingSetting2Df = GpSdfMappingSetting<float, 2>;
}  // namespace erl::sdf_mapping

#include "gp_sdf_mapping_setting.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting3Dd> : erl::sdf_mapping::GpSdfMappingSetting3Dd::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting3Df> : erl::sdf_mapping::GpSdfMappingSetting3Df::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting2Dd> : erl::sdf_mapping::GpSdfMappingSetting2Dd::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting2Df> : erl::sdf_mapping::GpSdfMappingSetting2Df::YamlConvertImpl {};
