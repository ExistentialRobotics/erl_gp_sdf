#pragma once

#include "sdf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    struct GpSdfMappingSetting : common::Yamlable<GpSdfMappingSetting<Dtype, Dim>> {

        using SdfGp = SdfGaussianProcess<Dtype, Dim>;
        using SdfGpSetting = typename SdfGp::Setting;

        struct TestQuery {
            Dtype max_test_valid_distance_var = 0.4f;  // maximum distance variance of prediction.
            Dtype search_area_half_size = 4.8f;        // the half-size of the search area.
            int num_neighbor_gps = 4;                  // Number of neighbors used for prediction.
            bool use_smallest = false;              // If true, use the smallest sdf for prediction.
            bool compute_gradient = true;           // If true, compute the sdf gradient.
            bool compute_gradient_variance = true;  // If true, compute the gradient variance.
            bool compute_covariance = true;  // If true, compute the covariance of prediction.
            bool use_gp_covariance = false;  // If true, compute variance with the GP.
            bool retrain_outdated = false;   // If true, retrain the trained GPs if outdated.
            bool use_global_buffer = false;  // If true, use the global buffer.
        };

        TestQuery test_query;                 // parameters used by Test.
        uint32_t num_threads = 64;            // number of threads for testing.
        Dtype update_hz = 20.0f;              // update frequency in Hz.
        Dtype sensor_noise = 0.01f;           // sensor noise for surface data.
        Dtype gp_sdf_area_scale = 4.0f;       // ratio between GP area and cluster area
        Dtype max_valid_gradient_var = 0.1f;  // max gradient variance valid for training.
        Dtype invalid_position_var = 2.0f;    // position variance when > max_valid_gradient_var.
        Dtype offset_distance = 0.04f;        // distance to shift for surface data.
        std::shared_ptr<SdfGpSetting> sdf_gp = std::make_shared<SdfGpSetting>();

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
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting3Dd>
    : erl::sdf_mapping::GpSdfMappingSetting3Dd::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting3Df>
    : erl::sdf_mapping::GpSdfMappingSetting3Df::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting2Dd>
    : erl::sdf_mapping::GpSdfMappingSetting2Dd::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingSetting2Df>
    : erl::sdf_mapping::GpSdfMappingSetting2Df::YamlConvertImpl {};
