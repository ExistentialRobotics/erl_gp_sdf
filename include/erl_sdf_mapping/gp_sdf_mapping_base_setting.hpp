#pragma once

#include "log_sdf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {
    struct GpSdfMappingBaseSetting : public common::Yamlable<GpSdfMappingBaseSetting> {
        struct TestQuery : public Yamlable<TestQuery> {
            double max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
            double search_area_half_size = 4.8;
            bool use_nearest_only = false;    // if true, only the nearest point will be used for prediction.
            bool compute_covariance = false;  // if true, compute covariance of prediction.
            bool recompute_variance = true;   // if true, compute variance using different method.
            double softmax_temperature = 10.;
        };

        uint32_t num_threads = 64;
        double update_hz = 20;                // frequency that Update() is called.
        double gp_sdf_area_scale = 4;         // ratio between GP area and Quadtree cluster area
        double offset_distance = 0.0;         // offset distance for surface points
        double max_valid_gradient_var = 0.1;  // maximum gradient variance qualified for training.
        double invalid_position_var = 2.;
        // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
        bool train_gp_immediately = false;
        std::shared_ptr<LogSdfGaussianProcess::Setting> gp_sdf = std::make_shared<LogSdfGaussianProcess::Setting>();
        std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by Test.
        bool log_timing = false;
    };
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery> {
    static Node
    encode(const erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery& rhs) {
        Node node;
        node["max_test_valid_distance_var"] = rhs.max_test_valid_distance_var;
        node["search_area_half_size"] = rhs.search_area_half_size;
        node["use_nearest_only"] = rhs.use_nearest_only;
        node["compute_covariance"] = rhs.compute_covariance;
        node["recompute_variance"] = rhs.recompute_variance;
        node["softmax_temperature"] = rhs.softmax_temperature;
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery& rhs) {
        if (!node.IsMap()) { return false; }
        rhs.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<double>();
        rhs.search_area_half_size = node["search_area_half_size"].as<double>();
        rhs.use_nearest_only = node["use_nearest_only"].as<bool>();
        rhs.compute_covariance = node["compute_covariance"].as<bool>();
        rhs.recompute_variance = node["recompute_variance"].as<bool>();
        rhs.softmax_temperature = node["softmax_temperature"].as<double>();
        return true;
    }
};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMappingBaseSetting> {
    static Node
    encode(const erl::sdf_mapping::GpSdfMappingBaseSetting& setting) {
        Node node;
        node["num_threads"] = setting.num_threads;
        node["update_hz"] = setting.update_hz;
        node["gp_sdf_area_scale"] = setting.gp_sdf_area_scale;
        node["offset_distance"] = setting.offset_distance;
        node["max_valid_gradient_var"] = setting.max_valid_gradient_var;
        node["invalid_position_var"] = setting.invalid_position_var;
        node["train_gp_immediately"] = setting.train_gp_immediately;
        node["gp_sdf"] = setting.gp_sdf;
        node["test_query"] = setting.test_query;
        node["log_timing"] = setting.log_timing;
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMappingBaseSetting& setting) {
        if (!node.IsMap()) { return false; }
        setting.num_threads = node["num_threads"].as<uint32_t>();
        setting.update_hz = node["update_hz"].as<double>();
        setting.gp_sdf_area_scale = node["gp_sdf_area_scale"].as<double>();
        setting.offset_distance = node["offset_distance"].as<double>();
        setting.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
        setting.invalid_position_var = node["invalid_position_var"].as<double>();
        setting.train_gp_immediately = node["train_gp_immediately"].as<bool>();
        setting.gp_sdf = node["gp_sdf"].as<std::shared_ptr<erl::sdf_mapping::LogSdfGaussianProcess::Setting>>();
        setting.test_query = node["test_query"].as<std::shared_ptr<erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery>>();
        setting.log_timing = node["log_timing"].as<bool>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
