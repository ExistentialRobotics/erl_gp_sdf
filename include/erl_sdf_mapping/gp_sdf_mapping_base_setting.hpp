#pragma once

#include "log_edf_gp.hpp"
#include "sign_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {
    struct GpSdfMappingBaseSetting : public common::Yamlable<GpSdfMappingBaseSetting> {
        struct TestQuery : public Yamlable<TestQuery> {
            double max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
            double search_area_half_size = 4.8;
            int num_neighbor_gps = 4;        // number of neighbors used for prediction.
            bool use_smallest = false;       // if true, use the smallest sdf for prediction.
            bool compute_covariance = true;  // if true, compute covariance of prediction.
            bool use_gp_covariance = false;  // if true, compute variance with the GP.
            double softmin_temperature = 10.;
        };

        uint32_t num_threads = 64;
        double update_hz = 20;                // frequency that Update() is called.
        double gp_sdf_area_scale = 4;         // ratio between GP area and Quadtree cluster area
        double max_valid_gradient_var = 0.1;  // maximum gradient variance qualified for training.
        double invalid_position_var = 2.;     // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
        // bool train_gp_immediately = false;    // if true, train GP immediately after loading data.
        double offset_distance = 0.04;  // distance to shift for surface data.
        bool use_occ_sign = false;      // if true, use sign from occupancy tree: unknown -> occupied, otherwise, free.
        std::shared_ptr<LogEdfGaussianProcess::Setting> edf_gp = std::make_shared<LogEdfGaussianProcess::Setting>();
        std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by Test.
        // bool log_timing = false;
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
        node["num_neighbor_gps"] = rhs.num_neighbor_gps;
        node["use_smallest"] = rhs.use_smallest;
        node["compute_covariance"] = rhs.compute_covariance;
        node["use_gp_covariance"] = rhs.use_gp_covariance;
        node["softmin_temperature"] = rhs.softmin_temperature;
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery& rhs) {
        if (!node.IsMap()) { return false; }
        rhs.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<double>();
        rhs.search_area_half_size = node["search_area_half_size"].as<double>();
        rhs.num_neighbor_gps = node["num_neighbor_gps"].as<int>();
        rhs.use_smallest = node["use_smallest"].as<bool>();
        rhs.compute_covariance = node["compute_covariance"].as<bool>();
        rhs.use_gp_covariance = node["use_gp_covariance"].as<bool>();
        rhs.softmin_temperature = node["softmin_temperature"].as<double>();
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
        node["max_valid_gradient_var"] = setting.max_valid_gradient_var;
        node["invalid_position_var"] = setting.invalid_position_var;
        // node["train_gp_immediately"] = setting.train_gp_immediately;
        node["offset_distance"] = setting.offset_distance;
        node["use_occ_sign"] = setting.use_occ_sign;
        node["edf_gp"] = setting.edf_gp;
        node["test_query"] = setting.test_query;
        // node["log_timing"] = setting.log_timing;
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMappingBaseSetting& setting) {
        if (!node.IsMap()) { return false; }
        setting.num_threads = node["num_threads"].as<uint32_t>();
        setting.update_hz = node["update_hz"].as<double>();
        setting.gp_sdf_area_scale = node["gp_sdf_area_scale"].as<double>();
        setting.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
        setting.invalid_position_var = node["invalid_position_var"].as<double>();
        // setting.train_gp_immediately = node["train_gp_immediately"].as<bool>();
        setting.offset_distance = node["offset_distance"].as<double>();
        setting.use_occ_sign = node["use_occ_sign"].as<bool>();
        setting.edf_gp = node["edf_gp"].as<std::shared_ptr<erl::sdf_mapping::LogEdfGaussianProcess::Setting>>();
        setting.test_query = node["test_query"].as<std::shared_ptr<erl::sdf_mapping::GpSdfMappingBaseSetting::TestQuery>>();
        // setting.log_timing = node["log_timing"].as<bool>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
