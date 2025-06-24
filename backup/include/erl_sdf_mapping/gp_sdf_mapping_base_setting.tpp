#pragma once

#include "log_edf_gp.hpp"

#include "erl_common/yaml.hpp"

namespace erl::gp_sdf {
    template<typename Dtype>
    YAML::Node
    GpSdfMappingBaseSetting<Dtype>::TestQuery::YamlConvertImpl::encode(const TestQuery& setting) {
        YAML::Node node;
        node["max_test_valid_distance_var"] = setting.max_test_valid_distance_var;
        node["search_area_half_size"] = setting.search_area_half_size;
        node["softmin_temperature"] = setting.softmin_temperature;
        node["num_neighbor_gps"] = setting.num_neighbor_gps;
        node["use_smallest"] = setting.use_smallest;
        node["compute_covariance"] = setting.compute_covariance;
        node["use_gp_covariance"] = setting.use_gp_covariance;
        return node;
    }

    template<typename Dtype>
    bool
    GpSdfMappingBaseSetting<Dtype>::TestQuery::YamlConvertImpl::decode(const YAML::Node& node, TestQuery& setting) {
        if (!node.IsMap()) { return false; }
        setting.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<Dtype>();
        setting.search_area_half_size = node["search_area_half_size"].as<Dtype>();
        setting.softmin_temperature = node["softmin_temperature"].as<Dtype>();
        setting.num_neighbor_gps = node["num_neighbor_gps"].as<int>();
        setting.use_smallest = node["use_smallest"].as<bool>();
        setting.compute_covariance = node["compute_covariance"].as<bool>();
        setting.use_gp_covariance = node["use_gp_covariance"].as<bool>();
        return true;
    }

    template<typename Dtype>
    YAML::Node
    GpSdfMappingBaseSetting<Dtype>::YamlConvertImpl::encode(const GpSdfMappingBaseSetting& setting) {
        YAML::Node node;
        node["num_threads"] = setting.num_threads;
        node["update_hz"] = setting.update_hz;
        node["gp_sdf_area_scale"] = setting.gp_sdf_area_scale;
        node["max_valid_gradient_var"] = setting.max_valid_gradient_var;
        node["invalid_position_var"] = setting.invalid_position_var;
        node["offset_distance"] = setting.offset_distance;
        node["use_occ_sign"] = setting.use_occ_sign;
        node["edf_gp"] = setting.edf_gp;
        node["test_query"] = setting.test_query;
        return node;
    }

    template<typename Dtype>
    bool
    GpSdfMappingBaseSetting<Dtype>::YamlConvertImpl::decode(const YAML::Node& node, GpSdfMappingBaseSetting& setting) {
        if (!node.IsMap()) { return false; }
        setting.num_threads = node["num_threads"].as<uint32_t>();
        setting.update_hz = node["update_hz"].as<Dtype>();
        setting.gp_sdf_area_scale = node["gp_sdf_area_scale"].as<Dtype>();
        setting.max_valid_gradient_var = node["max_valid_gradient_var"].as<Dtype>();
        setting.invalid_position_var = node["invalid_position_var"].as<Dtype>();
        setting.offset_distance = node["offset_distance"].as<Dtype>();
        setting.use_occ_sign = node["use_occ_sign"].as<bool>();
        setting.edf_gp = node["edf_gp"].as<std::shared_ptr<typename LogEdfGaussianProcess<Dtype>::Setting>>();
        setting.test_query = node["test_query"].as<std::shared_ptr<TestQuery>>();
        return true;
    }

}  // namespace erl::gp_sdf
