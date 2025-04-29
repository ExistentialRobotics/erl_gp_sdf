#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    YAML::Node
    GpSdfMappingSetting<Dtype, Dim>::YamlConvertImpl::encode(const GpSdfMappingSetting &setting) {
        YAML::Node node;

        YAML::Node test_query;
        test_query["max_test_valid_distance_var"] = setting.test_query.max_test_valid_distance_var;
        test_query["search_area_half_size"] = setting.test_query.search_area_half_size;
        test_query["softmin_temperature"] = setting.test_query.softmin_temperature;
        test_query["num_neighbor_gps"] = setting.test_query.num_neighbor_gps;
        test_query["use_smallest"] = setting.test_query.use_smallest;
        test_query["compute_covariance"] = setting.test_query.compute_covariance;
        test_query["use_gp_covariance"] = setting.test_query.use_gp_covariance;
        test_query["use_global_buffer"] = setting.test_query.use_global_buffer;

        node["test_query"] = test_query;
        node["num_threads"] = setting.num_threads;
        node["update_hz"] = setting.update_hz;
        node["sensor_noise"] = setting.sensor_noise;
        node["gp_sdf_area_scale"] = setting.gp_sdf_area_scale;
        node["max_valid_gradient_var"] = setting.max_valid_gradient_var;
        node["invalid_position_var"] = setting.invalid_position_var;
        node["offset_distance"] = setting.offset_distance;
        node["use_sign_from_surface_mapping"] = setting.use_sign_from_surface_mapping;
        node["sdf_gp"] = setting.sdf_gp;

        return node;
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMappingSetting<Dtype, Dim>::YamlConvertImpl::decode(const YAML::Node &node, GpSdfMappingSetting &setting) {
        const YAML::Node &test_query_node = node["test_query"];
        TestQuery &test_query = setting.test_query;
        test_query.max_test_valid_distance_var = test_query_node["max_test_valid_distance_var"].as<Dtype>();
        test_query.search_area_half_size = test_query_node["search_area_half_size"].as<Dtype>();
        test_query.softmin_temperature = test_query_node["softmin_temperature"].as<Dtype>();
        test_query.num_neighbor_gps = test_query_node["num_neighbor_gps"].as<int>();
        test_query.use_smallest = test_query_node["use_smallest"].as<bool>();
        test_query.compute_covariance = test_query_node["compute_covariance"].as<bool>();
        test_query.use_gp_covariance = test_query_node["use_gp_covariance"].as<bool>();
        test_query.use_global_buffer = test_query_node["use_global_buffer"].as<bool>();

        setting.num_threads = node["num_threads"].as<int>();
        setting.update_hz = node["update_hz"].as<Dtype>();
        setting.sensor_noise = node["sensor_noise"].as<Dtype>();
        setting.gp_sdf_area_scale = node["gp_sdf_area_scale"].as<Dtype>();
        setting.max_valid_gradient_var = node["max_valid_gradient_var"].as<Dtype>();
        setting.invalid_position_var = node["invalid_position_var"].as<Dtype>();
        setting.offset_distance = node["offset_distance"].as<Dtype>();
        setting.use_sign_from_surface_mapping = node["use_sign_from_surface_mapping"].as<bool>();
        setting.sdf_gp = node["sdf_gp"].as<decltype(setting.sdf_gp)>();

        return true;
    }

}  // namespace erl::sdf_mapping
