#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    YAML::Node
    GpSdfMappingSetting<Dtype, Dim>::YamlConvertImpl::encode(const GpSdfMappingSetting &setting) {
        YAML::Node tq_node;
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, max_test_valid_distance_var);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, search_area_half_size);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, num_neighbor_gps);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, use_smallest);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, compute_gradient);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, compute_gradient_variance);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, compute_covariance);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, use_gp_covariance);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, retrain_outdated);
        ERL_YAML_SAVE_ATTR(tq_node, setting.test_query, use_global_buffer);

        YAML::Node node;
        node["test_query"] = tq_node;
        ERL_YAML_SAVE_ATTR(node, setting, num_threads);
        ERL_YAML_SAVE_ATTR(node, setting, update_hz);
        ERL_YAML_SAVE_ATTR(node, setting, sensor_noise);
        ERL_YAML_SAVE_ATTR(node, setting, gp_sdf_area_scale);
        ERL_YAML_SAVE_ATTR(node, setting, max_valid_gradient_var);
        ERL_YAML_SAVE_ATTR(node, setting, invalid_position_var);
        ERL_YAML_SAVE_ATTR(node, setting, offset_distance);
        ERL_YAML_SAVE_ATTR(node, setting, sdf_gp);

        return node;
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMappingSetting<Dtype, Dim>::YamlConvertImpl::decode(
        const YAML::Node &node,
        GpSdfMappingSetting &setting) {
        const YAML::Node &tq_node = node["test_query"];
        TestQuery &tq = setting.test_query;
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, max_test_valid_distance_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, search_area_half_size, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, num_neighbor_gps, int);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, use_smallest, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, compute_gradient, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, compute_gradient_variance, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, compute_covariance, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, use_gp_covariance, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, retrain_outdated, bool);
        ERL_YAML_LOAD_ATTR_TYPE(tq_node, tq, use_global_buffer, bool);

        ERL_YAML_LOAD_ATTR_TYPE(node, setting, num_threads, int);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, update_hz, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, sensor_noise, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, gp_sdf_area_scale, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, max_valid_gradient_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, invalid_position_var, Dtype);
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, offset_distance, Dtype);
        ERL_YAML_LOAD_ATTR(node, setting, sdf_gp);

        return true;
    }
}  // namespace erl::sdf_mapping
