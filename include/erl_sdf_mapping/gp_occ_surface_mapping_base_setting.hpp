#pragma once
#include "abstract_surface_mapping.hpp"

#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {
    struct GpOccSurfaceMappingBaseSetting : common::Yamlable<GpOccSurfaceMappingBaseSetting, AbstractSurfaceMapping::Setting> {

        struct ComputeVariance : Yamlable<ComputeVariance> {
            double zero_gradient_position_var = 1.;  // position variance to use when the estimated gradient is almost zero.
            double zero_gradient_gradient_var = 1.;  // gradient variance to use when the estimated gradient is almost zero.
            double min_distance_var = 1.;            // minimum distance variance.
            double max_distance_var = 100.;          // maximum distance variance.
            double position_var_alpha = 0.01;        // scaling number of position variance.
            double min_gradient_var = 0.01;          // minimum gradient variance.
            double max_gradient_var = 1.;            // maximum gradient variance.
        };

        struct UpdateMapPoints : Yamlable<UpdateMapPoints> {
            double min_observable_occ = -0.1;     // points of OCC smaller than this value is considered unobservable, i.e. inside the object.
            double max_surface_abs_occ = 0.02;    // maximum absolute value of surface points' OCC, which should be zero ideally.
            double max_valid_gradient_var = 0.5;  // maximum valid gradient variance, above this threshold, it won't be used for the Bayes Update.
            int max_adjust_tries = 10;
            double max_bayes_position_var = 1.;   // if the position variance by Bayes Update is above this threshold, it will be discarded.
            double max_bayes_gradient_var = 0.6;  // if the gradient variance by Bayes Update is above this threshold, it will be discarded.
            double min_position_var = 0.001;      // minimum position variance.
            double min_gradient_var = 0.001;      // minimum gradient variance.
        };

        ComputeVariance compute_variance;   // parameters used by ComputeVariance.
        UpdateMapPoints update_map_points;  // parameters used by UpdateMapPoints.
        unsigned int cluster_level = 2;     // 2^2 times of the quadtree resolution.
        double perturb_delta = 0.01;
        double zero_gradient_threshold = 1.e-15;  // gradient below this threshold is considered zero.
        bool update_occupancy = true;
    };

}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::ComputeVariance> {

    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::ComputeVariance &rhs) {
        Node node;
        node["zero_gradient_position_var"] = rhs.zero_gradient_position_var;
        node["zero_gradient_gradient_var"] = rhs.zero_gradient_gradient_var;
        node["min_distance_var"] = rhs.min_distance_var;
        node["max_distance_var"] = rhs.max_distance_var;
        node["position_var_alpha"] = rhs.position_var_alpha;
        node["min_gradient_var"] = rhs.min_gradient_var;
        node["max_gradient_var"] = rhs.max_gradient_var;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::ComputeVariance &rhs) {
        if (!node.IsMap()) { return false; }
        rhs.zero_gradient_position_var = node["zero_gradient_position_var"].as<double>();
        rhs.zero_gradient_gradient_var = node["zero_gradient_gradient_var"].as<double>();
        rhs.min_distance_var = node["min_distance_var"].as<double>();
        rhs.max_distance_var = node["max_distance_var"].as<double>();
        rhs.position_var_alpha = node["position_var_alpha"].as<double>();
        rhs.min_gradient_var = node["min_gradient_var"].as<double>();
        rhs.max_gradient_var = node["max_gradient_var"].as<double>();
        return true;
    }
};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::UpdateMapPoints> {

    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::UpdateMapPoints &rhs) {
        Node node;
        node["min_observable_occ"] = rhs.min_observable_occ;
        node["max_surface_abs_occ"] = rhs.max_surface_abs_occ;
        node["max_valid_gradient_var"] = rhs.max_valid_gradient_var;
        node["max_adjust_tries"] = rhs.max_adjust_tries;
        node["max_bayes_position_var"] = rhs.max_bayes_position_var;
        node["max_bayes_gradient_var"] = rhs.max_bayes_gradient_var;
        node["min_position_var"] = rhs.min_position_var;
        node["min_gradient_var"] = rhs.min_gradient_var;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMappingBaseSetting::UpdateMapPoints &rhs) {
        if (!node.IsMap()) { return false; }
        rhs.min_observable_occ = node["min_observable_occ"].as<double>();
        rhs.max_surface_abs_occ = node["max_surface_abs_occ"].as<double>();
        rhs.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
        rhs.max_adjust_tries = node["max_adjust_tries"].as<int>();
        rhs.max_bayes_position_var = node["max_bayes_position_var"].as<double>();
        rhs.max_bayes_gradient_var = node["max_bayes_gradient_var"].as<double>();
        rhs.min_position_var = node["min_position_var"].as<double>();
        rhs.min_gradient_var = node["min_gradient_var"].as<double>();
        return true;
    }
};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting> {
    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMappingBaseSetting &setting) {
        Node node;
        node["compute_variance"] = setting.compute_variance;
        node["update_map_points"] = setting.update_map_points;
        node["cluster_level"] = setting.cluster_level;
        node["perturb_delta"] = setting.perturb_delta;
        node["zero_gradient_threshold"] = setting.zero_gradient_threshold;
        node["update_occupancy"] = setting.update_occupancy;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMappingBaseSetting &setting) {
        if (!node.IsMap()) { return false; }
        setting.compute_variance = node["compute_variance"].as<decltype(setting.compute_variance)>();
        setting.update_map_points = node["update_map_points"].as<decltype(setting.update_map_points)>();
        setting.cluster_level = node["cluster_level"].as<unsigned int>();
        setting.perturb_delta = node["perturb_delta"].as<double>();
        setting.zero_gradient_threshold = node["zero_gradient_threshold"].as<double>();
        setting.update_occupancy = node["update_occupancy"].as<bool>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
