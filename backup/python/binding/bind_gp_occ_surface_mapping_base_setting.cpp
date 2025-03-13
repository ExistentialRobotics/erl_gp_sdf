#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_base_setting.hpp"

void
BindGpOccSurfaceMappingBaseSetting(const py::module& m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<GpOccSurfaceMappingBaseSetting, AbstractSurfaceMapping::Setting, std::shared_ptr<GpOccSurfaceMappingBaseSetting>> setting(
        m,
        "GpOccSurfaceMappingBaseSetting");
    py::class_<GpOccSurfaceMappingBaseSetting::ComputeVariance, YamlableBase, std::shared_ptr<GpOccSurfaceMappingBaseSetting::ComputeVariance>>(
        setting,
        "ComputeVariance")
        .def_readwrite("zero_gradient_position_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::zero_gradient_position_var)
        .def_readwrite("zero_gradient_gradient_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::zero_gradient_gradient_var)
        .def_readwrite("min_distance_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::min_distance_var)
        .def_readwrite("max_distance_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::max_distance_var)
        .def_readwrite("position_var_alpha", &GpOccSurfaceMappingBaseSetting::ComputeVariance::position_var_alpha)
        .def_readwrite("min_gradient_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::min_gradient_var)
        .def_readwrite("max_gradient_var", &GpOccSurfaceMappingBaseSetting::ComputeVariance::max_gradient_var);
    py::class_<GpOccSurfaceMappingBaseSetting::UpdateMapPoints, YamlableBase, std::shared_ptr<GpOccSurfaceMappingBaseSetting::UpdateMapPoints>>(
        setting,
        "UpdateMapPoints")
        .def_readwrite("min_observable_occ", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::min_observable_occ)
        .def_readwrite("max_surface_abs_occ", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::max_surface_abs_occ)
        .def_readwrite("max_valid_gradient_var", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::max_valid_gradient_var)
        .def_readwrite("max_adjust_tries", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::max_adjust_tries)
        .def_readwrite("max_bayes_position_var", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::max_bayes_position_var)
        .def_readwrite("max_bayes_gradient_var", &GpOccSurfaceMappingBaseSetting::UpdateMapPoints::max_bayes_gradient_var);
    setting.def_readwrite("compute_variance", &GpOccSurfaceMappingBaseSetting::compute_variance)
        .def_readwrite("update_map_points", &GpOccSurfaceMappingBaseSetting::update_map_points)
        .def_readwrite("cluster_level", &GpOccSurfaceMappingBaseSetting::cluster_level)
        .def_readwrite("perturb_delta", &GpOccSurfaceMappingBaseSetting::perturb_delta)
        .def_readwrite("zero_gradient_threshold", &GpOccSurfaceMappingBaseSetting::zero_gradient_threshold)
        .def_readwrite("update_occupancy", &GpOccSurfaceMappingBaseSetting::update_occupancy);
}
