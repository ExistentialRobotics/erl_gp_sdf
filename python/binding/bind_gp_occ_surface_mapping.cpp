#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"

template<typename Dtype, int Dim>
void
BindGpOccSurfaceMappingImpl(const py::module &m, const char *name) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    using T = GpOccSurfaceMapping<Dtype, Dim>;
    using Setting = typename T::Setting;
    using SettingComputeVariance = typename Setting::ComputeVariance;
    using SettingUpdateMapPoints = typename Setting::UpdateMapPoints;

    py::class_<T, AbstractSurfaceMapping<Dtype, Dim>, std::shared_ptr<T>> surface_mapping(m, name);
    py::class_<Setting, YamlableBase, std::shared_ptr<Setting>> setting(surface_mapping, "Setting");
    py::class_<SettingComputeVariance>(setting, "ComputeVariance")
        .def_readwrite(
            "zero_gradient_position_var",
            &SettingComputeVariance::zero_gradient_position_var)
        .def_readwrite(
            "zero_gradient_gradient_var",
            &SettingComputeVariance::zero_gradient_gradient_var)
        .def_readwrite("position_var_alpha", &SettingComputeVariance::position_var_alpha)
        .def_readwrite("min_distance_var", &SettingComputeVariance::min_distance_var)
        .def_readwrite("max_distance_var", &SettingComputeVariance::max_distance_var)
        .def_readwrite("min_gradient_var", &SettingComputeVariance::min_gradient_var)
        .def_readwrite("max_gradient_var", &SettingComputeVariance::max_gradient_var);
    py::class_<SettingUpdateMapPoints>(surface_mapping, "UpdateMapPoints")
        .def_readwrite("max_adjust_tries", &SettingUpdateMapPoints::max_adjust_tries)
        .def_readwrite("min_observable_occ", &SettingUpdateMapPoints::min_observable_occ)
        .def_readwrite("min_position_var", &SettingUpdateMapPoints::min_position_var)
        .def_readwrite("min_gradient_var", &SettingUpdateMapPoints::min_gradient_var)
        .def_readwrite("max_surface_abs_occ", &SettingUpdateMapPoints::max_surface_abs_occ)
        .def_readwrite("max_valid_gradient_var", &SettingUpdateMapPoints::max_valid_gradient_var)
        .def_readwrite("max_bayes_position_var", &SettingUpdateMapPoints::max_bayes_position_var)
        .def_readwrite("max_bayes_gradient_var", &SettingUpdateMapPoints::max_bayes_gradient_var);
    setting.def_readwrite("compute_variance", &Setting::compute_variance)
        .def_readwrite("update_map_points", &Setting::update_map_points)
        .def_readwrite("sensor_gp", &Setting::sensor_gp)
        .def_readwrite("tree", &Setting::tree)
        .def_readwrite("scaling", &Setting::scaling)
        .def_readwrite("perturb_delta", &Setting::perturb_delta)
        .def_readwrite("zero_gradient_threshold", &Setting::zero_gradient_threshold)
        .def_readwrite("update_occupancy", &Setting::update_occupancy)
        .def_readwrite("cluster_depth", &Setting::cluster_depth);

    surface_mapping.def(py::init<const std::shared_ptr<Setting> &>(), py::arg("setting"))
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly("sensor_gp", &T::GetSensorGp)
        .def_property_readonly("tree", &T::GetTree)
        .def_property_readonly("surface_data_manager", &T::GetSurfaceDataManager)
        .def_property_readonly("scaling", &T::GetScaling)
        .def_property_readonly("cluster_size", &T::GetClusterSize)
        .def("get_cluster_center", &T::GetClusterCenter, py::arg("cluster_key"))
        .def_property_readonly(
            "changed_clusters",
            [](T &self) {
                auto cluster_keys = self.GetChangedClusters();
                return typename T::KeyVector(cluster_keys.begin(), cluster_keys.end());
            })
        .def_property_readonly("surface_data_buffer", &T::GetSurfaceDataBuffer)
        .def(
            "collect_surface_data_in_aabb",
            [](T &self, const typename T::Aabb &aabb) {
                std::vector<std::pair<Dtype, std::size_t>> surface_data_indices;
                self.CollectSurfaceDataInAabb(aabb, surface_data_indices);
                return surface_data_indices;
            },
            py::arg("aabb"))
        .def_property_readonly("map_boundary", &T::GetMapBoundary)
        .def(
            "is_in_free_space",
            [](T &self, const typename T::Positions &positions) {
                typename T::VectorX in_free_space;
                bool success = self.IsInFreeSpace(positions, in_free_space);
                return std::make_tuple(success, in_free_space);
            })
        .def(
            "write",
            [](const T *self, const char *filename) {
                return erl::common::Serialization<T>::Write(filename, self);
            },
            py::arg("filename"))
        .def(
            "read",
            [](T *self, const char *filename) {
                return erl::common::Serialization<T>::Read(filename, self);
            },
            py::arg("filename"));
    ;
}

void
BindGpOccSurfaceMapping(const py::module &m) {
    BindGpOccSurfaceMappingImpl<double, 2>(m, "GpOccSurfaceMapping2Dd");
    BindGpOccSurfaceMappingImpl<float, 2>(m, "GpOccSurfaceMapping2Df");
    BindGpOccSurfaceMappingImpl<double, 3>(m, "GpOccSurfaceMapping3Dd");
    BindGpOccSurfaceMappingImpl<float, 3>(m, "GpOccSurfaceMapping3Df");
}
