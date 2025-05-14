#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_setting.hpp"

template<typename Dtype, int Dim>
void
BindGpSdfMappingSettingImpl(const py::module &m, const char *name) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    using T = GpSdfMappingSetting<Dtype, Dim>;
    using TestQuery = typename T::TestQuery;

    py::class_<T, YamlableBase, std::shared_ptr<T>> sdf_mapping_setting(m, name);
    py::class_<TestQuery>(sdf_mapping_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &TestQuery::search_area_half_size)
        .def_readwrite("num_neighbor_gps", &TestQuery::num_neighbor_gps)
        .def_readwrite("use_smallest", &TestQuery::use_smallest)
        .def_readwrite("compute_gradient", &TestQuery::compute_gradient)
        .def_readwrite("compute_gradient_variance", &TestQuery::compute_gradient_variance)
        .def_readwrite("compute_covariance", &TestQuery::compute_covariance)
        .def_readwrite("use_gp_covariance", &TestQuery::use_gp_covariance)
        .def_readwrite("retrain_outdated", &TestQuery::retrain_outdated)
        .def_readwrite("use_global_buffer", &TestQuery::use_global_buffer);
    sdf_mapping_setting.def(py::init<>([]() { return std::make_shared<T>(); }))
        .def_readwrite("test_query", &T::test_query)
        .def_readwrite("num_threads", &T::num_threads)
        .def_readwrite("update_hz", &T::update_hz)
        .def_readwrite("sensor_noise", &T::sensor_noise)
        .def_readwrite("gp_sdf_area_scale", &T::gp_sdf_area_scale)
        .def_readwrite("max_valid_gradient_var", &T::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &T::invalid_position_var)
        .def_readwrite("sdf_gp", &T::sdf_gp);
}

void
BindGpSdfMappingSetting(const py::module &m) {
    BindGpSdfMappingSettingImpl<double, 2>(m, "GpSdfMappingSetting2Dd");
    BindGpSdfMappingSettingImpl<float, 2>(m, "GpSdfMappingSetting2Df");
    BindGpSdfMappingSettingImpl<double, 3>(m, "GpSdfMappingSetting3Dd");
    BindGpSdfMappingSettingImpl<float, 3>(m, "GpSdfMappingSetting3Df");
}
