#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_setting.hpp"

void
BindGpSdfMappingSetting(const py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMappingSetting, YamlableBase, std::shared_ptr<GpSdfMappingSetting>> sdf_mapping_setting(m, "GpSdfMappingSetting");
    py::class_<GpSdfMappingSetting::TestQuery, YamlableBase, std::shared_ptr<GpSdfMappingSetting::TestQuery>>(sdf_mapping_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &GpSdfMappingSetting::TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &GpSdfMappingSetting::TestQuery::search_area_half_size)
        .def_readwrite("use_nearest_only", &GpSdfMappingSetting::TestQuery::use_nearest_only)
        .def_readwrite("compute_covariance", &GpSdfMappingSetting::TestQuery::compute_covariance)
        .def_readwrite("recompute_variance", &GpSdfMappingSetting::TestQuery::recompute_variance)
        .def_readwrite("softmax_temperature", &GpSdfMappingSetting::TestQuery::softmax_temperature);
    sdf_mapping_setting.def(py::init<>([]() { return std::make_shared<GpSdfMappingSetting>(); }))
        .def_readwrite("num_threads", &GpSdfMappingSetting::num_threads)
        .def_readwrite("update_hz", &GpSdfMappingSetting::update_hz)
        .def_readwrite("gp_sdf_area_scale", &GpSdfMappingSetting::gp_sdf_area_scale)
        .def_readwrite("offset_distance", &GpSdfMappingSetting::offset_distance)
        .def_readwrite("max_valid_gradient_var", &GpSdfMappingSetting::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &GpSdfMappingSetting::invalid_position_var)
        .def_readwrite("train_gp_immediately", &GpSdfMappingSetting::train_gp_immediately)
        .def_readwrite("gp_sdf", &GpSdfMappingSetting::gp_sdf)
        .def_readwrite("test_query", &GpSdfMappingSetting::test_query)
        .def_readwrite("log_timing", &GpSdfMappingSetting::log_timing);
}
