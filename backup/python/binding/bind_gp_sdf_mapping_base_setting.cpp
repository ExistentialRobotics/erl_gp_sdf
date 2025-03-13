#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_base_setting.hpp"

void
BindGpSdfMappingSetting(const py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMappingBaseSetting, YamlableBase, std::shared_ptr<GpSdfMappingBaseSetting>> sdf_mapping_setting(m, "GpSdfMappingBaseSetting");
    py::class_<GpSdfMappingBaseSetting::TestQuery, YamlableBase, std::shared_ptr<GpSdfMappingBaseSetting::TestQuery>>(sdf_mapping_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &GpSdfMappingBaseSetting::TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &GpSdfMappingBaseSetting::TestQuery::search_area_half_size)
        .def_readwrite("num_neighbor_gps", &GpSdfMappingBaseSetting::TestQuery::num_neighbor_gps)
        .def_readwrite("use_smallest", &GpSdfMappingBaseSetting::TestQuery::use_smallest)
        .def_readwrite("compute_covariance", &GpSdfMappingBaseSetting::TestQuery::compute_covariance)
        .def_readwrite("use_gp_covariance", &GpSdfMappingBaseSetting::TestQuery::use_gp_covariance)
        .def_readwrite("softmax_temperature", &GpSdfMappingBaseSetting::TestQuery::softmin_temperature);
    sdf_mapping_setting.def(py::init<>([]() { return std::make_shared<GpSdfMappingBaseSetting>(); }))
        .def_readwrite("num_threads", &GpSdfMappingBaseSetting::num_threads)
        .def_readwrite("update_hz", &GpSdfMappingBaseSetting::update_hz)
        .def_readwrite("gp_sdf_area_scale", &GpSdfMappingBaseSetting::gp_sdf_area_scale)
        .def_readwrite("offset_distance", &GpSdfMappingBaseSetting::offset_distance)
        .def_readwrite("max_valid_gradient_var", &GpSdfMappingBaseSetting::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &GpSdfMappingBaseSetting::invalid_position_var)
        .def_readwrite("use_occ_sign", &GpSdfMappingBaseSetting::use_occ_sign)
        .def_readwrite("offset_distance", &GpSdfMappingBaseSetting::offset_distance)
        .def_readwrite("edf_gp", &GpSdfMappingBaseSetting::edf_gp)
        .def_readwrite("test_query", &GpSdfMappingBaseSetting::test_query);
}
