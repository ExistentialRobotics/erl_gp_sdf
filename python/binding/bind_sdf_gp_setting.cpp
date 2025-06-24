#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/sdf_gp.hpp"

template<typename Dtype>
void
BindSdfGpSettingImpl(const py::module &m, const char *name) {
    using namespace erl::common;
    using namespace erl::gp_sdf;
    using T = erl::gp_sdf::SdfGaussianProcessSetting<Dtype>;
    py::class_<T, YamlableBase, std::shared_ptr<T>>(m, name)
        .def_readwrite("sign_method", &T::sign_method)
        .def_readwrite("hybrid_sign_methods", &T::hybrid_sign_methods)
        .def_readwrite("hybrid_sign_threshold", &T::hybrid_sign_threshold)
        .def_readwrite("normal_scale", &T::normal_scale)
        .def_readwrite("softmin_temperature", &T::softmin_temperature)
        .def_readwrite("sign_gp_offset_distance", &T::sign_gp_offset_distance)
        .def_readwrite("edf_gp_offset_distance", &T::edf_gp_offset_distance)
        .def_readwrite("sign_gp", &T::sign_gp)
        .def_readwrite("edf_gp", &T::edf_gp);
}

void
BindSdfGpSetting(const py::module &m) {
    py::enum_<erl::gp_sdf::SignMethod>(m, "SignMethod")
        .value("kNone", erl::gp_sdf::SignMethod::kNone)
        .value("kSignGp", erl::gp_sdf::SignMethod::kSignGp)
        .value("kNormalGp", erl::gp_sdf::SignMethod::kNormalGp)
        .value("kExternal", erl::gp_sdf::SignMethod::kExternal)
        .value("kHybrid", erl::gp_sdf::SignMethod::kHybrid)
        .export_values();
    BindSdfGpSettingImpl<double>(m, "SdfGaussianProcessSettingD");
    BindSdfGpSettingImpl<float>(m, "SdfGaussianProcessSettingF");
}
