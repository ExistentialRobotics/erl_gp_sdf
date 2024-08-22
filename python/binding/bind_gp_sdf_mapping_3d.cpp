#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

void
BindGpSdfMapping3D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;
    using T = GpSdfMapping3D;

    py::class_<T, std::shared_ptr<T>> sdf_mapping(m, "GpSdfMapping3D");

    py::class_<T::Gp, std::shared_ptr<T::Gp>>(sdf_mapping, "Gp")
        .def_readonly("active", &T::Gp::active)
        .def_property_readonly("locked_for_test", [](const T::Gp &gp) { return gp.locked_for_test.load(); })
        .def_readonly("num_train_samples", &T::Gp::num_train_samples)
        .def_readonly("position", &T::Gp::position)
        .def_readonly("half_size", &T::Gp::half_size)
        .def_readonly("gp", &T::Gp::gp)
        .def("train", &T::Gp::Train);

    py::class_<T::Setting, GpSdfMappingBaseSetting, std::shared_ptr<T::Setting>>(sdf_mapping, "Setting")
        .def(py::init<>())
        .def_readwrite("surface_mapping_type", &T::Setting::surface_mapping_type)
        .def_readwrite("surface_mapping", &T::Setting::surface_mapping);

    sdf_mapping.def(py::init<std::shared_ptr<T::Setting>>(), py::arg("setting"))
        .def_property_readonly("setting", &T::GetSetting)
        .def_property_readonly("surface_mapping", &T::GetSurfaceMapping)
        .def("update", &T::Update, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def(
            "test",
            [](T &self, const Eigen::Ref<const Eigen::Matrix3Xd> &positions) {
                Eigen::VectorXd distances;
                Eigen::Matrix3Xd gradients;
                Eigen::Matrix4Xd variances_out;
                Eigen::Matrix6Xd covariances_out;

                if (!self.Test(positions, distances, gradients, variances_out, covariances_out)) {
                    return py::make_tuple(py::none(), py::none(), py::none(), py::none());
                }
                return py::make_tuple(distances, gradients, variances_out, covariances_out);
            },
            py::arg("positions"))
        .def_property_readonly("used_gps", &T::GetUsedGps, "GPs used by the last test call")
        .def_property_readonly("gps", &T::GetGpMap)
        .def_property_readonly("num_update_calls", &T::GetNumUpdateCalls)
        .def_property_readonly("num_test_calls", &T::GetNumTestCalls)
        .def_property_readonly("num_test_positions", &T::GetNumTestPositions)
        .def("__eq__", &T::operator==)
        .def("__ne__", &T::operator!=)
        .def("write", py::overload_cast<const std::string &>(&T::Write, py::const_), py::arg("filename"))
        .def("read", py::overload_cast<const std::string &>(&T::Read), py::arg("filename"));
}
