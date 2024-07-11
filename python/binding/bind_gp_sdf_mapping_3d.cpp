#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

void
BindGpSdfMapping3D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMapping3D, std::shared_ptr<GpSdfMapping3D>> sdf_mapping(m, "GpSdfMapping3D");

    py::class_<GpSdfMapping3D::Gp, std::shared_ptr<GpSdfMapping3D::Gp>>(sdf_mapping, "Gp")
        .def_readonly("active", &GpSdfMapping3D::Gp::active)
        .def_readonly("locked_for_test", &GpSdfMapping3D::Gp::locked_for_test)
        .def_readonly("num_train_samples", &GpSdfMapping3D::Gp::num_train_samples)
        .def_readonly("position", &GpSdfMapping3D::Gp::position)
        .def_readonly("half_size", &GpSdfMapping3D::Gp::half_size)
        .def_readonly("gp", &GpSdfMapping3D::Gp::gp)
        .def("train", &GpSdfMapping3D::Gp::Train);

    sdf_mapping.def(py::init<std::shared_ptr<AbstractSurfaceMapping3D>, std::shared_ptr<GpSdfMappingSetting>>(), py::arg("surface_mapping"), py::arg("setting"))
        .def_property_readonly("setting", &GpSdfMapping3D::GetSetting)
        .def_property_readonly("surface_mapping", &GpSdfMapping3D::GetSurfaceMapping)
        .def("update", &GpSdfMapping3D::Update, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def(
            "test",
            [](GpSdfMapping3D &self, const Eigen::Ref<const Eigen::Matrix3Xd> &positions) {
                Eigen::VectorXd distances;
                Eigen::Matrix3Xd gradients;
                Eigen::Matrix4Xd variances_out;
                Eigen::Matrix6Xd covariances_out;

                if (!self.Test(positions, distances, gradients, variances_out, covariances_out)) {
                    return py::make_tuple(py::none(), py::none(), py::none(), py::none());
                }
                return py::make_tuple(distances, gradients, variances_out, covariances_out);
            },
            py::arg("positions"));
}
