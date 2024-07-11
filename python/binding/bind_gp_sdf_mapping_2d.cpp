#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"

void
BindGpSdfMapping2D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMapping2D, std::shared_ptr<GpSdfMapping2D>> sdf_mapping(m, "GpSdfMapping2D");

    py::class_<GpSdfMapping2D::Gp, std::shared_ptr<GpSdfMapping2D::Gp>>(sdf_mapping, "Gp")
        .def_readonly("active", &GpSdfMapping2D::Gp::active)
        .def_readonly("locked_for_test", &GpSdfMapping2D::Gp::locked_for_test)
        .def_readonly("num_train_samples", &GpSdfMapping2D::Gp::num_train_samples)
        .def_readonly("position", &GpSdfMapping2D::Gp::position)
        .def_readonly("half_size", &GpSdfMapping2D::Gp::half_size)
        .def_readonly("gp", &GpSdfMapping2D::Gp::gp)
        .def("train", &GpSdfMapping2D::Gp::Train);

    sdf_mapping.def(py::init<std::shared_ptr<AbstractSurfaceMapping2D>, std::shared_ptr<GpSdfMappingSetting>>(), py::arg("surface_mapping"), py::arg("setting"))
        .def_property_readonly("setting", &GpSdfMapping2D::GetSetting)
        .def_property_readonly("surface_mapping", &GpSdfMapping2D::GetSurfaceMapping)
        .def("update", &GpSdfMapping2D::Update, py::arg("rotation"), py::arg("translation"), py::arg("ranges"))
        .def(
            "test",
            [](GpSdfMapping2D &self, const Eigen::Ref<const Eigen::Matrix2Xd> &positions) {
                Eigen::VectorXd distances;
                Eigen::Matrix2Xd gradients;

                if (Eigen::Matrix3Xd variances_out, covariances_out; self.Test(positions, distances, gradients, variances_out, covariances_out)) {
                    return py::make_tuple(distances, gradients, variances_out, covariances_out);
                }
                return py::make_tuple(py::none(), py::none(), py::none(), py::none());
            },
            py::arg("positions"));
}
