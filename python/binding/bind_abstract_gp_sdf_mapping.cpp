// #include "erl_common/pybind11.hpp"
// #include "erl_common/serialization.hpp"
// #include "erl_gp_sdf/gp_sdf_mapping.hpp"
//
// void
// BindAbstractGpSdfMapping(const py::module &m) {
//     using namespace erl::common;
//     using namespace erl::sdf_mapping;
//     using T = AbstractGpSdfMapping;
//
//     py::class_<T, std::shared_ptr<T>> abstract_gp_sdf_mapping(m, "AbstractGpSdfMapping");
//     abstract_gp_sdf_mapping
//         .def_static(
//             "create",
//             &T::Create,
//             py::arg("mapping_type"),
//             py::arg("surface_mapping_setting"),
//             py::arg("sdf_mapping_setting"))
//         .def(
//             "update",
//             &T::Update,
//             py::arg("rotation"),
//             py::arg("translation"),
//             py::arg("scan"),
//             py::arg("are_points"),
//             py::arg("are_local"))
//         .def(
//             "predict",
//             [](T *self,
//                const Eigen::Ref<const Eigen::MatrixXd> &positions) -> std::optional<py::dict> {
//                 Eigen::VectorXd distances;
//                 Eigen::MatrixXd gradients;
//                 Eigen::MatrixXd variances;
//                 Eigen::MatrixXd covariances;
//                 if (self->Predict(positions, distances, gradients, variances, covariances)) {
//                     py::dict result;
//                     result["distances"] = distances;
//                     result["gradients"] = gradients;
//                     result["variances"] = variances;
//                     result["covariances"] = covariances;
//                     return result;
//                 }
//                 return std::nullopt;
//             },
//             py::arg("positions"))
//         .def("__eq__", &T::operator==, py::arg("other"))
//         .def("__ne__", &T::operator!=, py::arg("other"))
//         .def(
//             "write",
//             [](const T *self, const std::string &filename) {
//                 return Serialization<T>::Write(filename, self);
//             },
//             py::arg("filename"))
//         .def(
//             "read",
//             [](T *self, const std::string &filename) {
//                 return Serialization<T>::Read(filename, self);
//             },
//             py::arg("filename"));
// }
