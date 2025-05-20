#include "erl_common/pybind11.hpp"
#include "erl_common/serialization.hpp"
#include "erl_sdf_mapping/abstract_surface_mapping.hpp"

void
BindAbstractSurfaceMapping(const py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;
    using T = AbstractSurfaceMapping;

    py::class_<T, std::shared_ptr<T>> abstract_surface_mapping(m, "AbstractSurfaceMapping");
    abstract_surface_mapping
        .def_static("create", &T::Create<>, py::arg("mapping_type"), py::arg("setting"))
        .def(
            "update",
            &T::Update,
            py::arg("rotation"),
            py::arg("translation"),
            py::arg("scan"),
            py::arg("are_points"),
            py::arg("are_local"))
        .def("__eq__", &T::operator==, py::arg("other"))
        .def("__ne__", &T::operator!=, py::arg("other"))
        .def(
            "write",
            [](const T *self, const std::string &filename) {
                return Serialization<T>::Write(filename, self);
            },
            py::arg("filename"))
        .def(
            "read",
            [](T *self, const std::string &filename) {
                return Serialization<T>::Read(filename, self);
            },
            py::arg("filename"));
}
