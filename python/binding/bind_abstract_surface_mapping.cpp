#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/abstract_surface_mapping.hpp"

void
BindAbstractSurfaceMapping(const py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<AbstractSurfaceMapping, std::shared_ptr<AbstractSurfaceMapping>> abstract_surface_mapping(m, "AbstractSurfaceMapping");
    abstract_surface_mapping.def_static("create", &AbstractSurfaceMapping::Create<>, py::arg("mapping_type"), py::arg("setting"))
        .def("__eq__", &AbstractSurfaceMapping::operator==, py::arg("other"))
        .def("__ne__", &AbstractSurfaceMapping::operator!=, py::arg("other"))
        .def("write", py::overload_cast<const std::string &>(&AbstractSurfaceMapping::Write, py::const_), py::arg("filename"))
        .def("read", py::overload_cast<const std::string &>(&AbstractSurfaceMapping::Read), py::arg("filename"));
}
