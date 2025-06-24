#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping_3d.hpp"

void
BindGpOccSurfaceMapping3D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::gp_sdf;

    py::class_<GpOccSurfaceMapping3D, AbstractSurfaceMapping3D, std::shared_ptr<GpOccSurfaceMapping3D>> surface_mapping(m, "GpOccSurfaceMapping3D");
    py::class_<GpOccSurfaceMapping3D::Setting, GpOccSurfaceMappingBaseSetting, std::shared_ptr<GpOccSurfaceMapping3D::Setting>>(surface_mapping, "Setting")
        .def_readwrite("sensor_gp", &GpOccSurfaceMapping3D::Setting::sensor_gp)
        .def_readwrite("octree", &GpOccSurfaceMapping3D::Setting::octree);

    surface_mapping.def(py::init<const std::shared_ptr<GpOccSurfaceMapping3D::Setting> &>(), py::arg("setting"))
        .def_property_readonly("setting", &GpOccSurfaceMapping3D::GetSetting)
        .def_property_readonly("octree", &GpOccSurfaceMapping3D::GetOctree)
        .def("update", &GpOccSurfaceMapping3D::Update, py::arg("rotation"), py::arg("translation"), py::arg("ranges"));
}
