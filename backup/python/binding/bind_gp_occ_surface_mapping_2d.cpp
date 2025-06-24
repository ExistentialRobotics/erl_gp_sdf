#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping_2d.hpp"

void
BindGpOccSurfaceMapping2D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::gp_sdf;

    py::class_<GpOccSurfaceMapping2D, AbstractSurfaceMapping2D, std::shared_ptr<GpOccSurfaceMapping2D>> surface_mapping(m, "GpOccSurfaceMapping2D");
    py::class_<GpOccSurfaceMapping2D::Setting, GpOccSurfaceMappingBaseSetting, std::shared_ptr<GpOccSurfaceMapping2D::Setting>>(surface_mapping, "Setting")
        .def_readwrite("sensor_gp", &GpOccSurfaceMapping2D::Setting::sensor_gp)
        .def_readwrite("quadtree", &GpOccSurfaceMapping2D::Setting::quadtree);

    surface_mapping.def(py::init<const std::shared_ptr<GpOccSurfaceMapping2D::Setting> &>(), py::arg("setting"))
        .def_property_readonly("setting", &GpOccSurfaceMapping2D::GetSetting);
}
