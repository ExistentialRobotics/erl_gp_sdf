#include "erl_common/pybind11.hpp"

void
BindSurfaceMappingQuadtreeNode(const py::module &m);

void
BindSurfaceMappingQuadtree(const py::module &m);

void
BindSurfaceMappingOctreeNode(const py::module &m);

void
BindSurfaceMappingOctree(const py::module &m);

void
BindLogEdfGaussianProcess(const py::module &m);

void
BindAbstractSurfaceMapping(const py::module &m);

void
BindGpOccSurfaceMappingBaseSetting(const py::module &m);

void
BindGpOccSurfaceMapping2D(const py::module &m);

void
BindGpOccSurfaceMapping3D(const py::module &m);

void
BindGpSdfMappingSetting(const py::module &m);

void
BindGpSdfMapping2D(const py::module &m);

void
BindGpSdfMapping3D(const py::module &m);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_sdf_mapping";
    BindSurfaceMappingQuadtreeNode(m);
    BindSurfaceMappingQuadtree(m);
    BindSurfaceMappingOctreeNode(m);
    BindSurfaceMappingOctree(m);
    BindLogEdfGaussianProcess(m);
    BindAbstractSurfaceMapping(m);
    BindGpOccSurfaceMappingBaseSetting(m);
    BindGpOccSurfaceMapping2D(m);
    BindGpOccSurfaceMapping3D(m);
    BindGpSdfMappingSetting(m);
    BindGpSdfMapping2D(m);
    BindGpSdfMapping3D(m);
}
