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
BindSurfaceDataManager(const py::module &m);

void
BindLogEdfGaussianProcess(const py::module &m);

void
BindSdfGp(const py::module &m);

void
BindAbstractSurfaceMapping(const py::module &m);

void
BindGpOccSurfaceMapping(const py::module &m);

void
BindGpSdfMappingSetting(const py::module &m);

void
BindGpSdfMapping(const py::module &m);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_sdf_mapping";

    BindSurfaceMappingQuadtreeNode(m);
    BindSurfaceMappingQuadtree(m);
    BindSurfaceMappingOctreeNode(m);
    BindSurfaceMappingOctree(m);

    BindSurfaceDataManager(m);
    BindLogEdfGaussianProcess(m);
    BindSdfGp(m);

    BindAbstractSurfaceMapping(m);
    BindGpOccSurfaceMapping(m);
    BindGpSdfMappingSetting(m);
    BindGpSdfMapping(m);
}
