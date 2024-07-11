#include "erl_common/pybind11.hpp"

void
BindLogSdfGaussianProcess(const py::module &m);

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

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_sdf_mapping";
    BindLogSdfGaussianProcess(m);
    BindGpOccSurfaceMappingBaseSetting(m);
    BindGpOccSurfaceMapping2D(m);
    BindGpOccSurfaceMapping3D(m);
    BindGpSdfMappingSetting(m);
    BindGpSdfMapping2D(m);
    BindGpSdfMapping3D(m);
}
