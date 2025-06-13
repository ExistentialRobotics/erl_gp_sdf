#include "erl_common/pybind11.hpp"

void
BindSurfaceDataManager(const py::module &m);

void
BindLogEdfGaussianProcess(const py::module &m);

void
BindSdfGpSetting(const py::module &m);

void
BindSdfGp(const py::module &m);

void
BindAbstractSurfaceMapping(const py::module &m);

void
BindGpOccSurfaceMapping(const py::module &m);

void
BindGpSdfMappingSetting(const py::module &m);

// void
// BindAbstractGpSdfMapping(const py::module &m);

void
BindGpSdfMapping(const py::module &m);

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_gp_sdf";

    BindSurfaceDataManager(m);
    BindLogEdfGaussianProcess(m);
    BindSdfGpSetting(m);
    BindSdfGp(m);

    BindAbstractSurfaceMapping(m);
    BindGpOccSurfaceMapping(m);
    BindGpSdfMappingSetting(m);
    // BindAbstractGpSdfMapping(m);
    BindGpSdfMapping(m);
}
