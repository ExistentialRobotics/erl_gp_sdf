#include "erl_common/pybind11.hpp"
#include "erl_common/pybind11_data_buffer_manager.hpp"
#include "erl_gp_sdf/surface_data_manager.hpp"

template<typename Dtype, int Dim>
void
BindSurfaceDataImpl(const py::module &m, const char *name) {
    using namespace erl::gp_sdf;
    using T = SurfaceData<Dtype, Dim>;
    using VectorD = typename T::VectorD;
    py::class_<T>(m, name)
        .def(py::init<>())
        .def(py::init<VectorD, VectorD, Dtype, Dtype>(), py::arg("position"), py::arg("normal"), py::arg("var_position"), py::arg("var_normal"))
        .def_readwrite("position", &T::position)
        .def_readwrite("normal", &T::normal)
        .def_readwrite("var_position", &T::var_position)
        .def_readwrite("var_normal", &T::var_normal);
}

template<typename Dtype, int Dim>
void
BindSurfaceDataManagerImpl(const py::module &m, const char *name) {
    using namespace erl::common;
    using namespace erl::gp_sdf;
    using Data = SurfaceData<Dtype, Dim>;
    // using Base = DataBufferManager<Data>;
    using T = SurfaceDataManager<Dtype, Dim>;

    auto data_buffer_manager = BindDataBufferManagerImpl<Data, std::vector<Data>>(m, (name + std::string("Base")).c_str());
    py::class_<T, DataBufferManager<Data>>(m, name);
}

void
BindSurfaceDataManager(const py::module &m) {
    BindSurfaceDataImpl<double, 3>(m, "SurfaceData3Dd");
    BindSurfaceDataImpl<float, 3>(m, "SurfaceData3Df");
    BindSurfaceDataImpl<double, 2>(m, "SurfaceData2Dd");
    BindSurfaceDataImpl<float, 2>(m, "SurfaceData2Df");

    BindSurfaceDataManagerImpl<double, 3>(m, "SurfaceDataManager3Dd");
    BindSurfaceDataManagerImpl<float, 3>(m, "SurfaceDataManager3Df");
    BindSurfaceDataManagerImpl<double, 2>(m, "SurfaceDataManager2Dd");
    BindSurfaceDataManagerImpl<float, 2>(m, "SurfaceDataManager2Df");
}
