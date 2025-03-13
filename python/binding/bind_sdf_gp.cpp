#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/sdf_gp.hpp"

template<typename Dtype, int Dim>
void
BindSdfGpImpl(const py::module &m, const char *name) {
    using namespace erl::sdf_mapping;

    using T = SdfGaussianProcess<Dtype, Dim>;

    py::class_<T, std::shared_ptr<T>>(m, name)
        .def_readwrite("setting", &T::setting)
        .def_readonly("active", &T::active)
        .def_property_readonly("locked_for_test", [](const T &gp) { return gp.locked_for_test.load(); })
        .def_readonly("num_edf_samples", &T::num_edf_samples)
        .def_readonly("position", &T::position)
        .def_readonly("half_size", &T::half_size)
        .def_readonly("edf_gp", &T::edf_gp)
        .def("activate", &T::Activate)
        .def("deactivate", &T::Deactivate)
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def(
            "intersects",
            py::overload_cast<const typename T::VectorD &, Dtype>(&T::Intersects, py::const_),
            py::arg("other_position"),
            py::arg("other_half_size"))
        .def(
            "intersects",
            py::overload_cast<const typename T::VectorD &, const typename T::VectorD &>(&T::Intersects, py::const_),
            py::arg("other_position"),
            py::arg("other_half_sizes"))
        .def(
            "load_surface_data",
            &T::LoadSurfaceData,
            py::arg("surface_data_indices"),
            py::arg("surface_data_vec"),
            py::arg("offset_distance"),
            py::arg("sensor_noise"),
            py::arg("max_valid_gradient_var"),
            py::arg("invalid_position_var"))
        .def("train", &T::Train)
        .def(
            "test",
            [](T &self,
               const typename T::VectorD &test_position,
               const Dtype offset_distance,
               const Dtype softmin_temperature,
               const bool use_gp_covariance,
               const bool compute_covariance) {
                Eigen::Vector<Dtype, Dim + 1> f;
                Eigen::Vector<Dtype, Dim + 1> var;
                Eigen::Vector<Dtype, Dim *(Dim + 1) / 2> covariance;

                return self.Test(test_position, f, var, covariance, offset_distance, softmin_temperature, use_gp_covariance, compute_covariance)
                           ? py::make_tuple(f, var, covariance)
                           : py::make_tuple(py::none(), py::none(), py::none());
            });
}

void
BindSdfGp(const py::module &m) {
    BindSdfGpImpl<double, 3>(m, "SdfGp3Dd");
    BindSdfGpImpl<float, 3>(m, "SdfGp3Df");
    BindSdfGpImpl<double, 2>(m, "SdfGp2Dd");
    BindSdfGpImpl<float, 2>(m, "SdfGp2Df");
}
