#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/log_edf_gp.hpp"

template<typename Dtype>
void
BindLogEdfGaussianProcessImpl(const py::module &m, const char *name) {

    using ParentT = erl::gaussian_process::NoisyInputGaussianProcess<Dtype>;
    using T = erl::sdf_mapping::LogEdfGaussianProcess<Dtype>;
    using ParentSetting = typename ParentT::Setting;
    using Setting = typename T::Setting;

    py::class_<T, ParentT, std::shared_ptr<T>> py_log_noisy_input_gp(m, name);

    py::class_<Setting, ParentSetting, std::shared_ptr<Setting>>(py_log_noisy_input_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("log_lambda", &T::Setting::log_lambda);

    py::class_<
        typename T::TestResult,
        typename ParentT::TestResult,
        std::shared_ptr<typename T::TestResult>>(py_log_noisy_input_gp, "TestResult")
        .def(
            "get_gradient_2d",
            [](const typename T::TestResult &self, const long index, const long y_index) {
                ERL_ASSERTM(self.GetDimX() == 2, "x_dim = {}, it should be {}.", self.GetDimX(), 2);
                Eigen::Vector2<Dtype> vec_grad_out(2);
                bool success = self.template GetGradientD<2>(index, y_index, vec_grad_out.data());
                return py::make_tuple(success, vec_grad_out);
            },
            py::arg("index"),
            py::arg("y_index"))
        .def(
            "get_gradient_3d",
            [](const typename T::TestResult &self, const long index, const long y_index) {
                ERL_ASSERTM(self.GetDimX() == 3, "x_dim = {}, it should be {}.", self.GetDimX(), 3);
                Eigen::Vector3<Dtype> vec_grad_out(3);
                bool success = self.template GetGradientD<3>(index, y_index, vec_grad_out.data());
                return py::make_tuple(success, vec_grad_out);
            },
            py::arg("index"),
            py::arg("y_index"));

    py_log_noisy_input_gp.def(py::init<std::shared_ptr<Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def_property_readonly("setting", &T::template GetSetting<Setting>)
        .def(
            "load_surface_data_2d",
            &T::template LoadSurfaceData<2>,
            py::arg("surface_data_indices"),
            py::arg("surface_data_vec"),
            py::arg("coord_origin"),
            py::arg("load_normals"),
            py::arg("normal_scale"),
            py::arg("offset_distance"),
            py::arg("sensor_noise"),
            py::arg("max_valid_gradient_var"),
            py::arg("invalid_position_var"))
        .def(
            "load_surface_data_3d",
            &T::template LoadSurfaceData<3>,
            py::arg("surface_data_indices"),
            py::arg("surface_data_vec"),
            py::arg("coord_origin"),
            py::arg("load_normals"),
            py::arg("normal_scale"),
            py::arg("offset_distance"),
            py::arg("sensor_noise"),
            py::arg("max_valid_gradient_var"),
            py::arg("invalid_position_var"))
        .def("test", &T::Test, py::arg("mat_x_test"), py::arg("predict_gradient"));
}

void
BindLogEdfGaussianProcess(const py::module &m) {
    BindLogEdfGaussianProcessImpl<double>(m, "LogEdfGaussianProcessD");
    BindLogEdfGaussianProcessImpl<float>(m, "LogEdfGaussianProcessF");
}
