#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

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

    py_log_noisy_input_gp.def(py::init<std::shared_ptr<Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def_property_readonly("setting", &T::template GetSetting<Setting>)
        .def(
            "load_surface_data_2d",
            &T::template LoadSurfaceData<2>,
            py::arg("surface_data_indices"),
            py::arg("surface_data_vec"),
            py::arg("coord_origin"),
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
            py::arg("offset_distance"),
            py::arg("sensor_noise"),
            py::arg("max_valid_gradient_var"),
            py::arg("invalid_position_var"))
        .def(
            "test",
            [](const T &self, const Eigen::Ref<const Eigen::MatrixX<Dtype>> &mat_x_test, const bool predict_gradient) {
                Eigen::MatrixX<Dtype> mat_f_out, mat_var_out, mat_cov_out;
                const long dim = mat_x_test.rows();
                const long n = mat_x_test.cols();
                mat_f_out.resize(dim + 1, n);
                mat_var_out.resize(dim + 1, n);
                mat_cov_out.resize(dim * (dim + 1) / 2, n);
                // TODO - allow varying predict_gradient
                if (!self.Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out, predict_gradient)) { return py::make_tuple(py::none(), py::none(), py::none()); }
                return py::make_tuple(mat_f_out, mat_var_out, mat_cov_out);
            },
            py::arg("mat_x_test"),
            py::arg("predict_gradient") = true);
}

void
BindLogEdfGaussianProcess(const py::module &m) {
    BindLogEdfGaussianProcessImpl<double>(m, "LogEdfGaussianProcessD");
    BindLogEdfGaussianProcessImpl<float>(m, "LogEdfGaussianProcessF");
}
