#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/log_sdf_gp.hpp"

void
BindLogSdfGaussianProcess(const py::module &m) {

    using ParentT = erl::gaussian_process::NoisyInputGaussianProcess;
    using T = erl::sdf_mapping::LogSdfGaussianProcess;

    auto py_log_noisy_input_gp = py::class_<T, ParentT, std::shared_ptr<T>>(m, "LogSdfGaussianProcess");

    py::class_<T::Setting, ParentT::Setting, std::shared_ptr<T::Setting>>(py_log_noisy_input_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("log_lambda", &T::Setting::log_lambda)
        .def_readwrite("edf_threshold", &T::Setting::edf_threshold)
        .def_readwrite("unify_scale", &T::Setting::unify_scale);

    py_log_noisy_input_gp.def(py::init<std::shared_ptr<T::Setting>>(), py::arg("setting").none(false))
        .def_property_readonly("setting", &T::GetSetting<T::Setting>)
        .def("reset", &T::Reset, py::arg("max_num_samples"), py::arg("x_dim"))
        .def_property_readonly("log_kernel", &T::GetLogKernel)
        .def_property_readonly("log_k_train", &T::GetLogKtrain)
        .def_property_readonly("log_alpha", &T::GetLogAlpha)
        .def_property_readonly("log_cholesky_k_train", &T::GetLogCholeskyDecomposition)
        .def_property_readonly("memory_usage", &T::GetMemoryUsage)
        .def(
            "train",
            [](T &self,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_x_train,
               const Eigen::Ref<const Eigen::VectorXd> &vec_y_train,
               const Eigen::Ref<const Eigen::MatrixXd> &mat_grad_train,
               const Eigen::Ref<const Eigen::VectorXl> &vec_grad_flag,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) {
                const long num_train_samples = mat_x_train.cols();
                const long x_dim = mat_x_train.rows();
                self.Reset(num_train_samples, x_dim);
                self.GetTrainInputSamplesBuffer().topLeftCorner(x_dim, num_train_samples) = mat_x_train;
                self.GetTrainInputSamplesVarianceBuffer().head(num_train_samples) = vec_var_x;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(num_train_samples) = vec_y_train;
                self.GetTrainOutputValueSamplesVarianceBuffer().head(num_train_samples) = vec_var_y;
                self.GetTrainOutputGradientSamplesBuffer().topLeftCorner(2, num_train_samples) = mat_grad_train;
                self.GetTrainGradientFlagsBuffer().head(num_train_samples) = vec_grad_flag;
                self.GetTrainOutputGradientSamplesVarianceBuffer().head(num_train_samples) = vec_var_grad;
                self.Train(num_train_samples);
            },
            py::arg("mat_x_train"),
            py::arg("vec_y_train"),
            py::arg("mat_grad_train"),
            py::arg("vec_grad_flag"),
            py::arg("vec_var_x"),
            py::arg("vec_var_y"),
            py::arg("vec_var_grad"))
        .def(
            "test",
            [](const T &self, const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test) {
                Eigen::MatrixXd mat_f_out, mat_var_out, mat_cov_out;
                const long dim = mat_x_test.rows();
                const long n = mat_x_test.cols();
                mat_f_out.resize(dim + 1, n);
                mat_var_out.resize(dim + 1, n);
                mat_cov_out.resize(dim * (dim + 1) / 2, n);
                self.Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out);
                return py::make_tuple(mat_f_out, mat_var_out, mat_cov_out);
            },
            py::arg("mat_x_test"));
}
