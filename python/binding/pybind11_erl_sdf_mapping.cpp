#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"
#include "erl_sdf_mapping/log_sdf_gp.hpp"

void
BindLogSdfGaussianProcess(const py::module &m) {

    using ParentT = erl::gaussian_process::NoisyInputGaussianProcess;
    using T = erl::sdf_mapping::LogSdfGaussianProcess;

    auto py_log_noisy_input_gp = py::class_<T, ParentT, std::shared_ptr<T>>(m, "LogSdfGaussianProcess");

    py::class_<T::Setting, ParentT::Setting, std::shared_ptr<T::Setting>>(py_log_noisy_input_gp, "Setting")
        .def(py::init<>())
        .def_readwrite("log_lambda", &T::Setting::log_lambda);

    py_log_noisy_input_gp.def(py::init<>([]() { return std::make_shared<T>(); }))
        .def(py::init<>([](std::shared_ptr<T::Setting> setting) { return std::make_shared<T>(std::move(setting)); }), py::arg("setting").none(false))
        .def_property_readonly("setting", &T::GetSetting)
        .def("reset", &T::Reset)
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

void
BindGpOccSurfaceMappingBaseSetting(const py::module &m);

static void
BindGpOccSurfaceMapping2D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    py::class_<GpOccSurfaceMapping2D, AbstractSurfaceMapping2D, std::shared_ptr<GpOccSurfaceMapping2D>> surface_mapping(m, "GpOccSurfaceMapping2D");
    py::class_<GpOccSurfaceMapping2D::Setting, GpOccSurfaceMappingBaseSetting, std::shared_ptr<GpOccSurfaceMapping2D::Setting>>(surface_mapping, "Setting")
        .def_readwrite("sensor_gp", &GpOccSurfaceMapping2D::Setting::sensor_gp)
        .def_readwrite("quadtree", &GpOccSurfaceMapping2D::Setting::quadtree);

    surface_mapping.def(py::init<const std::shared_ptr<GpOccSurfaceMapping2D::Setting> &>(), py::arg("setting"))
        .def_property_readonly("setting", &GpOccSurfaceMapping2D::GetSetting);

    // TODO: bind other methods of GpOccSurfaceMapping2D
}

static void
BindGpSdfMappingSetting(const py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMappingSetting, YamlableBase, std::shared_ptr<GpSdfMappingSetting>> sdf_mapping_setting(m, "GpSdfMappingSetting");
    py::class_<GpSdfMappingSetting::TestQuery, YamlableBase, std::shared_ptr<GpSdfMappingSetting::TestQuery>>(sdf_mapping_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &GpSdfMappingSetting::TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &GpSdfMappingSetting::TestQuery::search_area_half_size)
        .def_readwrite("use_nearest_only", &GpSdfMappingSetting::TestQuery::use_nearest_only)
        .def_readwrite("compute_covariance", &GpSdfMappingSetting::TestQuery::compute_covariance)
        .def_readwrite("recompute_variance", &GpSdfMappingSetting::TestQuery::recompute_variance)
        .def_readwrite("softmax_temperature", &GpSdfMappingSetting::TestQuery::softmax_temperature);
    sdf_mapping_setting.def(py::init<>([]() { return std::make_shared<GpSdfMappingSetting>(); }))
        .def_readwrite("num_threads", &GpSdfMappingSetting::num_threads)
        .def_readwrite("update_hz", &GpSdfMappingSetting::update_hz)
        .def_readwrite("gp_sdf_area_scale", &GpSdfMappingSetting::gp_sdf_area_scale)
        .def_readwrite("offset_distance", &GpSdfMappingSetting::offset_distance)
        .def_readwrite("max_valid_gradient_var", &GpSdfMappingSetting::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &GpSdfMappingSetting::invalid_position_var)
        .def_readwrite("train_gp_immediately", &GpSdfMappingSetting::train_gp_immediately)
        .def_readwrite("gp_sdf", &GpSdfMappingSetting::gp_sdf)
        .def_readwrite("test_query", &GpSdfMappingSetting::test_query);
}

static void
BindGpSdfMapping2D(const py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMapping2D, std::shared_ptr<GpSdfMapping2D>>(m, "GpSdfMapping2D")
        .def(py::init<std::shared_ptr<AbstractSurfaceMapping2D>, std::shared_ptr<GpSdfMappingSetting>>(), py::arg("surface_mapping"), py::arg("setting"))
        .def_property_readonly("setting", &GpSdfMapping2D::GetSetting)
        .def_property_readonly("surface_mapping", &GpSdfMapping2D::GetSurfaceMapping)
        .def("update", &GpSdfMapping2D::Update, py::arg("angles"), py::arg("distances"), py::arg("pose"))
        .def(
            "test",
            [](GpSdfMapping2D &self, const Eigen::Ref<const Eigen::Matrix2Xd> &xy) {
                Eigen::VectorXd distances;
                Eigen::Matrix2Xd gradients;

                if (Eigen::Matrix3Xd variances_out, covariances_out; self.Test(xy, distances, gradients, variances_out, covariances_out)) {
                    return py::make_tuple(distances, gradients, variances_out, covariances_out);
                }
                return py::make_tuple(py::none(), py::none(), py::none(), py::none());
            },
            py::arg("xy"));
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_sdf_mapping";
    BindLogSdfGaussianProcess(m);
    BindGpOccSurfaceMapping2D(m);
    BindGpSdfMappingSetting(m);
    BindGpSdfMapping2D(m);
    // TODO: bind other modules
}
