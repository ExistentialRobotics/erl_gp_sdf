#include "erl_common/pybind11.hpp"
#include "erl_sdf_mapping/gpis/gpis_map_2d.hpp"
#include "erl_sdf_mapping/gpis/log_gpis_map_2d.hpp"
#include "erl_sdf_mapping/gpis/node.hpp"
#include "erl_sdf_mapping/gpis/node_container.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"
#include "erl_sdf_mapping/log_sdf_gp.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree_base.hpp"

static void
BindGpSdf2D(py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping::gpis;

    // GpisData2D
    py::class_<GpisData2D, NodeData, std::shared_ptr<GpisData2D>>(m, ERL_AS_STRING(GpisData2D))
        .def_readwrite("distance", &GpisData2D::distance)
        .def_readwrite("gradient", &GpisData2D::gradient)
        .def_readwrite("var_position", &GpisData2D::var_position)
        .def_readwrite("var_gradient", &GpisData2D::var_gradient)
        .def("update_data", &GpisData2D::UpdateData, py::arg("new_distance"), py::arg("new_gradient"), py::arg("new_var_position"), py::arg("new_var_gradient"))
        .def("__str__", [](const GpisData2D &gpis_data) -> std::string {
            std::stringstream ss;
            gpis_data.Print(ss);
            return ss.str();
        });

    // GpisNode2D
    py::class_<GpisNode2D, Node, std::shared_ptr<GpisNode2D>>(m, ERL_AS_STRING(GpisNode2D)).def(py::init<Eigen::Vector<double, 2>>(), py::arg("position"));

    // GpisNodeContainer2D
    auto py_gpis_node_container = py::class_<GpisNodeContainer2D, NodeContainer, std::shared_ptr<GpisNodeContainer2D>>(m, ERL_AS_STRING(GpisNodeContainer2D));

    py::class_<GpisNodeContainer2D::Setting, YamlableBase, std::shared_ptr<GpisNodeContainer2D::Setting>>(py_gpis_node_container, "Setting")
        .def_readwrite("capacity", &GpisNodeContainer2D::Setting::capacity)
        .def_readwrite("min_squared_distance", &GpisNodeContainer2D::Setting::min_squared_distance);

    py_gpis_node_container.def(py::init(&GpisNodeContainer2D::Create), py::arg("setting").none(false))
        .def_property_readonly("setting", &GpisNodeContainer2D::GetSetting)
        .def("__len__", py::overload_cast<>(&GpisNodeContainer2D::Size, py::const_));

    // GpisMapBase2D
    auto py_gpis_map_base = py::class_<GpisMapBase2D>(m, ERL_AS_STRING(GpisMapBase2D));
    // GpisMapBase2D::Setting
    auto py_gpis_map_setting = py::class_<GpisMapBase2D::Setting, YamlableBase, std::shared_ptr<GpisMapBase2D::Setting>>(py_gpis_map_base, "Setting");
    // GpisMapBase2D::Setting::ComputeVariance
    py::class_<GpisMapBase2D::Setting::ComputeVariance, YamlableBase, std::shared_ptr<GpisMapBase2D::Setting::ComputeVariance>>(
        py_gpis_map_setting,
        "ComputeVariance")
        .def_readwrite("zero_gradient_position_var", &GpisMapBase2D::Setting::ComputeVariance::zero_gradient_position_var)
        .def_readwrite("zero_gradient_gradient_var", &GpisMapBase2D::Setting::ComputeVariance::zero_gradient_gradient_var)
        .def_readwrite("min_distance_var", &GpisMapBase2D::Setting::ComputeVariance::min_distance_var)
        .def_readwrite("max_distance_var", &GpisMapBase2D::Setting::ComputeVariance::max_distance_var)
        .def_readwrite("position_var_alpha", &GpisMapBase2D::Setting::ComputeVariance::position_var_alpha)
        .def_readwrite("min_gradient_var", &GpisMapBase2D::Setting::ComputeVariance::min_gradient_var)
        .def_readwrite("max_gradient_var", &GpisMapBase2D::Setting::ComputeVariance::max_gradient_var);
    // GpisMapBase2D::Setting::UpdateSurfacePoints
    py::class_<GpisMapBase2D::Setting::UpdateMapPoints, YamlableBase, std::shared_ptr<GpisMapBase2D::Setting::UpdateMapPoints>>(
        py_gpis_map_setting,
        "UpdateSurfacePoints")
        .def_readwrite("min_observable_occ", &GpisMapBase2D::Setting::UpdateMapPoints::min_observable_occ)
        .def_readwrite("max_surface_abs_occ", &GpisMapBase2D::Setting::UpdateMapPoints::max_surface_abs_occ)
        .def_readwrite("max_valid_gradient_var", &GpisMapBase2D::Setting::UpdateMapPoints::max_valid_gradient_var)
        .def_readwrite("max_adjust_tries", &GpisMapBase2D::Setting::UpdateMapPoints::max_adjust_tries)
        .def_readwrite("max_bayes_position_var", &GpisMapBase2D::Setting::UpdateMapPoints::max_bayes_position_var)
        .def_readwrite("max_bayes_gradient_var", &GpisMapBase2D::Setting::UpdateMapPoints::max_bayes_gradient_var);
    // GpisMapBase2D::Setting::UpdateGpSdf
    py::class_<GpisMapBase2D::Setting::UpdateGpSdf, YamlableBase, std::shared_ptr<GpisMapBase2D::Setting::UpdateGpSdf>>(py_gpis_map_setting, "UpdateGpSdf")
        .def_readwrite("add_offset_points", &GpisMapBase2D::Setting::UpdateGpSdf::add_offset_points)
        .def_readwrite("offset_distance", &GpisMapBase2D::Setting::UpdateGpSdf::offset_distance)
        .def_readwrite("search_area_scale", &GpisMapBase2D::Setting::UpdateGpSdf::search_area_scale)
        .def_readwrite("zero_gradient_threshold", &GpisMapBase2D::Setting::UpdateGpSdf::zero_gradient_threshold)
        .def_readwrite("max_valid_gradient_var", &GpisMapBase2D::Setting::UpdateGpSdf::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &GpisMapBase2D::Setting::UpdateGpSdf::invalid_position_var);
    // GpisMapBase2D::Setting::TestQuery
    py::class_<GpisMapBase2D::Setting::TestQuery, YamlableBase, std::shared_ptr<GpisMapBase2D::Setting::TestQuery>>(py_gpis_map_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &GpisMapBase2D::Setting::TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &GpisMapBase2D::Setting::TestQuery::search_area_half_size)
        .def_readwrite("use_nearest_only", &GpisMapBase2D::Setting::TestQuery::use_nearest_only);

    py_gpis_map_setting.def(py::init<>([]() { return std::make_shared<GpisMapBase2D::Setting>(); }))
        .def_readwrite("init_tree_half_size", &GpisMapBase2D::Setting::init_tree_half_size)
        .def_readwrite("perturb_delta", &GpisMapBase2D::Setting::perturb_delta)
        .def_readwrite("compute_variance", &GpisMapBase2D::Setting::compute_variance)
        .def_readwrite("update_map_points", &GpisMapBase2D::Setting::update_map_points)
        .def_readwrite("update_gp_sdf", &GpisMapBase2D::Setting::update_gp_sdf)
        .def_readwrite("gp_theta", &GpisMapBase2D::Setting::gp_theta)
        .def_readwrite("gp_sdf", &GpisMapBase2D::Setting::gp_sdf)
        .def_readwrite("node_container", &GpisMapBase2D::Setting::node_container)
        .def_readwrite("quadtree", &GpisMapBase2D::Setting::quadtree)
        .def_readwrite("test_query", &GpisMapBase2D::Setting::test_query);

    py_gpis_map_base.def_property_readonly("quadtree", &GpisMapBase2D::GetQuadtree)
        .def("update", &GpisMapBase2D::Update, py::arg("angles"), py::arg("distances"), py::arg("pose"))
        .def(
            "test",
            [](GpisMapBase2D &gpis_map, const Eigen::Ref<const Eigen::Matrix2Xd> &xy) {
                Eigen::VectorXd distances;
                Eigen::Matrix2Xd gradients;
                Eigen::VectorXd distance_variances;
                Eigen::Matrix2Xd gradient_variances;

                if (gpis_map.Test(xy, distances, gradients, distance_variances, gradient_variances)) {
                    return py::make_tuple(distances, gradients, distance_variances, gradient_variances);
                } else {
                    return py::make_tuple(py::none(), py::none(), py::none(), py::none());
                }
            },
            py::arg("xy"))
        .def(
            "compute_sddf_v2",
            &GpisMapBase2D::ComputeSddfV2,
            py::arg("positions"),
            py::arg("angles"),
            py::arg("threshold"),
            py::arg("max_distance"),
            py::arg("max_marching_steps"))
        .def("dump_quadtree_structure", &GpisMapBase2D::DumpQuadtreeStructure)
        .def("dump_surface_points", &GpisMapBase2D::DumpSurfacePoints)
        .def("dump_surface_normals", &GpisMapBase2D::DumpSurfaceNormals)
        .def("dump_surface_data", [](const GpisMapBase2D &gpis_map) {
            Eigen::Matrix2Xd surface_points;
            Eigen::Matrix2Xd surface_normals;
            Eigen::VectorXd points_variance;
            Eigen::VectorXd normals_variance;
            gpis_map.DumpSurfaceData(surface_points, surface_normals, points_variance, normals_variance);

            return py::make_tuple(surface_points, surface_normals, points_variance, normals_variance);
        });

    // GpisMap2D
    py::class_<GpisMap2D, GpisMapBase2D>(m, ERL_AS_STRING(GpisMap2D))
        .def(py::init<>())
        .def(py::init<const std::shared_ptr<GpisMap2D::Setting> &>())
        .def_property_readonly("setting", &GpisMap2D::GetSetting);

    // LogGpisMap2D
    auto py_log_gpis_map = py::class_<LogGpisMap2D, GpisMapBase2D>(m, ERL_AS_STRING(LogGpisMap2D));

    py::class_<LogGpisMap2D::Setting, GpisMapBase2D::Setting, std::shared_ptr<LogGpisMap2D::Setting>>(py_log_gpis_map, "Setting")
        .def(py::init<>([]() { return std::make_shared<LogGpisMap2D::Setting>(); }))
        .def_readwrite("gp_sdf", &LogGpisMap2D::Setting::gp_sdf);

    py_log_gpis_map.def(py::init<>())
        .def(py::init<const std::shared_ptr<LogGpisMap2D::Setting> &>(), py::arg("setting").none(false))
        .def_property_readonly("setting", &LogGpisMap2D::GetSetting);
}

static void
BindGpSdf3D(py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping::gpis;

    // GpisData3D
    py::class_<GpisData3D, NodeData, std::shared_ptr<GpisData3D>>(m, ERL_AS_STRING(GpisData3D))
        .def_readwrite("distance", &GpisData3D::distance)
        .def_readwrite("gradient", &GpisData3D::gradient)
        .def_readwrite("var_position", &GpisData3D::var_position)
        .def_readwrite("var_gradient", &GpisData3D::var_gradient)
        .def("update_data", &GpisData3D::UpdateData, py::arg("new_distance"), py::arg("new_gradient"), py::arg("new_var_position"), py::arg("new_var_gradient"))
        .def("__str__", [](const GpisData3D &gpis_data) -> std::string {
            std::stringstream ss;
            gpis_data.Print(ss);
            return ss.str();
        });

    py::class_<GpisNode3D, Node, std::shared_ptr<GpisNode3D>>(m, ERL_AS_STRING(GpisNode3D)).def(py::init<Eigen::Vector<double, 3>>(), py::arg("position"));

    // GpisNodeContainer3D
    auto py_gpis_node_container = py::class_<GpisNodeContainer3D, NodeContainer, std::shared_ptr<GpisNodeContainer3D>>(m, ERL_AS_STRING(GpisNodeContainer3D));

    py::class_<GpisNodeContainer3D::Setting, YamlableBase, std::shared_ptr<GpisNodeContainer3D::Setting>>(py_gpis_node_container, "Setting")
        .def_readwrite("capacity", &GpisNodeContainer3D::Setting::capacity)
        .def_readwrite("min_squared_distance", &GpisNodeContainer3D::Setting::min_squared_distance);

    py_gpis_node_container.def(py::init(&GpisNodeContainer3D::Create), py::arg("setting").none(false))
        .def_property_readonly("setting", &GpisNodeContainer3D::GetSetting)
        .def("__len__", py::overload_cast<>(&GpisNodeContainer3D::Size, py::const_));
}

void
BindLogSdfGaussianProcess(py::module &m) {

    using ParentT = erl::gaussian_process::NoisyInputGaussianProcess;
    using T = erl::sdf_mapping::LogSdfGaussianProcess;

    auto py_log_noisy_input_gp = py::class_<T, ParentT, std::shared_ptr<T>>(m, ERL_AS_STRING(LogSdfGaussianProcess));

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
               const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flag,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
               const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) {
                long num_train_samples = mat_x_train.cols();
                long x_dim = mat_x_train.rows();
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
                long dim = mat_x_test.rows();
                long n = mat_x_test.cols();
                mat_f_out.resize(dim + 1, n);
                mat_var_out.resize(dim + 1, n);
                mat_cov_out.resize(dim * (dim + 1) / 2, n);
                self.Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out);
                return py::make_tuple(mat_f_out, mat_var_out, mat_cov_out);
            },
            py::arg("mat_x_test"));
}

static void
BindGpOccSurfaceMapping2D(py::module &m) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    // TODO: bind SurfaceMappingQuadtreeNode
    py::class_<SurfaceMappingQuadtreeNode, OccupancyQuadtreeNode, std::shared_ptr<SurfaceMappingQuadtreeNode>>(m, "SurfaceMappingQuadtreeNode")
        .def("allow_update_log_odds", &SurfaceMappingQuadtreeNode::AllowUpdateLogOdds, py::arg("delta"));

    BindOccupancyQuadtree<SurfaceMappingQuadtree, SurfaceMappingQuadtreeNode>(m, "SurfaceMappingQuadtree");

    py::class_<AbstractSurfaceMapping2D, std::shared_ptr<AbstractSurfaceMapping2D>>(m, "AbstractSurfaceMapping2D")
        .def_property_readonly("quadtree", &AbstractSurfaceMapping2D::GetQuadtree)
        .def_property_readonly("sensor_noise", &AbstractSurfaceMapping2D::GetSensorNoise)
        .def("update", &AbstractSurfaceMapping2D::Update, py::arg("angles"), py::arg("distances"), py::arg("pose"));

    py::class_<GpOccSurfaceMapping2D, AbstractSurfaceMapping2D, std::shared_ptr<GpOccSurfaceMapping2D>> surface_mapping(m, "GpOccSurfaceMapping2D");
    py::class_<GpOccSurfaceMapping2D::Setting, std::shared_ptr<GpOccSurfaceMapping2D::Setting>> surface_mapping_setting(surface_mapping, "Setting");
    py::class_<GpOccSurfaceMapping2D::Setting::ComputeVariance, YamlableBase, std::shared_ptr<GpOccSurfaceMapping2D::Setting::ComputeVariance>>(
        surface_mapping_setting,
        "ComputeVariance")
        .def_readwrite("zero_gradient_position_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::zero_gradient_position_var)
        .def_readwrite("zero_gradient_gradient_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::zero_gradient_gradient_var)
        .def_readwrite("min_distance_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::min_distance_var)
        .def_readwrite("max_distance_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::max_distance_var)
        .def_readwrite("position_var_alpha", &GpOccSurfaceMapping2D::Setting::ComputeVariance::position_var_alpha)
        .def_readwrite("min_gradient_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::min_gradient_var)
        .def_readwrite("max_gradient_var", &GpOccSurfaceMapping2D::Setting::ComputeVariance::max_gradient_var);
    py::class_<GpOccSurfaceMapping2D::Setting::UpdateMapPoints, YamlableBase, std::shared_ptr<GpOccSurfaceMapping2D::Setting::UpdateMapPoints>>(
        surface_mapping_setting,
        "UpdateMapPoints")
        .def_readwrite("min_observable_occ", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::min_observable_occ)
        .def_readwrite("max_surface_abs_occ", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::max_surface_abs_occ)
        .def_readwrite("max_valid_gradient_var", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::max_valid_gradient_var)
        .def_readwrite("max_adjust_tries", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::max_adjust_tries)
        .def_readwrite("max_bayes_position_var", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::max_bayes_position_var)
        .def_readwrite("max_bayes_gradient_var", &GpOccSurfaceMapping2D::Setting::UpdateMapPoints::max_bayes_gradient_var);
    surface_mapping_setting.def_readwrite("gp_theta", &GpOccSurfaceMapping2D::Setting::gp_theta)
        .def_readwrite("compute_variance", &GpOccSurfaceMapping2D::Setting::compute_variance)
        .def_readwrite("update_map_points", &GpOccSurfaceMapping2D::Setting::update_map_points)
        .def_readwrite("quadtree", &GpOccSurfaceMapping2D::Setting::quadtree)
        .def_readwrite("cluster_level", &GpOccSurfaceMapping2D::Setting::cluster_level)
        .def_readwrite("perturb_delta", &GpOccSurfaceMapping2D::Setting::perturb_delta)
        .def_readwrite("update_occupancy", &GpOccSurfaceMapping2D::Setting::update_occupancy);

    surface_mapping.def(py::init<>())
        .def(py::init<const std::shared_ptr<GpOccSurfaceMapping2D::Setting> &>(), py::arg("setting"))
        .def_property_readonly("setting", &GpOccSurfaceMapping2D::GetSetting);

    // TODO: bind other methods of GpOccSurfaceMapping2D
}

static void
BindGpSdfMapping2D(py::module &m) {
    using namespace erl::common;
    using namespace erl::sdf_mapping;

    py::class_<GpSdfMapping2D, std::shared_ptr<GpSdfMapping2D>> sdf_mapping(m, ERL_AS_STRING(GpSdfMapping2D));
    py::class_<GpSdfMapping2D::Setting, YamlableBase, std::shared_ptr<GpSdfMapping2D::Setting>> sdf_mapping_setting(sdf_mapping, "Setting");
    py::class_<GpSdfMapping2D::Setting::TestQuery, YamlableBase, std::shared_ptr<GpSdfMapping2D::Setting::TestQuery>>(sdf_mapping_setting, "TestQuery")
        .def_readwrite("max_test_valid_distance_var", &GpSdfMapping2D::Setting::TestQuery::max_test_valid_distance_var)
        .def_readwrite("search_area_half_size", &GpSdfMapping2D::Setting::TestQuery::search_area_half_size)
        .def_readwrite("use_nearest_only", &GpSdfMapping2D::Setting::TestQuery::use_nearest_only)
        .def_readwrite("compute_covariance", &GpSdfMapping2D::Setting::TestQuery::compute_covariance)
        .def_readwrite("recompute_variance", &GpSdfMapping2D::Setting::TestQuery::recompute_variance)
        .def_readwrite("softmax_temperature", &GpSdfMapping2D::Setting::TestQuery::softmax_temperature);
    sdf_mapping_setting.def(py::init<>([]() { return std::make_shared<GpSdfMapping2D::Setting>(); }))
        .def_readwrite("num_threads", &GpSdfMapping2D::Setting::num_threads)
        .def_readwrite("update_hz", &GpSdfMapping2D::Setting::update_hz)
        .def_readwrite("gp_sdf_area_scale", &GpSdfMapping2D::Setting::gp_sdf_area_scale)
        .def_readwrite("offset_distance", &GpSdfMapping2D::Setting::offset_distance)
        .def_readwrite("zero_gradient_threshold", &GpSdfMapping2D::Setting::zero_gradient_threshold)
        .def_readwrite("max_valid_gradient_var", &GpSdfMapping2D::Setting::max_valid_gradient_var)
        .def_readwrite("invalid_position_var", &GpSdfMapping2D::Setting::invalid_position_var)
        .def_readwrite("train_gp_immediately", &GpSdfMapping2D::Setting::train_gp_immediately)
        .def_readwrite("gp_sdf", &GpSdfMapping2D::Setting::gp_sdf)
        .def_readwrite("test_query", &GpSdfMapping2D::Setting::test_query);

    sdf_mapping
        .def(py::init<std::shared_ptr<AbstractSurfaceMapping2D>, std::shared_ptr<GpSdfMapping2D::Setting>>(), py::arg("surface_mapping"), py::arg("setting"))
        .def_property_readonly("setting", &GpSdfMapping2D::GetSetting)
        .def_property_readonly("surface_mapping", &GpSdfMapping2D::GetSurfaceMapping)
        .def("update", &GpSdfMapping2D::Update, py::arg("angles"), py::arg("distances"), py::arg("pose"))
        .def(
            "test",
            [](GpSdfMapping2D &self, const Eigen::Ref<const Eigen::Matrix2Xd> &xy) {
                Eigen::VectorXd distances;
                Eigen::Matrix2Xd gradients;
                Eigen::Matrix3Xd variances_out;
                Eigen::Matrix3Xd covariances_out;

                if (self.Test(xy, distances, gradients, variances_out, covariances_out)) {
                    return py::make_tuple(distances, gradients, variances_out, covariances_out);
                } else {
                    return py::make_tuple(py::none(), py::none(), py::none(), py::none());
                }
            },
            py::arg("xy"));
}

PYBIND11_MODULE(PYBIND_MODULE_NAME, m) {
    m.doc() = "Python 3 Interface of erl_sdf_mapping";

    auto gpis = m.def_submodule("gpis", "Interface of erl_sdf_mapping");
    BindGpSdf2D(gpis);
    BindGpSdf3D(gpis);

    BindLogSdfGaussianProcess(m);
    BindGpOccSurfaceMapping2D(m);
    BindGpSdfMapping2D(m);
    // TODO: bind other modules
}
