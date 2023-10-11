#pragma once

#include <fstream>
#include <memory>

#include "erl_common/eigen.hpp"
#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/lidar_gp_1d.hpp"
#include "erl_gaussian_process/noisy_input_gp.hpp"
#include "erl_geometry/incremental_quadtree.hpp"
#include "erl_geometry/ray_marching.hpp"
#include "gpis_test_buffer.hpp"
#include "node.hpp"
#include "node_container.hpp"
#include "default_params.hpp"

namespace erl::sdf_mapping::gpis {

    class GpisMapBase2D {

    public:
        struct Setting : public common::Yamlable<Setting> {

            struct ComputeVariance : public common::Yamlable<ComputeVariance> {
                double zero_gradient_position_var = 1.;  // position variance to use when the estimated gradient is almost zero.
                double zero_gradient_gradient_var = 1.;  // gradient variance to use when the estimated gradient is almost zero.
                double min_distance_var = 1.;            // minimum distance variance.
                double max_distance_var = 100.;          // maximum distance variance.
                double position_var_alpha = 0.01;        // scaling number of position variance.
                double min_gradient_var = 0.01;          // minimum gradient variance.
                double max_gradient_var = 1.;            // maximum gradient variance.
            };

            struct UpdateMapPoints : public common::Yamlable<UpdateMapPoints> {
                double min_observable_occ = -0.1;     // points of OCC smaller than this value is considered unobservable, i.e. inside the object.
                double max_surface_abs_occ = 0.02;    // maximum absolute value of surface points' OCC, which should be zero ideally.
                double max_valid_gradient_var = 0.5;  // maximum valid gradient variance, above this threshold, it won't be used for the Bayes Update.
                int max_adjust_tries = 10;
                double max_bayes_position_var = 1.;   // if the position variance by Bayes Update is above this threshold, it will be discarded.
                double max_bayes_gradient_var = 0.6;  // if the gradient variance by Bayes Update is above this threshold, it will be discarded.
            };

            struct UpdateGpSdf : public common::Yamlable<UpdateGpSdf> {
                bool add_offset_points = false;
                double offset_distance = 0.02;
                double search_area_scale = 4.;           // scale to enlarge the area to search for data points for training.
                double zero_gradient_threshold = 1.e-6;  // gradient below this threshold is considered zero.
                double max_valid_gradient_var = 0.1;     // maximum gradient variance qualified for training.
                double invalid_position_var = 2.;        // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
            };

            struct TestQuery : public common::Yamlable<TestQuery> {
                double max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
                double search_area_half_size = 4.8;
                bool use_nearest_only = false;  // if true, only the nearest point will be used for prediction.
            };

            unsigned int num_threads = -1;      // max number of threads used for parallel computation.
            double init_tree_half_size = 12.8;  // Initial area GetSize of the quad tree root
            double perturb_delta = 0.01;        // Small value used for numerical differential to compute gradient
            std::shared_ptr<ComputeVariance> compute_variance = std::make_shared<ComputeVariance>();   // parameters used by GpisMap2D::compute_variance.
            std::shared_ptr<UpdateMapPoints> update_map_points = std::make_shared<UpdateMapPoints>();  // parameters used by GpisMap2D::update_map_points.
            std::shared_ptr<UpdateGpSdf> update_gp_sdf = std::make_shared<UpdateGpSdf>();              // parameters used by GpisMap2D::update_gp_sdf.
            std::shared_ptr<gaussian_process::LidarGaussianProcess1D::Setting> gp_theta = std::make_shared<gaussian_process::LidarGaussianProcess1D::Setting>();
            std::shared_ptr<gaussian_process::NoisyInputGaussianProcess::Setting> gp_sdf =
                std::make_shared<gaussian_process::NoisyInputGaussianProcess::Setting>();
            std::shared_ptr<GpisNodeContainer2D::Setting> node_container = std::make_shared<GpisNodeContainer2D::Setting>();
            std::shared_ptr<geometry::IncrementalQuadtree::Setting> quadtree = std::make_shared<geometry::IncrementalQuadtree::Setting>();
            std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by GpisMap2D::Test.
        };

#if defined(ERL_BUILD_TEST)
    public:
#else
    protected:
#endif
        std::shared_ptr<Setting> m_setting_;
        TestBuffer m_test_buffer_;                                              // buffer for test data processing.
        std::shared_ptr<gaussian_process::LidarGaussianProcess1D> m_gp_theta_;  // the GP of regression between angle and mapped distance

        // structure for storing, updating the map
        std::shared_ptr<geometry::IncrementalQuadtree> m_quadtree_ = nullptr;
        std::unordered_set<std::shared_ptr<geometry::IncrementalQuadtree>> m_active_clusters_;
        std::vector<std::shared_ptr<geometry::IncrementalQuadtree>> m_clusters_to_update_;
        std::function<std::shared_ptr<geometry::NodeContainer>()> m_node_container_constructor_;
        // other parameters
        const Eigen::Matrix24d mk_XyPerturb_;

    public:
        GpisMapBase2D();

        explicit GpisMapBase2D(const std::shared_ptr<Setting> &setting);

        virtual ~GpisMapBase2D() = default;

        std::shared_ptr<const geometry::IncrementalQuadtree>
        GetQuadtree() const {
            return m_quadtree_;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::VectorXd> &angles,
            const Eigen::Ref<const Eigen::VectorXd> &distances,
            const Eigen::Ref<const Eigen::Matrix23d> &pose);

        virtual bool
        Test(
            const TestBuffer::InBuffer &xy,
            TestBuffer::OutVectorBuffer::PlainMatrix &distances,
            TestBuffer::OutMatrixBuffer::PlainMatrix &gradients,
            TestBuffer::OutVectorBuffer::PlainMatrix &distance_variances,
            TestBuffer::OutMatrixBuffer::PlainMatrix &gradient_variances);

        Eigen::VectorXd
        ComputeSddfV2(
            const Eigen::Ref<const Eigen::Matrix2Xd> &positions,
            const Eigen::Ref<const Eigen::VectorXd> &angles,
            double threshold,
            double max_distance,
            int max_marching_steps);

        [[nodiscard]] std::string
        DumpQuadtreeStructure() const {
            std::stringstream ss;
            m_quadtree_->Print(ss);
            return ss.str();
        }

        Eigen::Matrix2Xd
        DumpSurfacePoints() const;

        Eigen::Matrix2Xd
        DumpSurfaceNormals() const;

        void
        DumpSurfaceData(
            Eigen::Matrix2Xd &surface_points,
            Eigen::Matrix2Xd &surface_normals,
            Eigen::VectorXd &points_variance,
            Eigen::VectorXd &normals_variance) const;

    protected:
        inline bool
        LaunchUpdate() {
            UpdateMapPoints();
            AddNewMeasurement();
            UpdateGpX();
            return true;
        }

        bool
        ComputeGradient1(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Ref<Eigen::Vector2d> gradient, double &occ_mean, double &distance_var);

        bool
        ComputeVariance(
            const Eigen::Ref<const Eigen::Vector2d> &xy_local,
            const double &distance,
            const double &distance_var,
            const double &occ_mean_abs,
            const double &occ_abs,
            bool new_point,
            Eigen::Ref<Eigen::Vector2d> grad_local,
            Eigen::Ref<Eigen::Vector2d> grad_global,
            double &var_position,
            double &var_gradient) const;

        void
        UpdateMapPoints();

        bool
        ComputeGradient2(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Ref<Eigen::Vector2d> gradient, double &occ_mean);

        void
        AddNewMeasurement();

        void
        UpdateGpX();

        virtual void
        UpdateGpXsThread(int thread_idx, int start_idx, int end_idx);

        virtual std::shared_ptr<void>
        TrainGpX(
            const Eigen::Ref<Eigen::MatrixXd> &mat_x,
            const Eigen::Ref<Eigen::VectorXb> &vec_grad_flag,
            const Eigen::Ref<const Eigen::VectorXd> &vec_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) = 0;

        void
        LaunchTest(size_t n);

        virtual void
        InferWithGpX(
            const std::shared_ptr<const void> &gp_ptr,
            const Eigen::Ref<const Eigen::Vector2d> &vec_xt,
            Eigen::Ref<Eigen::Vector3d> f,
            Eigen::Ref<Eigen::Vector3d> var) const = 0;

        virtual void
        TestThread(int j_1, size_t j_2, size_t end_idx);
    };
}  // namespace erl::sdf_mapping::gpis

namespace YAML {

    using namespace erl::sdf_mapping::gpis;

    template<>
    struct convert<GpisMapBase2D::Setting::ComputeVariance> {

        inline static Node
        encode(const GpisMapBase2D::Setting::ComputeVariance &rhs) {
            Node node;
            node["zero_gradient_position_var"] = rhs.zero_gradient_position_var;
            node["zero_gradient_gradient_var"] = rhs.zero_gradient_gradient_var;
            node["min_distance_var"] = rhs.min_distance_var;
            node["max_distance_var"] = rhs.max_distance_var;
            node["position_var_alpha"] = rhs.position_var_alpha;
            node["min_gradient_var"] = rhs.min_gradient_var;
            node["max_gradient_var"] = rhs.max_gradient_var;
            return node;
        }

        inline static bool
        decode(const Node &node, GpisMapBase2D::Setting::ComputeVariance &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.zero_gradient_position_var = node["zero_gradient_position_var"].as<double>();
            rhs.zero_gradient_gradient_var = node["zero_gradient_gradient_var"].as<double>();
            rhs.min_distance_var = node["min_distance_var"].as<double>();
            rhs.max_distance_var = node["max_distance_var"].as<double>();
            rhs.position_var_alpha = node["position_var_alpha"].as<double>();
            rhs.min_gradient_var = node["min_gradient_var"].as<double>();
            rhs.max_gradient_var = node["max_gradient_var"].as<double>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const GpisMapBase2D::Setting::ComputeVariance &rhs) {
        out << BeginMap;
        out << Key << "zero_gradient_position_var" << Value << rhs.zero_gradient_position_var;
        out << Key << "zero_gradient_gradient_var" << Value << rhs.zero_gradient_gradient_var;
        out << Key << "min_distance_var" << Value << rhs.min_distance_var;
        out << Key << "max_distance_var" << Value << rhs.max_distance_var;
        out << Key << "position_var_alpha" << Value << rhs.position_var_alpha;
        out << Key << "min_gradient_var" << Value << rhs.min_gradient_var;
        out << Key << "max_gradient_var" << Value << rhs.max_gradient_var;
        out << EndMap;
        return out;
    }

    template<>
    struct convert<GpisMapBase2D::Setting::UpdateMapPoints> {

        inline static Node
        encode(const GpisMapBase2D::Setting::UpdateMapPoints &rhs) {
            Node node;
            node["min_observable_occ"] = rhs.min_observable_occ;
            node["max_surface_abs_occ"] = rhs.max_surface_abs_occ;
            node["max_valid_gradient_var"] = rhs.max_valid_gradient_var;
            node["max_adjust_tries"] = rhs.max_adjust_tries;
            node["max_bayes_position_var"] = rhs.max_bayes_position_var;
            node["max_bayes_gradient_var"] = rhs.max_bayes_gradient_var;
            return node;
        }

        inline static bool
        decode(const Node &node, GpisMapBase2D::Setting::UpdateMapPoints &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.min_observable_occ = node["min_observable_occ"].as<double>();
            rhs.max_surface_abs_occ = node["max_surface_abs_occ"].as<double>();
            rhs.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
            rhs.max_adjust_tries = node["max_adjust_tries"].as<int>();
            rhs.max_bayes_position_var = node["max_bayes_position_var"].as<double>();
            rhs.max_bayes_gradient_var = node["max_bayes_gradient_var"].as<double>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const GpisMapBase2D::Setting::UpdateMapPoints &rhs) {
        out << BeginMap;
        out << Key << "min_observable_occ" << Value << rhs.min_observable_occ;
        out << Key << "max_surface_abs_occ" << Value << rhs.max_surface_abs_occ;
        out << Key << "max_valid_gradient_var" << Value << rhs.max_valid_gradient_var;
        out << Key << "max_adjust_tries" << Value << rhs.max_adjust_tries;
        out << Key << "max_bayes_position_var" << Value << rhs.max_bayes_position_var;
        out << Key << "max_bayes_gradient_var" << Value << rhs.max_bayes_gradient_var;
        out << EndMap;
        return out;
    }

    template<>
    struct convert<GpisMapBase2D::Setting::UpdateGpSdf> {

        inline static Node
        encode(const GpisMapBase2D::Setting::UpdateGpSdf &rhs) {
            Node node;
            node["add_offset_points"] = rhs.add_offset_points;
            node["offset_distance"] = rhs.offset_distance;
            node["search_area_scale"] = rhs.search_area_scale;
            node["zero_gradient_threshold"] = rhs.zero_gradient_threshold;
            node["max_valid_gradient_var"] = rhs.max_valid_gradient_var;
            node["invalid_position_var"] = rhs.invalid_position_var;
            return node;
        }

        inline static bool
        decode(const Node &node, GpisMapBase2D::Setting::UpdateGpSdf &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.add_offset_points = node["add_offset_points"].as<bool>();
            rhs.offset_distance = node["offset_distance"].as<double>();
            rhs.search_area_scale = node["search_area_scale"].as<double>();
            rhs.zero_gradient_threshold = node["zero_gradient_threshold"].as<double>();
            rhs.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
            rhs.invalid_position_var = node["invalid_position_var"].as<double>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const GpisMapBase2D::Setting::UpdateGpSdf &rhs) {
        out << BeginMap;
        out << Key << "add_offset_points" << Value << rhs.add_offset_points;
        out << Key << "offset_distance" << Value << rhs.offset_distance;
        out << Key << "search_area_scale" << Value << rhs.search_area_scale;
        out << Key << "zero_gradient_threshold" << Value << rhs.zero_gradient_threshold;
        out << Key << "max_valid_gradient_var" << Value << rhs.max_valid_gradient_var;
        out << Key << "invalid_position_var" << Value << rhs.invalid_position_var;
        out << EndMap;
        return out;
    }

    template<>
    struct convert<GpisMapBase2D::Setting::TestQuery> {

        inline static Node
        encode(const GpisMapBase2D::Setting::TestQuery &rhs) {
            Node node;
            node["max_test_valid_distance_var"] = rhs.max_test_valid_distance_var;
            node["search_area_half_size"] = rhs.search_area_half_size;
            node["use_nearest_only"] = rhs.use_nearest_only;
            return node;
        }

        inline static bool
        decode(const Node &node, GpisMapBase2D::Setting::TestQuery &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<double>();
            rhs.search_area_half_size = node["search_area_half_size"].as<double>();
            rhs.use_nearest_only = node["use_nearest_only"].as<bool>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const GpisMapBase2D::Setting::TestQuery &rhs) {
        out << BeginMap;
        out << Key << "max_test_valid_distance_var" << Value << rhs.max_test_valid_distance_var;
        out << Key << "search_area_half_size" << Value << rhs.search_area_half_size;
        out << Key << "use_nearest_only" << Value << rhs.use_nearest_only;
        out << EndMap;
        return out;
    }

    template<>
    struct convert<GpisMapBase2D::Setting> {

        inline static Node
        encode(const GpisMapBase2D::Setting &setting) {
            Node node;
            node["num_threads"] = setting.num_threads;
            node["init_tree_half_size"] = setting.init_tree_half_size;
            node["perturb_delta"] = setting.perturb_delta;
            node["compute_variance"] = *setting.compute_variance;
            node["update_map_points"] = *setting.update_map_points;
            node["update_gp_sdf"] = *setting.update_gp_sdf;
            node["gp_theta"] = *setting.gp_theta;
            node["gp_sdf"] = *setting.gp_sdf;
            node["node_container"] = *setting.node_container;
            node["quadtree"] = *setting.quadtree;
            node["test_query"] = *setting.test_query;
            return node;
        }

        inline static bool
        decode(const Node &node, GpisMapBase2D::Setting &setting) {
            if (!node.IsMap()) { return false; }
            setting.num_threads = node["num_threads"].as<unsigned int>();
            setting.init_tree_half_size = node["init_tree_half_size"].as<double>();
            setting.perturb_delta = node["perturb_delta"].as<double>();
            *setting.compute_variance = node["compute_variance"].as<GpisMapBase2D::Setting::ComputeVariance>();
            *setting.update_map_points = node["update_map_points"].as<GpisMapBase2D::Setting::UpdateMapPoints>();
            *setting.update_gp_sdf = node["update_gp_sdf"].as<GpisMapBase2D::Setting::UpdateGpSdf>();
            *setting.gp_theta = node["gp_theta"].as<erl::gaussian_process::LidarGaussianProcess1D::Setting>();
            *setting.gp_sdf = node["gp_sdf"].as<erl::gaussian_process::NoisyInputGaussianProcess::Setting>();
            *setting.node_container = node["node_container"].as<GpisNodeContainer2D::Setting>();
            *setting.quadtree = node["quadtree"].as<erl::geometry::IncrementalQuadtree::Setting>();
            *setting.test_query = node["test_query"].as<GpisMapBase2D::Setting::TestQuery>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const GpisMapBase2D::Setting &setting) {
        out << BeginMap;
        out << Key << "num_threads" << Value << setting.num_threads;
        out << Key << "init_tree_half_size" << Value << setting.init_tree_half_size;
        out << Key << "perturb_delta" << Value << setting.perturb_delta;
        out << Key << "compute_variance" << Value << *setting.compute_variance;
        out << Key << "update_map_points" << Value << *setting.update_map_points;
        out << Key << "update_gp_sdf" << Value << *setting.update_gp_sdf;
        out << Key << "gp_theta" << Value << *setting.gp_theta;
        out << Key << "gp_sdf" << Value << *setting.gp_sdf;
        out << Key << "node_container" << Value << *setting.node_container;
        out << Key << "quadtree" << Value << *setting.quadtree;
        out << Key << "test_query" << Value << *setting.test_query;
        out << EndMap;
        return out;
    }
}  // namespace YAML
