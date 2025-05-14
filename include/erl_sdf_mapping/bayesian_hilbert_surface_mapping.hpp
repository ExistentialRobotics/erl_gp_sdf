#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"

#include "erl_geometry/bayesian_hilbert_map.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"
#include "erl_geometry/occupancy_octree.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"
#include "erl_geometry/octree_key.hpp"
#include "erl_geometry/quadtree_key.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    struct LocalBayesianHilbertMapSetting
        : common::Yamlable<LocalBayesianHilbertMapSetting<Dtype>> {
        using Covariance = covariance::Covariance<Dtype>;
        using KernelSetting = typename Covariance::Setting;

        std::shared_ptr<geometry::BayesianHilbertMapSetting> bhm =
            std::make_shared<geometry::BayesianHilbertMapSetting>();
        std::string kernel_type = type_name<Covariance>();
        std::string kernel_setting_type = type_name<KernelSetting>();
        std::shared_ptr<KernelSetting> kernel = std::make_shared<KernelSetting>();
        long max_dataset_size = -1;                // maximum size of the dataset to store
        long hit_buffer_size = -1;                 // -1 means no limit, 0 means no hit buffer
        bool track_surface = true;                 // if true, track the surface points
        Dtype surface_resolution = 0.01f;          // resolution to track the surface points
        Dtype surface_occ_prob_threshold = 0.55f;  // threshold to consider a point as occupied
        Dtype surface_occ_prob_target = 0.99f;     // target occupancy probability for the surface
        Dtype surface_adjust_step = 0.01f;         // step for the surface adjustment
        Dtype var_scale = 1.0f;                    // scale for the variance

        struct YamlConvertImpl {
            static YAML::Node
            encode(const LocalBayesianHilbertMapSetting &setting);

            static bool
            decode(const YAML::Node &node, LocalBayesianHilbertMapSetting &setting);
        };
    };

    using LocalBayesianHilbertMapSettingF = LocalBayesianHilbertMapSetting<float>;
    using LocalBayesianHilbertMapSettingD = LocalBayesianHilbertMapSetting<double>;

    template<typename Dtype, int Dim>
    class BayesianHilbertSurfaceMapping : public AbstractSurfaceMapping {

        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

    public:
        // type definitions required by GpSdfMapping
        using Key = std::conditional_t<Dim == 2, geometry::QuadtreeKey, geometry::OctreeKey>;
        using KeySet = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeySet,
            geometry::OctreeKeySet>;
        using KeyVector = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeyVector,
            geometry::OctreeKeyVector>;
        using Tree = std::conditional_t<
            Dim == 2,
            geometry::OccupancyQuadtree<Dtype>,
            geometry::OccupancyOctree<Dtype>>;
        using TreeNode = std::conditional_t<  //
            Dim == 2,
            geometry::OccupancyQuadtreeNode,
            geometry::OccupancyOctreeNode>;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = typename SurfDataManager::Data;

        // other types
        using BayesianHilbertMap = geometry::BayesianHilbertMap<Dtype, Dim>;
        using TreeSetting = typename Tree::Setting;
        using Covariance = covariance::Covariance<Dtype>;
        using KernelSetting = typename Covariance::Setting;
        using Aabb = geometry::Aabb<Dtype, Dim>;
        using Kdtree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Vector<Dtype, Dim>;
        using Ranges = MatrixX;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Gradient = Position;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using Gradients = Positions;

        struct LocalBayesianHilbertMap {
            struct Surface {
                Position position = Position::Zero();  // position of the surface point
                Gradient normal = Gradient::Zero();    // normal of the surface point
                Dtype prob_occupied = 0.0f;  // probability of the surface point being occupied

                explicit Surface(Position position_)
                    : position(std::move(position_)) {}

                [[nodiscard]] bool
                operator==(const Surface &other) const;

                [[nodiscard]] bool
                operator!=(const Surface &other) const;
            };

            using Setting = LocalBayesianHilbertMapSetting<Dtype>;

            std::shared_ptr<Setting> setting = nullptr;  // settings for the local map
            Aabb tracked_surface_boundary{};             // boundary of the surface to track
            BayesianHilbertMap bhm;                      // local Bayesian Hilbert map
            Eigen::Vector<int, Dim> strides;             // strides for indexing the surface points
            std::vector<int> surface_indices{};          // indices of the surface points
            absl::flat_hash_map<int, Surface> surface;   // map of surface points and their normals
            long num_dataset_points = 0;                 // number of dataset points
            Positions dataset_points{};                  // [Dim, N] dataset points
            VectorX dataset_labels{};                    // [N, 1] dataset labels
            std::vector<long> hit_indices{};             // indices of the hit points in the dataset
            std::vector<Position> hit_buffer{};          // hit point buffer of M points
            long hit_buffer_head = 0;                    // head of the hit point buffer

            LocalBayesianHilbertMap(
                std::shared_ptr<Setting> setting_,
                Positions hinged_points,
                Aabb map_boundary,
                uint64_t seed,
                std::optional<Aabb> track_surface_boundary_ = std::nullopt);

            bool
            Update(
                const Eigen::Ref<const Position> &sensor_origin,
                const Eigen::Ref<const Positions> &points);

            [[nodiscard]] bool
            Write(std::ostream &s) const;

            [[nodiscard]] bool
            Read(std::istream &s);

            [[nodiscard]] bool
            operator==(const LocalBayesianHilbertMap &other) const;

            [[nodiscard]] bool
            operator!=(const LocalBayesianHilbertMap &other) const;

        private:
            bool
            UpdateBhm(
                const Eigen::Ref<const Position> &sensor_origin,
                const Eigen::Ref<const Positions> &points);

            void
            TrackSurface(const Eigen::Ref<const Positions> &points);
        };

        struct Setting : public common::Yamlable<Setting> {
            std::shared_ptr<typename LocalBayesianHilbertMap::Setting> local_bhm =
                std::make_shared<typename LocalBayesianHilbertMap::Setting>();
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();
            int hinged_grid_size = 11;
            // tree depth for the local Bayesian Hilbert map
            uint32_t bhm_depth = 14;
            // overlap between the Bayesian Hilbert maps
            Dtype bhm_overlap = 0.2f;
            // bhm_cluster_size * 0.5 + bhm_test_margin is the half-size of the local test region
            Dtype bhm_test_margin = 0.1f;
            // number of nearest neighboring local Bayesian Hilbert maps to use for one test point
            int test_knn = 1;
            // number of test points to process in one batch
            int test_batch_size = 128;
            // if true, build the Bayesian Hilbert map on hit, otherwise on node occupied
            bool build_bhm_on_hit = true;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<Tree> m_tree_ = nullptr;
        std::shared_ptr<Kdtree> m_bhm_kdtree_ = nullptr;
        bool m_bhm_kdtree_needs_update_ = true;
        Positions m_hinged_points_{};
        std::vector<std::pair<Key, Position>> m_key_bhm_positions_{};  // key -> center
        absl::flat_hash_map<Key, std::shared_ptr<LocalBayesianHilbertMap>> m_key_bhm_dict_{};

    public:
        explicit BayesianHilbertSurfaceMapping(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetTree() const;

        [[nodiscard]] const absl::flat_hash_map<Key, std::shared_ptr<LocalBayesianHilbertMap>> &
        GetLocalBayesianHilbertMaps() const;

        bool
        Update(
            const Eigen::Ref<const Position> &sensor_origin,
            const Eigen::Ref<const Positions> &points,
            bool parallel);

        /**
         *
         * @param points Matrix of points in the world frame. Each column is a point.
         * @param logodd If true, the output will be log-odds instead of probabilities.
         * @param faster If true, the computation will be faster but less accurate.
         * @param compute_gradient If true, the gradient will be computed.
         * @param gradient_with_sigmoid If true, the gradient will be multiplied by the sigmoid
         * function.
         * @param parallel If true, the computation will be parallelized.
         * @param prob_occupied Output vector of occupancy probabilities or log-odds.
         * @param gradient Output matrix of gradients. If compute_gradient is false, this will not
         * be used.
         */
        void
        Predict(
            const Eigen::Ref<const Positions> &points,
            bool logodd,
            bool faster,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            VectorX &prob_occupied,
            Gradients &gradient) const;

        void
        PredictGradient(
            const Eigen::Ref<const Positions> &points,
            bool faster,
            bool with_sigmoid,
            bool parallel,
            Gradients &gradient) const;

        // implement the methods required by AbstractSurfaceMapping for factory pattern
        [[nodiscard]] bool
        operator==(const AbstractSurfaceMapping &other) const override;

        using AbstractSurfaceMapping::Read;
        using AbstractSurfaceMapping::Write;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

    private:
        void
        GenerateHingedPoints();

        void
        BuildBhmKdtree() const;

        void
        PredictThread(
            const Dtype *points_ptr,
            long start,
            long end,
            bool logodd,
            bool faster,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            Dtype *prob_occupied_ptr,
            Dtype *gradient_ptr) const;
    };

    using BayesianHilbertSurfaceMapping2Df = BayesianHilbertSurfaceMapping<float, 2>;
    using BayesianHilbertSurfaceMapping3Df = BayesianHilbertSurfaceMapping<float, 3>;
    using BayesianHilbertSurfaceMapping2Dd = BayesianHilbertSurfaceMapping<double, 2>;
    using BayesianHilbertSurfaceMapping3Dd = BayesianHilbertSurfaceMapping<double, 3>;

}  // namespace erl::sdf_mapping

#include "bayesian_hilbert_surface_mapping.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::LocalBayesianHilbertMapSettingF>
    : erl::sdf_mapping::LocalBayesianHilbertMapSettingF::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::LocalBayesianHilbertMapSettingD>
    : erl::sdf_mapping::LocalBayesianHilbertMapSettingD::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::BayesianHilbertSurfaceMapping2Df::Setting>
    : erl::sdf_mapping::BayesianHilbertSurfaceMapping2Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::BayesianHilbertSurfaceMapping2Dd::Setting>
    : erl::sdf_mapping::BayesianHilbertSurfaceMapping2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::BayesianHilbertSurfaceMapping3Df::Setting>
    : erl::sdf_mapping::BayesianHilbertSurfaceMapping3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::BayesianHilbertSurfaceMapping3Dd::Setting>
    : erl::sdf_mapping::BayesianHilbertSurfaceMapping3Dd::Setting::YamlConvertImpl {};
