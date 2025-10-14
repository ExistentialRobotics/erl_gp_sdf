#pragma once

#include "abstract_surface_mapping.hpp"

#include "erl_geometry/bayesian_hilbert_map.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"
#include "erl_geometry/occupancy_octree.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"
#include "erl_geometry/octree_key.hpp"
#include "erl_geometry/quadtree_key.hpp"

namespace erl::gp_sdf {

    /**
     * Select 2D rays for Bayesian Hilbert Map.
     */
    template<typename Dtype>
    class RaySelector2D {

    public:
        struct Setting : common::Yamlable<Setting> {
            // minimum ray angle in radians in the local frame
            Dtype angle_min = -M_PI / 4.0;
            // maximum ray angle in radians in the local frame
            Dtype angle_max = M_PI / 4.0;
            // number of angles
            long num_angles = 91;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        using Vector2 = Eigen::Vector2<Dtype>;
        using Matrix2 = Eigen::Matrix2<Dtype>;
        using Matrix2X = Eigen::Matrix2X<Dtype>;

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Dtype m_angle_resolution_ = 0.0f;
        std::vector<std::vector<long>> m_ray_indices_;  // ray indices for each angle

    public:
        explicit RaySelector2D(std::shared_ptr<Setting> setting);

        void
        UpdateRays(
            const Vector2 &sensor_origin,
            const Matrix2 &sensor_rotation,
            const Eigen::Ref<const Matrix2X> &ray_end_points);

        void
        SelectRays(
            const Vector2 &sensor_origin,
            const Matrix2 &sensor_rotation,
            Vector2 point,
            Dtype radius,
            std::vector<long> &ray_indices) const;
    };

    template<typename Dtype>
    class RaySelector3D {
    public:
        struct Setting : common::Yamlable<Setting> {
            // minimum azimuth angle in radians
            Dtype azimuth_min = -M_PI;
            // maximum azimuth angle in radians
            Dtype azimuth_max = M_PI;
            // minimum elevation angle in radians
            Dtype elevation_min = -M_PI / 2.0;
            // maximum elevation angle in radians
            Dtype elevation_max = M_PI / 2.0;
            // number of azimuth angles
            Dtype num_azimuth_angles = 181;
            // number of elevation angles
            Dtype num_elevation_angles = 91;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        using Vector3 = Eigen::Vector3<Dtype>;
        using Matrix3 = Eigen::Matrix3<Dtype>;
        using Matrix3X = Eigen::Matrix3X<Dtype>;

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Dtype m_azimuth_res_ = 0.0f;
        Dtype m_elevation_res_ = 0.0f;
        Eigen::MatrixX<std::vector<long>> m_ray_indices_;  // ray indices for each angle

    public:
        explicit RaySelector3D(std::shared_ptr<Setting> setting);

        void
        UpdateRays(
            const Vector3 &sensor_origin,
            const Matrix3 &sensor_rotation,
            const Eigen::Ref<const Matrix3X> &ray_end_points);

        void
        SelectRays(
            const Vector3 &sensor_origin,
            const Matrix3 &sensor_rotation,
            Vector3 point,
            Dtype radius,
            std::vector<long> &ray_indices) const;
    };

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
        long max_dataset_size = -1;        // maximum size of the dataset to store
        long hit_buffer_size = -1;         // -1 means no limit, 0 means no hit buffer
        Dtype surface_resolution = 0.01f;  // resolution to track the surface points

        struct YamlConvertImpl {
            static YAML::Node
            encode(const LocalBayesianHilbertMapSetting &setting);

            static bool
            decode(const YAML::Node &node, LocalBayesianHilbertMapSetting &setting);
        };
    };

    template<typename Dtype, int Dim>
    class BayesianHilbertSurfaceMapping : public AbstractSurfaceMapping<Dtype, Dim> {

        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

    public:
        using Super = AbstractSurfaceMapping<Dtype, Dim>;
        using typename Super::Aabb;
        using typename Super::Key;
        using typename Super::KeySet;
        using typename Super::MatrixX;
        using typename Super::Position;
        using typename Super::Positions;
        using typename Super::Ranges;
        using typename Super::Rotation;
        using typename Super::SurfData;
        using typename Super::SurfDataManager;
        using typename Super::Translation;
        using typename Super::VectorX;

        using KeyVector = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeyVector,
            geometry::OctreeKeyVector>;
        using KeyVectorMap = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeyVectorMap,
            geometry::OctreeKeyVectorMap>;
        using Tree = std::conditional_t<
            Dim == 2,
            geometry::OccupancyQuadtree<Dtype>,
            geometry::OccupancyOctree<Dtype>>;
        using TreeNode = std::conditional_t<  //
            Dim == 2,
            geometry::OccupancyQuadtreeNode,
            geometry::OccupancyOctreeNode>;
        using RaySelector = std::conditional_t<  //
            Dim == 2,
            RaySelector2D<Dtype>,
            RaySelector3D<Dtype>>;

        // other types
        using BayesianHilbertMap = geometry::BayesianHilbertMap<Dtype, Dim>;
        using TreeSetting = typename Tree::Setting;
        using Covariance = covariance::Covariance<Dtype>;
        using KernelSetting = typename Covariance::Setting;
        using Kdtree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using Gradient = Position;
        using Gradients = Positions;

        struct LocalBayesianHilbertMap {
            using Setting = LocalBayesianHilbertMapSetting<Dtype>;

            std::shared_ptr<Setting> setting = nullptr;  // settings for the local map
            Aabb tracked_surface_boundary{};             // boundary of the surface to track
            BayesianHilbertMap bhm;                      // local Bayesian Hilbert map
            Eigen::Vector<int, Dim> grid_shape;          // shape of the grid for the surface points
            Eigen::Vector<int, Dim> strides;             // strides for indexing the surface points
            absl::flat_hash_map<int, std::size_t> surface_indices;  // local index -> buffer index
            long num_dataset_points = 0;                            // number of dataset points
            Positions dataset_points{};                             // [Dim, N] dataset points
            VectorX dataset_labels{};                               // [N, 1] dataset labels
            std::vector<long> hit_indices{};     // indices of the hit points in the dataset
            std::vector<Position> hit_buffer{};  // hit point buffer of M points
            long hit_buffer_head = 0;            // head of the hit point buffer
            bool active = true;                  // whether the local Bayesian Hilbert map is active

            LocalBayesianHilbertMap(
                std::shared_ptr<Setting> setting_,
                Positions hinged_points,
                Aabb map_boundary,
                uint64_t seed,
                std::optional<Aabb> track_surface_boundary_ = std::nullopt);

            void
            GenerateDataset(
                const Eigen::Ref<const Position> &sensor_origin,
                const Eigen::Ref<const Positions> &points,
                const std::vector<long> &point_indices);

            bool
            Update(
                const Eigen::Ref<const Position> &sensor_origin,
                const Eigen::Ref<const Positions> &points,
                const std::vector<long> &point_indices);

            void
            UpdateHitBuffer(const Eigen::Ref<const Positions> &points);

            [[nodiscard]] bool
            Write(std::ostream &s) const;

            [[nodiscard]] bool
            Read(std::istream &s);

            [[nodiscard]] bool
            operator==(const LocalBayesianHilbertMap &other) const;

            [[nodiscard]] bool
            operator!=(const LocalBayesianHilbertMap &other) const;
        };

        struct Setting : public common::Yamlable<Setting> {
            std::shared_ptr<typename LocalBayesianHilbertMap::Setting> local_bhm =
                std::make_shared<typename LocalBayesianHilbertMap::Setting>();
            std::shared_ptr<typename RaySelector::Setting> ray_selector =
                std::make_shared<typename RaySelector::Setting>();
            // the tree setting
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();
            // number of hinged points per axis
            int hinged_grid_size = 11;
            // threshold for stopping the adjustment
            Dtype surface_max_abs_logodd = 0.05f;
            // threshold for bad surface points to be removed
            Dtype surface_bad_abs_logodd = 0.1f;
            // step size for the surface adjustment
            Dtype surface_step_size = 0.01f;
            // if true, pass faster=true to the Bayesian Hilbert map predict methods, which assumes
            // that the weight covariance is very small.
            bool faster_prediction = false;
            // maximum number of tries to adjust the surface points
            int max_adjust_tries = 3;
            // scale for the variance
            Dtype var_scale = 1.0f;
            // maximum variance for the surface points/normals
            Dtype var_max = 2.0f;
            // scaling factor for the map
            Dtype scaling = 1.0f;
            // tree depth for the local Bayesian Hilbert map
            uint32_t bhm_depth = 14;
            // overlap between the Bayesian Hilbert maps
            Dtype bhm_overlap = 0.2f;
            // if true, update the local Bayesian Hilbert maps with CUDA
            bool update_with_cuda = false;
            // CUDA device ID to use for the local Bayesian Hilbert maps
            int cuda_device_id = 0;
            // number of local Bayesian Hilbert maps to update in one batch when using CUDA
            std::size_t update_batch_size = 128;
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
        SurfDataManager m_surf_data_manager_ = {};
        KeySet m_changed_clusters_{};  // keys of the clusters that have changed
        RaySelector m_ray_selector_;
        std::vector<std::vector<long>> m_ray_indices_;  // buffer for ray indices

        // buffers for the new and existing hit points

        struct PointInfo {
            int local_idx;
            std::size_t surf_idx;
            bool to_remove;
            int new_local_idx;

            PointInfo(
                const int local_idx_,
                const std::size_t surf_idx_,
                const bool to_remove_ = false,
                const int new_local_idx_ = -1)
                : local_idx(local_idx_),
                  surf_idx(surf_idx_),
                  to_remove(to_remove_),
                  new_local_idx(new_local_idx_) {}
        };

        // bhm_key, new_hit_points, existing_hit_points
        std::vector<std::tuple<Key, std::vector<PointInfo>, std::vector<PointInfo>>> m_hit_points_;

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
            const Eigen::Ref<const Rotation> &sensor_rotation,
            const Eigen::Ref<const Position> &sensor_origin,
            const Eigen::Ref<const Positions> &points,
            bool parallel);

        // [[nodiscard]] std::vector<SurfaceData<double, 3>>
        // GetSurfaceData() const override;

        typename SurfDataManager::Iterator
        BeginSurfaceData();

        typename SurfDataManager::Iterator
        EndSurfaceData();

        size_t
        GetSurfaceDataSize() const {
            return m_surf_data_manager_.Size();
        }

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

        // implement the methods required by AbstractSurfaceMapping

        bool
        Update(
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &scan,
            bool are_points,
            bool are_local) override;

        [[nodiscard]] Dtype
        GetScaling() const override;

        [[nodiscard]] Dtype
        GetClusterSize() const override;

        [[nodiscard]] Position
        GetClusterCenter(const Key &key) const override;

        [[nodiscard]] const KeySet &
        GetChangedClusters() const override;

        void
        IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback)
            const override;

        [[nodiscard]] const std::vector<SurfData> &
        GetSurfaceDataBuffer() const override;

        void
        CollectSurfaceDataInAabb(
            const Aabb &aabb,
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const override;

        [[nodiscard]] Aabb
        GetMapBoundary() const override;

        [[nodiscard]] bool
        IsInFreeSpace(const Positions &positions, VectorX &in_free_space) const override;

        [[nodiscard]] bool
        operator==(const Super &other) const override;

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

        void
        UpdateMapPoints(const Eigen::Ref<const Positions> &points);

        // void
        // InitMapPointThread(int thread_id, int start, int end);

        void
        InitMapPoint(BayesianHilbertMap &bhm, SurfData &surf_data, bool &to_remove) const;

        void
        UpdateMapPoint(BayesianHilbertMap &bhm, SurfData &surf_data, bool &to_remove) const;
    };

    using RaySelector2Df = RaySelector2D<float>;
    using RaySelector2Dd = RaySelector2D<double>;
    using RaySelector3Df = RaySelector3D<float>;
    using RaySelector3Dd = RaySelector3D<double>;
    using LocalBayesianHilbertMapSettingF = LocalBayesianHilbertMapSetting<float>;
    using LocalBayesianHilbertMapSettingD = LocalBayesianHilbertMapSetting<double>;
    using BayesianHilbertSurfaceMapping2Df = BayesianHilbertSurfaceMapping<float, 2>;
    using BayesianHilbertSurfaceMapping3Df = BayesianHilbertSurfaceMapping<float, 3>;
    using BayesianHilbertSurfaceMapping2Dd = BayesianHilbertSurfaceMapping<double, 2>;
    using BayesianHilbertSurfaceMapping3Dd = BayesianHilbertSurfaceMapping<double, 3>;

    extern template class RaySelector2D<float>;
    extern template class RaySelector2D<double>;
    extern template class RaySelector3D<float>;
    extern template class RaySelector3D<double>;
    extern template class LocalBayesianHilbertMapSetting<float>;
    extern template class LocalBayesianHilbertMapSetting<double>;
    extern template class BayesianHilbertSurfaceMapping<float, 2>;
    extern template class BayesianHilbertSurfaceMapping<float, 3>;
    extern template class BayesianHilbertSurfaceMapping<double, 2>;
    extern template class BayesianHilbertSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf

// #include "bayesian_hilbert_surface_mapping.tpp"

template<>
struct YAML::convert<erl::gp_sdf::RaySelector2Df::Setting>
    : erl::gp_sdf::RaySelector2Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::RaySelector2Dd::Setting>
    : erl::gp_sdf::RaySelector2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::RaySelector3Df::Setting>
    : erl::gp_sdf::RaySelector3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::RaySelector3Dd::Setting>
    : erl::gp_sdf::RaySelector3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::LocalBayesianHilbertMapSettingF>
    : erl::gp_sdf::LocalBayesianHilbertMapSettingF::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::LocalBayesianHilbertMapSettingD>
    : erl::gp_sdf::LocalBayesianHilbertMapSettingD::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::BayesianHilbertSurfaceMapping2Df::Setting>
    : erl::gp_sdf::BayesianHilbertSurfaceMapping2Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::BayesianHilbertSurfaceMapping2Dd::Setting>
    : erl::gp_sdf::BayesianHilbertSurfaceMapping2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::BayesianHilbertSurfaceMapping3Df::Setting>
    : erl::gp_sdf::BayesianHilbertSurfaceMapping3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::BayesianHilbertSurfaceMapping3Dd::Setting>
    : erl::gp_sdf::BayesianHilbertSurfaceMapping3Dd::Setting::YamlConvertImpl {};
