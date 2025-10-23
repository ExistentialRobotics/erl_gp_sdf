#pragma once

#include "abstract_surface_mapping.hpp"
#include "local_bayesian_hilbert_map.hpp"
#include "ray_selector_2d.hpp"
#include "ray_selector_3d.hpp"

#include "erl_geometry/bayesian_hilbert_map.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"
#include "erl_geometry/marching_cubes.hpp"
#include "erl_geometry/marching_squares.hpp"
#include "erl_geometry/occupancy_octree.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"
#include "erl_geometry/octree_key.hpp"
#include "erl_geometry/quadtree_key.hpp"

#include <boost/heap/d_ary_heap.hpp>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    class BayesianHilbertSurfaceMapping : public AbstractSurfaceMapping<Dtype, Dim> {

        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

    public:
        using Super = AbstractSurfaceMapping<Dtype, Dim>;
        using typename Super::Aabb;
        using typename Super::Face;
        using typename Super::Key;
        using typename Super::KeySet;
        using typename Super::KeyVector;
        using typename Super::MatrixX;
        using typename Super::Position;
        using typename Super::Positions;
        using typename Super::Ranges;
        using typename Super::Rotation;
        using typename Super::SurfData;
        using typename Super::SurfDataManager;
        using typename Super::Translation;
        using typename Super::VectorX;
        using GridShape = Eigen::Vector<long, Dim>;
        using GridIndex = Eigen::Vector<long, Dim + 1>;
        using SurfaceIndexMap = absl::flat_hash_map<GridIndex, std::size_t>;
        using SurfaceDataMap = absl::flat_hash_map<GridIndex, SurfData>;

        using KeyLongMap = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeyLongMap,
            geometry::OctreeKeyLongMap>;
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
        using RaySelectorSetting = typename RaySelector::Setting;

        // other types
        using LocalBhm = LocalBayesianHilbertMap<Dtype, Dim>;
        using LocalBhmSetting = typename LocalBhm::Setting;
        using TreeSetting = typename Tree::Setting;
        using Covariance = covariance::Covariance<Dtype>;
        using KernelSetting = typename Covariance::Setting;
        using Kdtree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;
        using MC = std::conditional_t<Dim == 2, geometry::MarchingSquares, geometry::MarchingCubes>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using Gradient = Position;
        using Gradients = Positions;

        struct Setting : public common::Yamlable<Setting> {

            struct UpdateTree {
                bool with_count = false;
                bool parallel = true;
                bool lazy_eval = true;
                bool discrete = true;
            };

            struct UpdateMap {
                // method for updating the map: 1=points, 2=marching-cubes
                int method = 1;
                // threshold for stopping the adjustment
                Dtype surface_max_abs_logodd = 0.05f;
                // threshold for bad surface points to be removed
                Dtype surface_bad_abs_logodd = 0.1f;
                // step size for the surface adjustment
                Dtype surface_step_size = 0.01f;
                // whether to automatically learn the surface log-odds
                bool auto_surface_log_odds = true;
                // maximum number of points to update, used when method=1
                int max_num_points = 100000;
                // maximum number of tries to adjust the surface points, used when method=1
                int max_adjust_tries = 3;
                // maximum number of voxels to update, used when method=2
                int max_num_voxels = 1000;
                // scale for the variance
                Dtype var_scale = 1.0f;
                // maximum variance for the surface points/normals
                Dtype var_max = 2.0f;
                // if true, update the local Bayesian Hilbert maps with CUDA
                bool update_with_cuda = false;
                // CUDA device ID to use for the local Bayesian Hilbert maps
                int cuda_device_id = 0;
                // number of local Bayesian Hilbert maps to update in one batch when using CUDA
                std::size_t update_batch_size = 128;
            };

            std::shared_ptr<LocalBhmSetting> local_bhm = std::make_shared<LocalBhmSetting>();
            std::shared_ptr<RaySelectorSetting> ray_selector =
                std::make_shared<RaySelectorSetting>();
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();

            UpdateTree update_tree;
            UpdateMap update_map;

            // scaling factor for the map
            Dtype scaling = 1.0f;
            // tree depth for the local Bayesian Hilbert map
            uint32_t bhm_depth = 14;
            // if true, synchronize shared weights across local Bayesian Hilbert maps
            bool weight_sync = false;
            // method used for weight synchronization: copy, mean, or bayesian
            std::string sync_method = "copy";
            // number of hinged points per axis
            int hinged_grid_size = 11;
            // overlap between the Bayesian Hilbert maps
            Dtype bhm_overlap = 0.2f;
            // overlap between the Bayesian Hilbert maps when weight_sync is enabled
            int bhm_overlap_sync = 1;
            // if true, build the Bayesian Hilbert map on hit, otherwise on node occupied
            bool build_bhm_on_hit = true;
            // bhm_cluster_size * 0.5 + bhm_test_margin is the half-size of the local test region
            Dtype bhm_test_margin = 0.1f;
            // number of nearest neighboring local Bayesian Hilbert maps to use for one test point
            int test_knn = 1;
            // number of test points to process in one batch
            int test_batch_size = 128;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        // the priority queue uses std::less<T> by default to make it a max-heap.
        // we want a min-heap, so we need to reverse the comparison.
        // and we want to prioritize maps that have no surface voxels.
        struct MarchingQueueItem {
            long priority = 0;
            Key key{};
        };

        struct MarchingOrder {  // greater comparison

            [[nodiscard]] bool
            operator()(const MarchingQueueItem &a, const MarchingQueueItem &b) const {
                return a.priority > b.priority;
            }
        };

        using PriorityQueue = boost::heap::d_ary_heap<
            MarchingQueueItem,
            boost::heap::mutable_<true>,
            boost::heap::stable<true>,
            boost::heap::arity<8>,
            boost::heap::compare<MarchingOrder>>;
        using KeyQueueMap = absl::flat_hash_map<Key, typename PriorityQueue::handle_type>;

        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<Tree> m_tree_ = nullptr;
        std::shared_ptr<Kdtree> m_bhm_kdtree_ = nullptr;
        bool m_bhm_kdtree_needs_update_ = true;
        Positions m_hinged_points_{};
        std::vector<std::pair<Key, Position>> m_key_bhm_positions_{};  // key -> center
        absl::flat_hash_map<Key, std::shared_ptr<LocalBhm>> m_key_bhm_dict_{};
        SurfDataManager m_surf_data_manager_ = {};
        KeySet m_changed_clusters_{};                   // keys of the changed clusters
        KeyVector m_clusters_to_update_{};              // keys of the clusters to update
        std::vector<int> m_updated_flags_{};            // flags: which BHMs are updated
        RaySelector m_ray_selector_;                    // selector for rays
        std::vector<std::vector<long>> m_ray_indices_;  // buffer for ray indices

        /* variables used when m_setting_->update_map.method = 1 */

        /**
         * @brief Struct to hold information about points during the update process.
         */
        struct PointInfo {
            GridIndex grid_idx = {};
            std::size_t surf_idx = -1;
            bool to_remove = false;
            GridIndex new_grid_idx = {};
            bool to_move = false;

            PointInfo() = default;

            PointInfo(const GridIndex grid_idx_, const std::size_t surf_idx_)
                : grid_idx(grid_idx_), surf_idx(surf_idx_) {}
        };

        // buffers for the new and existing hit points:
        // - Key: bhm_key
        // - std::vector<PointInfo>: new_hit_points
        // - std::vector<PointInfo>: existing_hit_points
        std::vector<std::tuple<Key, std::vector<PointInfo>, std::vector<PointInfo>>> m_hit_points_;

        /* variables used when m_setting_->update_map.method = 2 */
        KeyQueueMap m_marching_queue_keys_ = {};  // caching key in the queue
        PriorityQueue m_marching_queue_;          // queue BHMs, smaller cnt first

        // members for synchronizing weights

        // (tree key offset, idx in src, idx in dst)
        using WeightAddr = Eigen::Vector<int, Dim + 2>;
        std::vector<WeightAddr> m_core_indices_;              // core indices
        std::vector<WeightAddr> m_managed_share_indices_;     // managed share indices
        std::vector<WeightAddr> m_unmanaged_share_indices_;   // unmanaged indices
        std::vector<Eigen::Vector<int, Dim>> m_key_offsets_;  // key offsets of neighboring BHMs
        std::vector<long> m_hinged_point_order_;              // ordered index to original index
        std::vector<long> m_hinged_point_order_reverse_;      // original index to ordered index
        KeySet m_bhm_to_sync_;                                // set of keys of BHMs to sync weights
        KeyVector m_bhm_to_sync_vector_;                      // keys of BHMs to sync weights

        // frequently used intermediate values
        int m_block_size_ = 0;          // number of hinged points along one dimension of a block
        int m_sync_method_ = 0;         // method used for synchronizing weights
        Dtype m_bhm_node_size_ = 0.0f;  // size of a node that stores a BHM
        Dtype m_half_bhm_node_size_ = 0.0f;      // half the size of a BHM node
        Dtype m_half_bhm_size_ = 0.0f;           // half BHM size (with overlap)
        Dtype m_hinged_grid_res_ = 0.0f;         // resolution of the hinged grid
        Dtype m_ray_search_radius_ = 0.0f;       // radius to search rays for a BHM
        Dtype m_half_bhm_test_size_ = 0.0f;      // half size of the valid test boundary
        Dtype m_surface_res_ = 0.0f;             // resolution of the surface grid
        std::size_t m_surf_point_capacity_ = 0;  // capacity of the surface grid

    public:
        explicit BayesianHilbertSurfaceMapping(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetTree() const;

        [[nodiscard]] const absl::flat_hash_map<Key, std::shared_ptr<LocalBhm>> &
        GetLocalBhms() const;

        /**
         * @brief Update the Bayesian Hilbert map with a point cloud from sensor observation.
         * @param sensor_rotation The rotation of the sensor.
         * @param sensor_origin The origin of the sensor.
         * @param points The point cloud in the world frame.
         * @param parallel If true, the update will be parallelized.
         * @return True if the update was successful, false otherwise.
         */
        bool
        Update(
            const Eigen::Ref<const Rotation> &sensor_rotation,
            const Eigen::Ref<const Position> &sensor_origin,
            const Eigen::Ref<const Positions> &points,
            bool parallel);

        typename SurfDataManager::Iterator
        BeginSurfaceData();

        typename SurfDataManager::Iterator
        EndSurfaceData();

        /**
         *
         * @param points Matrix of points in the world frame. Each column is a point.
         * @param logodd If true, the output will be log-odds instead of probabilities.
         * @param compute_free_space If true, compute if the points are in free space.
         * @param compute_gradient If true, the gradient will be computed.
         * @param gradient_with_sigmoid If true, compute the gradient of sigmoid(logodd).
         * @param parallel If true, the computation will be parallelized.
         * @param prob_occupied Output vector of occupancy probabilities or log-odds.
         * @param in_free_space Output vector indicating if each point is in free space.
         * @param gradients Output matrix of gradients. If compute_gradient is false, this will not
         * be used.
         */
        void
        Predict(
            const Eigen::Ref<const Positions> &points,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            VectorX &prob_occupied,
            Eigen::VectorXb &in_free_space,
            Gradients &gradients) const;

        void
        PredictGradient(
            const Eigen::Ref<const Positions> &points,
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

        [[nodiscard]] KeySet
        GetAllClusters() const override;

        [[nodiscard]] Key
        GetClusterKey(const Eigen::Ref<const Position> &pos) const override;

        void
        IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback)
            const override;

        [[nodiscard]] const std::vector<SurfData> &
        GetSurfaceDataBuffer() const override;

        void
        CollectSurfaceDataInAabb(
            const Aabb &aabb,
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const override;

        void
        GetMesh(std::vector<Position> &vertices, std::vector<Face> &faces) const override;

        [[nodiscard]] Aabb
        GetMapBoundary() const override;

        [[nodiscard]] bool
        IsInFreeSpace(const Positions &positions, Eigen::VectorXb &in_free_space) const override;

        [[nodiscard]] bool
        operator==(const Super &other) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

        void
        ResetMarchingResults();

    private:
        void
        InitConstants();

        void
        GenerateHingedPoints();

        void
        GenerateWeightAddress();

        std::pair<typename absl::flat_hash_map<Key, std::shared_ptr<LocalBhm>>::iterator, bool>
        CreateBhm(const Key &key);

        void
        SyncBhmWeights(const Key &key);

        void
        BuildBhmKdtree() const;

        void
        PredictThread(
            const Dtype *points_ptr,
            long start,
            long end,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            Dtype *prob_occupied_ptr,
            bool *in_free_space_ptr,
            Dtype *gradient_ptr) const;

        void
        UpdateMapPoints(const Position &sensor_origin, const Eigen::Ref<const Positions> &points);

        void
        UpdateMapPoints1(const Position &sensor_origin, const Eigen::Ref<const Positions> &points);

        void
        UpdateSurfaceManager1();

        void
        InitMapPoint1(LocalBhm &local_bhm, SurfData &surf_data, bool &to_remove) const;

        void
        UpdateMapPoint1(LocalBhm &local_bhm, SurfData &surf_data, bool &to_remove) const;

        void
        UpdateMapPoints2(const Position &sensor_origin, const Eigen::Ref<const Positions> &points);

        void
        MarchingBhm(const Key &key, LocalBhm &local_bhm) const;

        void
        UpdateSurfaceManager2(std::vector<std::pair<Key, std::shared_ptr<LocalBhm>>> &local_bhms);

        void
        RunMarchingQueue(bool run_all);

        [[nodiscard]] Position
        GetUniqueVertex(Key key, GridIndex edge_idx, int buffer_idx) const;
    };

    using BayesianHilbertSurfaceMapping2Df = BayesianHilbertSurfaceMapping<float, 2>;
    using BayesianHilbertSurfaceMapping3Df = BayesianHilbertSurfaceMapping<float, 3>;
    using BayesianHilbertSurfaceMapping2Dd = BayesianHilbertSurfaceMapping<double, 2>;
    using BayesianHilbertSurfaceMapping3Dd = BayesianHilbertSurfaceMapping<double, 3>;

    extern template class BayesianHilbertSurfaceMapping<float, 2>;
    extern template class BayesianHilbertSurfaceMapping<float, 3>;
    extern template class BayesianHilbertSurfaceMapping<double, 2>;
    extern template class BayesianHilbertSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf

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
