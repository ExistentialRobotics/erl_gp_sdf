#pragma once

#include "abstract_surface_mapping.hpp"
#include "gp_sdf_mapping_setting.hpp"
#include "sdf_gp.hpp"
#include "surface_data_manager.hpp"

#include "erl_geometry/aabb.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <absl/container/flat_hash_map.h>
#include <boost/heap/d_ary_heap.hpp>

#include <vector>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    class GpSdfMapping {
    public:
        using SurfaceMapping = AbstractSurfaceMapping<Dtype, Dim>;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = SurfaceData<Dtype, Dim>;
        using SdfGp = SdfGaussianProcess<Dtype, Dim>;
        using Setting = GpSdfMappingSetting<Dtype, Dim>;
        using KdTree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;
        using KdTreePtr = std::shared_ptr<KdTree>;

        using Key = typename SurfaceMapping::Key;
        using KeySet = typename SurfaceMapping::KeySet;
        using VectorD = typename SurfaceMapping::VectorD;
        using VectorX = typename SurfaceMapping::VectorX;
        using MatrixDX = typename SurfaceMapping::MatrixDX;
        using MatrixX = typename SurfaceMapping::MatrixX;
        using Ranges = typename SurfaceMapping::Ranges;
        using Rotation = typename SurfaceMapping::Rotation;
        using Translation = typename SurfaceMapping::Translation;
        using Face = typename SurfaceMapping::Face;

        using KeyVector = std::vector<Key>;
        using Variances = Eigen::Matrix<Dtype, Dim + 1, Eigen::Dynamic>;
        using Covariances = Eigen::Matrix<Dtype, (Dim + 1) * Dim / 2, Eigen::Dynamic>;

    private:
        template<typename T>
        struct Less {
            [[nodiscard]] bool
            operator()(const T &lhs, const T &rhs) const {
                return lhs.priority < rhs.priority;
            }
        };

        using GpPtr = std::shared_ptr<SdfGp>;
        using KeyGpMap = absl::flat_hash_map<Key, GpPtr>;
        using KeyGpPair = std::pair<Key, GpPtr>;
        using Aabb = geometry::Aabb<Dtype, Dim>;

        struct PriorityQueueItem {
            Dtype priority = 0;
            KeyGpPair key_gp_pair{};
        };

        using PriorityQueue = boost::heap::d_ary_heap<
            PriorityQueueItem,
            boost::heap::mutable_<true>,
            boost::heap::stable<true>,
            boost::heap::arity<8>,
            boost::heap::compare<Less<PriorityQueueItem>>>;  // max-heap

        using KeyQueueMap = absl::flat_hash_map<Key, typename PriorityQueue::handle_type>;

        using UsedGps = std::array<std::shared_ptr<SdfGp>, static_cast<std::size_t>(Dim - 1) * 2>;

        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const MatrixDX>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<VectorX>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<MatrixDX>> gradients = nullptr;
            // var(d, grad.x, grad.y, grad.z)
            std::unique_ptr<Eigen::Ref<Variances>> variances = nullptr;
            // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)
            std::unique_ptr<Eigen::Ref<Covariances>> covariances = nullptr;
            // for caching intermediate results used for testing, the shape is
            // (num_neighbors * (2 * Dim + 1), num_queries).
            MatrixX gp_buffer{};

            [[nodiscard]] std::size_t
            Size() const {
                if (positions == nullptr) { return 0; }
                return positions->cols();
            }

            bool
            ConnectBuffers(
                const Eigen::Ref<const MatrixDX> &positions_in,
                VectorX &distances_out,
                MatrixDX &gradients_out,
                Variances &variances_out,
                Covariances &covariances_out,
                bool compute_covariance);

            void
            DisconnectBuffers();

            void
            PrepareGpBuffer(long num_queries, long num_neighbor_gps);
        };

        mutable std::mutex m_mutex_;
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SurfaceMapping> m_surface_mapping_ = nullptr;  // RACING CONDITION.
        KeyGpMap m_gp_map_ = {};                // key -> gp, RACING CONDITION.
        KeyQueueMap m_queue_keys_ = {};         // caching keys in the queue
        PriorityQueue m_load_data_queue_;       // queue for loading surface data
        double m_load_surf_data_time_us_ = 10;  // time spent for loading surface data
        double m_train_gp_time_us_ = 10;        // time spent for training GPs

        // temporary data for parallelism and test
        std::vector<std::vector<std::array<long, Dim>>> m_neighbor_offsets_;
        // KdTreePtr m_kdtree_surf_data_ = nullptr;  // for loading surface data
        // std::vector<std::vector<std::size_t>> m_surf_data_indices_;
        std::vector<std::vector<std::pair<Dtype, std::size_t>>> m_surf_data_dist_indices_;
        std::vector<std::size_t> m_gp_load_data_cnt_;         // counts of GPs that load data
        KeySet m_clusters_to_collect_data_;                   // stores clusters to collect data
        KeySet m_clusters_to_load_data_;                      // stores clusters to update
        std::vector<KeySet> m_key_sets_;                      // for multi-threading
        std::vector<KeyVector> m_key_vectors_;                // for multi-threading
        std::vector<KeyGpPair> m_gps_to_load_data_;           // GPs to load surface data
        std::vector<std::pair<long, GpPtr>> m_gps_to_train_;  // GPs to train, [priority, gp]
        std::vector<GpPtr> m_candidate_gps_;                  // for test
        KdTreePtr m_kdtree_candidate_gps_ = nullptr;          // for test to search candidate GPs
        Aabb m_map_boundary_{};                               // for test, boundary of the map
        std::vector<std::vector<GpPtr>> m_query_to_gps_;      // for test
        Eigen::VectorXb m_in_free_space_;                     // if queries are in free space
        std::vector<UsedGps> m_query_used_gps_;
        TestBuffer m_test_buffer_{};

    public:
        GpSdfMapping(
            std::shared_ptr<Setting> setting,
            std::shared_ptr<SurfaceMapping> surface_mapping);

        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard() const;

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<SurfaceMapping>
        GetSurfaceMapping() const;

        /**
         * Call this method to update the surface mapping and then update the GP SDF mapping.
         * You can call this method or call the surface mapping update first and then UpdateGpSdf
         * separately.
         * @param rotation The rotation of the sensor. For 2D, it is a 2x2 matrix. For 3D, it is a
         * 3x3 matrix.
         * @param translation The translation of the sensor. For 2D, it is a 2x1 vector. For 3D, it
         * is a 3x1 vector.
         * @param scan The observation that can be a point cloud or a range array.
         * @param are_points true if the scan is a point cloud. false if the scan is a range array.
         * @param are_local true if the points are in the local frame.
         * @return true if the update is successful.
         */
        [[nodiscard]] bool
        Update(
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &scan,
            bool are_points,
            bool are_local);

        bool
        UpdateGpSdf(double time_budget_us = 0);

        void
        TrainAllGps();

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const MatrixDX> &positions_in,
            VectorX &distances_out,
            MatrixDX &gradients_out,
            Variances &variances_out,
            Covariances &covariances_out);

        [[nodiscard]] const std::vector<UsedGps> &
        GetUsedGps() const {
            return m_query_used_gps_;
        }

        [[nodiscard]] const KeyGpMap &
        GetGpMap() const {
            return m_gp_map_;
        }

        void
        GetMesh(
            const VectorD &boundary_size,
            const Rotation &boundary_rotation,
            const VectorD &boundary_center,
            Dtype resolution,
            Dtype iso_value,
            std::vector<VectorD> &surface_points,
            std::vector<Face> &faces,
            std::vector<VectorD> &face_normals);

        [[nodiscard]] bool
        Write(std::ostream &stream) const;

        [[nodiscard]] bool
        Read(std::istream &stream);

        [[nodiscard]] bool
        operator==(const GpSdfMapping &other) const;

    private:
        void
        InitMultiThreading();

        Dtype
        GetDataCollectionRadius() const;

        Dtype
        GetDataCollectionAabbHalfSize() const;

        /**
         * Collect clusters that have changed in the surface mapping. m_clusters_to_load_data_ will
         * be updated.
         */
        void
        CollectChangedClusters();

        /**
         * Update the load data queue based on the clusters in m_clusters_to_load_data_.
         */
        void
        UpdateLoadDataQueue();

        /**
         * Consume the load data queue within the given time budget.
         * @param time_budget_us Time budget in microseconds.
         * @param ignore_budget If true, ignore the time budget and load data for all GPs in the
         * queue.
         */
        void
        RunLoadDataQueue(double time_budget_us, bool ignore_budget);

        void
        PrepareNeighborClusterOffsets();

        // Load surface data to the GPs in m_gps_to_load_data_
        void
        LoadSurfaceData();

        /**
         * Do the actual loading of surface data for GPs.
         * @param thread_idx the index of the thread
         * @param start_idx the start index (inclusive)
         * @param end_idx the end index (exclusive)
         */
        void
        LoadSurfaceDataThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        CollectGpsToTrain();

        // Train the GPs in m_gps_to_train_
        void
        TrainGps();

        /**
         * Do the actual training for GPs.
         * @param thread_idx the index of the thread
         * @param start_idx the start index (inclusive)
         * @param end_idx the end index (exclusive)
         */
        void
        TrainGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        SearchCandidateGps(const Eigen::Ref<const MatrixDX> &positions_in);

        void
        SearchGpThread(
            uint32_t thread_idx,
            std::size_t start_idx,
            std::size_t end_idx,
            std::vector<std::size_t> &no_gps_indices);

        void
        SearchGpFallback(const std::vector<std::size_t> &no_gps_indices);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        struct IndexedMetric {
            long idx = 0;
            std::size_t gp_idx = 0;
            Dtype metric = 0;

            IndexedMetric(long result_idx_, std::size_t gp_idx_, Dtype metric_)
                : idx(result_idx_), gp_idx(gp_idx_), metric(metric_) {}
        };

        template<int D>
        std::enable_if_t<D == 3, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<IndexedMetric> &indexed_metrics,
            const Eigen::Matrix<Dtype, 7, Eigen::Dynamic> &fs,
            const Variances &variances,
            const Covariances &covariances);

        template<int D>
        std::enable_if_t<D == 2, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<IndexedMetric> &indexed_metrics,
            const Eigen::Matrix<Dtype, 5, Eigen::Dynamic> &fs,
            const Variances &variances,
            const Covariances &covariances);
    };

    using GpSdfMapping2Df = GpSdfMapping<float, 2>;
    using GpSdfMapping2Dd = GpSdfMapping<double, 2>;
    using GpSdfMapping3Df = GpSdfMapping<float, 3>;
    using GpSdfMapping3Dd = GpSdfMapping<double, 3>;

    extern template class GpSdfMapping<float, 2>;
    extern template class GpSdfMapping<double, 2>;
    extern template class GpSdfMapping<float, 3>;
    extern template class GpSdfMapping<double, 3>;

}  // namespace erl::gp_sdf
