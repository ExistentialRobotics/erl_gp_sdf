#pragma once

#include "abstract_surface_mapping.hpp"
#include "gp_sdf_mapping_setting.hpp"
#include "sdf_gp.hpp"
#include "surface_data_manager.hpp"

#include "erl_geometry/aabb.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
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

        using Key = typename SurfaceMapping::Key;
        using KeySet = typename SurfaceMapping::KeySet;
        using MatrixX = typename SurfaceMapping::MatrixX;
        using Position = typename SurfaceMapping::Position;
        using Positions = typename SurfaceMapping::Positions;
        using Ranges = typename SurfaceMapping::Ranges;
        using Rotation = typename SurfaceMapping::Rotation;
        using Translation = typename SurfaceMapping::Translation;
        using VectorX = typename SurfaceMapping::VectorX;

        using KeyVector = std::vector<Key>;
        using Gradient = Eigen::Vector<Dtype, Dim>;
        using Distances = VectorX;
        using Gradients = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using Variances = Eigen::Matrix<Dtype, Dim + 1, Eigen::Dynamic>;
        using Covariances = Eigen::Matrix<Dtype, (Dim + 1) * Dim / 2, Eigen::Dynamic>;

    private:
        template<typename T>
        struct Greater {
            [[nodiscard]] bool
            operator()(const T& lhs, const T& rhs) const {
                return lhs.time_stamp > rhs.time_stamp;
            }
        };

        struct PriorityQueueItem {
            long time_stamp = 0;
            Key key{};
        };

        using PriorityQueue = boost::heap::d_ary_heap<
            PriorityQueueItem,
            boost::heap::mutable_<true>,
            boost::heap::stable<true>,
            boost::heap::arity<8>,
            boost::heap::compare<Greater<PriorityQueueItem>>>;

        using KeyQueueMap = absl::flat_hash_map<Key, typename PriorityQueue::handle_type>;
        using KeyGpMap = absl::flat_hash_map<Key, std::shared_ptr<SdfGp>>;
        using KeyGpPair = std::pair<Key, std::shared_ptr<SdfGp>>;
        using Aabb = geometry::Aabb<Dtype, Dim>;

        struct TestBuffer {

            std::unique_ptr<Eigen::Ref<const Positions>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Distances>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Gradients>> gradients = nullptr;
            // var(d, grad.x, grad.y, grad.z)
            std::unique_ptr<Eigen::Ref<Variances>> variances = nullptr;
            // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)
            std::unique_ptr<Eigen::Ref<Covariances>> covariances = nullptr;
            // for caching intermediate results used for testing, the shape is
            // (num_neighbors * (2 * Dim + 1), num_queries).
            MatrixX gp_buffer{};

            [[nodiscard]] std::size_t
            Size() const {
                if (positions == nullptr) return 0;
                return positions->cols();
            }

            bool
            ConnectBuffers(
                const Eigen::Ref<const Positions>& positions_in,
                Distances& distances_out,
                Gradients& gradients_out,
                Variances& variances_out,
                Covariances& covariances_out,
                bool compute_covariance);

            void
            DisconnectBuffers();

            void
            PrepareGpBuffer(long num_queries, long num_neighbor_gps);
        };

        std::mutex m_mutex_;
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SurfaceMapping> m_surface_mapping_ = nullptr;  // RACING CONDITION.
        KeyGpMap m_gp_map_ = {};                 // key -> gp, RACING CONDITION.
        KeyVector m_affected_clusters_ = {};     // stores clusters to update
        KeyQueueMap m_cluster_queue_keys_ = {};  // caching clusters in the queue to be updated
        PriorityQueue m_cluster_queue_;          // queue clusters, smaller time_stamp first (FIFO)
        std::vector<KeyGpPair> m_clusters_to_train_ = {};  // clusters to train, RACING CONDITION.
        std::vector<KeyGpPair> m_candidate_gps_ = {};      // for testing
        std::shared_ptr<KdTree> m_kdtree_candidate_gps_ = nullptr;  // to search candidate GPs
        Aabb m_map_boundary_ = {};  // for testing, boundary of the surface map
        std::vector<std::vector<std::pair<Dtype, KeyGpPair>>> m_query_to_gps_ = {};  // for testing
        VectorX m_query_signs_ = {};      // sign of query positions
        double m_train_gp_time_us_ = 10;  // time spent for training GPs
        TestBuffer m_test_buffer_ = {};
        std::vector<std::array<std::shared_ptr<SdfGp>, (Dim - 1) * 2>> m_query_used_gps_ = {};

    public:
        GpSdfMapping(
            std::shared_ptr<Setting> setting,
            std::shared_ptr<SurfaceMapping> surface_mapping);

        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard();

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
            const Eigen::Ref<const Rotation>& rotation,
            const Eigen::Ref<const Translation>& translation,
            const Eigen::Ref<const Ranges>& scan,
            bool are_points,
            bool are_local);

        bool
        UpdateGpSdf(double time_budget_us = 0);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Positions>& positions_in,
            Distances& distances_out,
            Gradients& gradients_out,
            Variances& variances_out,
            Covariances& covariances_out);

        [[nodiscard]] const std::vector<std::array<std::shared_ptr<SdfGp>, (Dim - 1) * 2>>&
        GetUsedGps() const {
            return m_query_used_gps_;
        }

        [[nodiscard]] const KeyGpMap&
        GetGpMap() const {
            return m_gp_map_;
        }

        [[nodiscard]] bool
        Write(std::ostream& s) const;

        [[nodiscard]] bool
        Read(std::istream& s);

        [[nodiscard]] bool
        operator==(const GpSdfMapping& other) const;

    private:
        void
        CollectChangedClusters();

        void
        UpdateClusterQueue();

        void
        TrainGps();

        void
        TrainGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        SearchCandidateGps(const Eigen::Ref<const Positions>& positions_in);

        void
        SearchGpThread(
            uint32_t thread_idx,
            std::size_t start_idx,
            std::size_t end_idx,
            std::vector<std::size_t>& no_gps_indices);

        void
        SearchGpFallback(const std::vector<std::size_t>& no_gps_indices);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        template<int D>
        std::enable_if_t<D == 3, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<std::pair<long, long>>& tested_idx,
            const Eigen::Matrix<Dtype, 7, Eigen::Dynamic>& fs,
            const Variances& variances,
            const Covariances& covariances);

        template<int D>
        std::enable_if_t<D == 2, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<std::pair<long, long>>& tested_idx,
            const Eigen::Matrix<Dtype, 5, Eigen::Dynamic>& fs,
            const Variances& variances,
            const Covariances& covariances);
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
