#pragma once

#include "gp_sdf_mapping_base_setting.hpp"
#include "sdf_gp.hpp"
#include "surface_data_manager.hpp"

#include "erl_geometry/aabb.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <boost/heap/d_ary_heap.hpp>

#include <vector>

namespace erl::sdf_mapping {

    template<int Dim, typename Dtype, typename SurfaceMapping>
    class GpSdfMapping {
    public:
        using SurfaceDataManager = SurfaceDataManager<Dim>;
        using SurfaceData = typename SurfaceDataManager::SurfaceData;
        using Gp = SdfGaussianProcess<Dim, SurfaceData>;
        using Key = typename SurfaceMapping::Key;
        using KeySet = absl::flat_hash_set<Key>;
        using KeyVector = std::vector<Key>;
        using Setting = GpSdfMappingBaseSetting;
        using KdTree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Matrix<Dtype, Dim, 1>;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Gradient = Eigen::Vector<Dtype, Dim>;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using Distances = Eigen::VectorX<Dtype>;
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
        using KeyGpMap = absl::flat_hash_map<Key, std::shared_ptr<Gp>>;
        using KeyGpPair = std::pair<Key, std::shared_ptr<Gp>>;

        using Aabb = geometry::Aabb<Dtype, Dim>;

        struct TestBuffer {

            std::unique_ptr<Eigen::Ref<const Positions>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Distances>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Gradients>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Variances>> variances = nullptr;      // var(d, grad.x, grad.y, grad.z)
            std::unique_ptr<Eigen::Ref<Covariances>> covariances = nullptr;  // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)

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
            DisconnectBuffers() {
                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                variances = nullptr;
                covariances = nullptr;
            }
        };

        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<SurfaceMapping> m_surface_mapping_ = nullptr;                // for getting surface points, racing condition.
        KeyGpMap m_gp_map_ = {};                                                     // for getting GP from Octree key, racing condition.
        KeyVector m_affected_clusters_ = {};                                         // stores clusters that are to be updated after a new observation
        KeyQueueMap m_cluster_queue_keys_ = {};                                      // caching keys of clusters in the queue to be updated
        PriorityQueue m_cluster_queue_;                                              // ordering clusters, smaller time_stamp first (FIFO)
        std::vector<KeyGpPair> m_clusters_to_train_ = {};                            // stores clusters that are to be trained
        std::vector<KeyGpPair> m_candidate_gps_ = {};                                // for testing
        std::shared_ptr<KdTree> m_kd_tree_candidate_gps_ = nullptr;                  // for testing to search candidate GPs
        std::vector<std::vector<std::pair<Dtype, KeyGpPair>>> m_query_to_gps_ = {};  // for testing
        Eigen::VectorX<Dtype> m_query_signs_ = {};                                   // sign of query positions
        double m_train_gp_time_us_ = 10;                                             // time spent for training GPs
        TestBuffer m_test_buffer_ = {};

#if defined(ERL_GP_SDF_MAPPING_TRACK_QUERY_GPS)
        std::vector<std::vector<std::shared_ptr<Gp>>> m_query_used_gps_ = {};  // for testing
#endif

    public:
        GpSdfMapping(std::shared_ptr<Setting> setting, std::shared_ptr<SurfaceMapping> surface_mapping);

        bool
        Update(
            const Eigen::Ref<const Rotation>& rotation,
            const Eigen::Ref<const Translation>& translation,
            const Eigen::Ref<const Eigen::MatrixX<Dtype>>& ranges);

        bool
        UpdateGpSdf(double time_budget_us = 0);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const Positions>& positions, Distances& distances_out) {
            Gradients gradients;
            Variances variances;
            Covariances covariances;
            return Test(positions, distances_out, gradients, variances, covariances);
        }

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Positions>& positions_in,
            Distances& distances_out,
            Gradients& gradients_out,
            Variances& variances_out,
            Covariances& covariances_out);

#if defined(ERL_GP_SDF_MAPPING_TRACK_QUERY_GPS)
        [[nodiscard]] const std::vector<std::vector<std::shared_ptr<Gp>>>&
        GetUsedGps() const {
            return m_query_used_gps_;
        }
#endif

        [[nodiscard]] const KeyGpMap&
        GetGpMap() const {
            return m_gp_map_;
        }

    private:
        void
        TrainGps();

        void
        TrainGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        SearchCandidateGps(const Eigen::Ref<const Positions>& positions_in);

        void
        SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        template<int D>
        std::enable_if_t<D == 3, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<std::pair<long, long>>& tested_idx,
            Eigen::Matrix<Dtype, 4, Eigen::Dynamic>& fs,
            Variances& variances,
            Covariances& covariances);

        template<int D>
        std::enable_if_t<D == 2, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<std::pair<long, long>>& tested_idx,
            Eigen::Matrix<Dtype, 3, Eigen::Dynamic>& fs,
            Variances& variances,
            Covariances& covariances);
    };

#include "gp_sdf_mapping.tpp"

}  // namespace erl::sdf_mapping
