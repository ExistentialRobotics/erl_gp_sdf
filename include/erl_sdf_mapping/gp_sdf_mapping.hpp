#pragma once

#include "gp_sdf_mapping_setting.hpp"
#include "sdf_gp.hpp"
#include "surface_data_manager.hpp"

#include "erl_geometry/aabb.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <boost/heap/d_ary_heap.hpp>

#include <vector>

namespace erl::sdf_mapping {

    class AbstractGpSdfMapping {
        inline static std::unordered_map<std::string, std::string>
            s_surface_mapping_to_sdf_mapping_ = {};

    public:
        using Factory = common::FactoryPattern<
            AbstractGpSdfMapping,                    // Base
            false,                                   // UniquePtr
            false,                                   // RawPtr
            std::shared_ptr<common::YamlableBase>,   // Args: surface_mapping_setting
            std::shared_ptr<common::YamlableBase>>;  // Args: sdf_mapping_setting

        [[nodiscard]] static std::string
        GetSdfMappingId(const std::string& surface_mapping_id) {
            const auto it = s_surface_mapping_to_sdf_mapping_.find(surface_mapping_id);
            if (it == s_surface_mapping_to_sdf_mapping_.end()) {
                ERL_WARN(
                    "Failed to find sdf mapping for surface mapping: {}. Do you register it?",
                    surface_mapping_id);
                return "";
            }
            return it->second;
        }

        static std::shared_ptr<AbstractGpSdfMapping>
        Create(
            const std::string& mapping_type,
            const std::shared_ptr<common::YamlableBase>& surface_mapping_setting,
            const std::shared_ptr<common::YamlableBase>& sdf_mapping_setting) {
            return Factory::GetInstance().Create(
                mapping_type,
                surface_mapping_setting,
                sdf_mapping_setting);
        }

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractGpSdfMapping, Derived>, bool>
        Register(const std::string& mapping_type = "") {
            using SurfaceMapping = typename Derived::SurfaceMappingType;

            s_surface_mapping_to_sdf_mapping_[type_name<SurfaceMapping>()] = mapping_type;

            return Factory::GetInstance().Register<Derived>(
                mapping_type,
                [](const std::shared_ptr<common::YamlableBase>& surface_mapping_setting,
                   const std::shared_ptr<common::YamlableBase>& sdf_mapping_setting)
                    -> std::shared_ptr<AbstractGpSdfMapping> {
                    using SurfaceMappingSetting = typename SurfaceMapping::Setting;

                    auto casted_surface_mapping_setting =
                        std::dynamic_pointer_cast<SurfaceMappingSetting>(surface_mapping_setting);
                    if (casted_surface_mapping_setting == nullptr) {
                        ERL_WARN(
                            "Failed to cast surface_mapping_setting to {}",
                            type_name<SurfaceMappingSetting>());
                        return nullptr;
                    }
                    auto surface_mapping =
                        std::make_shared<SurfaceMapping>(casted_surface_mapping_setting);

                    using SdfMappingSetting = typename Derived::Setting;
                    auto casted_sdf_mapping_setting =
                        std::dynamic_pointer_cast<SdfMappingSetting>(sdf_mapping_setting);
                    if (casted_sdf_mapping_setting == nullptr) {
                        ERL_WARN(
                            "Failed to cast sdf_mapping_setting to {}",
                            type_name<SdfMappingSetting>());
                        return nullptr;
                    }
                    return std::make_shared<Derived>(casted_sdf_mapping_setting, surface_mapping);
                });
        }
    };

    template<typename Dtype, int Dim, typename SurfaceMapping>
    class GpSdfMapping : public AbstractGpSdfMapping {
    public:
        using SurfaceMappingType = SurfaceMapping;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = SurfaceData<Dtype, Dim>;
        using SdfGp = SdfGaussianProcess<Dtype, Dim>;
        using Setting = GpSdfMappingSetting<Dtype, Dim>;
        using Key = typename SurfaceMapping::Key;
        using KeySet = absl::flat_hash_set<Key>;
        using KeyVector = std::vector<Key>;
        using KdTree = geometry::KdTreeEigenAdaptor<Dtype, Dim>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Matrix<Dtype, Dim, 1>;
        using Ranges = MatrixX;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Gradient = Eigen::Vector<Dtype, Dim>;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
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
            // for caching intermediate results used for testing
            MatrixX gp_buffer;  //(num_neighbors * (2 * Dim + 1), num_queries)

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

        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
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

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        /**
         * Call this method to update the surface mapping and then update the GP SDF mapping.
         * You can call this method or call the surface mapping update first and then UpdateGpSdf
         * separately.
         * @param rotation The rotation of the sensor. For 2D, it is a 2x2 matrix. For 3D, it is a
         * 3x3 matrix.
         * @param translation The translation of the sensor. For 2D, it is a 2x1 vector. For 3D, it
         * is a 3x1 vector.
         * @param ranges The observation of ranges assumed organized in the order of ray angles. For
         * 2D, it is a vector of ranges. For 3D, it is a matrix of ranges.
         * @return true if the update is successful.
         */
        bool
        Update(
            const Eigen::Ref<const Rotation>& rotation,
            const Eigen::Ref<const Translation>& translation,
            const Eigen::Ref<const Ranges>& ranges);

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

        [[nodiscard]] bool
        operator!=(const GpSdfMapping& other) const {
            return !(*this == other);
        }

    private:
        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard();

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
            Eigen::Ref<Eigen::Matrix<Dtype, 7, Eigen::Dynamic>> fs,
            Variances& variances,
            Covariances& covariances);

        template<int D>
        std::enable_if_t<D == 2, void>
        ComputeWeightedSum(
            uint32_t i,
            const std::vector<std::pair<long, long>>& tested_idx,
            Eigen::Ref<Eigen::Matrix<Dtype, 5, Eigen::Dynamic>> fs,
            Variances& variances,
            Covariances& covariances);
    };

}  // namespace erl::sdf_mapping

#include "gp_sdf_mapping.tpp"
