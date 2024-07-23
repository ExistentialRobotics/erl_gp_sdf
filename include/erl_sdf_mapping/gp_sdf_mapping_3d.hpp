#pragma once

#include "gp_sdf_mapping_base_setting.hpp"
#include "log_sdf_gp.hpp"

#include "erl_common/template_helper.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/abstract_surface_mapping_3d.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <boost/heap/d_ary_heap.hpp>

#include <memory>
#include <vector>

namespace erl::sdf_mapping {

    class GpSdfMapping3D {

    public:
        struct Setting : public common::OverrideYamlable<GpSdfMappingBaseSetting, Setting> {
            std::string surface_mapping_type = "GpOccSurfaceMapping3D";
            std::shared_ptr<geometry::AbstractSurfaceMapping3D::Setting> surface_mapping = nullptr;
        };

        struct Gp {
            bool active = false;
            std::atomic_bool locked_for_test = false;
            long num_train_samples = 0;
            Eigen::Vector3d position{};
            double half_size = 0;
            std::shared_ptr<LogSdfGaussianProcess> gp = {};

            [[nodiscard]] std::size_t
            GetMemoryUsage() const {
                std::size_t memory_usage = sizeof(Gp);
                if (gp != nullptr) { memory_usage += gp->GetMemoryUsage(); }
                return memory_usage;
            }

            void
            Train() const {
                gp->Train(num_train_samples);
            }

            [[nodiscard]] bool
            operator==(const Gp& other) const;

            [[nodiscard]] bool
            operator!=(const Gp& other) const {
                return !(*this == other);
            }

            [[nodiscard]] bool
            Write(std::ostream& s) const;

            [[nodiscard]] bool
            Read(std::istream& s, const std::shared_ptr<LogSdfGaussianProcess::Setting>& setting);
        };

        using OctreeKeyGpMap = std::unordered_map<geometry::OctreeKey, std::shared_ptr<Gp>, geometry::OctreeKey::KeyHash>;
        using SurfaceData = geometry::SurfaceMappingOctreeNode::SurfaceData;

    private:
        struct PriorityQueueItem {
            long time_stamp = 0;
            geometry::OctreeKey key{};
        };

        template<typename T>
        struct Greater {
            [[nodiscard]] bool
            operator()(const T& lhs, const T& rhs) const {
                return lhs.time_stamp > rhs.time_stamp;
            }
        };

        using PriorityQueue = boost::heap::d_ary_heap<
            PriorityQueueItem,
            boost::heap::mutable_<true>,
            boost::heap::stable<true>,
            boost::heap::arity<8>,
            boost::heap::compare<Greater<PriorityQueueItem>>>;
        using OctreeKeyPqMap = std::unordered_map<geometry::OctreeKey, PriorityQueue::handle_type, geometry::OctreeKey::KeyHash>;

        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<geometry::AbstractSurfaceMapping3D> m_surface_mapping_ = nullptr;       // for getting surface points, racing condition.
        std::vector<geometry::OctreeKey> m_clusters_to_update_ = {};                            // stores clusters that are to be updated by UpdateGpThread.
        OctreeKeyGpMap m_gp_map_ = {};                                                          // for getting GP from Octree key, racing condition.
        std::vector<std::pair<geometry::Aabb3D, std::shared_ptr<Gp>>> m_candidate_gps_ = {};    // for testing
        std::shared_ptr<geometry::KdTree3d> m_kd_tree_candidate_gps_ = nullptr;                 // for searching candidate GPs
        std::vector<std::vector<std::pair<double, std::shared_ptr<Gp>>>> m_query_to_gps_ = {};  // for testing, racing condition
        std::vector<std::array<std::shared_ptr<Gp>, 4>> m_query_used_gps_ = {};                 // for testing, racing condition
        OctreeKeyPqMap m_new_gp_keys_ = {};                                                     // caching keys of new GPs to be moved into m_gps_to_train_
        PriorityQueue m_new_gp_queue_;                                                          // ordering new GPs, smaller time_stamp first
        std::vector<std::shared_ptr<Gp>> m_gps_to_train_ = {};                                  // for training SDF GPs, racing condition in Update() and Test()
        double m_train_gp_time_ = 10;                                                           // us
        std::mutex m_log_mutex_;                                                                // for logging
        double m_travel_distance_ = 0;                                                          // for logging
        std::optional<Eigen::Vector3d> m_last_position_ = std::nullopt;                         // for logging
        std::ofstream m_train_log_file_;
        std::ofstream m_test_log_file_;

        // for testing
        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const Eigen::Matrix3Xd>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::VectorXd>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix3Xd>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix4Xd>> variances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix6Xd>> covariances = nullptr;

            [[nodiscard]] std::size_t
            Size() const {
                if (positions == nullptr) return 0;
                return positions->cols();
            }

            bool
            ConnectBuffers(
                const Eigen::Ref<const Eigen::Matrix3Xd>& positions_in,
                Eigen::VectorXd& distances_out,
                Eigen::Matrix3Xd& gradients_out,
                Eigen::Matrix4Xd& variances_out,
                Eigen::Matrix6Xd& covariances_out,
                bool compute_covariance);

            void
            DisconnectBuffers();
        };

        TestBuffer m_test_buffer_ = {};

    public:
        explicit GpSdfMapping3D(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] std::shared_ptr<geometry::AbstractSurfaceMapping3D>
        GetSurfaceMapping() const {
            return m_surface_mapping_;
        }

        [[nodiscard]] bool
        Update(
            const Eigen::Ref<const Eigen::Matrix3d>& rotation,
            const Eigen::Ref<const Eigen::Vector3d>& translation,
            const Eigen::Ref<const Eigen::MatrixXd>& ranges);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Eigen::Matrix3Xd>& positions_in,
            Eigen::VectorXd& distances_out,
            Eigen::Matrix3Xd& gradients_out,
            Eigen::Matrix4Xd& variances_out,
            Eigen::Matrix6Xd& covariances_out);

        [[nodiscard]] const std::vector<std::array<std::shared_ptr<Gp>, 4>>&
        GetUsedGps() const {
            return m_query_used_gps_;
        }

        [[nodiscard]] const OctreeKeyGpMap&
        GetGpMap() const {
            return m_gp_map_;
        }

        [[nodiscard]] bool
        operator==(const GpSdfMapping3D& other) const;

        [[nodiscard]] bool
        operator!=(const GpSdfMapping3D& other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(const std::string& filename) const;

        [[nodiscard]] bool
        Write(std::ostream& s) const;

        [[nodiscard]] bool
        Read(const std::string& filename);

        [[nodiscard]] bool
        Read(std::istream& s);

    private:
        void
        UpdateGps(double time_budget);

        void
        UpdateGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TrainGps();

        void
        SearchCandidateGps(const Eigen::Ref<const Eigen::Matrix3Xd>& positions_in);

        void
        SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);
    };
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMapping3D::Setting> {
    static Node
    encode(const erl::sdf_mapping::GpSdfMapping3D::Setting& rhs) {
        Node node = convert<erl::sdf_mapping::GpSdfMappingBaseSetting>::encode(rhs);
        node["surface_mapping_type"] = rhs.surface_mapping_type;
        node["surface_mapping"] = rhs.surface_mapping->AsYamlNode();
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMapping3D::Setting& rhs) {
        if (!convert<erl::sdf_mapping::GpSdfMappingBaseSetting>::decode(node, rhs)) { return false; }
        rhs.surface_mapping_type = node["surface_mapping_type"].as<std::string>();
        rhs.surface_mapping = erl::geometry::AbstractSurfaceMapping::Setting::Create(rhs.surface_mapping_type);
        if (rhs.surface_mapping == nullptr) {
            ERL_WARN("Failed to decode surface_mapping of type: {}", rhs.surface_mapping_type);
            return false;
        }
        return rhs.surface_mapping->FromYamlNode(node["surface_mapping"]);
    }
};

// ReSharper restore CppInconsistentNaming
