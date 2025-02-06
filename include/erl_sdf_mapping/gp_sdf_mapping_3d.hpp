#pragma once

#include "abstract_surface_mapping_3d.hpp"
#include "gp_sdf_mapping_base_setting.hpp"
#include "sdf_gp.hpp"

#include "erl_common/csv.hpp"
#include "erl_common/template_helper.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <boost/heap/d_ary_heap.hpp>

#include <memory>
#include <vector>

namespace erl::sdf_mapping {

    class GpSdfMapping3D {

    public:
        struct Setting : common::Yamlable<Setting, GpSdfMappingBaseSetting> {
            std::string surface_mapping_type = "erl::sdf_mapping::GpOccSurfaceMapping3D";
            std::string surface_mapping_setting_type = "erl::sdf_mapping::GpOccSurfaceMapping3D::Setting";
            std::shared_ptr<AbstractSurfaceMapping3D::Setting> surface_mapping = nullptr;
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();
        using SurfaceData = SurfaceDataManager<3>::SurfaceData;
        using Gp = SdfGaussianProcess<3, SurfaceData>;

    private:
        template<typename T>
        struct Greater {
            [[nodiscard]] bool
            operator()(const T& lhs, const T& rhs) const {
                return lhs.time_stamp > rhs.time_stamp;
            }
        };

        using Key = geometry::OctreeKey;
        using KeySet = geometry::OctreeKeySet;
        using KeyVector = geometry::OctreeKeyVector;

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
        using KeyQueueMap = std::unordered_map<Key, PriorityQueue::handle_type, Key::KeyHash>;
        using KeyGpMap = std::unordered_map<Key, std::shared_ptr<Gp>, Key::KeyHash>;
        using KeyGpPair = std::pair<Key, std::shared_ptr<Gp>>;

        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<AbstractSurfaceMapping3D> m_surface_mapping_ = nullptr;       // for getting surface points, racing condition.
        KeyGpMap m_gp_map_ = {};                                                      // for getting GP from Octree key, racing condition.
        KeyVector m_affected_clusters_ = {};                                          // stores clusters that are to be updated after a new observation
        KeyQueueMap m_cluster_queue_keys_ = {};                                       // caching keys of clusters in the queue to be updated
        PriorityQueue m_cluster_queue_;                                               // ordering clusters, smaller time_stamp first (FIFO)
        std::vector<KeyGpPair> m_clusters_to_train_ = {};                             // stores clusters that are to be trained
        std::vector<KeyGpPair> m_candidate_gps_ = {};                                 // for testing
        std::shared_ptr<geometry::KdTree3d> m_kd_tree_candidate_gps_ = nullptr;       // for testing to search candidate GPs
        std::vector<std::vector<std::pair<double, KeyGpPair>>> m_query_to_gps_ = {};  // for testing
        std::vector<std::array<std::shared_ptr<Gp>, 4>> m_query_used_gps_ = {};       // for testing
        Eigen::VectorXd m_query_signs_ = {};                                          // sign of query positions
        double m_train_gp_time_ = 10;                                                 // us

        // for testing
        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const Eigen::Matrix3Xd>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::VectorXd>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix3Xd>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix4Xd>> variances = nullptr;    // var(d, grad.x, grad.y, grad.z)
            std::unique_ptr<Eigen::Ref<Eigen::Matrix6Xd>> covariances = nullptr;  // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)

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

        [[nodiscard]] std::shared_ptr<AbstractSurfaceMapping3D>
        GetSurfaceMapping() const {
            return m_surface_mapping_;
        }

        [[nodiscard]] bool
        Update(
            const Eigen::Ref<const Eigen::Matrix3d>& rotation,
            const Eigen::Ref<const Eigen::Vector3d>& translation,
            const Eigen::Ref<const Eigen::MatrixXd>& ranges);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const Eigen::Matrix3Xd>& positions, Eigen::VectorXd& distances_out) {
            Eigen::Matrix3Xd gradients;
            Eigen::Matrix4Xd variances;
            Eigen::Matrix6Xd covariances;
            return Test(positions, distances_out, gradients, variances, covariances);
        }

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

        [[nodiscard]] const KeyGpMap&
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
        UpdateGps(double time_budget_us);

        void
        TrainGps();

        void
        TrainGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

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
        node["surface_mapping_setting_type"] = rhs.surface_mapping_setting_type;
        node["surface_mapping"] = rhs.surface_mapping->AsYamlNode();
        return node;
    }

    static bool
    decode(const Node& node, erl::sdf_mapping::GpSdfMapping3D::Setting& rhs) {
        if (!convert<erl::sdf_mapping::GpSdfMappingBaseSetting>::decode(node, rhs)) { return false; }
        rhs.surface_mapping_type = node["surface_mapping_type"].as<std::string>();
        rhs.surface_mapping_setting_type = node["surface_mapping_setting_type"].as<std::string>();
        using SettingBase = erl::sdf_mapping::AbstractSurfaceMapping3D::Setting;
        rhs.surface_mapping = SettingBase::Create<SettingBase>(rhs.surface_mapping_setting_type);
        if (rhs.surface_mapping == nullptr) {
            ERL_WARN("Failed to decode surface_mapping of type: {}", rhs.surface_mapping_setting_type);
            return false;
        }
        return rhs.surface_mapping->FromYamlNode(node["surface_mapping"]);
    }
};

// ReSharper restore CppInconsistentNaming
