#pragma once

#include "abstract_surface_mapping_3d.hpp"
#include "gp_sdf_mapping_base_setting.hpp"
#include "sdf_gp.hpp"

#include "erl_common/yaml.hpp"
#include "erl_geometry/kdtree_eigen_adaptor.hpp"

#include <boost/heap/d_ary_heap.hpp>

#include <memory>
#include <vector>

namespace erl::sdf_mapping {

    template<typename Dtype>
    class GpSdfMapping3D {

    public:
        struct Setting : common::Yamlable<Setting, GpSdfMappingBaseSetting<Dtype>> {
            std::string surface_mapping_type = "erl::sdf_mapping::GpOccSurfaceMapping3D";
            std::string surface_mapping_setting_type = "erl::sdf_mapping::GpOccSurfaceMapping3D::Setting";
            std::shared_ptr<typename AbstractSurfaceMapping3D<Dtype>::Setting> surface_mapping = nullptr;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting& setting);

                static bool
                decode(const YAML::Node& node, Setting& setting);
            };
        };

        using SurfaceDataType = typename SurfaceDataManager<Dtype, 3>::Data;
        using Gp = SdfGaussianProcess<Dtype, 3, SurfaceDataType>;
        using Matrix = Eigen::MatrixX<Dtype>;
        using Matrix3 = Eigen::Matrix3<Dtype>;
        using Matrix4 = Eigen::Matrix4<Dtype>;
        using Matrix3X = Eigen::Matrix3X<Dtype>;
        using Matrix4X = Eigen::Matrix4X<Dtype>;
        using Matrix6X = Eigen::Matrix<Dtype, 6, Eigen::Dynamic>;
        using Vector = Eigen::VectorX<Dtype>;
        using Vector3 = Eigen::Vector3<Dtype>;
        using Vector4 = Eigen::Vector4<Dtype>;
        using Vector6 = Eigen::Vector<Dtype, 6>;
        using Aabb = geometry::Aabb<Dtype, 3>;
        using Kdtree = geometry::KdTreeEigenAdaptor<Dtype, 3>;

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
        using KeyQueueMap = std::unordered_map<Key, typename PriorityQueue::handle_type, Key::KeyHash>;
        using KeyGpMap = std::unordered_map<Key, std::shared_ptr<Gp>, Key::KeyHash>;
        using KeyGpPair = std::pair<Key, std::shared_ptr<Gp>>;

        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<AbstractSurfaceMapping3D<Dtype>> m_surface_mapping_ = nullptr;  // for getting surface points, racing condition.
        KeyGpMap m_gp_map_ = {};                                                        // for getting GP from Octree key, racing condition.
        KeyVector m_affected_clusters_ = {};                                            // stores clusters that are to be updated after a new observation
        KeyQueueMap m_cluster_queue_keys_ = {};                                         // caching keys of clusters in the queue to be updated
        PriorityQueue m_cluster_queue_;                                                 // ordering clusters, smaller time_stamp first (FIFO)
        std::vector<KeyGpPair> m_clusters_to_train_ = {};                               // stores clusters that are to be trained
        std::vector<KeyGpPair> m_candidate_gps_ = {};                                   // for testing
        std::shared_ptr<Kdtree> m_kd_tree_candidate_gps_ = nullptr;                     // for testing to search candidate GPs
        std::vector<std::vector<std::pair<Dtype, KeyGpPair>>> m_query_to_gps_ = {};     // for testing
        std::vector<std::array<std::shared_ptr<Gp>, 4>> m_query_used_gps_ = {};         // for testing
        Vector m_query_signs_ = {};                                                     // sign of query positions
        double m_train_gp_time_ = 10;                                                   // us

        // for testing
        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const Matrix3X>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Vector>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Matrix3X>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Matrix4X>> variances = nullptr;    // var(d, grad.x, grad.y, grad.z)
            std::unique_ptr<Eigen::Ref<Matrix6X>> covariances = nullptr;  // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)

            [[nodiscard]] std::size_t
            Size() const {
                if (positions == nullptr) return 0;
                return positions->cols();
            }

            bool
            ConnectBuffers(
                const Eigen::Ref<const Matrix3X>& positions_in,
                Vector& distances_out,
                Matrix3X& gradients_out,
                Matrix4X& variances_out,
                Matrix6X& covariances_out,
                bool compute_covariance);

            void
            DisconnectBuffers();
        };

        TestBuffer m_test_buffer_ = {};

    public:
        explicit GpSdfMapping3D(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<AbstractSurfaceMapping3D<Dtype>>
        GetSurfaceMapping() const;

        [[nodiscard]] bool
        Update(const Eigen::Ref<const Matrix3>& rotation, const Eigen::Ref<const Vector3>& translation, const Eigen::Ref<const Matrix>& ranges);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const Matrix3X>& positions, Vector& distances_out) {
            Matrix3X gradients;
            Matrix4X variances;
            Matrix6X covariances;
            return Test(positions, distances_out, gradients, variances, covariances);
        }

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Matrix3X>& positions_in,
            Vector& distances_out,
            Matrix3X& gradients_out,
            Matrix4X& variances_out,
            Matrix6X& covariances_out);

        [[nodiscard]] const std::vector<std::array<std::shared_ptr<Gp>, 4>>&
        GetUsedGps() const;

        [[nodiscard]] const KeyGpMap&
        GetGpMap() const;

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
        SearchCandidateGps(const Eigen::Ref<const Matrix3X>& positions_in);

        void
        SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);
    };

    using GpSdfMapping3Dd = GpSdfMapping3D<double>;
    using GpSdfMapping3Df = GpSdfMapping3D<float>;
}  // namespace erl::sdf_mapping

#include "gp_sdf_mapping_3d.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMapping3Dd::Setting> : erl::sdf_mapping::GpSdfMapping3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpSdfMapping3Df::Setting> : erl::sdf_mapping::GpSdfMapping3Df::Setting::YamlConvertImpl {};
