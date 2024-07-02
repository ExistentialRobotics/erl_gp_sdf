#pragma once

#include "gp_sdf_mapping_setting.hpp"
#include "log_sdf_gp.hpp"

#include "erl_common/yaml.hpp"
#include "erl_geometry/abstract_surface_mapping_3d.hpp"

#include <memory>
#include <queue>

namespace erl::sdf_mapping {

    class GpSdfMapping3D {

    public:
        struct Gp {
            bool active = false;
            std::atomic_bool locked_for_test = false;
            long num_train_samples = 0;
            Eigen::Vector3d position;
            double half_size = 0;
            std::shared_ptr<LogSdfGaussianProcess> gp = {};

            void
            Train() const {
                gp->Train(num_train_samples);
            }
        };

        using OctreeKeyGpMap = std::unordered_map<geometry::OctreeKey, std::shared_ptr<Gp>, geometry::OctreeKey::KeyHash>;
        using Setting = GpSdfMappingSetting;

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<geometry::AbstractSurfaceMapping3D> m_surface_mapping_ = nullptr;       // for getting surface points, racing condition.
        std::vector<geometry::OctreeKey> m_clusters_to_update_ = {};                            // stores clusters that are to be updated by UpdateGpThread.
        OctreeKeyGpMap m_gp_map_ = {};                                                          // for getting GP from Octree key, racing condition.
        std::vector<std::vector<std::pair<double, std::shared_ptr<Gp>>>> m_query_to_gps_ = {};  // for testing, racing condition
        std::vector<std::array<std::shared_ptr<Gp>, 2>> m_query_used_gps_ = {};                 // for testing, racing condition
        std::list<std::shared_ptr<Gp>> m_new_gps_ = {};                                         // caching new GPs to be moved into m_gps_to_train_
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
                const bool compute_covariance) {

                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                variances = nullptr;
                covariances = nullptr;
                const long n = positions_in.cols();
                if (n == 0) return false;

                distances_out.resize(n);
                gradients_out.resize(3, n);
                variances_out.resize(4, n);
                if (compute_covariance) { covariances_out.resize(6, n); }
                this->positions = std::make_unique<Eigen::Ref<const Eigen::Matrix3Xd>>(positions_in);
                this->distances = std::make_unique<Eigen::Ref<Eigen::VectorXd>>(distances_out);
                this->gradients = std::make_unique<Eigen::Ref<Eigen::Matrix3Xd>>(gradients_out);
                this->variances = std::make_unique<Eigen::Ref<Eigen::Matrix4Xd>>(variances_out);
                this->covariances = std::make_unique<Eigen::Ref<Eigen::Matrix6Xd>>(covariances_out);
                return true;
            }

            void
            DisconnectBuffers() {
                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                variances = nullptr;
                covariances = nullptr;
            }
        };

        TestBuffer m_test_buffer_ = {};

    public:
        explicit GpSdfMapping3D(std::shared_ptr<geometry::AbstractSurfaceMapping3D> surface_mapping, std::shared_ptr<Setting> setting = nullptr)
            : m_setting_(std::move(setting)),
              m_surface_mapping_(std::move(surface_mapping)) {
            if (m_setting_ == nullptr) { m_setting_ = std::make_shared<Setting>(); }
            ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");

            // get log dir from env
            if (m_setting_->log_timing) {
                char* log_dir_env = std::getenv("LOG_DIR");
                const std::filesystem::path log_dir = log_dir_env == nullptr ? std::filesystem::current_path() : std::filesystem::path(log_dir_env);
                const std::filesystem::path train_log_file_name = log_dir / "gp_sdf_mapping_3d_train.csv";
                const std::filesystem::path test_log_file_name = log_dir / "gp_sdf_mapping_3d_test.csv";
                if (std::filesystem::exists(train_log_file_name)) { std::filesystem::remove(train_log_file_name); }
                if (std::filesystem::exists(test_log_file_name)) { std::filesystem::remove(test_log_file_name); }
                m_train_log_file_.open(train_log_file_name);
                m_test_log_file_.open(test_log_file_name);
                ERL_WARN_COND(!m_train_log_file_.is_open(), ("Failed to open " + train_log_file_name.string()).c_str());
                ERL_WARN_COND(!m_test_log_file_.is_open(), ("Failed to open " + test_log_file_name.string()).c_str());
                m_train_log_file_ << "travel_distance,surf_mapping_time(us),gp_data_update_time(us),gp_delay_cnt,"
                                  << "gp_train_time(us),total_gp_update_time(ms),total_update_time(ms)" << std::endl
                                  << std::flush;
                m_test_log_file_ << "travel_distance,gp_search_time(us),gp_train_time(us),gp_test_time(us),total_test_time(ms)" << std::endl << std::flush;
            }
        }

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] std::shared_ptr<geometry::AbstractSurfaceMapping3D>
        GetSurfaceMapping() const {
            return m_surface_mapping_;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::Matrix3d>& rotation,
            const Eigen::Ref<const Eigen::Vector3d>& translation,
            const Eigen::Ref<const Eigen::MatrixXd>& ranges);

        bool
        Test(
            const Eigen::Ref<const Eigen::Matrix3Xd>& positions_in,
            Eigen::VectorXd& distances_out,
            Eigen::Matrix3Xd& gradients_out,
            Eigen::Matrix4Xd& variances_out,
            Eigen::Matrix6Xd& covariances_out);

        const std::vector<std::array<std::shared_ptr<Gp>, 2>>&
        GetUsedGps() const {
            return m_query_used_gps_;
        }

    private:
        void
        UpdateGps(double time_budget);

        void
        UpdateGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TrainGps();

        void
        TrainGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) const;

        void
        SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx);
    };
}  // namespace erl::sdf_mapping
