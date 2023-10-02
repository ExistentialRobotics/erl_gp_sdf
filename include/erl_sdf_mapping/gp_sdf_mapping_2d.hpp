#pragma once

#include <memory>
#include <queue>
#include "erl_common/yaml.hpp"
#include "abstract_surface_mapping_2d.hpp"
#include "log_sdf_gp.hpp"

namespace erl::sdf_mapping {

    class GpSdfMapping2D {

    public:
        struct Setting : public common::Yamlable<Setting> {
            struct TestQuery : public common::Yamlable<TestQuery> {
                double max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
                double search_area_half_size = 4.8;
                bool use_nearest_only = false;  // if true, only the nearest point will be used for prediction.
                bool compute_covariance = false;
            };

            unsigned int num_threads = 64;
            double update_hz = 20;                   // frequency that Update() is called.
            double gp_sdf_area_scale = 4;            // ratio between GP area and Quadtree cluster area
            double offset_distance = 0.0;            // offset distance for surface points
            double zero_gradient_threshold = 1.e-6;  // gradient below this threshold is considered zero.
            double max_valid_gradient_var = 0.1;     // maximum gradient variance qualified for training.
            double invalid_position_var = 2.;        // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
            bool train_gp_immediately = false;
            std::shared_ptr<LogSdfGaussianProcess::Setting> gp_sdf = std::make_shared<LogSdfGaussianProcess::Setting>();
            std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by Test.
        };

        struct GP {
            bool active = false;
            std::atomic_bool locked_for_test = false;
            long num_train_samples = 0;
            long num_train_samples_with_grad = 0;
            std::shared_ptr<LogSdfGaussianProcess> gp = {};

            inline void
            Train() {
                gp->Reset(num_train_samples, 2);
                gp->Train(num_train_samples, num_train_samples_with_grad);
            }
        };

        using QuadtreeKeyGpMap = std::unordered_map<geometry::QuadtreeKey, std::shared_ptr<GP>, geometry::QuadtreeKey::KeyHash>;

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::mutex m_mutex_;
        std::shared_ptr<AbstractSurfaceMapping2D> m_surface_mapping_ = nullptr;  // for getting surface points, racing condition.
        std::vector<geometry::QuadtreeKey> m_clusters_to_update_ = {};           // stores clusters that are to be updated by UpdateGpThread().
        QuadtreeKeyGpMap m_gp_map_ = {};                                                        // for getting GP from Quadtree key, racing condition.
        std::vector<std::vector<std::pair<double, std::shared_ptr<GP>>>> m_query_to_gps_ = {};  // for testing, racing condition
        std::queue<std::shared_ptr<GP>> m_new_gps_ = {};        // caching new GPs to be moved into m_gps_to_train_
        std::vector<std::shared_ptr<GP>> m_gps_to_train_ = {};  // for training SDF GPs, racing condition in Update() and Test()
        double m_train_gp_time_ = 10;  // us

        // for testing
        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const Eigen::Matrix2Xd>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::VectorXd>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix2Xd>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix3Xd>> variances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix3Xd>> covariances = nullptr;

            [[nodiscard]] inline std::size_t
            Size() const {
                if (positions == nullptr) return 0;
                return positions->cols();
            }

            inline bool
            ConnectBuffers(
                const Eigen::Ref<const Eigen::Matrix2Xd>& positions_in,
                Eigen::VectorXd& distances_out,
                Eigen::Matrix2Xd& gradients_out,
                Eigen::Matrix3Xd& variances_out,
                Eigen::Matrix3Xd& covariances_out,
                bool compute_covariance) {
                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                variances = nullptr;
                covariances = nullptr;
                long n = positions_in.cols();
                if (n == 0) return false;
                distances_out.resize(n);
                gradients_out.resize(2, n);
                variances_out.resize(3, n);
                if (compute_covariance) { covariances_out.resize(3, n); }
                this->positions = std::make_unique<Eigen::Ref<const Eigen::Matrix2Xd>>(positions_in);
                this->distances = std::make_unique<Eigen::Ref<Eigen::VectorXd>>(distances_out);
                this->gradients = std::make_unique<Eigen::Ref<Eigen::Matrix2Xd>>(gradients_out);
                this->variances = std::make_unique<Eigen::Ref<Eigen::Matrix3Xd>>(variances_out);
                this->covariances = std::make_unique<Eigen::Ref<Eigen::Matrix3Xd>>(covariances_out);
                return true;
            }

            inline void
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
        explicit GpSdfMapping2D(std::shared_ptr<AbstractSurfaceMapping2D> surface_mapping, std::shared_ptr<Setting> setting = nullptr)
            : m_setting_(std::move(setting)),
              m_surface_mapping_(std::move(surface_mapping)) {
            if (m_setting_ == nullptr) { m_setting_ = std::make_shared<Setting>(); }
            ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
        }

        [[nodiscard]] std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

        [[nodiscard]] std::shared_ptr<AbstractSurfaceMapping2D>
        GetSurfaceMapping() const {
            return m_surface_mapping_;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::VectorXd>& angles,
            const Eigen::Ref<const Eigen::VectorXd>& distances,
            const Eigen::Ref<const Eigen::Matrix23d>& pose) {

            double time_budget = 1e6 / m_setting_->update_hz;  // us
            bool success;
            std::chrono::high_resolution_clock::time_point t0, t1;
            long dt;
            {
                std::lock_guard<std::mutex> lock(m_mutex_);
                t0 = std::chrono::high_resolution_clock::now();
                success = m_surface_mapping_->Update(angles, distances, pose);
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
                ERL_INFO("Surface mapping update time: %ld us.", dt);
            }
            time_budget -= double(dt);

            if (success) {
                std::lock_guard<std::mutex> lock(m_mutex_);
                t0 = std::chrono::high_resolution_clock::now();
                UpdateGps(time_budget);
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                ERL_INFO("GP update time: %ld ms.", dt);
                return true;
            }
            return false;
        }

        bool
        Test(
            const Eigen::Ref<const Eigen::Matrix2Xd>& positions_in,
            Eigen::VectorXd& distances_out,
            Eigen::Matrix2Xd& gradients_out,
            Eigen::Matrix3Xd& variances_out,
            Eigen::Matrix3Xd& covariances_out);

    private:
        void
        UpdateGps(double time_budget);

        void
        UpdateGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TrainGps() {
            ERL_INFO("Training %zu GPs ...", m_gps_to_train_.size());
            auto t0 = std::chrono::high_resolution_clock::now();
            unsigned int n = m_gps_to_train_.size();
            if (n == 0) return;
            unsigned int num_threads = std::min(n, std::thread::hardware_concurrency());
            num_threads = std::min(num_threads, m_setting_->num_threads);
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            std::size_t batch_size = n / num_threads;
            std::size_t leftover = n - batch_size * num_threads;
            std::size_t start_idx = 0, end_idx;
            for (unsigned int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                end_idx = start_idx + batch_size;
                if (thread_idx < leftover) { end_idx++; }
                threads.emplace_back(&GpSdfMapping2D::TrainGpThread, this, thread_idx, start_idx, end_idx);
                start_idx = end_idx;
            }
            for (auto& thread: threads) { thread.join(); }
            m_gps_to_train_.clear();
            auto t1 = std::chrono::high_resolution_clock::now();
            double time = double(std::chrono::duration<double, std::micro>(t1 - t0).count()) / double(n);
            m_train_gp_time_ = m_train_gp_time_ * 0.4 + time * 0.6;
            ERL_INFO("Per GP training time: %f us.", m_train_gp_time_);
        }

        void
        TrainGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        SearchGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TestGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx);
    };
}  // namespace erl::sdf_mapping

namespace YAML {

    using namespace erl::sdf_mapping;

    template<>
    struct convert<GpSdfMapping2D::Setting::TestQuery> {

        inline static Node
        encode(const GpSdfMapping2D::Setting::TestQuery& rhs) {
            Node node;
            node["max_test_valid_distance_var"] = rhs.max_test_valid_distance_var;
            node["search_area_half_size"] = rhs.search_area_half_size;
            node["use_nearest_only"] = rhs.use_nearest_only;
            node["compute_covariance"] = rhs.compute_covariance;
            return node;
        }

        inline static bool
        decode(const Node& node, GpSdfMapping2D::Setting::TestQuery& rhs) {
            if (!node.IsMap()) { return false; }
            rhs.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<double>();
            rhs.search_area_half_size = node["search_area_half_size"].as<double>();
            rhs.use_nearest_only = node["use_nearest_only"].as<bool>();
            rhs.compute_covariance = node["compute_covariance"].as<bool>();
            return true;
        }
    };

    inline Emitter&
    operator<<(Emitter& out, const GpSdfMapping2D::Setting::TestQuery& rhs) {
        out << BeginMap;
        out << Key << "max_test_valid_distance_var" << Value << rhs.max_test_valid_distance_var;
        out << Key << "search_area_half_size" << Value << rhs.search_area_half_size;
        out << Key << "use_nearest_only" << Value << rhs.use_nearest_only;
        out << Key << "compute_covariance" << Value << rhs.compute_covariance;
        out << EndMap;
        return out;
    }

    template<>
    struct convert<GpSdfMapping2D::Setting> {
        static Node
        encode(const GpSdfMapping2D::Setting& setting) {
            Node node;
            node["num_threads"] = setting.num_threads;
            node["update_hz"] = setting.update_hz;
            node["gp_sdf_area_scale"] = setting.gp_sdf_area_scale;
            node["offset_distance"] = setting.offset_distance;
            node["zero_gradient_threshold"] = setting.zero_gradient_threshold;
            node["max_valid_gradient_var"] = setting.max_valid_gradient_var;
            node["invalid_position_var"] = setting.invalid_position_var;
            node["train_gp_immediately"] = setting.train_gp_immediately;
            node["gp_sdf"] = setting.gp_sdf;
            node["test_query"] = setting.test_query;
            return node;
        }

        static bool
        decode(const Node& node, GpSdfMapping2D::Setting& setting) {
            if (!node.IsMap()) { return false; }
            setting.num_threads = node["num_threads"].as<unsigned int>();
            setting.update_hz = node["update_hz"].as<double>();
            setting.gp_sdf_area_scale = node["gp_sdf_area_scale"].as<double>();
            setting.offset_distance = node["offset_distance"].as<double>();
            setting.zero_gradient_threshold = node["zero_gradient_threshold"].as<double>();
            setting.max_valid_gradient_var = node["max_valid_gradient_var"].as<double>();
            setting.invalid_position_var = node["invalid_position_var"].as<double>();
            setting.train_gp_immediately = node["train_gp_immediately"].as<bool>();
            setting.gp_sdf = node["gp_sdf"].as<decltype(setting.gp_sdf)>();
            setting.test_query = node["test_query"].as<decltype(setting.test_query)>();
            return true;
        }
    };

    inline Emitter&
    operator<<(Emitter& out, const GpSdfMapping2D::Setting& setting) {
        out << BeginMap;
        out << Key << "num_threads" << Value << setting.num_threads;
        out << Key << "update_hz" << Value << setting.update_hz;
        out << Key << "gp_sdf_area_scale" << Value << setting.gp_sdf_area_scale;
        out << Key << "offset_distance" << Value << setting.offset_distance;
        out << Key << "zero_gradient_threshold" << Value << setting.zero_gradient_threshold;
        out << Key << "max_valid_gradient_var" << Value << setting.max_valid_gradient_var;
        out << Key << "invalid_position_var" << Value << setting.invalid_position_var;
        out << Key << "train_gp_immediately" << Value << setting.train_gp_immediately;
        out << Key << "gp_sdf" << Value << setting.gp_sdf;
        out << Key << "test_query" << Value << setting.test_query;
        out << EndMap;
        return out;
    }

}  // namespace YAML
