#pragma once

#include <memory>
#include "erl_common/yaml.hpp"
#include "abstract_surface_mapping_2d.hpp"
#include "erl_gaussian_process/log_noisy_input_gp.hpp"

namespace erl::sdf_mapping {

    class GpSdfMapping2D {

    public:
        using GpSdf = gaussian_process::LogNoisyInputGaussianProcess;

        struct Setting : public common::Yamlable<Setting> {
            struct TestQuery : public common::Yamlable<TestQuery> {
                double max_test_valid_distance_var = 0.4;  // maximum distance variance of prediction.
                double search_area_half_size = 4.8;
                bool use_nearest_only = false;             // if true, only the nearest point will be used for prediction.
            };

            unsigned int num_threads = 64;
            double update_hz = 50;
            double gp_sdf_area_scale = 4;            // ratio between GP area and Quadtree cluster area
            double offset_distance = 0.02;
            double zero_gradient_threshold = 1.e-6;  // gradient below this threshold is considered zero.
            double max_valid_gradient_var = 0.1;     // maximum gradient variance qualified for training.
            double invalid_position_var = 2.;        // position variance of points whose gradient is labeled invalid, i.e. > max_valid_gradient_var.
            bool train_gp_immediately = false;
            std::shared_ptr<GpSdf::Setting> gp_sdf = std::make_shared<GpSdf::Setting>();
            std::shared_ptr<TestQuery> test_query = std::make_shared<TestQuery>();  // parameters used by Test.
        };

        struct GP {
            Eigen::Matrix2Xd mat_x = {};
            Eigen::VectorXd vec_y = {};
            Eigen::VectorXd vec_sigma_x = {};
            Eigen::VectorXd vec_sigma_grad = {};
            Eigen::VectorXb vec_grad_flag = {};
            std::shared_ptr<GpSdf> gp = {};
        };

        using QuadtreeKeyGpMap = std::unordered_map<geometry::QuadtreeKey, std::shared_ptr<GP>, geometry::QuadtreeKey::KeyHash>;

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<AbstractSurfaceMapping2D> m_surface_mapping_ = nullptr;
        std::vector<geometry::QuadtreeKey> m_clusters_to_update_ = {};
        QuadtreeKeyGpMap m_gp_map_ = {};
        std::vector<std::vector<std::pair<double, std::shared_ptr<GP>>>> m_query_to_gps_ = {};
        std::vector<std::shared_ptr<GP>> m_gps_to_train_ = {};
        bool m_might_need_training_ = false;
        double m_train_gp_time_ = 10;  // us

        // for testing
        struct TestBuffer {
            std::unique_ptr<Eigen::Ref<const Eigen::Matrix2Xd>> positions = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::VectorXd>> distances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix2Xd>> gradients = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::VectorXd>> distance_variances = nullptr;
            std::unique_ptr<Eigen::Ref<Eigen::Matrix2Xd>> gradient_variances = nullptr;

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
                Eigen::VectorXd& distance_variances_out,
                Eigen::Matrix2Xd& gradient_variances_out) {
                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                distance_variances = nullptr;
                gradient_variances = nullptr;
                long n = positions_in.cols();
                if (n == 0) return false;
                distances_out.resize(n);
                gradients_out.resize(2, n);
                distance_variances_out.resize(n);
                gradient_variances_out.resize(2, n);
                this->positions = std::make_unique<Eigen::Ref<const Eigen::Matrix2Xd>>(positions_in);
                this->distances = std::make_unique<Eigen::Ref<Eigen::VectorXd>>(distances_out);
                this->gradients = std::make_unique<Eigen::Ref<Eigen::Matrix2Xd>>(gradients_out);
                this->distance_variances = std::make_unique<Eigen::Ref<Eigen::VectorXd>>(distance_variances_out);
                this->gradient_variances = std::make_unique<Eigen::Ref<Eigen::Matrix2Xd>>(gradient_variances_out);
                return true;
            }

            inline void
            DisconnectBuffers() {
                positions = nullptr;
                distances = nullptr;
                gradients = nullptr;
                distance_variances = nullptr;
                gradient_variances = nullptr;
            }
        };

        TestBuffer m_test_buffer_ = {};

    public:
        explicit GpSdfMapping2D(std::shared_ptr<AbstractSurfaceMapping2D> surface_mapping)
            : m_surface_mapping_(std::move(surface_mapping)) {
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

        std::shared_ptr<GP>
        GetGp(const geometry::QuadtreeKey& key) {
            auto it = m_gp_map_.find(key);
            if (it == m_gp_map_.end()) return nullptr;
            return it->second;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::VectorXd>& angles,
            const Eigen::Ref<const Eigen::VectorXd>& distances,
            const Eigen::Ref<const Eigen::Matrix23d>& pose) {

            double time_budget = 1e6 / m_setting_->update_hz;  // us
            auto t0 = std::chrono::high_resolution_clock::now();
            bool success = m_surface_mapping_->Update(angles, distances, pose);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            ERL_INFO("Surface mapping update time: %ld us.\n", dt);
            time_budget -= dt;
            if (success) {
                t0 = std::chrono::high_resolution_clock::now();
                UpdateGps(time_budget);
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                ERL_INFO("GP update time: %ld ms.\n", dt);
                return true;
            }
            return false;
        }

        bool
        Test(
            const Eigen::Ref<const Eigen::Matrix2Xd>& positions_in,
            Eigen::VectorXd& distances_out,
            Eigen::Matrix2Xd& gradients_out,
            Eigen::VectorXd& distance_variances_out,
            Eigen::Matrix2Xd& gradient_variances_out);

    private:
        void
        UpdateGps(double time_budget);

        void
        UpdateGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx);

        void
        TrainGps() {
            ERL_INFO("Training %zu GPs ...\n", m_gps_to_train_.size());
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
            double time = double(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / double(n);
            m_train_gp_time_ = m_train_gp_time_ * 0.4 + time * 0.6;
            ERL_INFO("Per GP training time: %f us.\n", m_train_gp_time_);
        }

        void
        TrainGp(const std::shared_ptr<GP>& gp);

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
            return node;
        }

        inline static bool
        decode(const Node& node, GpSdfMapping2D::Setting::TestQuery& rhs) {
            if (!node.IsMap()) { return false; }
            rhs.max_test_valid_distance_var = node["max_test_valid_distance_var"].as<double>();
            rhs.search_area_half_size = node["search_area_half_size"].as<double>();
            rhs.use_nearest_only = node["use_nearest_only"].as<bool>();
            return true;
        }
    };

    inline Emitter&
    operator<<(Emitter& out, const GpSdfMapping2D::Setting::TestQuery& rhs) {
        out << YAML::BeginMap;
        out << YAML::Key << "max_test_valid_distance_var" << YAML::Value << rhs.max_test_valid_distance_var;
        out << YAML::Key << "search_area_half_size" << YAML::Value << rhs.search_area_half_size;
        out << YAML::Key << "use_nearest_only" << YAML::Value << rhs.use_nearest_only;
        out << YAML::EndMap;
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
        out << YAML::BeginMap;
        out << YAML::Key << "num_threads" << YAML::Value << setting.num_threads;
        out << YAML::Key << "update_hz" << YAML::Value << setting.update_hz;
        out << YAML::Key << "gp_sdf_area_scale" << YAML::Value << setting.gp_sdf_area_scale;
        out << YAML::Key << "offset_distance" << YAML::Value << setting.offset_distance;
        out << YAML::Key << "zero_gradient_threshold" << YAML::Value << setting.zero_gradient_threshold;
        out << YAML::Key << "max_valid_gradient_var" << YAML::Value << setting.max_valid_gradient_var;
        out << YAML::Key << "invalid_position_var" << YAML::Value << setting.invalid_position_var;
        out << YAML::Key << "train_gp_immediately" << YAML::Value << setting.train_gp_immediately;
        out << YAML::Key << "gp_sdf" << YAML::Value << setting.gp_sdf;
        out << YAML::Key << "test_query" << YAML::Value << setting.test_query;
        out << YAML::EndMap;
        return out;
    }

}  // namespace YAML
