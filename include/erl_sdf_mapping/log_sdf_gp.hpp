#pragma once

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <utility>

namespace erl::sdf_mapping {

    class LogSdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess {

    public:
        struct Setting : public common::Yamlable<Setting, NoisyInputGaussianProcess::Setting> {
            double log_lambda = 40.0;
            double edf_threshold = 0.1;
            bool unify_scale = true;  // make gpis and log_sdf_gp have the same scale
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<covariance::Covariance> m_kernel_ = nullptr;
        Eigen::MatrixXd m_mat_log_k_train_ = {};
        Eigen::MatrixXd m_mat_log_l_ = {};
        Eigen::VectorXd m_vec_log_alpha_ = {};

    public:
        explicit LogSdfGaussianProcess(std::shared_ptr<Setting> setting)
            : NoisyInputGaussianProcess(setting),
              m_setting_(std::move(setting)) {
            if (!m_trained_) { AllocateMemory2(m_setting_->max_num_samples, m_setting_->kernel->x_dim); }
        }

        void
        Reset(long max_num_samples, long x_dim) override;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const override;

        void
        Train(long num_train_samples) override;

        void
        Test(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
            Eigen::Ref<Eigen::MatrixXd> mat_f_out,
            Eigen::Ref<Eigen::MatrixXd> mat_var_out,
            Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const override;

        [[nodiscard]] bool
        operator==(const LogSdfGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const LogSdfGaussianProcess &other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(const std::string &filename) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(const std::string &filename) override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

    protected:
        bool
        AllocateMemory2(long max_num_samples, long x_dim);
    };
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::LogSdfGaussianProcess::Setting> {
    static Node
    encode(const erl::sdf_mapping::LogSdfGaussianProcess::Setting &setting) {
        Node node = convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::encode(setting);
        node["log_lambda"] = setting.log_lambda;
        node["edf_threshold"] = setting.edf_threshold;
        node["unify_scale"] = setting.unify_scale;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::LogSdfGaussianProcess::Setting &setting) {
        if (!node.IsMap()) { return false; }
        convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::decode(node, setting);
        setting.log_lambda = node["log_lambda"].as<double>();
        setting.edf_threshold = node["edf_threshold"].as<double>();
        setting.unify_scale = node["unify_scale"].as<bool>();
        return true;
    }
};  // namespace YAML

// ReSharper restore CppInconsistentNaming
