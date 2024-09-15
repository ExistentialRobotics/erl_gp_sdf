#pragma once

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <utility>

namespace erl::sdf_mapping {

    class LogEdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess {

    public:
        struct Setting : public common::Yamlable<Setting, NoisyInputGaussianProcess::Setting> {
            double log_lambda = 40.0;
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    public:
        explicit LogEdfGaussianProcess(std::shared_ptr<Setting> setting)
            : NoisyInputGaussianProcess([setting]() -> std::shared_ptr<Setting> {
                  setting->kernel->scale = std::sqrt(3.) / setting->log_lambda;
                  setting->no_gradient_observation = true;
                  return setting;
              }()),
              m_setting_(std::move(setting)) {}

        LogEdfGaussianProcess(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess(LogEdfGaussianProcess &&other) = default;
        LogEdfGaussianProcess &
        operator=(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess &
        operator=(LogEdfGaussianProcess &&other) = default;

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
        operator==(const LogEdfGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const LogEdfGaussianProcess &other) const {
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
    };
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::LogEdfGaussianProcess::Setting> {
    static Node
    encode(const erl::sdf_mapping::LogEdfGaussianProcess::Setting &setting) {
        Node node = convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::encode(setting);
        node["log_lambda"] = setting.log_lambda;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::LogEdfGaussianProcess::Setting &setting) {
        if (!node.IsMap()) { return false; }
        convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::decode(node, setting);
        setting.log_lambda = node["log_lambda"].as<double>();
        return true;
    }
};  // namespace YAML

// ReSharper restore CppInconsistentNaming
