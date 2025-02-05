#pragma once

#include "surface_data_manager.hpp"

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

        using SurfaceData2D = SurfaceDataManager<2>::SurfaceData;
        using SurfaceData3D = SurfaceDataManager<3>::SurfaceData;

        template<int Dim>
        long
        LoadSurfaceData(
            std::vector<std::pair<double, std::size_t>> &surface_data_indices,
            const std::vector<typename SurfaceDataManager<Dim>::SurfaceData> &surface_data_vec,
            const Eigen::Vector<double, Dim> &coord_origin,
            const double offset_distance,
            const double sensor_noise,
            const double max_valid_gradient_var,
            const double invalid_position_var) {

            SetKernelCoordOrigin(coord_origin);

            const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
            Reset(max_num_samples, Dim);

            std::sort(surface_data_indices.begin(), surface_data_indices.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

            long count = 0;
            for (auto &[distance, surface_data_index]: surface_data_indices) {
                auto &surface_data = surface_data_vec[surface_data_index];
                m_mat_x_train_.col(count) = surface_data.position - offset_distance * surface_data.normal;
                m_vec_var_h_[count] = sensor_noise;
                m_vec_var_x_[count] = surface_data.var_position;
                if ((surface_data.var_normal > max_valid_gradient_var) ||                       // invalid gradient
                    (surface_data.normal.norm() < 0.9)) {                                       // invalid normal
                    m_vec_var_x_[count] = std::max(m_vec_var_x_[count], invalid_position_var);  // position is unreliable
                }
                if (++count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples
            }
            m_num_train_samples_ = count;
            m_vec_y_train_.setOnes(m_num_train_samples_);
            if (m_reduced_rank_kernel_) { UpdateKtrain(m_num_train_samples_); }
            return count;
        }

        // long
        // LoadSurfaceData(
        //     std::vector<std::pair<double, std::size_t>> &surface_data_indices,
        //     const std::vector<SurfaceDataManager<3>::SurfaceData> &surface_data_vec,
        //     const Eigen::Vector3d &coord_origin,
        //     double offset_distance,
        //     double sensor_noise,
        //     double max_valid_gradient_var,
        //     double invalid_position_var);

        [[nodiscard]] bool
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
