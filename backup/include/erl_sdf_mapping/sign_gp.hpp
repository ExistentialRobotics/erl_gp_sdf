#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <memory>

namespace erl::sdf_mapping {

    class SignGaussianProcess : public gaussian_process::NoisyInputGaussianProcess {

    public:
        struct Setting : public common::Yamlable<Setting, NoisyInputGaussianProcess::Setting> {
            double offset_distance = 0.1;
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    public:
        explicit SignGaussianProcess(std::shared_ptr<Setting> setting)
            : NoisyInputGaussianProcess([setting]() -> std::shared_ptr<Setting> {
                  setting->no_gradient_observation = false;
                  return setting;
              }()),
              m_setting_(std::move(setting)) {}

        SignGaussianProcess(const SignGaussianProcess &other) = default;
        SignGaussianProcess(SignGaussianProcess &&other) = default;
        SignGaussianProcess &
        operator=(const SignGaussianProcess &other) = default;
        SignGaussianProcess &
        operator=(SignGaussianProcess &&other) = default;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const override;

        using SurfaceData2D = SurfaceDataManager<2>::SurfaceData;
        using SurfaceData3D = SurfaceDataManager<3>::SurfaceData;

        long
        LoadSurfaceData(
            std::vector<std::pair<double, std::shared_ptr<SurfaceData2D>>> &surface_data_vec,
            const Eigen::Vector2d &coord_origin,
            double sensor_noise,
            double max_valid_gradient_var,
            double invalid_position_var);

        long
        LoadSurfaceData(
            std::vector<std::pair<double, std::shared_ptr<SurfaceData3D>>> &surface_data_vec,
            const Eigen::Vector3d &coord_origin,
            double sensor_noise,
            double max_valid_gradient_var,
            double invalid_position_var);

        [[nodiscard]] bool
        Test(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
            Eigen::Ref<Eigen::MatrixXd> mat_f_out,
            Eigen::Ref<Eigen::MatrixXd> mat_var_out,
            Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const override;

        [[nodiscard]] bool
        operator==(const SignGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const SignGaussianProcess &other) const {
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
struct YAML::convert<erl::sdf_mapping::SignGaussianProcess::Setting> {
    static Node
    encode(const erl::sdf_mapping::SignGaussianProcess::Setting &setting) {
        Node node = convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::encode(setting);
        node["offset_distance"] = setting.offset_distance;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::SignGaussianProcess::Setting &setting) {
        if (!node.IsMap()) { return false; }
        convert<erl::gaussian_process::NoisyInputGaussianProcess::Setting>::decode(node, setting);
        setting.offset_distance = node["offset_distance"].as<double>();
        return true;
    }
};  // namespace YAML

// ReSharper restore CppInconsistentNaming
