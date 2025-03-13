#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <utility>

namespace erl::sdf_mapping {

    template<typename Dtype>
    class LogEdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess<Dtype> {

    public:
        using Super = gaussian_process::NoisyInputGaussianProcess<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;

        struct Setting : common::Yamlable<Setting, typename Super::Setting> {
            Dtype log_lambda = 40.0;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<LogEdfGaussianProcess>());

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;

    public:
        explicit LogEdfGaussianProcess(std::shared_ptr<Setting> setting);

        LogEdfGaussianProcess(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess(LogEdfGaussianProcess &&other) = default;
        LogEdfGaussianProcess &
        operator=(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess &
        operator=(LogEdfGaussianProcess &&other) = default;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const override;

        template<int Dim>
        long
        LoadSurfaceData(
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
            const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
            const Eigen::Vector<Dtype, Dim> &coord_origin,
            Dtype offset_distance,
            Dtype sensor_noise,
            Dtype max_valid_gradient_var,
            Dtype invalid_position_var);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, Eigen::Ref<MatrixX> mat_f_out, Eigen::Ref<MatrixX> mat_var_out, Eigen::Ref<MatrixX> mat_cov_out)
            const override;

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

    using LogEdfGaussianProcessD = LogEdfGaussianProcess<double>;
    using LogEdfGaussianProcessF = LogEdfGaussianProcess<float>;
}  // namespace erl::sdf_mapping

#include "log_edf_gp.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::LogEdfGaussianProcessD::Setting> : erl::sdf_mapping::LogEdfGaussianProcessD::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::LogEdfGaussianProcessF::Setting> : erl::sdf_mapping::LogEdfGaussianProcessF::Setting::YamlConvertImpl {};
