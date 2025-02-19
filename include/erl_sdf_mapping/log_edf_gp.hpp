#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <utility>

namespace erl::sdf_mapping {

    template<typename Dtype>
    class LogEdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess<Dtype> {

    public:
        using Super = gaussian_process::NoisyInputGaussianProcess<Dtype>;
        using Matrix = Eigen::MatrixX<Dtype>;

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
        inline static const std::string kFileHeader = fmt::format("# erl::sdf_mapping::LogEdfGaussianProcess<{}>", type_name<Dtype>());

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
            const std::vector<typename SurfaceDataManager<Dtype, Dim>::Data> &surface_data_vec,
            const Eigen::Vector<Dtype, Dim> &coord_origin,
            Dtype offset_distance,
            Dtype sensor_noise,
            Dtype max_valid_gradient_var,
            Dtype invalid_position_var);

        [[nodiscard]] bool
        Test(const Eigen::Ref<const Matrix> &mat_x_test, Eigen::Ref<Matrix> mat_f_out, Eigen::Ref<Matrix> mat_var_out, Eigen::Ref<Matrix> mat_cov_out)
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

    using LogEdfGaussianProcess_d = LogEdfGaussianProcess<double>;
    using LogEdfGaussianProcess_f = LogEdfGaussianProcess<float>;
}  // namespace erl::sdf_mapping

#include "log_edf_gp.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::LogEdfGaussianProcess_d::Setting> : erl::sdf_mapping::LogEdfGaussianProcess_d::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::LogEdfGaussianProcess_f::Setting> : erl::sdf_mapping::LogEdfGaussianProcess_f::Setting::YamlConvertImpl {};
