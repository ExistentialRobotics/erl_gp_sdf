#pragma once
#include "log_edf_gp.hpp"
#include "sign_gp.hpp"
#include "surface_data_manager.hpp"

#include <atomic>
#include <memory>

namespace erl::sdf_mapping {

    template<typename Dtype>
    struct SdfGaussianProcessSetting : common::Yamlable<SdfGaussianProcessSetting<Dtype>> {
        using SignGpSetting = typename SignGaussianProcess<Dtype>::Setting;
        using EdfGpSetting = typename LogEdfGaussianProcess<Dtype>::Setting;

        bool enable_sign_gp = true;
        std::shared_ptr<SignGpSetting> sign_gp = std::make_shared<SignGpSetting>();
        std::shared_ptr<EdfGpSetting> edf_gp = std::make_shared<EdfGpSetting>();

        struct YamlConvertImpl {
            static YAML::Node
            encode(const SdfGaussianProcessSetting& setting);

            static bool
            decode(const YAML::Node& node, SdfGaussianProcessSetting& setting);
        };
    };

    using SdfGaussianProcessSettingD = SdfGaussianProcessSetting<double>;
    using SdfGaussianProcessSettingF = SdfGaussianProcessSetting<float>;

    template<typename Dtype, int Dim>
    struct SdfGaussianProcess {
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<SdfGaussianProcess>());

        using SignGp = SignGaussianProcess<Dtype>;
        using EdfGp = LogEdfGaussianProcess<Dtype>;
        using Setting = SdfGaussianProcessSetting<Dtype>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using VectorX = Eigen::VectorX<Dtype>;

        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

        std::shared_ptr<Setting> setting = nullptr;
        bool active = false;
        std::atomic_bool locked_for_test = false;
        long num_sign_samples = 0;
        long num_edf_samples = 0;
        VectorD position{};
        Dtype half_size = 0;
        std::shared_ptr<SignGp> sign_gp = nullptr;
        std::shared_ptr<EdfGp> edf_gp = nullptr;

        explicit SdfGaussianProcess(std::shared_ptr<Setting> setting);

        SdfGaussianProcess(const SdfGaussianProcess& other);

        SdfGaussianProcess(SdfGaussianProcess&& other) noexcept;

        SdfGaussianProcess&
        operator=(const SdfGaussianProcess& other);

        SdfGaussianProcess&
        operator=(SdfGaussianProcess&& other) noexcept;

        void
        Activate();

        void
        Deactivate();

        [[nodiscard]] std::size_t
        GetMemoryUsage() const;

        [[nodiscard]] bool
        Intersects(const VectorD& other_position, Dtype other_half_size) const;

        [[nodiscard]] bool
        Intersects(const VectorD& other_position, const VectorD& other_half_sizes) const;

        void
        LoadSurfaceData(
            std::vector<std::pair<Dtype, std::size_t>>& surface_data_indices,
            const std::vector<SurfaceData<Dtype, Dim>>& surface_data_vec,
            Dtype offset_distance,
            Dtype sensor_noise,
            Dtype max_valid_gradient_var,
            Dtype invalid_position_var);

        void
        Train() const;

        [[nodiscard]] bool
        Test(
            const VectorD& test_position,
            Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> f,
            Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
            Eigen::Ref<Eigen::Vector<Dtype, Dim*(Dim + 1) / 2>> covariance,
            Dtype offset_distance,
            Dtype softmin_temperature,
            bool use_gp_covariance,
            bool compute_covariance) const;

        [[nodiscard]] bool
        operator==(const SdfGaussianProcess& other) const;

        [[nodiscard]] bool
        operator!=(const SdfGaussianProcess& other) const;

        [[nodiscard]] bool
        Write(std::ostream& s) const;

        [[nodiscard]] bool
        Read(std::istream& s, const std::shared_ptr<typename EdfGp::Setting>& edf_gp_setting);
    };

    using SdfGp3Dd = SdfGaussianProcess<double, 3>;
    using SdfGp3Df = SdfGaussianProcess<float, 3>;
    using SdfGp2Dd = SdfGaussianProcess<double, 2>;
    using SdfGp2Df = SdfGaussianProcess<float, 2>;

}  // namespace erl::sdf_mapping

#include "sdf_gp.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::SdfGaussianProcessSettingD> : erl::sdf_mapping::SdfGaussianProcessSettingD::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::SdfGaussianProcessSettingF> : erl::sdf_mapping::SdfGaussianProcessSettingF::YamlConvertImpl {};
