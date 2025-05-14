#pragma once

#include "log_edf_gp.hpp"
#include "sign_gp.hpp"
#include "surface_data_manager.hpp"

#include <atomic>
#include <memory>

namespace erl::sdf_mapping {

    enum SignMethod {
        kNone = 0,      // No sign prediction.
        kSignGp = 1,    // Use sign gp.
        kNormalGp = 2,  // Use normal gp.
        kExternal = 3,  // Use external sign prediction.
        kHybrid = 4,    // Use two methods switched by hybrid_sign_threshold.
    };

    template<typename Dtype>
    struct SdfGaussianProcessSetting : common::Yamlable<SdfGaussianProcessSetting<Dtype>> {
        using SignGpSetting = typename SignGaussianProcess<Dtype>::Setting;
        using EdfGpSetting = typename LogEdfGaussianProcess<Dtype>::Setting;

        SignMethod sign_method = kNormalGp;
        std::pair<SignMethod, SignMethod> hybrid_sign_methods = {kNormalGp, kExternal};
        Dtype hybrid_sign_threshold = 0.2f;
        Dtype normal_scale = 1.0f;  // scale for normal gp
        Dtype softmin_temperature = 1.0f;
        Dtype sign_gp_offset_distance = 0.01f;  // distance to shift for surface data for sign_gp.
        Dtype edf_gp_offset_distance = 0.0f;    // distance to shift for surface data for edf_gp.
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
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

        using SignGp = SignGaussianProcess<Dtype>;
        using EdfGp = LogEdfGaussianProcess<Dtype>;
        using Setting = SdfGaussianProcessSetting<Dtype>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using VectorX = Eigen::VectorX<Dtype>;

        std::shared_ptr<Setting> setting = nullptr;
        bool active = false;
        bool outdated = true;  // whether the GP is outdated and needs to be retrained
        bool use_normal_gp = false;
        Dtype offset_distance = 0;
        std::atomic_bool locked_for_test = false;
        VectorD position{};
        Dtype half_size = 0;
        std::shared_ptr<SignGp> sign_gp = nullptr;
        std::shared_ptr<EdfGp> edf_gp = nullptr;  // initialized in Activate().

        explicit SdfGaussianProcess(std::shared_ptr<Setting> setting_);

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

        void
        MarkOutdated();

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
            Dtype sensor_noise,
            Dtype max_valid_gradient_var,
            Dtype invalid_position_var);

        [[nodiscard]] bool
        IsTrained() const;

        void
        Train();

        [[nodiscard]] bool
        Test(
            const VectorD& test_position,
            Eigen::Ref<Eigen::Vector<Dtype, 2 * Dim + 1>> f,
            Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
            Eigen::Ref<Eigen::Vector<Dtype, Dim*(Dim + 1) / 2>> covariance,
            Dtype external_sign,
            bool compute_gradient,
            bool compute_gradient_variance,
            bool compute_covariance,
            bool use_gp_covariance) const;

        [[nodiscard]] bool
        operator==(const SdfGaussianProcess& other) const;

        [[nodiscard]] bool
        operator!=(const SdfGaussianProcess& other) const;

        [[nodiscard]] bool
        Write(std::ostream& s) const;  // TODO: check implementation

        [[nodiscard]] bool
        Read(std::istream& s);

    private:
        void
        EstimateVariance(
            const VectorD& test_position,
            Dtype edf_pred,
            bool compute_gradient_variance,
            bool compute_covariance,
            Dtype* var,
            Dtype* covariance) const;
    };

    using SdfGp3Dd = SdfGaussianProcess<double, 3>;
    using SdfGp3Df = SdfGaussianProcess<float, 3>;
    using SdfGp2Dd = SdfGaussianProcess<double, 2>;
    using SdfGp2Df = SdfGaussianProcess<float, 2>;

}  // namespace erl::sdf_mapping

#include "sdf_gp.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::SdfGaussianProcessSettingD>
    : erl::sdf_mapping::SdfGaussianProcessSettingD::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::SdfGaussianProcessSettingF>
    : erl::sdf_mapping::SdfGaussianProcessSettingF::YamlConvertImpl {};
