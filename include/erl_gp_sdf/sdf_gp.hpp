#pragma once

#include "log_edf_gp.hpp"
#include "sign_gp.hpp"
#include "surface_data_manager.hpp"

#include "erl_common/enum_parse.hpp"

#include <atomic>
#include <memory>

namespace erl::gp_sdf {

    enum class SignMethod {
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

        SignMethod sign_method = SignMethod::kNormalGp;
        std::pair<SignMethod, SignMethod> hybrid_sign_methods = {
            SignMethod::kNormalGp,
            SignMethod::kExternal};
        Dtype hybrid_sign_threshold = 0.2f;
        Dtype normal_scale = 1.0f;              // scale for normal gp
        Dtype softmin_temperature = 1.0f;       // temperature for softmin
        Dtype sign_gp_offset_distance = 0.01f;  // distance to shift for surface data for sign_gp.
        Dtype edf_gp_offset_distance = 0.0f;    // distance to shift for surface data for edf_gp.
        std::shared_ptr<SignGpSetting> sign_gp = std::make_shared<SignGpSetting>();
        std::shared_ptr<EdfGpSetting> edf_gp = std::make_shared<EdfGpSetting>();

        ERL_REFLECT_SCHEMA(
            SdfGaussianProcessSetting,
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, sign_method),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, hybrid_sign_methods),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, hybrid_sign_threshold),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, normal_scale),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, softmin_temperature),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, sign_gp_offset_distance),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, edf_gp_offset_distance),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, sign_gp),
            ERL_REFLECT_MEMBER(SdfGaussianProcessSetting, edf_gp));
    };

    template<typename Dtype, int Dim>
    struct SdfGaussianProcess {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

        using SignGp = SignGaussianProcess<Dtype>;
        using EdfGp = LogEdfGaussianProcess<Dtype>;
        using Setting = SdfGaussianProcessSetting<Dtype>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using VectorX = Eigen::VectorX<Dtype>;

        std::shared_ptr<Setting> setting = nullptr;
        bool active = false;                     // true if the GP is active
        long time_stamp = 0;                     // last update timestamp
        long buf_outdated_count = 10000;         // C1: num of times buffer marked outdated
        std::atomic_long gp_outdated_count = 0;  // C2: num of times GP marked outdated
        std::atomic_long query_count = 0;        // C3: num of times queried
        bool use_normal_gp = false;              // true if normal gp is used for sign prediction
        VectorD position{};                      // center position of the GP
        std::atomic<std::array<Dtype, Dim>> running_mean_position{};  // mean pos of training data
        Dtype running_num_samples = 0;              // number of training data accumulated
        Dtype half_size = 0;                        // half-size of the GP area
        std::shared_ptr<SignGp> sign_gp = nullptr;  // initialized in Activate().
        std::shared_ptr<EdfGp> edf_gp = nullptr;    // initialized in Activate().

        explicit SdfGaussianProcess(std::shared_ptr<Setting> setting_);

        SdfGaussianProcess(const SdfGaussianProcess &other);

        SdfGaussianProcess(SdfGaussianProcess &&other) noexcept;

        SdfGaussianProcess &
        operator=(const SdfGaussianProcess &other);

        SdfGaussianProcess &
        operator=(SdfGaussianProcess &&other) noexcept;

        ~SdfGaussianProcess() = default;

        void
        Activate();

        void
        Deactivate();

        void
        MarkBufferOutdated();

        void
        MarkGpOutdated();

        void
        MarkQueried();

        [[nodiscard]]
        bool
        BufferOutdated() const;

        [[nodiscard]]
        bool
        GpOutdated() const;

        [[nodiscard]] Dtype
        GetLoadingPriority(Dtype query_count_weight) const;

        [[nodiscard]] Dtype
        GetRetrainPriority(Dtype query_count_weight) const;

        void
        SetMeanPosition(const VectorD &mean_position);

        [[nodiscard]] VectorD
        GetMeanPosition() const;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const;

        [[nodiscard]] bool
        Intersects(const VectorD &other_position, Dtype other_half_size) const;

        [[nodiscard]] bool
        Intersects(const VectorD &other_position, const VectorD &other_half_sizes) const;

        bool
        LoadSurfaceData(
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
            const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
            bool data_sorted,
            Dtype sensor_noise,
            Dtype max_valid_gradient_var,
            Dtype invalid_position_var);

        [[nodiscard]] bool
        IsTrained() const;

        void
        Train();

        [[nodiscard]] bool
        Test(
            const VectorD &test_position,
            Eigen::Ref<Eigen::Vector<Dtype, 2 * Dim + 1>> f,
            Eigen::Ref<Eigen::Vector<Dtype, Dim + 1>> var,
            Eigen::Ref<Eigen::Vector<Dtype, Dim *(Dim + 1) / 2>> covariance,
            Dtype &sign,
            bool compute_gradient,
            bool compute_gradient_variance,
            bool compute_covariance,
            bool use_gp_covariance) const;

        [[nodiscard]] bool
        operator==(const SdfGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const SdfGaussianProcess &other) const;

        [[nodiscard]] bool
        Write(std::ostream &stream) const;

        [[nodiscard]] bool
        Read(std::istream &stream);

    private:
        void
        EstimateVariance(
            const VectorD &test_position,
            Dtype edf_pred,
            bool compute_gradient_variance,
            bool compute_covariance,
            Dtype *var,
            Dtype *covariance) const;
    };

    using SdfGaussianProcessSettingD = SdfGaussianProcessSetting<double>;
    using SdfGaussianProcessSettingF = SdfGaussianProcessSetting<float>;
    using SdfGp3Dd = SdfGaussianProcess<double, 3>;
    using SdfGp3Df = SdfGaussianProcess<float, 3>;
    using SdfGp2Dd = SdfGaussianProcess<double, 2>;
    using SdfGp2Df = SdfGaussianProcess<float, 2>;

    extern template struct SdfGaussianProcessSetting<double>;
    extern template struct SdfGaussianProcessSetting<float>;
    extern template struct SdfGaussianProcess<double, 3>;
    extern template struct SdfGaussianProcess<float, 3>;
    extern template struct SdfGaussianProcess<double, 2>;
    extern template struct SdfGaussianProcess<float, 2>;

}  // namespace erl::gp_sdf

ERL_REFLECT_ENUM_SCHEMA(
    erl::gp_sdf::SignMethod,
    5,
    ERL_REFLECT_ENUM_MEMBER("none", erl::gp_sdf::SignMethod::kNone),
    ERL_REFLECT_ENUM_MEMBER("sign_gp", erl::gp_sdf::SignMethod::kSignGp),
    ERL_REFLECT_ENUM_MEMBER("normal_gp", erl::gp_sdf::SignMethod::kNormalGp),
    ERL_REFLECT_ENUM_MEMBER("external", erl::gp_sdf::SignMethod::kExternal),
    ERL_REFLECT_ENUM_MEMBER("hybrid", erl::gp_sdf::SignMethod::kHybrid));

ERL_PARSE_ENUM(erl::gp_sdf::SignMethod, 5);
