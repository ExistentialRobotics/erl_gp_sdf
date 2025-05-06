#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <memory>

namespace erl::sdf_mapping {

    template<typename Dtype>
    class SignGaussianProcess : public gaussian_process::NoisyInputGaussianProcess<Dtype> {

    public:
        using Super = gaussian_process::NoisyInputGaussianProcess<Dtype>;
        using Setting = typename Super::Setting;
        using MatrixX = Eigen::MatrixX<Dtype>;

        struct TestResult : Super::TestResult {
            TestResult(
                const SignGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test,
                bool will_predict_gradient);

            void
            GetMean(long y_index, Eigen::Ref<Eigen::VectorX<Dtype>> vec_f_out) const override;

            void
            GetMean(long index, long y_index, Dtype &f) const override;
        };

        explicit SignGaussianProcess(std::shared_ptr<Setting> setting);

        SignGaussianProcess(const SignGaussianProcess &other) = default;
        SignGaussianProcess(SignGaussianProcess &&other) = default;
        SignGaussianProcess &
        operator=(const SignGaussianProcess &other) = default;
        SignGaussianProcess &
        operator=(SignGaussianProcess &&other) = default;

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

        [[nodiscard]] std::shared_ptr<typename Super::TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, bool predict_gradient) const override;
    };

    using SignGaussianProcessD = SignGaussianProcess<double>;
    using SignGaussianProcessF = SignGaussianProcess<float>;
}  // namespace erl::sdf_mapping

#include "sign_gp.tpp"
