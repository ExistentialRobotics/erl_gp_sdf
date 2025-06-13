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
            GetMean(long y_index, Eigen::Ref<Eigen::VectorX<Dtype>> vec_f_out, bool parallel)
                const override;

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
            Dtype invalid_position_var) {

            this->SetKernelCoordOrigin(coord_origin);

            const long max_num_samples = std::min(  //
                this->m_setting_->max_num_samples,
                static_cast<long>(surface_data_vec.size()));
            this->Reset(max_num_samples, Dim, 1);

            std::sort(
                surface_data_indices.begin(),
                surface_data_indices.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
            typename Super::TrainSet &train_set = this->m_train_set_;
            const bool load_gradient = !this->m_setting_->no_gradient_observation;
            long count = 0;
            long count_grad = 0;
            for (auto &[distance, surface_data_index]: surface_data_indices) {
                auto &surface_data = surface_data_vec[surface_data_index];

                train_set.x.col(count) = surface_data.position;
                train_set.y.col(0)[count] = offset_distance;
                train_set.var_x[count] = surface_data.var_position;
                train_set.var_y[count] = sensor_noise;
                if (load_gradient) {
                    train_set.grad.col(count) = surface_data.normal;
                    train_set.var_grad[count] = surface_data.var_normal;
                }
                train_set.grad_flag[count] = load_gradient;
                ++count_grad;

                if ((surface_data.var_normal > max_valid_gradient_var) ||  // invalid gradient
                    (surface_data.normal.norm() < 0.9f)) {                 // invalid normal
                    train_set.var_x[count] = std::max(train_set.var_x[count], invalid_position_var);
                    train_set.grad_flag[count] = false;
                    --count_grad;  // revert gradient count
                }
                if (++count >= train_set.x.cols()) { break; }  // reached max_num_samples
            }
            train_set.num_samples = count;
            train_set.num_samples_with_grad = load_gradient ? count_grad : 0;

            // for GPIS, y=0 does not work, as y* = ktest * K^-1 * y
            // the trick is to set y = offset_distance, then y* = ktest * K^-1 * y - offset_distance

            if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(); }
            return count;
        }

        [[nodiscard]] std::shared_ptr<typename Super::TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, bool predict_gradient) const override;
    };

    using SignGaussianProcessD = SignGaussianProcess<double>;
    using SignGaussianProcessF = SignGaussianProcess<float>;

    extern template class SignGaussianProcess<double>;
    extern template class SignGaussianProcess<float>;
}  // namespace erl::sdf_mapping

// #include "sign_gp.tpp"
