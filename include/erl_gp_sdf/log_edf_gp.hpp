#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <absl/container/flat_hash_set.h>

#include <utility>

namespace erl::gp_sdf {

    template<typename Dtype>
    class LogEdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess<Dtype> {

    public:
        using Super = gaussian_process::NoisyInputGaussianProcess<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;

        struct Setting : public common::Yamlable<Setting, typename Super::Setting> {
            Dtype log_lambda = 40.0f;   // log-edf parameter
            bool use_exp_bias = true;  // whether to use a bias term in exp for numerical stability

            ERL_REFLECT_SCHEMA(
                Setting,
                ERL_REFLECT_MEMBER(Setting, log_lambda),
                ERL_REFLECT_MEMBER(Setting, use_exp_bias));

            bool
            PostDeserialization() override;
        };

        struct TestResult final : Super::TestResult {
            Dtype exp_bias = 0.0;

            TestResult(
                const LogEdfGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test,
                Dtype exp_bias,
                bool will_predict_gradient);

            void
            GetMean(long y_index, Eigen::Ref<Eigen::VectorX<Dtype>> vec_f_out, bool parallel)
                const override;

            void
            GetMean(long index, long y_index, Dtype &f) const override;

            [[nodiscard]] Eigen::VectorXb
            GetGradient(long y_index, Eigen::Ref<MatrixX> mat_grad_out, bool parallel)
                const override;

            [[nodiscard]] bool
            GetGradient(long index, long y_index, Dtype *grad) const override;

            template<int Dim>
            [[nodiscard]] bool
            GetGradientD(const long index, const long y_index, Dtype *grad) const {
                ERL_DEBUG_ASSERT(
                    this->m_x_dim_ == Dim,
                    "x_dim = {}, it should be {}.",
                    this->m_x_dim_,
                    Dim);
                const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
                const long &num_test = this->m_num_test_;
                const auto &mat_k_test = this->m_mat_k_test_;
                const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
                for (long j = 0, jj = index + num_test; j < Dim; ++j, jj += num_test) {
                    grad[j] = mat_k_test.col(jj).dot(alpha);
                    if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                }
                if (y_index != 0) { return true; }
                Dtype norm = 0.0f;
                if (exp_bias == 0.0f) {
                    Dtype max_abs_comp = 0.0f;
                    for (long j = 0; j < Dim; ++j) {
                        if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
                    }
                    for (long j = 0; j < Dim; ++j) {
                        grad[j] /= max_abs_comp;  // normalize to avoid zero division
                        if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                        norm += grad[j] * grad[j];
                    }
                } else {
                    for (long j = 0; j < Dim; ++j) {
                        if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                        norm += grad[j] * grad[j];
                    }
                }
                norm = -std::sqrt(norm);
                for (long j = 0; j < Dim; ++j) {
                    grad[j] /= norm;
                    if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                }
                return true;  // valid gradient
            }
        };

    protected:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        bool m_use_matern32_ = false;

    public:
        explicit LogEdfGaussianProcess(std::shared_ptr<Setting> setting);

        LogEdfGaussianProcess(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess(LogEdfGaussianProcess &&other) noexcept = default;
        LogEdfGaussianProcess &
        operator=(const LogEdfGaussianProcess &other) = default;
        LogEdfGaussianProcess &
        operator=(LogEdfGaussianProcess &&other) noexcept = default;

        ~LogEdfGaussianProcess() override = default;

        [[nodiscard]] std::size_t
        GetMemoryUsage() const override;

        template<int Dim>
        long
        LoadSurfaceData(
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
            const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
            const Eigen::Vector<Dtype, Dim> &coord_origin,
            const bool data_sorted,
            const bool load_normals,
            const Dtype normal_scale,
            const Dtype offset_distance,
            const Dtype sensor_noise,
            const Dtype max_valid_gradient_var,
            const Dtype invalid_position_var) {

            ERL_ASSERTM(
                offset_distance >= 0.0f,
                "offset_distance must be non-negative for log_edf.");

            this->SetKernelCoordOrigin(coord_origin);
            const long max_num_samples =
                std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));

            // We are going to modify the buffer. Lock it here to prevent buffer swaps.
            auto lock = this->GetBufferLock();
            (void) lock;

            this->Reset(max_num_samples, Dim, load_normals ? Dim + 1 : 1);
            if (!data_sorted) {
                std::sort(
                    surface_data_indices.begin(),
                    surface_data_indices.end(),
                    [](const auto &a, const auto &b) { return a.first < b.first; });
            }

            long count = 0;
            typename Super::TrainBuf &buf = this->m_buf_loading_;

            for (auto &[distance, surface_data_index]: surface_data_indices) {
                auto &surf_data = surface_data_vec[surface_data_index];
                if (surf_data.var_position >= 1.0e6f) { continue; }  // skip invalid position

                if (offset_distance == 0.0f) {
                    buf.x.col(count) = surf_data.position;
                } else {
                    buf.x.col(count) = surf_data.position - offset_distance * surf_data.normal;
                }
                buf.y.col(0)[count] = 1.0f;
                if (load_normals) {
                    for (long i = 0; i < Dim; ++i) {
                        buf.y.col(i + 1)[count] = normal_scale * surf_data.normal[i];
                    }
                }
                buf.var_x[count] = surf_data.var_position;
                if ((surf_data.var_normal > max_valid_gradient_var) ||  // invalid gradient
                    (surf_data.normal.norm() < 0.9f)) {                 // invalid normal
                    buf.var_x[count] = std::max(buf.var_x[count], invalid_position_var);
                }
                buf.var_y[count] = sensor_noise;
                if (++count >= buf.x.cols()) { break; }  // reached max_num_samples
            }
            buf.num_samples = count;
            buf.num_samples_with_grad = 0;
            return count;
        }

        [[nodiscard]] std::shared_ptr<typename Super::TestResult>
        Test(const Eigen::Ref<const MatrixX> &mat_x_test, bool predict_gradient) const override;

        [[nodiscard]] bool
        operator==(const LogEdfGaussianProcess &other) const;

        [[nodiscard]] bool
        operator!=(const LogEdfGaussianProcess &other) const {
            return !(*this == other);
        }
    };

    using LogEdfGaussianProcessD = LogEdfGaussianProcess<double>;
    using LogEdfGaussianProcessF = LogEdfGaussianProcess<float>;

    extern template class LogEdfGaussianProcess<float>;
    extern template class LogEdfGaussianProcess<double>;
}  // namespace erl::gp_sdf
