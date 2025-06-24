#pragma once

#include "surface_data_manager.hpp"

#include "erl_gaussian_process/noisy_input_gp.hpp"

#include <utility>

namespace erl::gp_sdf {

    template<typename Dtype>
    class LogEdfGaussianProcess : public gaussian_process::NoisyInputGaussianProcess<Dtype> {

    public:
        using Super = gaussian_process::NoisyInputGaussianProcess<Dtype>;
        using MatrixX = Eigen::MatrixX<Dtype>;

        struct Setting : public common::Yamlable<Setting, typename Super::Setting> {
            Dtype log_lambda = 40.0f;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        struct TestResult : Super::TestResult {
            TestResult(
                const LogEdfGaussianProcess *gp,
                const Eigen::Ref<const MatrixX> &mat_x_test,
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
            GetGradientD(long index, long y_index, Dtype *grad) const {
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
                Dtype max_abs_comp = 0.0f;
                for (long j = 0; j < Dim; ++j) {
                    if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
                }
                Dtype norm = 0.0f;
                for (long j = 0; j < Dim; ++j) {
                    grad[j] /= max_abs_comp;  // normalize to avoid zero division
                    if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                    norm += grad[j] * grad[j];
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
            this->Reset(max_num_samples, Dim, load_normals ? Dim + 1 : 1);
            std::sort(
                surface_data_indices.begin(),
                surface_data_indices.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });

            long count = 0;
            typename Super::TrainSet &train_set = this->m_train_set_;
            for (auto &[distance, surface_data_index]: surface_data_indices) {
                auto &surf_data = surface_data_vec[surface_data_index];
                if (offset_distance == 0.0f) {
                    train_set.x.col(count) = surf_data.position;
                } else {
                    train_set.x.col(count) =
                        surf_data.position - offset_distance * surf_data.normal;
                }
                train_set.y.col(0)[count] = 1.0f;
                if (load_normals) {
                    for (long i = 0; i < Dim; ++i) {
                        train_set.y.col(i + 1)[count] = normal_scale * surf_data.normal[i];
                    }
                }
                train_set.var_x[count] = surf_data.var_position;
                if ((surf_data.var_normal > max_valid_gradient_var) ||  // invalid gradient
                    (surf_data.normal.norm() < 0.9f)) {                 // invalid normal
                    train_set.var_x[count] = std::max(train_set.var_x[count], invalid_position_var);
                }
                train_set.var_y[count] = sensor_noise;
                if (++count >= train_set.x.cols()) { break; }  // reached max_num_samples
            }
            train_set.num_samples = count;
            train_set.num_samples_with_grad = 0;
            if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(); }
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

template<>
struct YAML::convert<erl::gp_sdf::LogEdfGaussianProcessD::Setting>
    : erl::gp_sdf::LogEdfGaussianProcessD::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::LogEdfGaussianProcessF::Setting>
    : erl::gp_sdf::LogEdfGaussianProcessF::Setting::YamlConvertImpl {};
