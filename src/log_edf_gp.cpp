#include "erl_gp_sdf/log_edf_gp.hpp"

#include "erl_covariance/matern32.hpp"
#include "erl_covariance/radial_bias_function.hpp"
#include "erl_covariance/reduced_rank_covariance.hpp"

namespace erl::gp_sdf {

    template<typename Dtype>
    static Dtype
    GetEdfWithMatern32(Dtype f_log_gpis, Dtype a, Dtype exp_bias) {
        // f_log_gpis = std::min(std::abs(f_log_gpis), static_cast<Dtype>(1.0));
        f_log_gpis = std::log(std::abs(f_log_gpis)) - exp_bias;
        return std::abs(a * f_log_gpis);
        // return std::max(a * f_log_gpis, static_cast<Dtype>(0.0));
    }

    template<typename Dtype>
    static Dtype
    GetEdfWithRbf(Dtype f_log_gpis, Dtype a, Dtype exp_bias) {
        // f_log_gpis = std::min(std::abs(f_log_gpis), static_cast<Dtype>(1.0));
        f_log_gpis = std::log(std::abs(f_log_gpis)) - exp_bias;
        return std::sqrt(std::abs(a * f_log_gpis));  // a bit better
        // return std::sqrt(std::max(a * f_log_gpis, static_cast<Dtype>(0.0)));
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Setting::PostDeserialization() {
        if (!Super::Setting::PostDeserialization()) { return false; }
        ERL_ASSERTM_RETURN(
            log_lambda > 0.0f,
            false,
            "log_lambda must be positive, but got {}.",
            log_lambda);
        ERL_ASSERTM_RETURN(
            duplicate_epsilon >= 0.0f,
            false,
            "duplicate_epsilon must be non-negative, but got {}.",
            duplicate_epsilon);
        ERL_ASSERTM_RETURN(
            this->kernel != nullptr,
            false,
            "setting->kernel should not be nullptr.");

        const auto &x_dim = this->kernel->x_dim;
        if (x_dim != 2 && x_dim != 3) {
            ERL_ERROR("x_dim should be either 2 or 3 but got {}.", x_dim);
            return false;
        }

        const auto &kernel_type = this->kernel_type;
        using Matern32_2D = covariance::Matern32<Dtype, 2>;
        using Matern32_3D = covariance::Matern32<Dtype, 3>;
        using Rbf_2D = covariance::RadialBiasFunction<Dtype, 2>;
        using Rbf_3D = covariance::RadialBiasFunction<Dtype, 3>;
        if (x_dim == 2) {
            ERL_ASSERTM_RETURN(
                kernel_type == type_name<Matern32_2D>() || kernel_type == type_name<Rbf_2D>(),
                false,
                "kernel_type should be either {} or {} for x_dim = 2 but got {}.",
                type_name<Matern32_2D>(),
                type_name<Rbf_2D>(),
                kernel_type);
        } else {
            ERL_ASSERTM_RETURN(
                kernel_type == type_name<Matern32_3D>() || kernel_type == type_name<Rbf_3D>(),
                false,
                "kernel_type should be either {} or {} for x_dim = 3 but got {}.",
                type_name<Matern32_3D>(),
                type_name<Rbf_3D>(),
                kernel_type);
        }

        // make sure the kernel setting works
        using Covariance = covariance::Covariance<Dtype>;
        auto kernel = Covariance::CreateCovariance(kernel_type, this->kernel);
        ERL_ASSERTM_RETURN(
            kernel != nullptr,
            false,
            "Failed to create covariance of type {}.",
            kernel_type);

        // adjust the kernel scale according to log_lambda and kernel type
        if (const std::string &kernel_name = kernel->GetCovarianceName();
            kernel_name == "Matern32") {
            this->kernel->scale = std::sqrt(3.0f) / log_lambda;
        } else if (kernel_name == "RadialBiasFunction") {
            this->kernel->scale = std::sqrt(0.5f / log_lambda);
        } else {
            ERL_WARN("No auto scale adjustment for kernel type {}", kernel_type);
        }

        this->no_gradient_observation = true;
        return true;
    }

    template<typename Dtype>
    LogEdfGaussianProcess<Dtype>::TestResult::TestResult(
        const LogEdfGaussianProcess *gp,
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const Dtype exp_bias_in,
        const bool will_predict_gradient)
        : Super::TestResult(gp, mat_x_test, exp_bias_in, will_predict_gradient),
          exp_bias(exp_bias_in) {}

    template<typename Dtype>
    void
    LogEdfGaussianProcess<Dtype>::TestResult::GetMean(
        const long y_index,
        Eigen::Ref<Eigen::VectorX<Dtype>> vec_f_out,
        const bool parallel) const {
        (void) parallel;
        const long &num_test = this->m_num_test_;
#ifndef NDEBUG
        const long &y_dim = this->m_y_dim_;
#endif
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < y_dim,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            y_dim);
        ERL_DEBUG_ASSERT(
            vec_f_out.size() >= num_test,
            "vec_f_out.size() = {}, it should be >= {}.",
            vec_f_out.size(),
            num_test);
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        const auto &mat_k_test = this->m_mat_k_test_;
        Dtype *f = vec_f_out.data();
        if (y_index == 0) {
            if (gp->m_kernel_->GetCovarianceName() == "Matern32") {
                const Dtype a = -1.0f / gp->m_setting_->log_lambda;
#pragma omp parallel for if (parallel) schedule(static) default(none) \
    shared(num_test, mat_k_test, f, a, alpha)
                for (long index = 0; index < num_test; ++index) {
                    Dtype f_log_gpis = mat_k_test.col(index).dot(alpha);
                    f[index] = GetEdfWithMatern32(f_log_gpis, a, exp_bias);
                }
                return;
            }
            if (gp->m_kernel_->GetCovarianceName() == "RadialBiasFunction") {
                const Dtype a = -1.0f / gp->m_setting_->log_lambda;
#pragma omp parallel for if (parallel) schedule(static) default(none) \
    shared(num_test, mat_k_test, f, a, alpha)
                for (long index = 0; index < num_test; ++index) {
                    Dtype f_log_gpis = mat_k_test.col(index).dot(alpha);
                    f[index] = GetEdfWithRbf(f_log_gpis, a, exp_bias);
                }
                return;
            }
            ERL_WARN(
                "No log-edf reverse transformation implemented for kernel type {}",
                gp->m_kernel_->GetCovarianceName());
        }
#pragma omp parallel for if (parallel) schedule(static) default(none) \
    shared(num_test, mat_k_test, f, alpha)
        for (long index = 0; index < num_test; ++index) {
            f[index] = mat_k_test.col(index).dot(alpha);
        }
    }

    template<typename Dtype>
    void
    LogEdfGaussianProcess<Dtype>::TestResult::GetMean(
        const long index,
        const long y_index,
        Dtype &f) const {
        const auto &mat_k_test = this->m_mat_k_test_;
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        f = mat_k_test.col(index).dot(alpha);
        // we only apply the log transformation to the first output dimension
        if (y_index == 0) {
            if (gp->m_kernel_->GetCovarianceName() == "Matern32") {
                f = GetEdfWithMatern32(f, -1.0f / gp->m_setting_->log_lambda, exp_bias);
                return;
            }
            if (gp->m_kernel_->GetCovarianceName() == "RadialBiasFunction") {
                f = GetEdfWithRbf(f, -1.0f / gp->m_setting_->log_lambda, exp_bias);
                return;
            }
            ERL_WARN(
                "No log-edf reverse transformation implemented for kernel type {}",
                gp->m_kernel_->GetCovarianceName());
        }
    }

    template<typename Dtype>
    Eigen::VectorXb
    LogEdfGaussianProcess<Dtype>::TestResult::GetGradient(
        const long y_index,
        Eigen::Ref<MatrixX> mat_grad_out,
        const bool parallel) const {
        (void) parallel;
        const long &num_test = this->m_num_test_;
        const long &x_dim = this->m_x_dim_;
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        const auto &mat_k_test = this->m_mat_k_test_;
        Eigen::VectorXb valid_gradients(num_test);
        valid_gradients.setConstant(true);  // assume all gradients are valid
#pragma omp parallel for if (parallel) default(none) schedule(static) \
    shared(num_test, mat_grad_out, x_dim, mat_k_test, alpha, y_index, valid_gradients)
        for (long index = 0; index < num_test; ++index) {
            Dtype *grad = mat_grad_out.col(index).data();
            for (long j = 0, jj = index + num_test; j < x_dim; ++j, jj += num_test) {
                grad[j] = mat_k_test.col(jj).dot(alpha);
                if (!std::isfinite(grad[j])) {
                    valid_gradients[index] = false;  // invalid gradient
                    break;                           // no need to compute further
                }
            }
            if (!valid_gradients[index]) { continue; }  // skip invalid gradients
            if (y_index != 0) { continue; }
            // compute the norm of the gradient
            Dtype norm = 0.0f;
            if (exp_bias == 0.0f) {
                // without exp_bias, we need to be more careful about numerical issues
                Dtype max_abs_comp = 0.0f;
                for (long j = 0; j < x_dim; ++j) {
                    if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
                }
                for (long j = 0; j < x_dim; ++j) {
                    grad[j] /= max_abs_comp;  // normalize to avoid zero division
                    if (!std::isfinite(grad[j])) {
                        valid_gradients[index] = false;  // invalid gradient
                        break;                           // no need to compute further
                    }
                    norm += grad[j] * grad[j];
                }
            } else {
                // with exp_bias != 0, the numerical condition is better.
                for (long j = 0; j < x_dim; ++j) {
                    if (!std::isfinite(grad[j])) {
                        valid_gradients[index] = false;
                        break;  // no need to compute further
                    }
                    norm += grad[j] * grad[j];
                }
            }
            if (!valid_gradients[index]) { continue; }  // skip invalid gradients
            norm = -std::sqrt(norm);                    // normalize the gradient
            for (long j = 0; j < x_dim; ++j) {
                grad[j] /= norm;
                if (!std::isfinite(grad[j])) {
                    valid_gradients[index] = false;  // invalid gradient
                    break;                           // no need to compute further
                }
            }
        }
        return valid_gradients;  // return the validity of gradients
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::TestResult::GetGradient(
        const long index,
        const long y_index,
        Dtype *grad) const {
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const long &num_test = this->m_num_test_;
        const long &x_dim = this->m_x_dim_;
        const auto &mat_k_test = this->m_mat_k_test_;
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        // d = -ln(f)/lambda, grad_d = -1/(lambda*f)*grad_f
        // SDF gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
        // So, we only need the normalized grad_d.
        // It is fine that we don't know the f value.
        for (long j = 0, jj = index + num_test; j < x_dim; ++j, jj += num_test) {
            grad[j] = mat_k_test.col(jj).dot(alpha);
            if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
        }
        if (y_index != 0) { return true; }
        Dtype norm = 0.0f;
        if (exp_bias == 0.0f) {
            Dtype max_abs_comp = 0.0f;
            for (long j = 0; j < x_dim; ++j) {
                if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
            }
            for (long j = 0; j < x_dim; ++j) {
                grad[j] /= max_abs_comp;                        // normalize to avoid zero division
                if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                norm += grad[j] * grad[j];
            }
        } else {
            for (long j = 0; j < x_dim; ++j) {
                if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
                norm += grad[j] * grad[j];
            }
        }
        norm = -std::sqrt(norm);  // normalize the gradient
        for (long j = 0; j < x_dim; ++j) {
            grad[j] /= norm;
            if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
        }
        return true;  // valid gradient
    }

    template<typename Dtype>
    LogEdfGaussianProcess<Dtype>::LogEdfGaussianProcess(std::shared_ptr<Setting> setting)
        : Super(setting), m_setting_(std::move(setting)) {

        const auto kernel_setting = m_setting_->kernel;
        const auto &x_dim = kernel_setting->x_dim;
        const auto &kernel_type = m_setting_->kernel_type;
        using Matern32_2D = covariance::Matern32<Dtype, 2>;
        using Matern32_3D = covariance::Matern32<Dtype, 3>;
        using Matern32_XD = covariance::Matern32<Dtype, Eigen::Dynamic>;
        using Rbf_2D = covariance::RadialBiasFunction<Dtype, 2>;
        using Rbf_3D = covariance::RadialBiasFunction<Dtype, 3>;
        using Rbf_XD = covariance::RadialBiasFunction<Dtype, Eigen::Dynamic>;
        if (x_dim == 2) {
            ERL_ASSERT(
                kernel_type == type_name<Matern32_2D>() || kernel_type == type_name<Rbf_2D>());
        } else if (x_dim == 3) {
            ERL_ASSERT(
                kernel_type == type_name<Matern32_3D>() || kernel_type == type_name<Rbf_3D>());
        } else {
            ERL_ASSERT(
                kernel_type == type_name<Matern32_XD>() || kernel_type == type_name<Rbf_XD>());
        }
        using Covariance = covariance::Covariance<Dtype>;
        auto kernel = Covariance::CreateCovariance(m_setting_->kernel_type, kernel_setting);
        ERL_ASSERTM(
            kernel != nullptr,
            "Failed to create covariance of type {}.",
            m_setting_->kernel_type);
        if (kernel->GetCovarianceName() == "Matern32") {
            kernel_setting->scale = std::sqrt(3.0f) / m_setting_->log_lambda;
            m_use_matern32_ = true;
        } else if (kernel->GetCovarianceName() == "RadialBiasFunction") {
            kernel_setting->scale = std::sqrt(0.5f / m_setting_->log_lambda);
            m_use_matern32_ = false;
        } else {
            ERL_WARN("No auto scale adjustment for kernel type {}", kernel_type);
        }

        m_setting_->no_gradient_observation = true;
    }

    template<typename Dtype>
    std::size_t
    LogEdfGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = Super::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(Super);
        return memory_usage;
    }

    template<typename Dtype>
    std::shared_ptr<typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TestResult>
    LogEdfGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        bool predict_gradient) const {
        if (m_setting_->use_exp_bias) {
            Dtype d = (mat_x_test.col(0) - this->m_buf_train_.x.col(0)).squaredNorm();
            if (m_use_matern32_) { d = std::sqrt(d); }
            const Dtype exp_bias = m_setting_->log_lambda * d;
            return std::make_shared<TestResult>(this, mat_x_test, exp_bias, predict_gradient);
        }
        return std::make_shared<TestResult>(this, mat_x_test, 0.0, predict_gradient);
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::operator==(const LogEdfGaussianProcess &other) const {
        if (!Super::operator==(other)) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        return true;
    }

    template class LogEdfGaussianProcess<float>;
    template class LogEdfGaussianProcess<double>;
}  // namespace erl::gp_sdf
