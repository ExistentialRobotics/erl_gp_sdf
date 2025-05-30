#pragma once

#include "log_edf_gp.hpp"

#include "erl_covariance/reduced_rank_covariance.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    YAML::Node
    LogEdfGaussianProcess<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node = YAML::convert<typename Super::Setting>::encode(setting);
        ERL_YAML_SAVE_ATTR(node, setting, log_lambda);
        return node;
    }

    template<typename Dtype>
    bool
    LogEdfGaussianProcess<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        YAML::convert<typename Super::Setting>::decode(node, setting);
        ERL_YAML_LOAD_ATTR(node, setting, log_lambda);
        return true;
    }

    template<typename Dtype>
    LogEdfGaussianProcess<Dtype>::TestResult::TestResult(
        const LogEdfGaussianProcess *gp,
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const bool will_predict_gradient)
        : Super::TestResult(gp, mat_x_test, will_predict_gradient) {}

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
            const Dtype a = -1.0f / gp->m_setting_->log_lambda;
#pragma omp parallel for if (parallel) default(none) shared(num_test, mat_k_test, f, a, alpha)
            for (long index = 0; index < num_test; ++index) {
                const Dtype f_log_gpis = mat_k_test.col(index).dot(alpha);
                f[index] = a * std::log(std::abs(f_log_gpis));
            }
            return;
        }
#pragma omp parallel for if (parallel) default(none) shared(num_test, mat_k_test, f, alpha)
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
        f = mat_k_test.col(index).dot(alpha);  // std::log(std::abs(f_log_gpis)) / -log_lambda
        // we only apply the log transformation to the first output dimension
        if (y_index == 0) { f = std::log(std::abs(f)) / -gp->m_setting_->log_lambda; }
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
#pragma omp parallel for if (parallel) default(none) \
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
            Dtype max_abs_comp = 0.0f;
            for (long j = 0; j < x_dim; ++j) {
                if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
            }
            Dtype norm = 0.0f;
            for (long j = 0; j < x_dim; ++j) {
                grad[j] /= max_abs_comp;  // normalize to avoid zero division
                if (!std::isfinite(grad[j])) {
                    valid_gradients[index] = false;  // invalid gradient
                    break;                           // no need to compute further
                }
                norm += grad[j] * grad[j];
            }
            if (!valid_gradients[index]) { continue; }  // skip invalid gradients
            norm = -std::sqrt(norm);
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
        Dtype max_abs_comp = 0.0f;
        for (long j = 0; j < x_dim; ++j) {
            if (std::abs(grad[j]) > max_abs_comp) { max_abs_comp = std::abs(grad[j]); }
        }
        Dtype norm = 0.0f;
        for (long j = 0; j < x_dim; ++j) {
            grad[j] /= max_abs_comp;                        // normalize to avoid zero division
            if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
            norm += grad[j] * grad[j];
        }
        norm = -std::sqrt(norm);
        for (long j = 0; j < x_dim; ++j) {
            grad[j] /= norm;
            if (!std::isfinite(grad[j])) { return false; }  // invalid gradient
        }
        return true;  // valid gradient
    }

    template<typename Dtype>
    template<int Dim>
    bool
    LogEdfGaussianProcess<Dtype>::TestResult::GetGradientD(
        const long index,
        const long y_index,
        Dtype *grad) const {
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
            grad[j] /= max_abs_comp;                        // normalize to avoid zero division
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

    template<typename Dtype>
    LogEdfGaussianProcess<Dtype>::LogEdfGaussianProcess(std::shared_ptr<Setting> setting)
        : Super([setting]() -> std::shared_ptr<Setting> {
              setting->kernel->scale = std::sqrt(3.) / setting->log_lambda;
              setting->no_gradient_observation = true;
              return setting;
          }()),
          m_setting_(std::move(setting)) {}

    template<typename Dtype>
    std::size_t
    LogEdfGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = Super::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(Super);
        return memory_usage;
    }

    template<typename Dtype>
    template<int Dim>
    long
    LogEdfGaussianProcess<Dtype>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        const Eigen::Vector<Dtype, Dim> &coord_origin,
        const bool load_normals,
        const Dtype normal_scale,
        const Dtype offset_distance,
        const Dtype sensor_noise,
        const Dtype max_valid_gradient_var,
        const Dtype invalid_position_var) {

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
                train_set.x.col(count) = surf_data.position - offset_distance * surf_data.normal;
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

    template<typename Dtype>
    std::shared_ptr<typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TestResult>
    LogEdfGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        bool predict_gradient) const {
        return std::make_shared<TestResult>(this, mat_x_test, predict_gradient);
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
}  // namespace erl::sdf_mapping
