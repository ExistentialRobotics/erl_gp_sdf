#pragma once

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
        ERL_YAML_LOAD_ATTR_TYPE(node, setting, log_lambda, Dtype);
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
        Eigen::Ref<Eigen::VectorX<Dtype>> vec_f_out) const {
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
        const Dtype a = -1.0f / gp->m_setting_->log_lambda;
        Dtype *f = vec_f_out.data();
        for (long index = 0; index < num_test; ++index, ++f) {
            const Dtype f_log_gpis = mat_k_test.col(index).dot(alpha);
            *f = a * std::log(std::abs(f_log_gpis));
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
    void
    LogEdfGaussianProcess<Dtype>::TestResult::GetGradient(
        const long y_index,
        Eigen::Ref<MatrixX> mat_grad_out) const {
        const long &num_test = this->m_num_test_;
        const long &x_dim = this->m_x_dim_;
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const Dtype a = 1.0e15f / gp->m_mat_alpha_(0, y_index);
        using VectorX = Eigen::VectorX<Dtype>;
        const VectorX alpha = a * gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        const auto &mat_k_test = this->m_mat_k_test_;
        for (long index = 0; index < num_test; ++index) {
            Dtype *grad = mat_grad_out.col(index).data();
            Dtype norm = 0.0f;
            for (long j = 0, jj = index + num_test; j < x_dim; ++j, jj += num_test) {
                grad[j] = mat_k_test.col(jj).dot(alpha);
                if (y_index == 0) { norm += grad[j] * grad[j]; }
            }
            if (y_index != 0) { continue; }
            norm = -std::sqrt(norm);
            for (long j = 0; j < x_dim; ++j) { grad[j] /= norm; }
        }
    }

    template<typename Dtype>
    void
    LogEdfGaussianProcess<Dtype>::TestResult::GetGradient(
        const long index,
        const long y_index,
        Dtype *grad) const {
        const auto gp = reinterpret_cast<const LogEdfGaussianProcess *>(this->m_gp_);
        const long &num_test = this->m_num_test_;
        const long &x_dim = this->m_x_dim_;
        const auto &mat_k_test = this->m_mat_k_test_;
        using VectorX = Eigen::VectorX<Dtype>;
        // `a` is a scale factor to avoid zero division.
        const Dtype a = 1.0e15f / gp->m_mat_alpha_(0, y_index);
        const VectorX alpha = a * gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        // d = -ln(f)/lambda, grad_d = -1/(lambda*f)*grad_f
        // SDF gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
        // So, we only need the normalized grad_d.
        // It is fine that we don't know the f value.
        Dtype norm = 0.0f;
        for (long j = 0, jj = index + num_test; j < x_dim; ++j, jj += num_test) {
            grad[j] = mat_k_test.col(jj).dot(alpha);
            norm += grad[j] * grad[j];
        }
        norm = -std::sqrt(norm);
        for (long j = 0; j < x_dim; ++j) { grad[j] /= norm; }
    }

    template<typename Dtype>
    template<int Dim>
    void
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
        using VectorX = Eigen::VectorX<Dtype>;
        const Dtype a = 1.0e15f / gp->m_mat_alpha_(0, y_index);
        const VectorX alpha = a * gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        Dtype norm = 0.0f;
        for (long j = 0, jj = index + num_test; j < Dim; ++j, jj += num_test) {
            grad[j] = mat_k_test.col(jj).dot(alpha);
            norm += grad[j] * grad[j];
        }
        norm = -std::sqrt(norm);
        for (long j = 0; j < Dim; ++j) { grad[j] /= norm; }
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
        typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set =
            this->m_train_set_;
        for (auto &[distance, surface_data_index]: surface_data_indices) {
            auto &surface_data = surface_data_vec[surface_data_index];
            if (offset_distance == 0.0f) {
                train_set.x.col(count) = surface_data.position;
            } else {
                train_set.x.col(count) =
                    surface_data.position - offset_distance * surface_data.normal;
            }
            train_set.y.col(0)[count] = 1.0f;
            if (load_normals) {
                for (long i = 0; i < Dim; ++i) {
                    train_set.y.col(i + 1)[count] = normal_scale * surface_data.normal[i];
                }
            }
            train_set.var_x[count] = surface_data.var_position;
            if ((surface_data.var_normal > max_valid_gradient_var) ||  // invalid gradient
                (surface_data.normal.norm() < 0.9f)) {                 // invalid normal
                train_set.var_x[count] = std::max(
                    train_set.var_x[count],
                    invalid_position_var);  // position is unreliable
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
