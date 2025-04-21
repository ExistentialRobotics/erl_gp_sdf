#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype>
    SignGaussianProcess<Dtype>::SignGaussianProcess(std::shared_ptr<Setting> setting)
        : Super([setting]() {
              ERL_ASSERTM(setting != nullptr, "Settings must not be nullptr");
              setting->no_gradient_observation = false;
              return setting;
          }()) {}

    template<typename Dtype>
    std::size_t
    SignGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = Super::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(Super);
        return memory_usage;
    }

    template<typename Dtype>
    template<int Dim>
    long
    SignGaussianProcess<Dtype>::LoadSurfaceData(
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices,
        const std::vector<SurfaceData<Dtype, Dim>> &surface_data_vec,
        const Eigen::Vector<Dtype, Dim> &coord_origin,
        Dtype offset_distance,
        Dtype sensor_noise,
        Dtype max_valid_gradient_var,
        Dtype invalid_position_var) {

        this->SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(this->m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        this->Reset(max_num_samples, Dim);

        std::sort(surface_data_indices.begin(), surface_data_indices.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        long count = 0;
        for (auto &[distance, surface_data_index]: surface_data_indices) {
            auto &surface_data = surface_data_vec[surface_data_index];
            this->m_mat_x_train_.col(count) = surface_data.position;
            this->m_mat_grad_train_.col(count) = surface_data.normal;
            this->m_vec_var_x_[count] = surface_data.var_position;
            this->m_vec_var_grad_[count] = surface_data.var_normal;
            this->m_vec_var_h_[count] = sensor_noise;
            this->m_vec_grad_flag_[count] = true;

            if ((surface_data.var_normal > max_valid_gradient_var) ||                                   // invalid gradient
                (surface_data.normal.norm() < 0.9f)) {                                                   // invalid normal
                this->m_vec_var_x_[count] = std::max(this->m_vec_var_x_[count], invalid_position_var);  // position is unreliable
                this->m_vec_grad_flag_[count] = false;
            }
            if (++count >= this->m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        this->m_num_train_samples_ = count;
        // for GPIS, y=0 does not work, as y* = k' * K^-1 * y
        // the trick is to set y = offset_distance, then y* = k' * K^-1 * y - offset_distance
        this->m_vec_y_train_.setConstant(this->m_num_train_samples_, offset_distance);
        if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(this->m_num_train_samples_); }
        return count;
    }

    template<typename Dtype>
    bool
    SignGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        Eigen::Ref<MatrixX> mat_f_out,
        Eigen::Ref<MatrixX> mat_var_out,
        Eigen::Ref<MatrixX> mat_cov_out,
        const bool predict_gradient) const {

        if (!Super::Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out, predict_gradient)) { return false; }
        const long dim = mat_x_test.rows();
        const long n = mat_x_test.cols();
        const Dtype offset_distance = this->m_vec_y_train_[0];
        for (long i = 0; i < n; ++i) {
            Dtype *f = mat_f_out.col(i).data();
            f[0] -= offset_distance;  // sdf = h(x) - offset_distance
            Dtype norm = 0;
            for (long j = 1; j <= dim; ++j) {  // gradient
                Dtype &grad = f[j];            // grad_sdf
                norm += grad * grad;
            }
            norm = std::sqrt(norm);
            if (norm > 1.e-15) {                                   // avoid zero division
                for (long j = 1; j <= dim; ++j) { f[j] /= norm; }  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
            }
        }
        return true;
    }

}  // namespace erl::sdf_mapping
