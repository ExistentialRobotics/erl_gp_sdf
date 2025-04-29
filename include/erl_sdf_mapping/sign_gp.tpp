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
        const Dtype offset_distance,
        const Dtype sensor_noise,
        const Dtype max_valid_gradient_var,
        const Dtype invalid_position_var) {

        this->SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(this->m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        this->Reset(max_num_samples, Dim, 1);

        std::sort(surface_data_indices.begin(), surface_data_indices.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TrainSet &train_set = this->GetTrainSet();
        long count = 0;
        long count_grad = 0;
        for (auto &[distance, surface_data_index]: surface_data_indices) {
            auto &surface_data = surface_data_vec[surface_data_index];

            train_set.x.col(count) = surface_data.position;
            train_set.y.col(0)[count] = offset_distance;
            train_set.grad.col(count) = surface_data.normal;
            train_set.var_x[count] = surface_data.var_position;
            train_set.var_y[count] = sensor_noise;
            train_set.var_grad[count] = surface_data.var_normal;
            train_set.grad_flag[count] = true;
            ++count_grad;

            // this->m_mat_x_train_.col(count) = surface_data.position;
            // this->m_mat_grad_train_.col(count) = surface_data.normal;
            // this->m_vec_var_x_[count] = surface_data.var_position;
            // this->m_vec_var_grad_[count] = surface_data.var_normal;
            // this->m_vec_var_h_[count] = sensor_noise;
            // this->m_vec_grad_flag_[count] = true;

            if ((surface_data.var_normal > max_valid_gradient_var) ||                             // invalid gradient
                (surface_data.normal.norm() < 0.9f)) {                                            // invalid normal
                train_set.var_x[count] = std::max(train_set.var_x[count], invalid_position_var);  // position is unreliable
                train_set.grad_flag[count] = false;                                               // gradient is unreliable
                --count_grad;                                                                     // revert gradient count
            }
            if (++count >= train_set.x.cols()) { break; }  // reached max_num_samples
        }
        train_set.num_samples = count;
        train_set.num_samples_with_grad = count_grad;

        // for GPIS, y=0 does not work, as y* = k' * K^-1 * y
        // the trick is to set y = offset_distance, then y* = k' * K^-1 * y - offset_distance

        if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(); }
        return count;
    }

    template<typename Dtype>
    bool
    SignGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        const std::vector<std::pair<long, bool>> &y_index_grad_pairs,
        Eigen::Ref<MatrixX> mat_f_out,
        Eigen::Ref<MatrixX> mat_var_out,
        Eigen::Ref<MatrixX> mat_cov_out) const {

        if (!Super::Test(mat_x_test, y_index_grad_pairs, mat_f_out, mat_var_out, mat_cov_out)) { return false; }
        const long dim = mat_x_test.rows();
        const long n = mat_x_test.cols();
        const Dtype offset_distance = this->GetTrainSet().y.data()[0];
        for (long i = 0; i < n; ++i) {
            Dtype *f = mat_f_out.col(i).data();
            f[0] -= offset_distance;  // sdf = h(x) - offset_distance
            Dtype norm = 0;
            for (long j = 1; j <= dim; ++j) {  // gradient
                Dtype &grad = f[j];            // grad_sdf
                norm += grad * grad;
            }
            norm = std::sqrt(norm);
            if (norm > 1.0e-6) {                                   // avoid zero division
                for (long j = 1; j <= dim; ++j) { f[j] /= norm; }  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
            }
        }
        return true;
    }

}  // namespace erl::sdf_mapping
