#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype>
    SignGaussianProcess<Dtype>::TestResult::TestResult(
        const SignGaussianProcess *gp,
        const Eigen::Ref<const MatrixX> &mat_x_test,
        bool will_predict_gradient)
        : Super::TestResult(gp, mat_x_test, will_predict_gradient) {}

    template<typename Dtype>
    void
    SignGaussianProcess<Dtype>::TestResult::GetMean(
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
        const auto gp = reinterpret_cast<const SignGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        const auto &mat_k_test = this->m_mat_k_test_;
        const Dtype offset = gp->m_train_set_.y.data()[0];
        Dtype *f = vec_f_out.data();
        for (long index = 0; index < num_test; ++index, ++f) {
            *f = mat_k_test.col(index).dot(alpha) - offset;
        }
    }

    template<typename Dtype>
    void
    SignGaussianProcess<Dtype>::TestResult::GetMean(const long index, const long y_index, Dtype &f)
        const {
#ifndef NDEBUG
        const long &num_test = this->m_num_test_;
        const long &y_dim = this->m_y_dim_;
#endif
        ERL_DEBUG_ASSERT(
            index >= 0 && index < num_test,
            "index = {}, it should be in [0, {}).",
            index,
            num_test);
        ERL_DEBUG_ASSERT(
            y_index >= 0 && y_index < y_dim,
            "y_index = {}, it should be in [0, {}).",
            y_index,
            y_dim);
        const auto gp = reinterpret_cast<const SignGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        f = this->m_mat_k_test_.col(index).dot(alpha);  // h_{y_index}(x_{index})
        f -= gp->m_train_set_.y.data()[0];
    }

    template<typename Dtype>
    SignGaussianProcess<Dtype>::SignGaussianProcess(std::shared_ptr<Setting> setting)
        : Super(setting) {}

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

        const long max_num_samples = std::min(  //
            this->m_setting_->max_num_samples,
            static_cast<long>(surface_data_vec.size()));
        this->Reset(max_num_samples, Dim, 1);

        std::sort(
            surface_data_indices.begin(),
            surface_data_indices.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        typename Super::TrainSet &train_set = this->GetTrainSet();
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

            if ((surface_data.var_normal > max_valid_gradient_var) ||  // invalid gradient
                (surface_data.normal.norm() < 0.9f)) {                 // invalid normal
                train_set.var_x[count] = std::max(
                    train_set.var_x[count],
                    invalid_position_var);           // position is unreliable
                train_set.grad_flag[count] = false;  // gradient is unreliable
                --count_grad;                        // revert gradient count
            }
            if (++count >= train_set.x.cols()) { break; }  // reached max_num_samples
        }
        train_set.num_samples = count;
        train_set.num_samples_with_grad = count_grad;

        // for GPIS, y=0 does not work, as y* = ktest * K^-1 * y
        // the trick is to set y = offset_distance, then y* = ktest * K^-1 * y - offset_distance

        if (this->m_reduced_rank_kernel_) { this->UpdateKtrain(); }
        return count;
    }

    template<typename Dtype>
    std::shared_ptr<typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TestResult>
    SignGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        bool predict_gradient) const {
        return std::make_shared<TestResult>(this, mat_x_test, predict_gradient);
    }

}  // namespace erl::sdf_mapping
