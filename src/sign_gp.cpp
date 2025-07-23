#include "erl_gp_sdf/sign_gp.hpp"

#include <utility>

namespace erl::gp_sdf {

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
        const auto gp = reinterpret_cast<const SignGaussianProcess *>(this->m_gp_);
        const auto alpha = gp->m_mat_alpha_.col(y_index).head(gp->m_k_train_cols_);
        const auto &mat_k_test = this->m_mat_k_test_;
        const Dtype offset = gp->m_train_set_.y.data()[0];
        Dtype *f = vec_f_out.data();
#pragma omp parallel for if (parallel) default(none) \
    shared(num_test, mat_k_test, f, offset, alpha) schedule(static)
        for (long index = 0; index < num_test; ++index) {
            f[index] = mat_k_test.col(index).dot(alpha) - offset;
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
        : Super(std::move(setting)) {}

    template<typename Dtype>
    std::size_t
    SignGaussianProcess<Dtype>::GetMemoryUsage() const {
        std::size_t memory_usage = Super::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(Super);
        return memory_usage;
    }

    template<typename Dtype>
    std::shared_ptr<typename gaussian_process::NoisyInputGaussianProcess<Dtype>::TestResult>
    SignGaussianProcess<Dtype>::Test(
        const Eigen::Ref<const MatrixX> &mat_x_test,
        bool predict_gradient) const {
        return std::make_shared<TestResult>(this, mat_x_test, predict_gradient);
    }

    template class SignGaussianProcess<double>;
    template class SignGaussianProcess<float>;
}  // namespace erl::gp_sdf
