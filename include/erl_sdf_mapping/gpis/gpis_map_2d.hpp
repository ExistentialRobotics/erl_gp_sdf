#pragma once

#include "gpis_map_base_2d.hpp"

namespace erl::sdf_mapping::gpis {
    class GpisMap2D : public GpisMapBase2D {
    public:
        explicit GpisMap2D()
            : GpisMap2D(std::make_shared<Setting>()) {}

        explicit GpisMap2D(const std::shared_ptr<Setting> &setting)
            : GpisMapBase2D(setting) {}

        std::shared_ptr<Setting>
        GetSetting() const {
            return m_setting_;
        }

    private:
        inline std::shared_ptr<void>
        TrainGpX(
            const Eigen::Ref<const Eigen::MatrixXd> &mat_x_train,
            const Eigen::Ref<const Eigen::VectorXd> &vec_y,
            const Eigen::Ref<const Eigen::MatrixXd> &mat_grad_train,
            const Eigen::Ref<const Eigen::VectorXb> &vec_grad_flag,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_var_grad) final {

            auto gp = std::make_shared<gaussian_process::NoisyInputGaussianProcess>(m_setting_->gp_sdf);
            long num_train_samples = mat_x_train.cols();
            gp->Reset(num_train_samples, 2);
            gp->GetTrainInputSamplesBuffer().topLeftCorner(2, num_train_samples) = mat_x_train;
            gp->GetTrainInputSamplesVarianceBuffer().head(num_train_samples) = vec_var_x;
            gp->GetTrainOutputSamplesBuffer().head(num_train_samples) = vec_y;
            gp->GetTrainOutputValueSamplesVarianceBuffer().head(num_train_samples) = vec_var_y;
            gp->GetTrainOutputGradientSamplesBuffer().topLeftCorner(2, num_train_samples) = mat_grad_train;
            gp->GetTrainGradientFlagsBuffer().head(num_train_samples) = vec_grad_flag;
            gp->GetTrainOutputGradientSamplesVarianceBuffer().head(num_train_samples) = vec_var_grad;
            gp->Train(mat_x_train.cols());
            return gp;
        }

        inline void
        InferWithGpX(
            const std::shared_ptr<const void> &gp_ptr,
            const Eigen::Ref<const Eigen::Vector2d> &vec_xt,
            Eigen::Ref<Eigen::Vector3d> f,
            Eigen::Ref<Eigen::Vector3d> var) const final {

            auto gp = std::static_pointer_cast<const gaussian_process::NoisyInputGaussianProcess>(gp_ptr);
            Eigen::Matrix3Xd mat_cov_out;
            gp->Test(vec_xt, f, var, mat_cov_out);
        }
    };
}  // namespace erl::sdf_mapping::gpis
