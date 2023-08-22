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
            const Eigen::Ref<Eigen::MatrixXd> &mat_x_train,
            const Eigen::Ref<Eigen::VectorXb> &vec_grad_flag,
            const Eigen::Ref<const Eigen::VectorXd> &vec_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_x,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_y,
            const Eigen::Ref<const Eigen::VectorXd> &vec_sigma_grad) final {

            auto gp = std::make_shared<gaussian_process::NoisyInputGaussianProcess>(m_setting_->gp_sdf);
            gp->Reset(mat_x_train.cols(), 2);
            gp->GetTrainInputSamplesBuffer() = mat_x_train;
            gp->GetTrainGradientFlagsBuffer() = vec_grad_flag;
            gp->GetTrainOutputSamplesBuffer() = vec_y;
            gp->GetTrainInputSamplesVarianceBuffer() = vec_sigma_x;
            gp->GetTrainOutputValueSamplesVarianceBuffer() = vec_sigma_y;
            gp->GetTrainOutputGradientSamplesVarianceBuffer() = vec_sigma_grad;
            gp->Train(mat_x_train.cols(), vec_grad_flag.cast<long>().sum());
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
