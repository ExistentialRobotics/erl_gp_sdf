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

            auto gp = gaussian_process::NoisyInputGaussianProcess::Create(m_setting_->gp_sdf);
            gp->Train(mat_x_train, vec_grad_flag, vec_y, vec_sigma_x, vec_sigma_y, vec_sigma_grad);
            return gp;
        }

        inline void
        InferWithGpX(
            const std::shared_ptr<const void> &gp_ptr,
            const Eigen::Ref<const Eigen::Vector2d> &vec_xt,
            Eigen::Ref<Eigen::Vector3d> f,
            Eigen::Ref<Eigen::Vector3d> var) const final {

            auto gp = std::static_pointer_cast<const gaussian_process::NoisyInputGaussianProcess>(gp_ptr);
            gp->Test(vec_xt, f, var);
        }
    };
}  // namespace erl::sdf_mapping::gpis
