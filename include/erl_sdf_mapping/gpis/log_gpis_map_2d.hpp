#pragma once

#include <fstream>

#include "erl_gaussian_process/log_noisy_input_gp.hpp"
#include "gpis_map_base_2d.hpp"

namespace erl::sdf_mapping::gpis {

    class LogGpisMap2D : public GpisMapBase2D {

    public:
        struct Setting : public GpisMapBase2D::Setting {
            std::shared_ptr<gaussian_process::LogNoisyInputGaussianProcess::Setting> gp_sdf =
                std::make_shared<gaussian_process::LogNoisyInputGaussianProcess::Setting>();
        };

    private:
        std::shared_ptr<Setting> m_setting_;

    public:
        explicit LogGpisMap2D()
            : LogGpisMap2D(std::make_shared<Setting>()) {}

        explicit LogGpisMap2D(const std::shared_ptr<Setting> &setting)
            : GpisMapBase2D(setting),
              m_setting_(setting) {}

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

            auto gp = gaussian_process::LogNoisyInputGaussianProcess::Create(m_setting_->gp_sdf);
            gp->Train(mat_x_train, vec_grad_flag, vec_y, vec_sigma_x, vec_sigma_y, vec_sigma_grad);
            return gp;
        }

        inline void
        InferWithGpX(
            const std::shared_ptr<const void> &gp_ptr,
            const Eigen::Ref<const Eigen::Vector2d> &mat_x_test,
            Eigen::Ref<Eigen::Vector3d> vec_f_out,
            Eigen::Ref<Eigen::Vector3d> vec_var_out) const final {

            auto gp = std::static_pointer_cast<const gaussian_process::LogNoisyInputGaussianProcess>(gp_ptr);
            gp->Test(mat_x_test, vec_f_out, vec_var_out);
        }
    };

}  // namespace erl::sdf_mapping::gpis
