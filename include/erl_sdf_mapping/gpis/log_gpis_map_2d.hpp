#pragma once

#include <fstream>

#include "erl_sdf_mapping/log_sdf_gp.hpp"
#include "gpis_map_base_2d.hpp"

namespace erl::sdf_mapping::gpis {

    class LogGpisMap2D : public GpisMapBase2D {

    public:
        struct Setting : public GpisMapBase2D::Setting {
            std::shared_ptr<LogSdfGaussianProcess::Setting> gp_sdf = std::make_shared<LogSdfGaussianProcess::Setting>();
        };

    private:
        std::shared_ptr<Setting> m_setting_ = {};

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

            auto gp = std::make_shared<LogSdfGaussianProcess>(m_setting_->gp_sdf);
            gp->Reset(mat_x_train.cols(), 2);
            gp->GetTrainInputSamplesBuffer() = mat_x_train;
            gp->GetTrainInputSamplesVarianceBuffer() = vec_sigma_x;
            gp->GetTrainOutputSamplesBuffer() = vec_y;
            gp->GetTrainOutputValueSamplesVarianceBuffer() = vec_sigma_y;
            gp->GetTrainOutputGradientSamplesVarianceBuffer() = vec_sigma_grad;
            gp->GetTrainGradientFlagsBuffer() = vec_grad_flag;
            gp->Train(mat_x_train.cols(), vec_grad_flag.cast<long>().sum());
            return gp;
        }

        inline void
        InferWithGpX(
            const std::shared_ptr<const void> &gp_ptr,
            const Eigen::Ref<const Eigen::Vector2d> &mat_x_test,
            Eigen::Ref<Eigen::Vector3d> vec_f_out,
            Eigen::Ref<Eigen::Vector3d> vec_var_out) const final {

            auto gp = std::static_pointer_cast<const LogSdfGaussianProcess>(gp_ptr);
            Eigen::Matrix3Xd mat_cov_out;
            gp->Test(mat_x_test, vec_f_out, vec_var_out, mat_cov_out);
        }
    };

}  // namespace erl::sdf_mapping::gpis
