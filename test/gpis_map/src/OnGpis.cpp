/*
 * GPisMap - Online Continuous Mapping using Gaussian Process Implicit Surfaces
 * https://github.com/leebhoram/GPisMap
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License v3 as published by
 * the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of any FITNESS FOR A
 * PARTICULAR PURPOSE. See the GNU General Public License v3 for more details.
 *
 * You should have received a copy of the GNU General Public License v3
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-3.0.html.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 *          Huang Zonghao<ac@hzh.io>
 */

#include "OnGpis.h"

#include <Eigen/Cholesky>

#include "covFnc.h"

// #define SQRT_3  double(1.732051)

void
OnGpis::Reset() {
    m_n_samples_ = 0;
    m_trained_ = false;
}

void
OnGpis::Train(const VecNode &samples) {
    Reset();

    auto n = int(samples.size());
    int dim = 2;

    if (n > 0) {
        m_n_samples_ = n;
        m_x_ = EMatrixX::Zero(dim, n);
        EMatrixX grad = EMatrixX::Zero(dim, n);
        EVectorX f = EVectorX::Zero(n);

#if defined(BUILD_TEST)
        m_sigx_ = EVectorX::Zero(n);
        m_siggrad_ = EVectorX::Zero(n);
#else
        EVectorX m_sigx_ = EVectorX::Zero(n);
        EVectorX m_siggrad_ = EVectorX::Zero(n);
#endif

        m_gradflag_.clear();
        m_gradflag_.resize(n, double(0.0));

        EMatrixX grad_valid(n, 2);

        int k = 0;
        int count = 0;
        for (auto it = samples.begin(); it != samples.end(); it++, k++) {
            m_x_(0, k) = (*it)->GetPosX();
            m_x_(1, k) = (*it)->GetPosY();
            grad(0, k) = (*it)->GetGradX();
            grad(1, k) = (*it)->GetGradY();
            f(k) = (*it)->GetVal();
            m_sigx_(k) = (*it)->GetPosNoise();
            m_siggrad_(k) = (*it)->GetGradNoise();
            if (m_siggrad_(k) > double(0.1001) ||
                (fabs(grad(0, k)) < double(1e-6) && fabs(grad(1, k)) < double(1e-6))) {  // gradient is almost 0, or its variance is big, unreliable
                m_gradflag_[k] = double(0.0);
                m_sigx_(k) = double(2.0);
            } else {
                m_gradflag_[k] = 1.;
                grad_valid(count, 0) = grad(0, k);
                grad_valid(count, 1) = grad(1, k);
                count++;
            }
        }
        grad_valid.conservativeResize(count, 2);

#if defined(BUILD_TEST)
        m_y_.resize(n + 2 * count);
        m_y_ << f, grad_valid.col(0), grad_valid.col(1);  // the observation is distance and gradient
        m_k_ = Matern32SparseDeriv1(m_x_, m_gradflag_, m_param_.scale, m_sigx_, m_param_.noise, m_siggrad_);
#else
        EVectorX m_y_(n + 2 * count);
        m_y_ << f, grad_valid.col(0), grad_valid.col(1);
        EMatrixX m_k_ = Matern32SparseDeriv1(m_x_, m_gradflag_, m_param_.scale, m_sigx_, m_param_.noise, m_siggrad_);
#endif

        m_l_ = m_k_.llt().matrixL();

        m_alpha_ = m_y_;
        m_l_.template triangularView<Eigen::Lower>().solveInPlace(m_alpha_);
        m_l_.transpose().template triangularView<Eigen::Upper>().solveInPlace(m_alpha_);

        m_trained_ = true;
    }
}

void
OnGpis::Train(const VecNode3 &samples) {
    Reset();

    auto n = int(samples.size());
    int dim = 3;

    if (n > 0) {
        m_n_samples_ = n;
        m_x_ = EMatrixX::Zero(dim, n);
        EMatrixX grad = EMatrixX::Zero(dim, n);
        EVectorX f = EVectorX::Zero(n);
        EVectorX sigx = EVectorX::Zero(n);
        EVectorX siggrad = EVectorX::Zero(n);

        m_gradflag_.clear();
        m_gradflag_.resize(n, double(0.0));

        EMatrixX grad_valid(n, dim);

        int k = 0;
        int count = 0;
        for (auto it = samples.begin(); it != samples.end(); it++, k++) {
            m_x_(0, k) = (*it)->GetPosX();
            m_x_(1, k) = (*it)->GetPosY();
            m_x_(2, k) = (*it)->GetPosZ();
            grad(0, k) = (*it)->GetGradX();
            grad(1, k) = (*it)->GetGradY();
            grad(2, k) = (*it)->GetGradZ();
            f(k) = (*it)->GetVal();
            sigx(k) = (*it)->GetPosNoise();
            siggrad(k) = (*it)->GetGradNoise();
            if (siggrad(k) > double(0.1001) || (fabs(grad(0, k)) < double(1e-6) && fabs(grad(1, k)) < double(1e-6) && fabs(grad(2, k)) < double(1e-6))) {
                m_gradflag_[k] = double(0.0);
                sigx(k) = double(2.0);
            } else {
                m_gradflag_[k] = 1.0;
                grad_valid(count, 0) = grad(0, k);
                grad_valid(count, 1) = grad(1, k);
                grad_valid(count, 2) = grad(2, k);
                count++;
            }
        }
        grad_valid.conservativeResize(count, 3);
        EVectorX y(n + dim * count);
        y << f, grad_valid.col(0), grad_valid.col(1), grad_valid.col(2);
        EMatrixX mat_k = Matern32SparseDeriv1(m_x_, m_gradflag_, m_param_.scale, sigx, m_param_.noise, siggrad);

        m_l_ = mat_k.llt().matrixL();

        m_alpha_ = y;
        m_l_.template triangularView<Eigen::Lower>().solveInPlace(m_alpha_);
        m_l_.transpose().template triangularView<Eigen::Upper>().solveInPlace(m_alpha_);

        m_trained_ = true;
    }
}

void
OnGpis::TestSinglePoint(const EVectorX &xt, double &val, double grad[], double var[]) {
    if (!IsTrained()) return;

    if (m_x_.rows() != xt.size()) return;

#if defined(BUILD_TEST)
    m_k_ = Matern32SparseDeriv1(m_x_, m_gradflag_, xt, m_param_.scale);
#else
    EMatrixX m_k_ = Matern32SparseDeriv1(m_x_, m_gradflag_, xt, m_param_.scale);
#endif

    EVectorX res = m_k_.transpose() * m_alpha_;
    val = res(0);
    if (res.size() == 3) {
        grad[0] = res(1);
        grad[1] = res(2);
    } else if (res.size() == 4) {
        grad[0] = res(1);
        grad[1] = res(2);
        grad[2] = res(3);
    }

    m_l_.template triangularView<Eigen::Lower>().solveInPlace(m_k_);
    m_k_ = m_k_.array().pow(2);
    EVectorX v = m_k_.colwise().sum();

    if (v.size() == 3) {
        var[0] = double(1.01) - v(0);
        var[1] = m_three_over_scale_ + double(0.1) - v(1);
        var[2] = m_three_over_scale_ + double(0.1) - v(2);
    } else if (v.size() == 4) {  // Noise m_param_!
        var[0] = double(1.001) - v(0);
        var[1] = m_three_over_scale_ + double(0.001) - v(1);
        var[2] = m_three_over_scale_ + double(0.001) - v(2);
        var[3] = m_three_over_scale_ + double(0.001) - v(3);
    }
}

void
OnGpis::Test2DPoint(const EVectorX &xt, double &val, double &gradx, double &grady, double &var_val, double &var_gradx, double &var_grady) {
    if (!IsTrained()) { return; }

    EMatrixX K_ = Matern32SparseDeriv1(m_x_, m_gradflag_, xt, m_param_.scale);

    EVectorX res = K_.transpose() * m_alpha_;
    val = res(0);
    gradx = res(1);
    grady = res(2);

    m_l_.template triangularView<Eigen::Lower>().solveInPlace(K_);
    K_ = K_.array().pow(2);
    EVectorX v = K_.colwise().sum();

    var_val = 1. - v(0);
    var_gradx = m_three_over_scale_ - v(1);
    var_grady = m_three_over_scale_ - v(2);
}
