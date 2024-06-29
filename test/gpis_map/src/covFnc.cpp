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
 * covFnc.cpp
 * This file implements covariance functions for Gaussian Processes.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 */
#include "covFnc.h"

#include <cmath>

using namespace Eigen;

// inline functions for Matern32SparseDeriv1
inline double
kf(double r, double a) {
    return (1.0f + a * r) * std::exp(-a * r);
}

inline double
kf1(double r, double dx, double a) {
    return (a * a) * dx * std::exp(-a * r);
}

inline double
kf2(double r, double dx1, double dx2, double delta, double a) {
    return (a * a) * (delta - a * dx1 * dx2 / r) * std::exp(-a * r);
}

// Dimension-specific implementations are required for covariances with derivatives.
// 2D
// FIXME: sigma of output noise should also be included
// static EMatrixX matern32_sparse_deriv1_2D(EMatrixX const& x1, std::vector<double> m_gradflag_,
//                                           double scale_param, EVectorX const& m_sigx_, EVectorX const& m_siggrad_);
static EMatrixX
matern32_sparse_deriv1_2D(
    EMatrixX const &x1,
    std::vector<double> gradflag,
    double scale_param,
    EVectorX const &sigx,
    double const &sigy,
    EVectorX const &siggrad);  // FIXED
static EMatrixX
matern32_sparse_deriv1_2D(EMatrixX const &x1, std::vector<double> gradflag, EMatrixX const &x2, double scale_param);

// 3D
// FIXME: sigma of output noise should also be included
// static EMatrixX matern32_sparse_deriv1_3D(EMatrixX const& x1, std::vector<double> m_gradflag_,
//                                           double scale_param, EVectorX const& m_sigx_, EVectorX const& m_siggrad_);
static EMatrixX
matern32_sparse_deriv1_3D(
    EMatrixX const &x1,
    std::vector<double> gradflag,
    double scale_param,
    EVectorX const &sigx,
    double const &sigy,
    EVectorX const &siggrad);  // FIXED
static EMatrixX
matern32_sparse_deriv1_3D(EMatrixX const &x1, std::vector<double> gradflag, EMatrixX const &x2, double scale_param);

EMatrixX
OrnsteinUhlenbeck(EMatrixX const &x_1, double scale_param, double sigx) {
    auto n = x_1.cols();
    double a = 1.f / scale_param;
    EMatrixX K = EMatrixX::Zero(n, n);

    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j++) {
            if (k == j) {
                K(k, k) = 1.0f + sigx;
            } else {
                double r = (x_1.col(k) - x_1.col(j)).norm();
                K(k, j) = std::exp(-a * r);
                K(j, k) = K(k, j);
            }
        }
    }

    return K;
}

EMatrixX
OrnsteinUhlenbeck(EMatrixX const &x_1, double scale_param, EVectorX const &sigx) {
    auto n = x_1.cols();
    double a = 1.f / scale_param;
    EMatrixX K = EMatrixX::Zero(n, n);

    for (int k = 0; k < n; k++) {
        for (int j = k; j < n; j++) {
            if (k == j) {
                K(k, k) = 1. + sigx(k);
            } else {
                double r = (x_1.col(k) - x_1.col(j)).norm();
                K(k, j) = std::exp(-a * r);
                K(j, k) = K(k, j);
            }
        }
    }

    return K;
}

EMatrixX
OrnsteinUhlenbeck(EMatrixX const &x_1, EMatrixX const &x_2, double scale_param) {
    auto n = x_1.cols();
    auto m = x_2.cols();
    auto a = double(-1.) / scale_param;
    EMatrixX K = EMatrixX::Zero(n, m);

    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) {
            double r = (x_1.col(k) - x_2.col(j)).norm();
            K(k, j) = 1.f * std::exp(a * r);
        }
    }

    return K;
}

EMatrixX
Matern32SparseDeriv1(EMatrixX const &x_1, std::vector<double> gradflag, double scale_param, EVectorX const &sigx, double const &sigy, EVectorX const &siggrad) {
    auto dim = x_1.rows();

    EMatrixX K;

    if (dim == 2) K = matern32_sparse_deriv1_2D(x_1, gradflag, scale_param, sigx, sigy, siggrad);
    else if (dim == 3)
        K = matern32_sparse_deriv1_3D(x_1, gradflag, scale_param, sigx, sigy, siggrad);

    return K;
}

EMatrixX
Matern32SparseDeriv1(EMatrixX const &x_1, std::vector<double> gradflag, EMatrixX const &x_2, double scale_param) {
    auto dim = x_1.rows();

    EMatrixX K;

    if (dim == 2) {
        K = matern32_sparse_deriv1_2D(x_1, gradflag, x_2, scale_param);
    } else if (dim == 3) {
        K = matern32_sparse_deriv1_3D(x_1, gradflag, x_2, scale_param);
    }

    return K;
}

// 3D
// FIXME: sigma of output noise should also be included
// EMatrixX matern32_sparse_deriv1_3D(EMatrixX const& x1, std::vector<double> m_gradflag_,
//    double scale_param, EVectorX const& m_sigx_, EVectorX const& m_siggrad_)
EMatrixX
matern32_sparse_deriv1_3D(
    EMatrixX const &x1,
    std::vector<double> gradflag,
    double scale_param,
    EVectorX const &sigx,
    double const &sigy,
    EVectorX const &siggrad)  // FIXED
{
    auto dim = x1.rows();
    auto n = x1.cols();
    double sqr3L = std::sqrt(double(3)) / scale_param;
    double sqr3L2 = sqr3L * sqr3L;
    EMatrixX K;

    int ng = 0;
    for (double &flag: gradflag) {
        if (flag > double(0.5)) {
            flag = double(ng);
            ng++;
        } else {
            flag = double(-1.0);
        }
    }

    K = EMatrixX::Zero(n + ng * dim, n + ng * dim);

    for (int k = 0; k < n; k++) {
        int kind1 = int(gradflag[k]) + int(n);
        int kind2 = kind1 + ng;
        int kind3 = kind2 + ng;

        for (int j = k; j < n; j++) {
            if (k == j) {
                // FIXME: sigma of output noise should also be included
                // m_k_(k,k) = 1.0+m_sigx_(k);
                K(k, k) = 1. + sigx(k) + sigy;  // FIXED
                if (gradflag[k] > double(-0.5)) {
                    K(k, kind1) = double(0.0);
                    K(kind1, k) = double(0.0);
                    K(k, kind2) = double(0.0);
                    K(kind2, k) = double(0.0);
                    K(k, kind3) = double(0.0);
                    K(kind3, k) = double(0.0);

                    K(kind1, kind1) = sqr3L2 + siggrad(k);
                    K(kind1, kind2) = double(0.0);
                    K(kind1, kind3) = double(0.0);
                    K(kind2, kind1) = double(0.0);
                    K(kind2, kind2) = sqr3L2 + siggrad(k);
                    K(kind2, kind3) = double(0.0);
                    K(kind3, kind1) = double(0.0);
                    K(kind3, kind2) = double(0.0);
                    K(kind3, kind3) = sqr3L2 + siggrad(k);
                }
            } else {
                double r = (x1.col(k) - x1.col(j)).norm();
                K(k, j) = kf(r, sqr3L);
                K(j, k) = K(k, j);
                if (gradflag[k] > double(-1.)) {

                    K(kind1, j) = -kf1(r, x1(0, k) - x1(0, j), sqr3L);
                    K(j, kind1) = K(kind1, j);
                    K(kind2, j) = -kf1(r, x1(1, k) - x1(1, j), sqr3L);
                    K(j, kind2) = K(kind2, j);
                    K(kind3, j) = -kf1(r, x1(2, k) - x1(2, j), sqr3L);
                    K(j, kind3) = K(kind3, j);

                    if (gradflag[j] > double(-1.)) {
                        int jind1 = int(gradflag[j]) + int(n);
                        int jind2 = jind1 + ng;
                        int jind3 = jind2 + ng;
                        K(k, jind1) = -K(j, kind1);
                        K(jind1, k) = K(k, jind1);
                        K(k, jind2) = -K(j, kind2);
                        K(jind2, k) = K(k, jind2);
                        K(k, jind3) = -K(j, kind3);
                        K(jind3, k) = K(k, jind3);

                        K(kind1, jind1) = kf2(r, x1(0, k) - x1(0, j), x1(0, k) - x1(0, j), 1., sqr3L);
                        K(jind1, kind1) = K(kind1, jind1);
                        K(kind1, jind2) = kf2(r, x1(0, k) - x1(0, j), x1(1, k) - x1(1, j), double(0.0), sqr3L);
                        K(jind1, kind2) = K(kind1, jind2);
                        K(kind1, jind3) = kf2(r, x1(0, k) - x1(0, j), x1(2, k) - x1(2, j), double(0.0), sqr3L);
                        K(jind1, kind3) = K(kind1, jind3);

                        K(kind2, jind1) = K(kind1, jind2);
                        K(jind2, kind1) = K(kind1, jind2);
                        K(kind2, jind2) = kf2(r, x1(1, k) - x1(1, j), x1(1, k) - x1(1, j), 1., sqr3L);
                        K(jind2, kind2) = K(kind2, jind2);
                        K(kind2, jind3) = kf2(r, x1(1, k) - x1(1, j), x1(2, k) - x1(2, j), double(0.0), sqr3L);
                        K(jind2, kind3) = K(kind2, jind3);

                        K(kind3, jind1) = K(kind1, jind3);
                        K(jind3, kind1) = K(kind1, jind3);
                        K(kind3, jind2) = K(kind2, jind3);
                        K(jind3, kind2) = K(kind2, jind3);
                        K(kind3, jind3) = kf2(r, x1(2, k) - x1(2, j), x1(2, k) - x1(2, j), 1., sqr3L);
                        K(jind3, kind3) = K(kind3, jind3);
                    }
                } else if (gradflag[j] > double(-1.)) {
                    int jind1 = int(gradflag[j]) + int(n);
                    int jind2 = jind1 + ng;
                    int jind3 = jind2 + ng;
                    K(k, jind1) = kf1(r, x1(0, k) - x1(0, j), sqr3L);
                    K(jind1, k) = K(k, jind1);
                    K(k, jind2) = kf1(r, x1(1, k) - x1(1, j), sqr3L);
                    K(jind2, k) = K(k, jind2);
                    K(k, jind3) = kf1(r, x1(2, k) - x1(2, j), sqr3L);
                    K(jind3, k) = K(k, jind3);
                }
            }
        }
    }

    return K;
}

EMatrixX
matern32_sparse_deriv1_3D(EMatrixX const &x1, std::vector<double> gradflag, EMatrixX const &x2, double scale_param) {
    auto dim = x1.rows();
    auto n = x1.cols();
    auto sqr3L = std::sqrt(double(3.)) / scale_param;
    EMatrixX K;

    int ng = 0;
    for (double &flag: gradflag) {
        if (flag > double(0.5)) {
            flag = double(ng);
            ng++;
        } else {
            flag = double(-1.0);
        }
    }

    int m = x2.cols();
    int m2 = m + m;
    int m3 = m2 + m;

    K = EMatrixX::Zero(n + ng * dim, m * (1 + dim));

    for (int k = 0; k < n; k++) {
        int kind1 = int(gradflag[k]) + int(n);
        int kind2 = kind1 + ng;
        int kind3 = kind2 + ng;
        for (int j = 0; j < m; j++) {
            double r = (x1.col(k) - x2.col(j)).norm();

            K(k, j) = kf(r, sqr3L);
            K(k, j + m) = kf1(r, x1(0, k) - x2(0, j), sqr3L);
            K(k, j + m2) = kf1(r, x1(1, k) - x2(1, j), sqr3L);
            K(k, j + m3) = kf1(r, x1(2, k) - x2(2, j), sqr3L);
            if (gradflag[k] > double(-0.5)) {
                K(kind1, j) = -K(k, j + m);
                K(kind2, j) = -K(k, j + m2);
                K(kind3, j) = -K(k, j + m3);
                K(kind1, j + m) = kf2(r, x1(0, k) - x2(0, j), x1(0, k) - x2(0, j), 1., sqr3L);
                K(kind1, j + m2) = kf2(r, x1(0, k) - x2(0, j), x1(1, k) - x2(1, j), double(0.0), sqr3L);
                K(kind1, j + m3) = kf2(r, x1(0, k) - x2(0, j), x1(2, k) - x2(2, j), double(0.0), sqr3L);
                K(kind2, j + m) = K(kind1, j + m2);
                K(kind2, j + m2) = kf2(r, x1(1, k) - x2(1, j), x1(1, k) - x2(1, j), 1., sqr3L);
                K(kind2, j + m3) = kf2(r, x1(1, k) - x2(1, j), x1(2, k) - x2(2, j), double(0.0), sqr3L);
                K(kind3, j + m) = K(kind1, j + m3);
                K(kind3, j + m2) = K(kind2, j + m3);
                K(kind3, j + m3) = kf2(r, x1(2, k) - x2(2, j), x1(2, k) - x2(2, j), 1., sqr3L);
            }
        }
    }

    return K;
}

// 2D
// FIXME: sigma of output noise should also be included
// EMatrixX matern32_sparse_deriv1_2D(EMatrixX const& x1,std::vector<double> m_gradflag_, double scale_param,
//                                 EVectorX const& m_sigx_,EVectorX const& m_siggrad_)
EMatrixX
matern32_sparse_deriv1_2D(
    EMatrixX const &x1,
    std::vector<double> gradflag,
    double scale_param,
    EVectorX const &sigx,
    double const &sigy,
    EVectorX const &siggrad)  // FIXED
{
    auto dim = x1.rows();
    auto n = x1.cols();
    double sqr3L = std::sqrt(double(3.)) / scale_param;
    double sqr3L2 = sqr3L * sqr3L;
    EMatrixX K;
    // label m_gradflag_=1 by numbers from 0, otherwise -1
    int ng = 0;  // # of m_gradflag_=1
    for (double &flag: gradflag) {
        if (flag > double(0.5)) {
            flag = double(ng);
            ng++;
        } else {
            flag = double(-1.0);
        }
    }

    K = EMatrixX::Zero(n + ng * dim, n + ng * dim);

    for (int k = 0; k < n; k++) {
        int kind1 = int(gradflag[k]) + int(n);
        int kind2 = kind1 + ng;

        for (int j = k; j < n; j++) {
            if (k == j) {
                // FIXME: sigma of output noise should also be included
                // m_k_(k,k) = 1.0 + m_sigx_(k);
                K(k, k) = 1. + sigx(k) + sigy;  // FIXED
                if (gradflag[k] > double(-0.5)) {
                    K(k, kind1) = double(0.0);
                    K(kind1, k) = double(0.0);
                    K(k, kind2) = double(0.0);
                    K(kind2, k) = double(0.0);
                    K(kind1, kind1) = sqr3L2 + siggrad(k);  // FIXED
                    K(kind1, kind2) = double(0.0);
                    K(kind2, kind1) = double(0.0);
                    K(kind2, kind2) = sqr3L2 + siggrad(k);
                }
            } else {
                // variance between mPosition of point k and mPosition of point j
                double r = (x1.col(k) - x1.col(j)).norm();
                K(k, j) = kf(r, sqr3L);
                K(j, k) = K(k, j);
                if (gradflag[k] > double(-1.)) {
                    // check papers:
                    // Gaussian Process training with input noise
                    // Derivative Observations in Gaussian Process Models of Dynamics Systems, etc.
                    // variance between gradient of point k and mPosition of point j
                    K(kind1, j) = -kf1(r, x1(0, k) - x1(0, j), sqr3L);
                    K(j, kind1) = K(kind1, j);
                    K(kind2, j) = -kf1(r, x1(1, k) - x1(1, j), sqr3L);
                    K(j, kind2) = K(kind2, j);

                    if (gradflag[j] > -1) {
                        // variance between gradient of point j and mPosition of point k
                        int jind1 = int(gradflag[j]) + int(n);
                        int jind2 = jind1 + ng;
                        K(k, jind1) = -K(j, kind1);
                        K(jind1, k) = K(k, jind1);
                        K(k, jind2) = -K(j, kind2);
                        K(jind2, k) = K(k, jind2);

                        K(kind1, jind1) = kf2(r, x1(0, k) - x1(0, j), x1(0, k) - x1(0, j), 1., sqr3L);
                        K(jind1, kind1) = K(kind1, jind1);
                        K(kind1, jind2) = kf2(r, x1(0, k) - x1(0, j), x1(1, k) - x1(1, j), double(0.0), sqr3L);
                        K(jind1, kind2) = K(kind1, jind2);
                        K(kind2, jind1) = K(kind1, jind2);
                        K(jind2, kind1) = K(kind1, jind2);
                        K(kind2, jind2) = kf2(r, x1(1, k) - x1(1, j), x1(1, k) - x1(1, j), 1., sqr3L);
                        K(jind2, kind2) = K(kind2, jind2);
                    }
                } else if (gradflag[j] > double(-1.)) {
                    int jind1 = int(gradflag[j]) + int(n);
                    int jind2 = jind1 + ng;
                    K(k, jind1) = kf1(r, x1(0, k) - x1(0, j), sqr3L);
                    K(jind1, k) = K(k, jind1);
                    K(k, jind2) = kf1(r, x1(1, k) - x1(1, j), sqr3L);
                    K(jind2, k) = K(k, jind2);
                }
            }
        }
    }

    return K;
}

EMatrixX
matern32_sparse_deriv1_2D(EMatrixX const &x1, std::vector<double> gradflag, EMatrixX const &x2, double scale_param) {
    auto dim = x1.rows();
    auto n = x1.cols();
    double sqr3L = std::sqrt(double(3.)) / scale_param;
    EMatrixX K;

    int ng = 0;
    for (double &flag: gradflag) {
        if (flag > double(0.5)) {
            flag = double(ng);
            ng++;
        } else {
            flag = double(-1.0);
        }
    }

    int m = x2.cols();
    int m2 = m * 2;

    K = EMatrixX::Zero(n + ng * dim, m * (1 + dim));
    for (int k = 0; k < n; k++) {
        int kind1 = int(gradflag[k]) + int(n);
        int kind2 = kind1 + ng;
        for (int j = 0; j < m; j++) {
            double r = (x1.col(k) - x2.col(j)).norm();

            K(k, j) = kf(r, sqr3L);
            K(k, j + m) = kf1(r, x1(0, k) - x2(0, j), sqr3L);
            K(k, j + m2) = kf1(r, x1(1, k) - x2(1, j), sqr3L);
            if (gradflag[k] > double(-0.5)) {
                K(kind1, j) = -K(k, j + m);
                K(kind2, j) = -K(k, j + m2);
                K(kind1, j + m) = kf2(r, x1(0, k) - x2(0, j), x1(0, k) - x2(0, j), 1., sqr3L);
                K(kind1, j + m2) = kf2(r, x1(0, k) - x2(0, j), x1(1, k) - x2(1, j), double(0.0), sqr3L);
                K(kind2, j + m) = K(kind1, j + m2);
                K(kind2, j + m2) = kf2(r, x1(1, k) - x2(1, j), x1(1, k) - x2(1, j), 1., sqr3L);
            }
        }
    }

    return K;
}
