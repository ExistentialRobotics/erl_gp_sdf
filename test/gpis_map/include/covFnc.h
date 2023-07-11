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
 * covFnc.h
 * This header includes definitions of covariance functions for Gaussian Processes.
 *
 * Authors: Bhoram Lee <bhoram.lee@gmail.com>
 */

#pragma once

#include <Eigen/Dense>
#include <vector>

#include "strct.h"

typedef Eigen::MatrixX<double> EMatrixX;
typedef Eigen::VectorX<double> EVectorX;

//////////////////////////////////////////
// Covariance matrix computation using the Ornstein-Uhlenbeck cov function.
// Note:
//       - Dims(x) MUST BE 2 or 3 (See covFnc.cpp)
//       - Used for GP Implicit Surface (GPIS) jointly with Gradient
//       - m_gradflag_ indicates whether the gradient is available or not for each point
//////////////////////////////////////////

// covariances for x1 (input points) with different noise params for inputs
EMatrixX
Matern32SparseDeriv1(
    EMatrixX const& x_1,
    std::vector<double> gradflag,  // FIXME: sigma of output noise should also be considered!
    double scale_param,
    EVectorX const& sigx,
    double const& sigy,
    EVectorX const& siggrad);

// covariances for x_1 (input points) and x2 (test points)
EMatrixX
Matern32SparseDeriv1(EMatrixX const& x_1, std::vector<double> gradflag, EMatrixX const& x_2, double scale_param);

//////////////////////////////////////////
// Covariance matrix computation using the Ornstein-Uhlenbeck cov function.
// Note:
//       - Dims(x) > 0
//       - Used for Observation Regression
//////////////////////////////////////////

// covariances for x1 (input points) with a constant noise m_param_
EMatrixX
OrnsteinUhlenbeck(EMatrixX const& x_1, double scale_param, double sigx);

// covariances for x1 (input points) with different noise params for inputs
EMatrixX
OrnsteinUhlenbeck(EMatrixX const& x_1, double scale_param, EVectorX const& sigx);

// covariances for x1 (input points) and x2 (test points)
EMatrixX
OrnsteinUhlenbeck(EMatrixX const& x_1, EMatrixX const& x_2, double scale_param);
