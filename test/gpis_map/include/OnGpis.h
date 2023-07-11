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

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "params.h"
#include "strct.h"

typedef Eigen::MatrixX<double> EMatrixX;
typedef Eigen::VectorX<double> EVectorX;
typedef Eigen::RowVectorX<double> ERowVectorX;

typedef std::vector<std::shared_ptr<Node>> VecNode;
typedef std::vector<std::shared_ptr<Node3>> VecNode3;

class OnGpis {
#if defined(BUILD_TEST)
public:
    EVectorX m_y_;
    EMatrixX m_k_;
    EVectorX m_sigx_;
    EVectorX m_siggrad_;
#endif
    EMatrixX m_x_;
    EMatrixX m_l_;
    EVectorX m_alpha_;
    std::vector<double> m_gradflag_;

    OnGpisParam m_param_;  // defined in strct.h
    // currently noise m_param_ is not effective
    double m_three_over_scale_;
    bool m_trained_;
    int m_n_samples_;

public:
    OnGpis()
        : m_param_(DEFAULT_MAP_SCALE_PARAM, DEFAULT_MAP_NOISE_PARAM, DEFAULT_MAP_NOISE_PARAM),
          m_three_over_scale_(double(3.0) / (DEFAULT_MAP_SCALE_PARAM * DEFAULT_MAP_SCALE_PARAM)),
          m_trained_(false),
          m_n_samples_(0) {}

    OnGpis(double s, double n)
        : m_param_(s, n, n),
          m_three_over_scale_(double(3.0) / (s * s)),
          m_trained_(false),
          m_n_samples_(0) {}

    void
    Reset();

    [[nodiscard]] bool
    IsTrained() const {
        return m_trained_;
    }

    void
    SetGpScaleParam(double l) {
        m_param_.scale = l;
    }

    void
    Train(const VecNode &samples);

    void
    Train(const VecNode3 &samples);

    //    void
    //    test(const EMatrixX &xt, EVectorX &val, EMatrixX &gradval, EVectorX &var);

    void  // used in 3D
    TestSinglePoint(const EVectorX &xt, double &val, double grad[], double var[]);

    void
    Test2DPoint(const EVectorX &xt, double &val, double &gradx, double &grady, double &var_val, double &var_gradx, double &var_grady);
};
