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

#include <memory>
#include <vector>

#include "params.h"

enum NodeType { kNone = 0, kHit = 1, kFree, kCluster };

template<typename T>
struct Point {
    T x = 0;
    T y = 0;

    Point(T x, T y)
        : x(x),
          y(y) {}

    Point() = default;
};

template<typename T>
struct Point3 {
    T x = 0;
    T y = 0;
    T z = 0;

    Point3(T x, T y, T z)
        : x(x),
          y(y),
          z(z) {}

    Point3() = default;
};

class Node {
#if defined(BUILD_TEST)
public:
#endif
    Point<double> m_pos_;
    Point<double> m_grad_;
    double m_val_;
    double m_pose_sig_;
    double m_grad_sig_;
    NodeType m_nt_;

public:
    Node(Point<double> pos, double val, double pose_sig, Point<double> grad, double grad_sig, NodeType n);

    explicit Node(Point<double> pos, NodeType nt = NodeType::kNone);

    Node();

    void
    UpdateData(double val, double pose_sig, Point<double> grad, double grad_sig, NodeType n);

    void
    UpdateNoise(double pose_sig, double grad_sig);

    [[nodiscard]] const Point<double>&
    GetPos() const {
        return m_pos_;
    }

    [[nodiscard]] const Point<double>&
    GetGrad() const {
        return m_grad_;
    }

    [[nodiscard]] double
    GetPosX() const {
        return m_pos_.x;
    }

    [[nodiscard]] double
    GetPosY() const {
        return m_pos_.y;
    }

    [[nodiscard]] double
    GetGradX() const {
        return m_grad_.x;
    }

    [[nodiscard]] double
    GetGradY() const {
        return m_grad_.y;
    }

    [[nodiscard]] double
    GetVal() const {
        return m_val_;
    }

    [[nodiscard]] double
    GetPosNoise() const {
        return m_pose_sig_;
    };

    [[nodiscard]] double
    GetGradNoise() const {
        return m_grad_sig_;
    };

    [[nodiscard]] NodeType
    GetType() const {
        return m_nt_;
    }
};

class Node3 {
    Point3<double> m_pos_;
    Point3<double> m_grad_;
    double m_val_;
    double m_pose_sig_;
    double m_grad_sig_;
    NodeType m_nt_;

public:
    Node3(Point3<double> pos, double val, double pose_sig, Point3<double> grad, double grad_sig, NodeType n = NodeType::kNone);

    explicit Node3(Point3<double> pos, NodeType nt = NodeType::kNone);

    Node3();

    void
    UpdateData(double val, double pose_sig, Point3<double> grad, double grad_sig, NodeType n = NodeType::kNone);

    void
    UpdateNoise(double pose_sig, double grad_sig);

    const Point3<double>&
    GetPos() {
        return m_pos_;
    }

    const Point3<double>&
    GetGrad() {
        return m_grad_;
    }

    [[nodiscard]] double
    GetPosX() const {
        return m_pos_.x;
    }

    [[nodiscard]] double
    GetPosY() const {
        return m_pos_.y;
    }

    [[nodiscard]] double
    GetPosZ() const {
        return m_pos_.z;
    }

    [[nodiscard]] double
    GetGradX() const {
        return m_grad_.x;
    }

    [[nodiscard]] double
    GetGradY() const {
        return m_grad_.y;
    }

    [[nodiscard]] double
    GetGradZ() const {
        return m_grad_.z;
    }

    [[nodiscard]] double
    GetVal() const {
        return m_val_;
    }

    [[nodiscard]] double
    GetPosNoise() const {
        return m_pose_sig_;
    };

    [[nodiscard]] double
    GetGradNoise() const {
        return m_grad_sig_;
    };

    NodeType
    GetType() {
        return m_nt_;
    }
};

//////////////////////////////////////////////////////////////////////////
// Parameters

// Observation GP
typedef struct ObsGpParam_ {
    // Npte:
    // ObsGP is implemented to use the Ornstein-Uhlenbeck covariance function,
    // which has a form of k(r)=exp(-r/l) (See covFnc.h)
    double scale;  // the scale parameter l
    double noise;  // the noise parameter of the measurement
    // currently use a constant value
    // could be potentially modified to have heteroscedastic noise
    // Note:
    // ObsGP is implemented to have overlapping partitioned GPs.
    double margin;  // used to decide if valid range
    // (don't use if too close to boundary
    //  because the derivates are hard to sample)
    int overlap;     // the overlapping parameters: number of samples to overlap
    int group_size;  // the number of samples to group together
    // (the actual group size will be (group_size+overlap)
    ObsGpParam_() = default;

    ObsGpParam_(double s, double n, double m, int ov, int gsz)
        : scale(s),
          noise(n),
          margin(m),
          overlap(ov),
          group_size(gsz) {}
} ObsGpParam;

// GPIS (SDF)
typedef struct OnGpisParam_ {
    // Note:
    // OnlineGPIS is implemented to use the Matern class covariance function with (nu=2/3),
    // which has a form of k(r)=(1+sqrt(3)*r/l)exp(-sqrt(3)*r/l) (See covFnc.h)
    double scale;  // the scale parameter l
    double noise;  // the default noise parameter of the measurement
    // currently use heteroscedastic noise acoording to a noise model
    double noise_deriv;  // the default noise parameter of the derivative measurement
    // currently use a noise model by numerical computation.
    OnGpisParam_() = default;

    OnGpisParam_(double s, double n, double nd)
        : scale(s),
          noise(n),
          noise_deriv(nd) {}
} OnGpisParam;

// QuadTree (2D) and OcTree (3D)
typedef struct tree_param_ {
    double initroot_halfleng;
    double min_halfleng;  // minimum (leaf) resolution of tree
    double min_halfleng_sqr;
    double max_halfleng;  // maximum (root) resolution of tree
    double max_halfleng_sqr;
    double cluster_halfleng;  // the resolution of GP clusters
    double cluster_halfleng_sqr;

public:
    tree_param_()
        : initroot_halfleng(DEFAULT_TREE_INIT_ROOT_HALFLENGTH),
          min_halfleng(DEFAULT_TREE_MIN_HALFLENGTH),
          min_halfleng_sqr(DEFAULT_TREE_MIN_HALFLENGTH * DEFAULT_TREE_MIN_HALFLENGTH),
          max_halfleng(DEFAULT_TREE_MAX_HALFLENGTH),
          max_halfleng_sqr(DEFAULT_TREE_MAX_HALFLENGTH * DEFAULT_TREE_MAX_HALFLENGTH),
          cluster_halfleng(DEFAULT_TREE_CLUSTER_HALFLENGTH),
          cluster_halfleng_sqr(DEFAULT_TREE_CLUSTER_HALFLENGTH * DEFAULT_TREE_CLUSTER_HALFLENGTH) {}

    tree_param_(double mi, double ma, double ini, double c)
        : initroot_halfleng(ini),
          min_halfleng(mi),
          min_halfleng_sqr(mi * mi),
          max_halfleng(ma),
          max_halfleng_sqr(ma * ma),
          cluster_halfleng(c),
          cluster_halfleng_sqr(c * c) {}
} TreeParam;
