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

#include "params.h"
#include "strct.h"
#include <Eigen/Dense>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

typedef Eigen::MatrixX<double> EMatrixX;
typedef Eigen::VectorX<double> EVectorX;
typedef Eigen::RowVectorX<double> ERowVectorX;

// This class builds a GP regressor using the Ornstein-Uuhlenbeck covariance function.
// NOTE: See covFnc.h)
class GPou {
#if defined(BUILD_TEST)
    public:
#endif
    EMatrixX x;
    EMatrixX L;
    EVectorX alpha;  // inv(m_k_+Kx) * y

    int dim; // need this?
    const double scale = DEFAULT_OBSGP_SCALE_PARAM;
    const double noise = DEFAULT_OBSGP_NOISE_PARAM;
    bool trained = false;

public:

    GPou() = default;

    void
    reset() { trained = false; }

    [[nodiscard]] bool
    isTrained() const { return trained; }

    int
    getNumSamples() { return x.cols(); }

    void
    train(const EMatrixX &xt, const EVectorX &f);

    void
    test(const EMatrixX &xt, EVectorX &f, EVectorX &var);

};

// This is a base class to build a partitioned GP regressor, holding multiple local GPs using GPou.
class ObsGP {
#if defined(BUILD_TEST)
    public:
#else
protected:
#endif
    bool trained{};

    std::vector<std::shared_ptr<GPou>> gps;  // pointer to the local GPs

public:
    ObsGP() = default;

    virtual ~ObsGP() = default;

    [[nodiscard]] bool
    isTrained() const { return trained; }

    virtual void
    Reset();

    virtual void
    train(double xt[], double f[], int N[]) = 0;

    virtual void
    test(const EMatrixX &xt, EVectorX &val, EVectorX &var) = 0;
};

// This class implements ObsGP for 1D input.
class ObsGP1D : public ObsGP {
#if defined(BUILD_TEST)
    public:
#endif
    int nGroup{};         // number of local GPs
    int nSamples;       // number of total input points.

    std::vector<double> range;   // partitioned range for test

    const ObsGpParam param = {DEFAULT_OBSGP_SCALE_PARAM,
                              DEFAULT_OBSGP_NOISE_PARAM,
                              DEFAULT_OBSGP_MARGIN,
                              DEFAULT_OBSGP_OVERLAP_SZ,
                              DEFAULT_OBSGP_GROUP_SZ};

public:
    ObsGP1D() : nSamples(0) {}

    void
    Reset() override;

    // NOTE: In 1D, it must be f > 0.
    void
    train(double xt[], double f[], int N[]) override;

    void
    test(const EMatrixX &xt, EVectorX &val, EVectorX &var) override;

};

// This class implements ObsGP for regular 2D input.
class ObsGP2D : public ObsGP {
    int nGroup[2];       // dimension of local GPs
    int szSamples[2];    // dimension of input data
    bool repartition;

    // pre-computed partition indices
    std::vector<int> Ind_i0;
    std::vector<int> Ind_i1;
    std::vector<int> Ind_j0;
    std::vector<int> Ind_j1;

    // pre-computed partition values
    std::vector<double> Val_i;
    std::vector<double> Val_j;

    void
    ClearGPs();

    void
    ComputePartition(double val[], int ni, int nj);

    void
    TrainValidPoints(double xt[], double f[]);

    const ObsGpParam param = {DEFAULT_OBSGP_SCALE_PARAM,
                              DEFAULT_OBSGP_NOISE_PARAM,
                              DEFAULT_OBSGP_MARGIN2,
                              DEFAULT_OBSGP_OVERLAP_SZ2,
                              DEFAULT_OBSGP_GROUP_SZ2};
public:

    void
    Reset() override;

    void
    GetNumValidPoints(std::vector<int> &n_pts);

    // NOTE: In 2D, the input xt must be a regular 2D array of size N[0] x N[1].
    //       If not f > 0, the point is considered invalid.
    void
    train(double xt[], double f[], int N[]) override;

    void
    Train(double xt[], double f[], int n[], std::vector<int> &num_samples);

    void
    test(const EMatrixX &xt, EVectorX &val, EVectorX &var) override;

private:
    void
    TestKernel(int thread_idx,
                int start_idx,
                int end_idx,
                const EMatrixX &xt,
                EVectorX &val,
                EVectorX &var);
};
