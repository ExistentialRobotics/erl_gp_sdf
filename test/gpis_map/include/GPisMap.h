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

#include "ObsGP.h"
#include "OnGpis.h"
#include "params.h"
#include "quadtree.h"

typedef struct GPisMapParam_ {
    double delx;   // numerical step delta (e.g. surface normal sampling)
    double fbias;  // constant map bias values (mean of GP)
    double sensor_offset[2] = {0, 0};
    double angle_obs_limit[2] = {0, 0};
    double obs_var_thre;  // threshold for variance of ObsGP
    //  - If var(prediction) > v_thre, then don't rely on the prediction.
    double min_position_noise;
    double min_grad_noise;

    double map_scale_param;
    double map_noise_param;

    GPisMapParam_() {
        delx = GPISMAP_DELX;
        fbias = GPISMAP_FBIAS;
        obs_var_thre = GPISMAP_OBS_VAR_THRE;
        sensor_offset[0] = GPISMAP_SENSOR_OFFSET_0;
        sensor_offset[1] = GPISMAP_SENSOR_OFFSET_1;
        angle_obs_limit[0] = GPISMAP_ANGLE_OBS_LIMIT_0;
        angle_obs_limit[1] = GPISMAP_ANGLE_OBS_LIMIT_1;
        min_position_noise = GPISMAP_MIN_POS_NOISE;
        min_grad_noise = GPISMAP_MIN_GRAD_NOISE;
        map_scale_param = GPISMAP_MAP_SCALE;
        map_noise_param = GPISMAP_MAP_NOISE;
    }

    GPisMapParam_(GPisMapParam_ &par) {
        delx = par.delx;
        fbias = par.fbias;
        obs_var_thre = par.obs_var_thre;
        sensor_offset[0] = par.sensor_offset[0];
        sensor_offset[1] = par.sensor_offset[1];
        min_position_noise = par.min_position_noise;
        min_grad_noise = par.min_grad_noise;
        map_scale_param = par.map_scale_param;
        map_noise_param = par.map_noise_param;
    }
} GPisMapParam;

class GPisMap {
#if defined(ERL_BUILD_TEST)
public:
#else
protected:
#endif
    GPisMapParam m_setting_;

    QuadTree *m_tree_;
    std::unordered_set<QuadTree *> m_active_set_;
    const int m_map_dimension_ = 2;

    void
    Init();

    bool
    PreproData(double *datax, double *dataf, int n, std::vector<double> &pose);

    bool
    RegressObs();

    void
    UpdateMapPoints();

    void
    ReEvalPoints(std::vector<std::shared_ptr<Node> > &nodes);

    void
    EvalPoints();

    void
    AddNewMeas();

    void
    UpdateGPs();

    ObsGP1D *m_gpo_;
    std::vector<double> m_obs_theta_;
    std::vector<double> m_obs_range_;
    std::vector<double> m_obs_f_;
    std::vector<double> m_obs_xy_local_;
    std::vector<double> m_obs_xy_global_;
    std::vector<double> m_pose_tr_;
    std::vector<double> m_pose_r_;
    int m_obs_numdata_;
    double m_range_obs_max_ = 0.;

public:
    GPisMap();

    explicit GPisMap(GPisMapParam par);

    ~GPisMap();

    void
    Reset();

    void
    Update(double *datax, double *dataf, int n, std::vector<double> &pose);

    bool
    Test(double *x, int dim, size_t len, double *res);

    int
    GetMapDimension() const {
        return m_map_dimension_;
    }

private:
    void
    TestKernel(int i_1, int i_2, int end_idx, double *x, double *res) const;

    void
    UpdateGPsKernel(int thread_idx, int start_idx, int end_idx, QuadTree **nodes_to_update) const;
};
