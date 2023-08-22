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

#include "ObsGP.h"

#include <Eigen/Cholesky>
#include <thread>

#include "covFnc.h"

using namespace Eigen;

///////////////////////////////////////////////////////////
// GPou
///////////////////////////////////////////////////////////

void
GPou::train(const EMatrixX &xt, const EVectorX &f) {
    //    int dim = xt.rows();
    int n = xt.cols();

    if (n > 0) {
        x = xt;
        K = OrnsteinUhlenbeck(xt, scale, noise);
        // GPIS paper: eq(10)
        L = K.llt().matrixL();
        alpha = f;
        L.template triangularView<Lower>().solveInPlace(alpha);
        L.transpose().template triangularView<Upper>().solveInPlace(alpha);

        trained = true;
    }
}

void
GPou::test(const EMatrixX &xt, EVectorX &f, EVectorX &var) {

    EMatrixX K = OrnsteinUhlenbeck(x, xt, scale);
    f = K.transpose() * alpha;
    // calculate m_l_.inv() * m_k_
    L.template triangularView<Lower>().solveInPlace(K);
    // calculate m_k_.transpose() * m_l_.inv().transpose() * m_l_.inv() * m_k_
    K = K.array().pow(2);
    EVectorX v = K.colwise().sum();
    // FIXME: for test points, I don't think we should add `noise`, which is the observation noise
    // var = 1 + noise -v.head(xt.cols()).array();
    var = 1 - v.head(xt.cols()).array();  // FIXED
}

///////////////////////////////////////////////////////////
// ObsGP
///////////////////////////////////////////////////////////

void
ObsGP::Reset() {
    trained = false;
    gps.clear();
}

///////////////////////////////////////////////////////////
// ObsGP 1D
///////////////////////////////////////////////////////////

void
ObsGP1D::Reset() {
    ObsGP::Reset();
    range.clear();
    nSamples = 0;
}

void
ObsGP1D::train(double xt[], double f[], int N[]) {
    Reset();

    if ((N[0] > 0) && (xt != nullptr)) {
        nSamples = N[0];
        nGroup = nSamples / (param.group_size) + 1;

        range.push_back(xt[0]);
        for (int n = 0; n < (nGroup - 1); n++) {
            // Make sure there are enough overlap

            if (n < nGroup - 2) {
                int i1 = n * param.group_size;
                int i2 = i1 + param.group_size + param.overlap;

                range.push_back(xt[i2 - param.overlap / 2]);

                Map<ERowVectorX> x_(xt + i1, param.group_size + param.overlap);
                Map<EVectorX> f_(f + i1, param.group_size + param.overlap);
                // Train each gp group
                std::shared_ptr<GPou> g(new GPou());
                g->train(x_, f_);

                gps.push_back(std::move(g));

            } else {  // the last two groups split in half
                // the second to last
                int i_1 = n * param.group_size;
                int i_2 = i_1 + (nSamples - i_1 + param.overlap) / 2;
                range.push_back(xt[i_2 - param.overlap / 2]);

                Map<ERowVectorX> x_(xt + i_1, i_2 - i_1);
                Map<EVectorX> f_(f + i_1, i_2 - i_1);
                std::shared_ptr<GPou> g(new GPou());
                g->train(x_, f_);
                gps.push_back(std::move(g));
                n++;

                // the last one
                i_1 = i_1 + (nSamples - i_1 - param.overlap) / 2;
                i_2 = nSamples - 1;
                range.push_back(xt[i_2]);
                new (&x_) Map<ERowVectorX>(xt + i_1, i_2 - i_1 + 1);
                new (&f_) Map<EVectorX>(f + i_1, i_2 - i_1 + 1);

                std::shared_ptr<GPou> g_last(new GPou());
                g_last->train(x_, f_);
                gps.push_back(std::move(g_last));
            }
        }

        trained = true;
    }
}

void
ObsGP1D::test(const EMatrixX &xt, EVectorX &val, EVectorX &var) {

    if (!isTrained()) { return; }

    auto dim = xt.rows();
    auto n = xt.cols();

    if (dim == 1) {
        double lim_l = (*(range.begin()) + param.margin);
        double lim_r = (*(range.end() - 1) - param.margin);
        for (int k = 0; k < n; k++) {

            EVectorX f = val.segment(k, 1);
            EVectorX v = var.segment(k, 1);
            var(k) = 1e6;
            // find the corresponding group
            if (xt(0, k) < lim_l) {  // boundary 1
                ;
            } else if (xt(0, k) > lim_r) {  // boundary 2
                ;
            } else {  // in-between
                int j = 0;
                for (auto it = (range.begin() + 1); it != range.end(); it++, j++) {
                    if (xt(0, k) >= *(it - 1) && xt(0, k) <= *it) {
                        // and test
                        if (gps[j]->isTrained()) {
                            gps[j]->test(xt.block(0, k, 1, 1), f, v);
                            val(k) = f(0);
                            var(k) = v(0);
                        }
                        break;
                    }
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////
// ObsGP 2D
///////////////////////////////////////////////////////////

void
ObsGP2D::ClearGPs() {
    ObsGP::Reset();
}

void
ObsGP2D::Reset() {
    ClearGPs();

    repartition = true;
}

void
ObsGP2D::ComputePartition(double val[], int ni, int nj) {
    // number of data grid
    szSamples[0] = ni;
    szSamples[1] = nj;

    // number group gp grid
    nGroup[0] = (szSamples[0] - param.overlap) / param.group_size + 1;
    nGroup[1] = (szSamples[1] - param.overlap) / param.group_size + 1;

    Ind_i0.clear();
    Ind_i1.clear();
    Ind_j0.clear();
    Ind_j1.clear();
    Val_i.clear();
    Val_j.clear();

    // [0]-Range for each group
    Val_i.push_back(val[0]);
    for (int n = 0; n < nGroup[0]; n++) {

        int i_0 = n * param.group_size;
        int i_1 = i_0 + param.group_size + param.overlap - 1;
        if (n < nGroup[0] - 1) {
            Val_i.push_back(val[2 * (i_1 - param.overlap / 2)]);
        } else if (n == nGroup[0] - 1) {
            i_1 = szSamples[0] - 1;
            Val_i.push_back(val[2 * i_1]);
        }
        Ind_i0.push_back(i_0);
        Ind_i1.push_back(i_1);
    }

    // [1]-Range for each group
    Val_j.push_back(val[1]);
    for (int m = 0; m < nGroup[1]; m++) {

        int j_0 = m * param.group_size;
        int j_1 = j_0 + param.group_size + param.overlap - 1;
        if (m < nGroup[1] - 1) {
            Val_j.push_back(val[2 * (j_1 - param.overlap / 2) * szSamples[0] + 1]);
        } else {
            // the last one
            j_1 = szSamples[1] - 1;
            Val_j.push_back(val[2 * j_1 * szSamples[0] + 1]);
        }

        Ind_j0.push_back(j_0);
        Ind_j1.push_back(j_1);
    }

    if ((!Ind_i0.empty()) && (!Ind_i1.empty()) && (!Ind_j0.empty()) && (!Ind_j1.empty())) repartition = false;
}

void
ObsGP2D::GetNumValidPoints(std::vector<int> &n_pts) {
    n_pts.clear();
    for (auto &gp: gps) {
        if (gp != nullptr) n_pts.push_back(gp->getNumSamples());
        else
            n_pts.push_back(0);
    }
}

void
ObsGP2D::TrainValidPoints(double xt[], double f[]) {
    if (repartition) return;

    ClearGPs();
    gps.resize(nGroup[0] * nGroup[1], nullptr);

    auto itj_0 = Ind_j0.begin();
    auto itj_1 = Ind_j1.begin();
    int m = 0;
    for (; (itj_0 != Ind_j0.end() && itj_1 != Ind_j1.end() && m < nGroup[1]); itj_0++, itj_1++, m++) {

        auto iti_0 = Ind_i0.begin();
        auto iti_1 = Ind_i1.begin();
        int n = 0;
        for (; (iti_0 != Ind_i0.end() && iti_1 != Ind_i1.end() && n < nGroup[0]); iti_0++, iti_1++, n++) {
            // Dynamic array for valid inputs
            std::vector<double> x_valid;  // 2D array
            std::vector<double> f_valid;

            for (int j = *itj_0; j <= *itj_1; j++) {
                for (int i = *iti_0; i <= *iti_1; i++) {
                    int ind = j * szSamples[0] + i;
                    if (f[ind] > 0) {
                        x_valid.push_back(xt[ind * 2]);
                        x_valid.push_back(xt[ind * 2 + 1]);
                        f_valid.push_back(f[ind]);
                    }
                }
            }

            // If not empty
            if (x_valid.size() > 1) {
                // matrix/vector map from vector
                Map<EMatrixX> x_val(x_valid.data(), 2, f_valid.size());
                Map<EVectorX> f_val(f_valid.data(), f_valid.size());

                // Train each gp group
                std::shared_ptr<GPou> g(new GPou());
                g->train(x_val, f_val);

                gps[m * nGroup[0] + n] = std::move(g);
            }
        }
    }

    trained = true;
}

void
ObsGP2D::train(double xt[], double f[], int N[]) {
    if ((N[0] > 0) && (N[1] > 0) && (xt != 0)) {

        if ((szSamples[0] != N[0]) || (szSamples[1] != N[1]) || repartition) { ComputePartition(xt, N[0], N[1]); }
        TrainValidPoints(xt, f);
    }
}

void
ObsGP2D::Train(double xt[], double f[], int n[], std::vector<int> &num_samples) {
    train(xt, f, n);
    GetNumValidPoints(num_samples);
}

void
ObsGP2D::TestKernel(int thread_idx, int start_idx, int end_idx, const EMatrixX &xt, EVectorX &val, EVectorX &var) {
    (void) thread_idx;

    for (int k = start_idx; k < end_idx; ++k) {

        EVectorX f = val.segment(k, 1);
        EVectorX v = var.segment(k, 1);
        var(k) = 1e6;
        // find the corresponding group
        if (xt(0, k) < *(Val_i.begin()) + param.margin) {  // boundary 1
            ;
        } else if (xt(0, k) > *(Val_i.end() - 1) - param.margin) {  // boundary 2
            ;
        } else if (xt(1, k) < *(Val_j.begin()) + param.margin) {  // boundary 1
            ;
        } else if (xt(1, k) > *(Val_j.end() - 1) - param.margin) {  // boundary 2
            ;
        } else {  // in-between

            int n = 0;
            for (auto it = (Val_i.begin() + 1); it != Val_i.end(); it++, n++) {
                if (xt(0, k) < *it) { break; }
            }

            int m = 0;
            for (auto it = (Val_j.begin() + 1); it != Val_j.end(); it++, m++) {
                if (xt(1, k) < *it) { break; }
            }

            size_t gp_ind = m * nGroup[0] + n;
            if (gp_ind < gps.size() && gps[gp_ind] != nullptr) {
                if (gps[gp_ind]->isTrained()) {
                    // and test
                    gps[gp_ind]->test(xt.block(0, k, 2, 1), f, v);
                    val(k) = f(0);
                    var(k) = v(0);
                }
            }
        }
    }
}

void
ObsGP2D::test(const EMatrixX &xt, EVectorX &val, EVectorX &var) {

    if (!isTrained() || xt.rows() != 2) { return; }

    long n = xt.cols();

    long num_threads = std::thread::hardware_concurrency();
    long num_threads_to_use;
    if (n < num_threads) {
        num_threads_to_use = n;
    } else {
        num_threads_to_use = num_threads;
    }
    auto *threads = new std::thread[num_threads_to_use];

    long num_leftovers = n % num_threads_to_use;
    long batch_size = n / num_threads_to_use;
    long element_cursor = 0;

    for (int i = 0; i < num_leftovers; ++i) {
        threads[i] = std::thread(&ObsGP2D::TestKernel, this, i, element_cursor, element_cursor + batch_size + 1, std::ref(xt), std::ref(val), std::ref(var));
        element_cursor += batch_size + 1;
    }
    for (long i = num_leftovers; i < num_threads_to_use; ++i) {
        threads[i] = std::thread(&ObsGP2D::TestKernel, this, i, element_cursor, element_cursor + batch_size, std::ref(xt), std::ref(val), std::ref(var));
        element_cursor += batch_size;
    }

    for (int i = 0; i < num_threads_to_use; ++i) { threads[i].join(); }

    delete[] threads;
}
