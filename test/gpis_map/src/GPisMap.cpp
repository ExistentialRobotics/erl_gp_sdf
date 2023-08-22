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

#include "GPisMap.h"

#include <algorithm>
#include <numeric>
#include <thread>
#include <fstream>
#include "params.h"

TreeParam QuadTree::m_param_ =
    TreeParam(GPISMAP_TREE_MIN_HALF_LENGTH, GPISMAP_TREE_MAX_HALF_LENGTH, GPISMAP_TREE_INIT_ROOT_HALF_LENGTH, GPISMAP_TREE_CLUSTER_HALF_LENGTH);

#define MAX_RANGE double(3.e1)
#define MIN_RANGE double(2.e-1)

static inline bool
IsRangeValid(double r) {
    return (r < MAX_RANGE) && (r > MIN_RANGE);
}

static inline double
OccTest(double r_inv, double r_inv_0, double a) {
    return 2.0 * (1. / (1. + std::exp(-a * (r_inv - r_inv_0))) - 0.5);
}

static inline double
Saturate(double val, double min_val, double max_val) {
    return std::min(std::max(val, min_val), max_val);
}

static inline void
Polar2Cart(double a, double r, double &x, double &y) {
    x = r * std::cos(a);
    y = r * std::sin(a);
}

static inline void
Cart2Polar(double x, double y, double &a, double &r) {
    a = std::atan2(y, x);
    r = std::sqrt(x * x + y * y);
}

GPisMap::GPisMap()
    : m_tree_(nullptr),
      m_gpo_(nullptr),
      m_obs_numdata_(0) {
    Init();
}

GPisMap::GPisMap(GPisMapParam par)
    : m_setting_(par),
      m_tree_(nullptr),
      m_gpo_(nullptr),
      m_obs_numdata_(0) {
    Init();
}

GPisMap::~GPisMap() { Reset(); }

void
GPisMap::Init() {
    m_pose_tr_.resize(2);
    m_pose_r_.resize(4);
}

void
GPisMap::Reset() {
    if (m_tree_ != nullptr) {
        delete m_tree_;
        m_tree_ = nullptr;
    }

    if (m_gpo_ != nullptr) {
        delete m_gpo_;
        m_gpo_ = nullptr;
    }

    m_obs_numdata_ = 0;

    m_active_set_.clear();
}

bool
GPisMap::PreproData(double *datax, double *dataf, int n, std::vector<double> &pose) {
    if (datax == nullptr || dataf == nullptr || n < 1) return false;

    m_obs_theta_.clear();
    m_obs_range_.clear();
    m_obs_f_.clear();
    m_obs_xy_local_.clear();
    m_obs_xy_global_.clear();
    m_range_obs_max_ = 0.;

    if (pose.size() != 6) return false;

    std::copy(pose.begin(), pose.begin() + 2, m_pose_tr_.begin());  // 2 elements
    std::copy(pose.begin() + 2, pose.end(), m_pose_r_.begin());     // 4 elements

    m_obs_numdata_ = 0;
    for (int k = 0; k < n; k++) {
        double x_loc = 0.;
        double y_loc = 0.;
        if (IsRangeValid(dataf[k])) {
            if (m_range_obs_max_ < dataf[k]) { m_range_obs_max_ = dataf[k]; }
            m_obs_theta_.push_back(datax[k]);
            m_obs_range_.push_back(dataf[k]);
            m_obs_f_.push_back(1. / std::sqrt(dataf[k]));
            Polar2Cart(datax[k], dataf[k], x_loc, y_loc);
            m_obs_xy_local_.push_back(x_loc);
            m_obs_xy_local_.push_back(y_loc);
            x_loc += m_setting_.sensor_offset[0];
            y_loc += m_setting_.sensor_offset[1];
            /*
                R = [m_pose_r_[0], m_pose_r_[2]]
                    [m_pose_r_[1], m_pose_r_[3]]
             */
            m_obs_xy_global_.push_back(m_pose_r_[0] * x_loc + m_pose_r_[2] * y_loc + m_pose_tr_[0]);  // indices based on the matlab convention
            m_obs_xy_global_.push_back(m_pose_r_[1] * x_loc + m_pose_r_[3] * y_loc + m_pose_tr_[1]);
            m_obs_numdata_++;
        }
    }

    if (m_obs_numdata_ > 1) return true;

    return false;
}

void
GPisMap::Update(double *datax, double *dataf, int n, std::vector<double> &pose) {
    if (!PreproData(datax, dataf, n, pose)) return;

    // Step 1
    if (RegressObs()) {
        // Step 2
        UpdateMapPoints();
        // Step 3
        AddNewMeas();
        // Step 4
        UpdateGPs();
    }
}

bool
GPisMap::RegressObs() {
    int n[2];
    if (m_gpo_ == nullptr) { m_gpo_ = new ObsGP1D(); }

    n[0] = m_obs_numdata_;  // number of valid obstacle points, from PreproData
    m_gpo_->Reset();
    m_gpo_->train(m_obs_theta_.data(), m_obs_f_.data(), n);
    return m_gpo_->isTrained();
}

/**
 * @brief Find mNodes in the quad tree, and Update the map with the points of these mNodes.
 *
 */
void
GPisMap::UpdateMapPoints() {
    if (m_tree_ != nullptr && m_gpo_ != nullptr) {

        Aabb search_bb(m_pose_tr_[0], m_pose_tr_[1], m_range_obs_max_);
        std::vector<QuadTree *> quads;
        m_tree_->QueryNonEmptyLevelC(search_bb, quads);

        if (!quads.empty()) {

            double r_sq = m_range_obs_max_ * m_range_obs_max_;
            int k = 0;
            for (auto it = quads.begin(); it != quads.end(); it++, k++) {

                Point<double> ct = (*it)->GetCenter();
                double l = (*it)->GetHalfLength();
                double sqr_range = (ct.x - m_pose_tr_[0]) * (ct.x - m_pose_tr_[0]) + (ct.y - m_pose_tr_[1]) * (ct.y - m_pose_tr_[1]);

                if (sqr_range > (r_sq + 2. * l * l)) {  // out_of_range
                    continue;
                }

                std::vector<Point<double>> ext;
                ext.push_back((*it)->GetNw());
                ext.push_back((*it)->GetNe());
                ext.push_back((*it)->GetSw());
                ext.push_back((*it)->GetSe());

                int within_angle = 0;
                for (auto &itr: ext) {
                    double x_loc = m_pose_r_[0] * (itr.x - m_pose_tr_[0]) + m_pose_r_[1] * (itr.y - m_pose_tr_[1]);
                    double y_loc = m_pose_r_[2] * (itr.x - m_pose_tr_[0]) + m_pose_r_[3] * (itr.y - m_pose_tr_[1]);
                    x_loc -= m_setting_.sensor_offset[0];
                    y_loc -= m_setting_.sensor_offset[1];
                    double ang = 0.;
                    double r = 0.;
                    Cart2Polar(x_loc, y_loc, ang, r);
                    within_angle += int((ang > m_setting_.angle_obs_limit[0]) && (ang < m_setting_.angle_obs_limit[1]));
                }

                if (within_angle == 0) { continue; }

                // Get all the mNodes
                std::vector<std::shared_ptr<Node>> nodes;
                (*it)->GetAllChildrenNonEmptyNodes(nodes);

                ReEvalPoints(nodes);
            }
        }
    }
}

void
GPisMap::ReEvalPoints(std::vector<std::shared_ptr<Node>> &nodes) {
    // placeholders
    EMatrixX amx(1, 1);
    EVectorX r_inv_0(1);
    EVectorX var(1);
    double ang = 0.;
    double r = 0.;

    // For each point
    for (auto &node: nodes) {

        Point<double> pos = node->GetPos();

        double x_loc = m_pose_r_[0] * (pos.x - m_pose_tr_[0]) + m_pose_r_[1] * (pos.y - m_pose_tr_[1]);
        double y_loc = m_pose_r_[2] * (pos.x - m_pose_tr_[0]) + m_pose_r_[3] * (pos.y - m_pose_tr_[1]);
        x_loc -= m_setting_.sensor_offset[0];
        y_loc -= m_setting_.sensor_offset[1];
        Cart2Polar(x_loc, y_loc, ang, r);

        amx(0) = ang;
        m_gpo_->test(amx, r_inv_0, var);

        // If unobservable, continue
        if (var(0) > m_setting_.obs_var_thre) continue;

        double oc = OccTest(1. / std::sqrt(r), r_inv_0(0), r * 30.);

        // If unobservable, continue
        if (oc < -0.1) { continue; }

        // gradient in the local coord.
        Point<double> grad = node->GetGrad();
        double grad_loc[2];
        grad_loc[0] = m_pose_r_[0] * grad.x + m_pose_r_[1] * grad.y;
        grad_loc[1] = m_pose_r_[2] * grad.x + m_pose_r_[3] * grad.y;

        /// Compute a new mPosition
        // Iteratively move along the normal direction.
        double abs_oc = std::fabs(oc);
        double dx = m_setting_.delx;
        double x_new[2] = {x_loc, y_loc};
        double r_new = r;
        for (int i = 0; i < 10 && abs_oc > 0.02; i++) {
            // move one step
            // (the direction is determined by the occupancy sign,
            //  the step size is heuristically determined according to iteration.)
            if (oc < 0) {
                x_new[0] += grad_loc[0] * dx;
                x_new[1] += grad_loc[1] * dx;
            } else {
                x_new[0] -= grad_loc[0] * dx;
                x_new[1] -= grad_loc[1] * dx;
            }

            // test the new point

            Cart2Polar(x_new[0], x_new[1], ang, r_new);
            amx(0) = ang;
            m_gpo_->test(amx, r_inv_0, var);

            if (var(0) > m_setting_.obs_var_thre) break;
            else {
                double oc_new = OccTest(1. / std::sqrt(r_new), r_inv_0(0), r_new * double(30.0));
                double abs_oc_new = std::fabs(oc_new);

                if (abs_oc_new < 0.02) {
                    abs_oc = abs_oc_new;
                    break;
                } else if (oc * oc_new < 0.) {
                    dx = 0.5 * dx;
                } else {
                    dx = 1.1 * dx;
                }

                abs_oc = abs_oc_new;
                oc = oc_new;
            }
        }

        // Compute its gradient and uncertainty
        double x_perturb[4] = {1., -1., 0., 0.};
        double y_perturb[4] = {0., 0., 1., -1.};
        double occ[4] = {-1., -1., -1., -1.};
        double occ_mean = 0.;
        double r_0_sum = 0.;
        double r_0_sqr_sum = 0.;
        for (int i = 0; i < 4; i++) {
            x_perturb[i] = x_new[0] + m_setting_.delx * x_perturb[i];
            y_perturb[i] = x_new[1] + m_setting_.delx * y_perturb[i];

            double d;
            Cart2Polar(x_perturb[i], y_perturb[i], ang, d);
            amx(0, 0) = ang;
            m_gpo_->test(amx, r_inv_0, var);

            if (var(0) > m_setting_.obs_var_thre) { break; }
            occ[i] = OccTest(1. / std::sqrt(d), r_inv_0(0), d * double(30.0));
            occ_mean += occ[i];
            double r_0 = 1. / (r_inv_0(0) * r_inv_0(0));
            r_0_sqr_sum += r_0 * r_0;
            r_0_sum += r_0;
        }

        if (var(0) > m_setting_.obs_var_thre)  // invalid
            continue;

        occ_mean *= 0.25;
        Point<double> grad_new_loc, grad_new;
        double norm_grad_new;

        grad_new_loc.x = (occ[0] - occ[1]) / m_setting_.delx;
        grad_new_loc.y = (occ[2] - occ[3]) / m_setting_.delx;
        norm_grad_new = std::sqrt(grad_new_loc.x * grad_new_loc.x + grad_new_loc.y * grad_new_loc.y);

        if (norm_grad_new < double(1.e-6)) {  // uncertainty increased
            node->UpdateNoise(2.0 * node->GetPosNoise(), 2.0 * node->GetGradNoise());
            continue;
        }

        double r_var = (r_0_sqr_sum - r_0_sum * r_0_sum * 0.25) / (double(3.0) * m_setting_.delx);
        double grad_noise = 1.;
        grad_new_loc.x = grad_new_loc.x / norm_grad_new;
        grad_new_loc.y = grad_new_loc.y / norm_grad_new;
        double dist_noise = Saturate(r_new * r_new, 1., double(100.));

        double dist = std::sqrt(x_new[0] * x_new[0] + x_new[1] * x_new[1]);
        double view_ang = -(x_new[0] * grad_new_loc.x + x_new[1] * grad_new_loc.y) / dist;
        double view_sq_ang = std::max(view_ang * view_ang, double(1.e-2));
        double view_noise = (1. - view_sq_ang) / view_sq_ang;

        double noise = m_setting_.min_position_noise * (dist_noise + view_noise) + abs_oc;  // extra term: abs_oc
        // extra term: r_var and 0.1 * view_noise
        grad_noise = Saturate(std::fabs(occ_mean) + r_var, m_setting_.min_grad_noise, grad_noise) + double(0.1) * view_noise;

        // local to global coord.
        Point<double> pos_new;
        x_new[0] += m_setting_.sensor_offset[0];
        x_new[1] += m_setting_.sensor_offset[1];
        pos_new.x = m_pose_r_[0] * x_new[0] + m_pose_r_[2] * x_new[1] + m_pose_tr_[0];
        pos_new.y = m_pose_r_[1] * x_new[0] + m_pose_r_[3] * x_new[1] + m_pose_tr_[1];
        grad_new.x = m_pose_r_[0] * grad_new_loc.x + m_pose_r_[2] * grad_new_loc.y;
        grad_new.y = m_pose_r_[1] * grad_new_loc.x + m_pose_r_[3] * grad_new_loc.y;

        double noise_old = node->GetPosNoise();
        double grad_noise_old = node->GetGradNoise();

        double pos_noise_sum = (noise_old + noise);
        double grad_noise_sum = grad_noise_old + grad_noise;

        // Now, Update
        // if (grad_noise_old > 0.5 || grad_noise_old > 0.6){  // FIXME: redundant condition
        if (grad_noise_old > 0.5) {
        } else {
            // Bayes Update
            // Position Update
            pos_new.x = (noise * pos.x + noise_old * pos_new.x) / pos_noise_sum;
            pos_new.y = (noise * pos.y + noise_old * pos_new.y) / pos_noise_sum;
            double new_dist = std::sqrt((pos.x - pos_new.x) * (pos.x - pos_new.x) + (pos.y - pos_new.y) * (pos.y - pos_new.y)) * 0.5;

            // Normal Update
            Point<double> tempv;
            // NOTE: both grad and grad_new are unit vectors
            tempv.x = grad.x * grad_new.x + grad.y * grad_new.y;   // inner_prod(grad, grad_new) --> cos
            tempv.y = -grad.y * grad_new.x + grad.x * grad_new.y;  // cross_prod(grad, grad_new) --> sin
            double ang_dist = std::atan2(tempv.y, tempv.x) * noise / pos_noise_sum;
            double sina = std::sin(ang_dist);
            double cosa = std::cos(ang_dist);
            grad_new.x = cosa * grad.x - sina * grad.y;
            grad_new.y = sina * grad.x + cosa * grad.y;

            // Noise Update
            grad_noise = std::min((double) 1., std::max(grad_noise * grad_noise_old / grad_noise_sum + new_dist, m_setting_.map_noise_param));

            noise = std::max((noise * noise_old / pos_noise_sum + new_dist), m_setting_.map_noise_param);
        }
        // Remove
        m_tree_->Remove(node, m_active_set_);

        if (noise > 1. && grad_noise > double(0.6)) {
            continue;
        } else {
            // try inserting
            std::shared_ptr<Node> p(new Node(pos_new));
            std::unordered_set<QuadTree *> vec_inserted;

            bool succeeded = false;
            if (!m_tree_->IsNotNew(p)) {
                succeeded = m_tree_->Insert(p, vec_inserted);
                if (succeeded) {
                    if (!m_tree_->IsRoot()) { m_tree_ = m_tree_->GetRoot(); }
                }
            }

            if ((succeeded == 0) || vec_inserted.empty())  // if failed, then continue to test the next point
                continue;

            // Update the point
            p->UpdateData(m_setting_.fbias, noise, grad_new, grad_noise, NodeType::kHit);

            for (auto itv: vec_inserted) m_active_set_.insert(itv);
        }
    }
}

void
GPisMap::AddNewMeas() {
    // Create if not initialized
    if (m_tree_ == nullptr) { m_tree_ = new QuadTree(Point<double>(0., 0.)); }
    EvalPoints();
}

void
GPisMap::EvalPoints() {

    if (m_tree_ == nullptr || m_obs_numdata_ < 1) return;

    std::ofstream ofs("GpisMap_EvalPoints.txt");

    // For each point
    for (int k = 0; k < m_obs_numdata_; k++) {
        int k_2 = 2 * k;

        // placeholder;
        EVectorX r_inv_0(1);
        EVectorX var(1);
        EMatrixX amx(1, 1);

        amx(0, 0) = m_obs_theta_[k];
        m_gpo_->test(amx, r_inv_0, var);

        if (var(0) > m_setting_.obs_var_thre) { continue; }

        /////////////////////////////////////////////////////////////////
        // Try inserting
        Point<double> pt(m_obs_xy_global_[k_2], m_obs_xy_global_[k_2 + 1]);

        std::shared_ptr<Node> p(new Node(pt));
        std::unordered_set<QuadTree *> vec_inserted;

        bool succeeded;
        succeeded = m_tree_->Insert(p, vec_inserted);
        if (succeeded) {
            if (!m_tree_->IsRoot()) { m_tree_ = m_tree_->GetRoot(); }
        }

        if ((succeeded == 0) || vec_inserted.empty())  // if failed, then continue to test the next point
            continue;                                  // may fail if the point to Insert is too close to existing ones.

        /////////////////////////////////////////////////////////////////
        // if succeeded, then compute surface normal and uncertainty
        double x_perturb[4] = {1., -1., 0., 0.};
        double y_perturb[4] = {0., 0., 1., -1.};
        double occ[4] = {-1., -1., -1., -1.};
        double occ_mean = 0.;
        int i = 0;
        for (; i < 4; i++) {
            x_perturb[i] = m_obs_xy_local_[k_2] + m_setting_.delx * x_perturb[i];
            y_perturb[i] = m_obs_xy_local_[k_2 + 1] + m_setting_.delx * y_perturb[i];

            double a, r;
            Cart2Polar(x_perturb[i], y_perturb[i], a, r);

            amx(0, 0) = a;
            m_gpo_->test(amx, r_inv_0, var);

            if (var(0) > m_setting_.obs_var_thre) { break; }
            occ[i] = OccTest(1. / std::sqrt(r), r_inv_0(0), r * double(30.0));
            occ_mean += occ[i];
        }
        occ_mean *= 0.25;

        if (var(0) > m_setting_.obs_var_thre) {  // fail to calculate the gradient
            m_tree_->Remove(p);
            continue;
        }

        double noise;
        double grad_noise = 1.;
        Point<double> grad;

        grad.x = (occ[0] - occ[1]) / m_setting_.delx;
        grad.y = (occ[2] - occ[3]) / m_setting_.delx;
        double norm_grad = grad.x * grad.x + grad.y * grad.y;
        norm_grad = std::sqrt(norm_grad);
        if (norm_grad > double(1.e-6)) {
            double grad_loc_x = grad.x / norm_grad;
            double grad_loc_y = grad.y / norm_grad;

            grad.x = m_pose_r_[0] * grad_loc_x + m_pose_r_[2] * grad_loc_y;
            grad.y = m_pose_r_[1] * grad_loc_x + m_pose_r_[3] * grad_loc_y;

            double dist_noise = Saturate(m_obs_range_[k] * m_obs_range_[k], 1., double(100.));
            grad_noise = Saturate(std::fabs(occ_mean), m_setting_.min_grad_noise, grad_noise);
            double dist = std::sqrt(m_obs_xy_local_[k_2] * m_obs_xy_local_[k_2] + m_obs_xy_local_[k_2 + 1] * m_obs_xy_local_[k_2 + 1]);
            double view_ang = -(m_obs_xy_local_[k_2] * grad_loc_x + m_obs_xy_local_[k_2 + 1] * grad_loc_y) / dist;
            double view_sq_ang = std::max(view_ang * view_ang, double(1.e-2));
            double view_noise = (1. - view_sq_ang) / view_sq_ang;  // var(angle) = alpha_angle * tan(r_angle)^2, different from the paper
            noise = m_setting_.min_position_noise * (dist_noise + view_noise);
        } else {
            noise = 1.;
        }

        /////////////////////////////////////////////////////////////////
        // Update the point
        p->UpdateData(m_setting_.fbias, noise, grad, grad_noise, NodeType::kHit);

        for (auto it: vec_inserted) { m_active_set_.insert(it); }

        ofs << k << '\t' << occ_mean << '\t' << grad.x << '\t' << grad.y << '\t' << noise << '\t' << grad_noise << std::endl;
    }

    ofs.close();
}

void
GPisMap::UpdateGPsKernel(int thread_idx, int start_idx, int end_idx, QuadTree **nodes_to_update) const {
    (void) thread_idx;

    std::vector<std::shared_ptr<Node>> res;
    for (int i = start_idx; i < end_idx; ++i) {
        if (nodes_to_update[i] != nullptr) {
            Point<double> ct = (nodes_to_update[i])->GetCenter();
            double l = (nodes_to_update[i])->GetHalfLength();
            Aabb search_bb(ct.x, ct.y, l * double(4.0));
            res.clear();
            m_tree_->QueryRange(search_bb, res);
            if (!res.empty()) {
                std::shared_ptr<OnGpis> gp(new OnGpis(m_setting_.map_scale_param, m_setting_.map_noise_param));
                gp->Train(res);
                (nodes_to_update[i])->Update(gp);
            }
        }
    }
}

void
GPisMap::UpdateGPs() {
    std::unordered_set<QuadTree *> update_set(m_active_set_);

    size_t num_threads = std::thread::hardware_concurrency();
    auto *threads = new std::thread[num_threads];

    size_t num_threads_to_use;

    for (auto it: m_active_set_) {

        Point<double> ct = it->GetCenter();
        double l = it->GetHalfLength();
        Aabb search_bb(ct.x, ct.y, double(4.0) * l);
        std::vector<QuadTree *> qs;
        m_tree_->QueryNonEmptyLevelC(search_bb, qs);
        if (!qs.empty()) {
            for (auto &q: qs) { update_set.insert(q); }
        }
    }

    size_t num_elements = update_set.size();
    auto **nodes_to_update = new QuadTree *[num_elements];
    int it_counter = 0;
    for (auto it = update_set.begin(); it != update_set.end(); ++it, ++it_counter) { nodes_to_update[it_counter] = *it; }

    if (num_elements < num_threads) {
        num_threads_to_use = num_elements;
    } else {
        num_threads_to_use = num_threads;
    }
    size_t num_leftovers = num_elements % num_threads_to_use;
    size_t batch_size = num_elements / num_threads_to_use;
    size_t element_cursor = 0;
    for (size_t i = 0; i < num_leftovers; ++i) {
        threads[i] = std::thread(&GPisMap::UpdateGPsKernel, this, i, element_cursor, element_cursor + batch_size + 1, nodes_to_update);
        element_cursor += batch_size + 1;
    }
    for (size_t i = num_leftovers; i < num_threads_to_use; ++i) {
        threads[i] = std::thread(&GPisMap::UpdateGPsKernel, this, i, element_cursor, element_cursor + batch_size, nodes_to_update);
        element_cursor += batch_size;
    }

    for (size_t i = 0; i < num_threads_to_use; ++i) { threads[i].join(); }

    delete[] nodes_to_update;
    delete[] threads;
    // clear active set once all the jobs for Update are done.
    m_active_set_.clear();
}

void
GPisMap::TestKernel(int thread_idx, int start_idx, int end_idx, double *x, double *res) const {
    (void) thread_idx;

    auto var_thre = double(0.4);  // TO-DO

    for (int i = start_idx; i < end_idx; ++i) {
        EVectorX xt(2);
        xt << x[2 * i], x[2 * i + 1];

        int k_6 = 6 * i;

        // query Cs
        auto search_bb_half_size = m_setting_.map_scale_param * double(4.0);
        std::vector<QuadTree *> quads;
        std::vector<double> sqdst;
        while (search_bb_half_size < m_tree_->GetHalfLength()) {
            Aabb search_bb(xt(0), xt(1), search_bb_half_size);
            quads.clear();
            sqdst.clear();
            m_tree_->QueryNonEmptyLevelC(search_bb, quads, sqdst);
            if (!quads.empty()) { break; }
            search_bb_half_size *= double(2.0);
        }

        if (quads.empty()) { continue; }

        res[k_6] = 0.;
        res[k_6 + 3] = 1. + m_setting_.map_noise_param;  // variance of sdf value
        if (quads.size() == 1) {
            std::shared_ptr<OnGpis> gp = quads[0]->GetGp();
            if (gp != nullptr) { gp->Test2DPoint(xt, res[k_6], res[k_6 + 1], res[k_6 + 2], res[k_6 + 3], res[k_6 + 4], res[k_6 + 5]); }
        } else if (sqdst.size() > 1) {

            // sort by distance
            std::vector<int> idx(sqdst.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(std::begin(idx), std::end(idx), [&](int i_1, int i_2) { return sqdst[i_1] < sqdst[i_2]; });

            // get THE FIRST gp pointer
            std::shared_ptr<OnGpis> gp = quads[idx[0]]->GetGp();
            if (gp != nullptr) { gp->Test2DPoint(xt, res[k_6], res[k_6 + 1], res[k_6 + 2], res[k_6 + 3], res[k_6 + 4], res[k_6 + 5]); }

            if ((res[k_6 + 3] > var_thre) && (sqdst.size() > 1)) {
                double f_2[4];
                double grad_2[4 * 2];
                double var_2[4 * 3];

                var_2[0] = res[k_6 + 3];
                size_t numc = sqdst.size() - 1;
                if (numc > 3) numc = 3;
                bool need_wsum = true;
                for (size_t m = 0; m < numc; m++) {
                    size_t m_1 = m + 1;
                    size_t m_2 = m_1 * 2;  // 2, 4, 6
                    size_t m_3 = m_1 * 3;  // 3, 6, 9
                    gp = quads[idx[m_1]]->GetGp();
                    gp->Test2DPoint(xt, f_2[m_1], grad_2[m_2], grad_2[m_2 + 1], var_2[m_3], var_2[m_3 + 1], var_2[m_3 + 2]);
                }

                if (need_wsum) {
                    f_2[0] = res[k_6];
                    grad_2[0] = res[k_6 + 1];
                    grad_2[1] = res[k_6 + 2];
                    var_2[1] = res[k_6 + 4];
                    var_2[2] = res[k_6 + 5];
                    idx.resize(numc + 1);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::sort(std::begin(idx), std::end(idx), [&](int i_1, int i_2) { return var_2[i_1 * 3] < var_2[i_2 * 3]; });

                    if (var_2[idx[0] * 3] < var_thre) {
                        res[k_6] = f_2[idx[0]];
                        res[k_6 + 1] = grad_2[idx[0] * 2];
                        res[k_6 + 2] = grad_2[idx[0] * 2 + 1];
                        res[k_6 + 3] = var_2[idx[0] * 3];
                        res[k_6 + 4] = var_2[idx[0] * 3 + 1];
                        res[k_6 + 5] = var_2[idx[0] * 3 + 2];
                    } else {
                        double w_1 = (var_2[idx[0] * 3] - var_thre);
                        double w_2 = (var_2[idx[1] * 3] - var_thre);

                        double w_12 = w_1 + w_2;

                        res[k_6] = (w_2 * f_2[idx[0]] + w_1 * f_2[idx[1]]) / w_12;
                        res[k_6 + 1] = (w_2 * grad_2[idx[0] * 2] + w_1 * grad_2[idx[1] * 2]) / w_12;
                        res[k_6 + 2] = (w_2 * grad_2[idx[0] * 2 + 1] + w_1 * grad_2[idx[1] * 2 + 1]) / w_12;
                        res[k_6 + 3] = (w_2 * var_2[idx[0] * 3] + w_1 * var_2[idx[1] * 3]) / w_12;
                        res[k_6 + 4] = (w_2 * var_2[idx[0] * 3 + 1] + w_1 * var_2[idx[1] * 3 + 1]) / w_12;
                        res[k_6 + 5] = (w_2 * var_2[idx[0] * 3 + 2] + w_1 * var_2[idx[1] * 3 + 2]) / w_12;
                    }
                }
            }
        }

        res[k_6] -= m_setting_.fbias;
    }
}

bool
GPisMap::Test(double *x, int dim, size_t len, double *res) {
    if (x == nullptr || dim != m_map_dimension_ || len < 1) return false;

    size_t num_threads = std::thread::hardware_concurrency();
    size_t num_threads_to_use;
    if (len < num_threads) {
        num_threads_to_use = len;
    } else {
        num_threads_to_use = num_threads;
    }

    auto *threads = new std::thread[num_threads_to_use];

    size_t num_leftovers = len % num_threads_to_use;
    size_t batch_size = len / num_threads_to_use;
    size_t element_cursor = 0;

    for (size_t i = 0; i < num_leftovers; ++i) {
        threads[i] = std::thread(&GPisMap::TestKernel, this, i, element_cursor, element_cursor + batch_size + 1, x, res);
        element_cursor += batch_size + 1;
    }
    for (size_t i = num_leftovers; i < num_threads_to_use; ++i) {
        threads[i] = std::thread(&GPisMap::TestKernel, this, i, element_cursor, element_cursor + batch_size, x, res);
        element_cursor += batch_size;
    }

    for (size_t i = 0; i < num_threads_to_use; ++i) { threads[i].join(); }

    delete[] threads;

    return true;
}
