#include "erl_sdf_mapping/gpis/gpis_map_base_2d.hpp"

#include <algorithm>
#include <numeric>
#include <thread>
#include <memory>
#include <unordered_set>

namespace erl::sdf_mapping::gpis {

    static double
    Clip(const double x, const double min_x, const double max_x) {
        return std::max(std::min(x, max_x), min_x);
    }

    static void
    Cartesian2Polar(const Eigen::Ref<const Eigen::Vector2d> &xy, double &r, double &angle) {
        r = xy.norm();
        angle = std::atan2(xy.y(), xy.x());
    }

    GpisMapBase2D::GpisMapBase2D()
        : GpisMapBase2D(std::make_shared<Setting>()) {}

    GpisMapBase2D::GpisMapBase2D(const std::shared_ptr<Setting> &setting)
        : m_setting_(setting),
          m_gp_theta_(gaussian_process::LidarGaussianProcess1D::Create(setting->gp_theta)),
          m_node_container_constructor_([&]() { return GpisNodeContainer2D::Create(m_setting_->node_container); }),
          m_xy_perturb_{[&]() -> Eigen::Matrix24d {
              Eigen::Matrix24d out;
              // clang-format off
              out << m_setting_->perturb_delta, -m_setting_->perturb_delta, 0., 0.,
                     0., 0., m_setting_->perturb_delta, -m_setting_->perturb_delta;
              // clang-format on
              return out;
          }()} {}

    bool
    GpisMapBase2D::Update(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const Eigen::Ref<const Eigen::VectorXd> &distances,
        const Eigen::Ref<const Eigen::Matrix23d> &pose) {

        // regress observation
        m_gp_theta_->Train(angles, distances, pose);
        if (m_gp_theta_->IsTrained()) { return LaunchUpdate(); }
        return false;
    }

    bool
    GpisMapBase2D::ComputeGradient1(
        const Eigen::Ref<const Eigen::Vector2d> &xy_local,
        Eigen::Ref<Eigen::Vector2d> gradient,
        double &occ_mean,
        double &distance_var) {
        Eigen::Scalard distance_pred, var;
        double occ[4];
        double distance_sum, distance_square_mean;
        occ_mean = 0.;
        distance_sum = 0.;
        distance_square_mean = 0.;
        gradient.x() = 0.;
        gradient.y() = 0.;

        double distance;
        Eigen::Scalard angle;
        for (int j = 0; j < 4; ++j) {
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            if (m_gp_theta_->ComputeOcc(angle, distance, distance_pred, var, occ[j])) {
                occ_mean += occ[j];
            } else {
                return false;
            }

            distance_sum += distance_pred[0];
            distance_square_mean += distance_pred[0] * distance_pred[0];
        }

        occ_mean *= 0.25;
        // 4 samples in total, to calculate the unbiased variance
        // var(r) = sum((r_i - mean(r))^2) / 3. = (mean(r^2) - mean(r) * mean(r)) * 4. / 3. = (sum(r^2) - sum(r) * sum(r) * 0.25) / 3.
        // to Remove the numerical approximation's influence, let var(r) = var(r) / delta
        distance_var = (distance_square_mean - distance_sum * distance_sum * 0.25) / (double(3.) * m_setting_->perturb_delta);

        gradient.x() = (occ[0] - occ[1]) / m_setting_->perturb_delta;
        gradient.y() = (occ[2] - occ[3]) / m_setting_->perturb_delta;

        return true;
    }

    bool
    GpisMapBase2D::ComputeVariance(
        const Eigen::Ref<const Eigen::Vector2d> &xy_local,
        const double &distance,
        const double &distance_var,
        const double &occ_mean_abs,
        const double &occ_abs,
        const bool new_point,
        Eigen::Ref<Eigen::Vector2d> grad_local,
        Eigen::Ref<Eigen::Vector2d> grad_global,
        double &var_position,
        double &var_gradient) const {

        const double gradient_norm = grad_local.norm();

        if (gradient_norm <= double(1.e-6)) {
            grad_global << grad_local;
            var_position = m_setting_->compute_variance->zero_gradient_position_var;
            var_gradient = m_setting_->compute_variance->zero_gradient_gradient_var;
            return false;
        }

        grad_local.x() /= gradient_norm;
        grad_local.y() /= gradient_norm;
        grad_global << m_gp_theta_->LocalToGlobalSo2(grad_local);

        const double var_distance = Clip(distance * distance, m_setting_->compute_variance->min_distance_var, m_setting_->compute_variance->max_distance_var);
        const double cos_view_angle = -xy_local.dot(grad_local) / xy_local.norm();
        const double cos2_view_angle = std::max(cos_view_angle * cos_view_angle, 1.e-2);  // avoid zero division
        const double var_direction = (1. - cos2_view_angle) / cos2_view_angle;            // tan^2(theta), the bigger the angle, the bigger the variance

        if (new_point) {
            var_position = m_setting_->compute_variance->position_var_alpha * (var_distance + var_direction);
            var_gradient = Clip(occ_mean_abs, m_setting_->compute_variance->min_gradient_var, m_setting_->compute_variance->max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = m_setting_->compute_variance->position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient = Clip(occ_mean_abs + distance_var, m_setting_->compute_variance->min_gradient_var, m_setting_->compute_variance->max_gradient_var) +
                           0.1 * var_direction;
        }

        return true;
    }

    void
    GpisMapBase2D::UpdateMapPoints() {
        // first observation, bad observation or failed to regress the observation
        if (m_quadtree_ == nullptr || !m_gp_theta_->IsTrained()) { return; }

        auto &kTrainBuffer = m_gp_theta_->GetTrainBuffer();
        std::vector<std::shared_ptr<geometry::IncrementalQuadtree>> clusters_to_update;
        geometry::Aabb2D observed_area(kTrainBuffer.position, kTrainBuffer.max_distance);
        m_quadtree_->CollectNonEmptyClusters(observed_area, clusters_to_update);
        if (clusters_to_update.empty()) { return; }

        double r_2 = kTrainBuffer.max_distance * kTrainBuffer.max_distance;
        for (auto &cluster: clusters_to_update) {
            auto &kArea = cluster->GetArea();
            double square_dist = (kTrainBuffer.position - kArea.center).squaredNorm();
            double l = kArea.half_sizes[0];

            // approximately out of m_range_
            if (square_dist > (r_2 + 2 * l * l)) { continue; }

            // auto corners = kArea.getCorners();
            double distance_old;
            Eigen::Scalard angle, distance_pred, var;
            std::vector<std::shared_ptr<geometry::Node>> nodes;
            for (int corner_idx = 0; corner_idx < 4; ++corner_idx) {
                Eigen::Vector2d corner = kArea.corner(geometry::Aabb2D::CornerType(corner_idx));
                // global frame to local frame
                double r;
                Cartesian2Polar(m_gp_theta_->GlobalToLocalSe2(corner), r, angle[0]);

                if ((angle[0] >= m_gp_theta_->GetSetting()->train_buffer->valid_angle_min) &&
                    (angle[0] <= m_gp_theta_->GetSetting()->train_buffer->valid_angle_max)) {
                    nodes.clear();
                    cluster->CollectNodes(nodes);
                    // cluster->CollectNodesOfType(Node::Type::SURFACE, m_nodes_);

                    // re-evaluate map points
                    double occ, occ_abs;

                    for (auto &node: nodes) {
                        auto &xy_global_old = std::dynamic_pointer_cast<GpisNode2D>(node)->position;
                        auto xy_local_old = m_gp_theta_->GlobalToLocalSe2(xy_global_old);
                        Cartesian2Polar(xy_local_old, distance_old, angle[0]);
                        if (!m_gp_theta_->ComputeOcc(angle, distance_old, distance_pred, var, occ)) { continue; }

                        if (occ < m_setting_->update_map_points->min_observable_occ) { continue; }

                        auto node_data = node->GetData<GpisData2D>();
                        auto grad_local_old = m_gp_theta_->GlobalToLocalSo2(node_data->gradient);

                        // compute a new position for the point
                        occ_abs = std::fabs(occ);
                        auto delta = m_setting_->perturb_delta;
                        auto xy_local_new = xy_local_old;
                        double distance_new = distance_old;
                        if (occ_abs > m_setting_->update_map_points->max_surface_abs_occ) {  // we need to adjust the point to make it closer to the surface.
                            for (int i = 0; i < m_setting_->update_map_points->max_adjust_tries; ++i) {
                                // move one step
                                // the direction is determined by the occupancy sign, the step GetSize is heuristically determined according to iteration.
                                if (occ < 0.) {
                                    // point is inside the obstacle
                                    xy_local_new += grad_local_old * delta;
                                } else if (occ > 0.) {
                                    // point is outside the obstacle
                                    xy_local_new -= grad_local_old * delta;
                                }

                                // test the new point
                                double occ_new;
                                Cartesian2Polar(xy_local_new, distance_new, angle[0]);
                                if (m_gp_theta_->ComputeOcc(angle, distance_new, distance_pred, var, occ_new)) {
                                    auto occ_abs_new = std::fabs(occ_new);
                                    if (occ_abs_new < m_setting_->update_map_points->max_surface_abs_occ) {
                                        occ_abs = occ_abs_new;
                                        break;
                                    } else if (occ * occ_new < 0.) {
                                        delta *= double(0.5);  // too big, make it smaller
                                    } else {
                                        delta *= 1.1;
                                    }

                                    occ_abs = occ_abs_new;
                                    occ = occ_new;
                                } else {
                                    break;  // fail to estimate occ
                                }
                            }
                        }

                        // compute new gradient and uncertainty
                        double occ_mean, var_distance;
                        Eigen::Vector2d grad_local_new;
                        if (!ComputeGradient1(xy_local_new, grad_local_new, occ_mean, var_distance)) { continue; }

                        Eigen::Vector2d grad_global_new;
                        double var_position_new, var_gradient_new;
                        if (!ComputeVariance(
                                xy_local_new,
                                distance_new,
                                var_distance,
                                std::fabs(occ_mean),
                                occ_abs,
                                false,
                                grad_local_new,
                                grad_global_new,
                                var_position_new,
                                var_gradient_new)) {
                            node_data->var_position *= 2.;
                            node_data->var_gradient *= 2.;
                            continue;
                        }

                        // Remove the node and mark this cluster as active such that it will be updated by UpdateGpSdf
                        m_quadtree_->Remove(node);  // will Insert a new node if variance is not too high
                        m_active_clusters_.insert(cluster);

                        auto xy_global_new = m_gp_theta_->LocalToGlobalSe2(xy_local_new);
                        double &var_position_old = node_data->var_position;
                        double &var_gradient_old = node_data->var_gradient;

                        if (var_gradient_old <= m_setting_->update_map_points->max_valid_gradient_var) {
                            // do bayes Update only when the old result is not too bad
                            double var_position_sum = var_position_new + var_position_old;
                            double var_gradient_sum = var_gradient_new + var_gradient_old;

                            // position Update
                            xy_global_new = (xy_global_new * var_position_old + xy_global_old * var_position_new) / var_position_sum;
                            double distance = (xy_global_new - xy_global_old).norm() * double(0.5);

                            // gradient Update
                            auto &grad_global_old = node_data->gradient;
                            double angle_dist = std::atan2(
                                                    grad_global_old.x() * grad_global_new.y() - grad_global_old.y() * grad_global_new.x(),
                                                    grad_global_old.x() * grad_global_new.x() + grad_global_old.y() * grad_global_new.y()) *
                                                var_position_new / var_position_sum;
                            double sin = std::sin(angle_dist);
                            double cos = std::cos(angle_dist);
                            // rotate grad_global_old by angle_dist
                            grad_global_new.x() = cos * grad_global_old.x() - sin * grad_global_old.y();
                            grad_global_new.y() = sin * grad_global_old.x() + cos * grad_global_old.y();

                            // variance Update
                            var_position_new =
                                std::max((var_position_new * var_position_old) / var_position_sum + distance, m_setting_->gp_theta->sensor_range_var);
                            var_gradient_new = Clip(
                                (var_gradient_new * var_gradient_old) / var_gradient_sum + distance,
                                m_setting_->compute_variance->min_gradient_var,
                                m_setting_->compute_variance->max_gradient_var);
                        }

                        if ((var_position_new > m_setting_->update_map_points->max_bayes_position_var) &&
                            (var_gradient_new > m_setting_->update_map_points->max_bayes_gradient_var)) {
                            continue;  // too bad, skip
                        }

                        // Insert the point to the tree
                        auto new_node = std::make_shared<GpisNodeContainer2D::Node>(xy_global_new);
                        std::shared_ptr<geometry::IncrementalQuadtree> inserted_leaf = m_quadtree_->Insert(new_node, m_quadtree_);
                        if (inserted_leaf == nullptr) { continue; }  // may fail if the point to Insert is too close to existing ones.
                        m_active_clusters_.insert(inserted_leaf->GetCluster());
                        double distance;
                        if (m_setting_->update_gp_sdf->add_offset_points) {
                            distance = 0.;  // except for the surface point, add one more pair of offset points. So, set surface distance to be zero.
                        } else {
                            distance = m_setting_->update_gp_sdf->offset_distance;  // offset the surface point
                        }
                        new_node->GetData<GpisData2D>()->UpdateData(distance, grad_global_new, var_position_new, var_gradient_new);
                    }

                    // this observed cluster is updated
                    break;
                }
            }
        }
    }

    inline bool
    GpisMapBase2D::ComputeGradient2(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Ref<Eigen::Vector2d> gradient, double &occ_mean) {
        Eigen::Scalard distance_pred, var;
        double occ[4];
        occ_mean = 0.;
        gradient.x() = 0.;
        gradient.y() = 0.;

        double distance;
        Eigen::Scalard angle;
        for (int j = 0; j < 4; ++j) {
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            if (m_gp_theta_->ComputeOcc(angle, distance, distance_pred, var, occ[j])) {
                occ_mean += occ[j];
            } else {
                return false;
            }
        }

        occ_mean *= 0.25;
        gradient.x() = (occ[0] - occ[1]) / m_setting_->perturb_delta;
        gradient.y() = (occ[2] - occ[3]) / m_setting_->perturb_delta;

        return true;
    }

    void
    GpisMapBase2D::AddNewMeasurement() {
        if (m_quadtree_ == nullptr) {
            m_quadtree_ = geometry::IncrementalQuadtree::Create(
                m_setting_->quadtree,
                geometry::Aabb2D({0., 0.}, m_setting_->init_tree_half_size),
                m_node_container_constructor_);
        }

        auto &kTrainBuffer = m_gp_theta_->GetTrainBuffer();
        auto n = kTrainBuffer.Size();
        Eigen::Scalard angle, f, var;
        for (ssize_t i = 0; i < n; ++i) {
            angle[0] = kTrainBuffer.vec_angles[i];
            m_gp_theta_->Test(angle, f, var, true);

            if (var[0] > m_setting_->gp_theta->max_valid_range_var) { continue; }  // uncertain point, drop it

            // Insert the point to the tree
            auto node = std::make_shared<GpisNodeContainer2D::Node>(kTrainBuffer.mat_xy_global.col(i));
            std::shared_ptr<geometry::IncrementalQuadtree> inserted_leaf = m_quadtree_->Insert(node, m_quadtree_);
            if (inserted_leaf == nullptr) { continue; }  // may fail if the point to Insert is too close to existing ones.

            double occ_mean;
            Eigen::Vector2d grad_local;
            if (!ComputeGradient2(kTrainBuffer.mat_xy_local.col(i), grad_local, occ_mean)) {
                m_quadtree_->Remove(node);  // failed to estimate the grad_local
                continue;
            }

            // mark this cluster as active
            m_active_clusters_.insert(inserted_leaf->GetCluster());

            Eigen::Vector2d grad_global;
            double var_position, var_gradient;
            ComputeVariance(
                kTrainBuffer.mat_xy_local.col(i),
                kTrainBuffer.vec_ranges[i],
                0.,
                std::fabs(occ_mean),
                0.,
                true,
                grad_local,
                grad_global,
                var_position,
                var_gradient);  // distanceVar and occAbs are not used for a new point

            double distance;
            if (m_setting_->update_gp_sdf->add_offset_points) {
                distance = 0.;  // except for the surface point, add one more pair of offset points. So, set surface distance to be zero.
            } else {
                distance = m_setting_->update_gp_sdf->offset_distance;  // offset the surface point
            }
            node->GetData<GpisData2D>()->UpdateData(distance, grad_global, var_position, var_gradient);
        }
    }

    void
    GpisMapBase2D::UpdateGpX() {
        if (m_active_clusters_.empty()) { return; }

        std::unordered_set<std::shared_ptr<geometry::IncrementalQuadtree>> active_set(m_active_clusters_);

        for (auto &kCluster: m_active_clusters_) {
            auto &kArea = kCluster->GetArea();
            geometry::Aabb2D search_area(kArea.center, kArea.half_sizes[0] * m_setting_->update_gp_sdf->search_area_scale);
            std::vector<std::shared_ptr<geometry::IncrementalQuadtree>> clusters;
            m_quadtree_->CollectNonEmptyClusters(search_area, clusters);
            for (auto &cluster: clusters) { active_set.insert(cluster); }
        }

        m_clusters_to_update_.clear();
        m_clusters_to_update_.insert(m_clusters_to_update_.begin(), active_set.begin(), active_set.end());
        active_set.clear();
        auto num_clusters = m_clusters_to_update_.size();

        auto num_threads = std::min(std::thread::hardware_concurrency(), m_setting_->num_threads);
        num_threads = num_clusters < num_threads ? num_clusters : num_threads;
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        size_t batch_size = num_clusters / num_threads;
        auto left_over = long(num_clusters - batch_size * num_threads);  // must be signed!
        size_t start_idx = 0;
        size_t end_idx = 0;
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            end_idx = start_idx + batch_size + (left_over-- > 0);
            threads.emplace_back(&GpisMapBase2D::UpdateGpXsThread, this, thread_idx, start_idx, end_idx);
            start_idx = end_idx;
        }

        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }

        m_active_clusters_.clear();
        m_clusters_to_update_.clear();
    }

    void
    GpisMapBase2D::UpdateGpXsThread(int thread_idx, int start_idx, int end_idx) {

        long num_nodes;
        std::vector<std::shared_ptr<geometry::Node>> nodes;
        Eigen::VectorXb grad_flag;

        for (int i = start_idx; i < end_idx; ++i) {
            if (m_clusters_to_update_[i] == nullptr) {
                ERL_WARN("Thread {}: empty shared pointer at index {} of m_clusters_to_update_.", thread_idx, i);
                continue;
            }

            auto &cluster = m_clusters_to_update_[i];
            auto &kArea = cluster->GetArea();
            geometry::Aabb2D search_area(kArea.center, kArea.half_sizes[0] * m_setting_->update_gp_sdf->search_area_scale);
            nodes.clear();
            m_quadtree_->CollectNodesOfTypeInArea(0, search_area, nodes);

            if (!nodes.empty()) {
                // generate training data
                num_nodes = long(nodes.size());
                long n = num_nodes;
                if (m_setting_->update_gp_sdf->add_offset_points) { n *= 3; }

                // TODO: use gp buffer instead of these variables
                Eigen::Matrix2Xd mat_x(2, n);
                Eigen::VectorXd vec_f(n);
                Eigen::Matrix2Xd mat_f_grad(2, n);
                Eigen::VectorXd vec_var_x(n);     // this is var(x)
                Eigen::VectorXd vec_var_y(n);     // this is var(y)
                Eigen::VectorXd vec_var_grad(n);  // var(grad_f)
                grad_flag.resize(n);

                int point_cnt = 0;
                // int valid_grad_cnt = 0;
                for (auto &node: nodes) {
                    auto node_data = node->GetData<GpisData2D>();
                    // auto node_data = node->getData<GpisData>();
                    auto &node_position = std::dynamic_pointer_cast<GpisNode2D>(node)->position;
                    mat_x.col(point_cnt) = node_position;

                    vec_f[point_cnt] = node_data->distance;
                    vec_var_y[point_cnt] = m_setting_->gp_theta->sensor_range_var;
                    vec_var_grad[point_cnt] = node_data->var_gradient;

                    if ((node_data->var_gradient > m_setting_->update_gp_sdf->max_valid_gradient_var) ||
                        ((std::fabs(node_data->gradient.x()) < m_setting_->update_gp_sdf->zero_gradient_threshold) &&
                         (std::fabs(node_data->gradient.y()) < m_setting_->update_gp_sdf->zero_gradient_threshold))) {

                        vec_var_x[point_cnt] = m_setting_->update_gp_sdf->invalid_position_var;
                        grad_flag[point_cnt] = false;
                        mat_f_grad.col(point_cnt++).setZero();
                        continue;
                    }

                    vec_var_x[point_cnt] = node_data->var_position;
                    // grad_flag[point_cnt++] = true;
                    grad_flag[point_cnt] = true;

                    auto gradient = node_data->gradient;
                    mat_f_grad.col(point_cnt++) = gradient;
                    // mat_f_grad.col(valid_grad_cnt) = gradient;
                    // valid_grad_cnt++;

                    if (m_setting_->update_gp_sdf->add_offset_points) {
                        mat_x.col(point_cnt) = node_position + m_setting_->update_gp_sdf->offset_distance * gradient;
                        vec_f[point_cnt] = node_data->distance + m_setting_->update_gp_sdf->offset_distance;
                        vec_var_y[point_cnt] = m_setting_->gp_theta->sensor_range_var;
                        vec_var_grad[point_cnt] = node_data->var_gradient;
                        vec_var_x[point_cnt] = m_setting_->update_gp_sdf->invalid_position_var;
                        grad_flag[point_cnt++] = false;

                        mat_x.col(point_cnt) = node_position - m_setting_->update_gp_sdf->offset_distance * gradient;
                        vec_f[point_cnt] = node_data->distance - m_setting_->update_gp_sdf->offset_distance;
                        vec_var_y[point_cnt] = m_setting_->gp_theta->sensor_range_var;
                        vec_var_grad[point_cnt] = node_data->var_gradient;
                        vec_var_x[point_cnt] = m_setting_->update_gp_sdf->invalid_position_var;
                        grad_flag[point_cnt++] = false;
                    }
                }
                mat_x.conservativeResize(2, point_cnt);
                vec_f.conservativeResize(point_cnt);
                // mat_f_grad.conservativeResize(2, valid_grad_cnt);
                mat_f_grad.conservativeResize(2, point_cnt);
                vec_var_x.conservativeResize(point_cnt);
                vec_var_y.conservativeResize(point_cnt);
                vec_var_grad.conservativeResize(point_cnt);
                grad_flag.conservativeResize(point_cnt);

                // Train a noisy input GP
                cluster->SetData(TrainGpX(mat_x, vec_f, mat_f_grad, grad_flag, vec_var_x, vec_var_y, vec_var_grad));
            }
        }
    }

    /**
     * Compute Gaussian process inference of SDF, surface normal, and the variance of given positions.
     * @param xy row-major storage of Nx2 numpy array of 2D positions
     * @param distances
     * @param gradients
     * @param variances
     * @return
     */
    bool
    GpisMapBase2D::Test(
        const TestBuffer::InBuffer &xy,
        TestBuffer::OutVectorBuffer::PlainMatrix &distances,
        TestBuffer::OutMatrixBuffer::PlainMatrix &gradients,
        TestBuffer::OutVectorBuffer::PlainMatrix &distance_variances,
        TestBuffer::OutMatrixBuffer::PlainMatrix &gradient_variances) {

        if (m_test_buffer_.ConnectBuffers(xy, distances, gradients, distance_variances, gradient_variances)) {
            LaunchTest(m_test_buffer_.Size());
            m_test_buffer_.DisconnectBuffers();
            return true;
        }

        return false;
    }

    Eigen::VectorXd
    GpisMapBase2D::ComputeSddfV2(
        const Eigen::Ref<const Eigen::Matrix2Xd> &positions,
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        double threshold,
        double max_distance,
        int max_marching_steps) {

        ERL_ASSERTM(threshold > 0, "threshold must be positive.");

        long num_rays = positions.cols();
        Eigen::VectorXd sddf_values = Eigen::VectorXd::Zero(num_rays);
        Eigen::Matrix2Xd directions(2, num_rays);
        for (long i = 0; i < num_rays; ++i) {
            directions(0, i) = std::cos(angles[i]);
            directions(1, i) = std::sin(angles[i]);
        }
        Eigen::VectorXd sdf_values(num_rays);
        Eigen::Matrix2Xd gradients(2, num_rays);
        Eigen::VectorXd distance_variances(num_rays);
        Eigen::Matrix2Xd gradient_variances(2, num_rays);
        Eigen::Matrix2Xd test_positions(2, num_rays);
        std::vector<long> ray_indices_0(num_rays);
        std::iota(ray_indices_0.begin(), ray_indices_0.end(), 0);
        std::vector<long> ray_indices_1;
        ray_indices_1.reserve(num_rays);

        int num_marching_steps = 0;
        while (!ray_indices_0.empty() && (max_marching_steps < 0 || num_marching_steps < max_marching_steps)) {
            // compute sdf values
            long cnt = 0;
            for (long &i: ray_indices_0) {
                test_positions(0, cnt) = positions(0, i) + sddf_values[i] * directions(0, i);
                test_positions(1, cnt) = positions(1, i) + sddf_values[i] * directions(1, i);
                cnt++;
            }
            test_positions.conservativeResize(2, cnt);
            Test(test_positions, sdf_values, gradients, distance_variances, gradient_variances);
            // update sddf_values, pick rays that do not hit obstacles yet
            ray_indices_1.clear();
            cnt = 0;
            for (long &i: ray_indices_0) {
                double &sdf_value = sdf_values[cnt];
                double &sddf_value = sddf_values[i];
                sddf_value += sdf_value;
                cnt++;
                if ((std::abs(sdf_value) >= threshold) && (max_distance <= 0 || std::abs(sddf_value) <= max_distance)) { ray_indices_1.push_back(i); }
            }
            ray_indices_0.swap(ray_indices_1);
            num_marching_steps++;
        }

        return sddf_values;
    }

    void
    GpisMapBase2D::LaunchTest(size_t n) {

        auto num_threads = std::min(std::thread::hardware_concurrency(), m_setting_->num_threads);
        if (n < num_threads) { num_threads = n; }

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        size_t batch_size = n / num_threads;
        size_t start_idx = 0;
        size_t end_idx = 0;
        int left_over = int(n - batch_size * num_threads);
        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            end_idx = start_idx + batch_size + (left_over-- > 0);
            threads.emplace_back(&GpisMapBase2D::TestThread, this, thread_idx, start_idx, end_idx);
            start_idx = end_idx;
        }

        for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }
    }

    void
    GpisMapBase2D::TestThread(int thread_idx, size_t start_idx, size_t end_idx) {

        std::vector<std::shared_ptr<geometry::IncrementalQuadtree>> clusters;
        std::vector<double> square_distances;
        std::vector<size_t> idx;
        const int kMaxTries = 4;

        Eigen::Matrix3Xd fs(3, kMaxTries);    // f, fGrad1, fGrad2
        Eigen::Matrix3Xd vars(3, kMaxTries);  // variances of f, fGrad1, fGrad2
        std::vector<long> tested_idx;
        tested_idx.reserve(kMaxTries);

        for (long i = (long) start_idx; i < (long) end_idx; ++i) {
            const auto &kPosition = m_test_buffer_.positions->col(i);

            // search the corresponding cluster
            auto search_area_half_size = m_setting_->test_query->search_area_half_size;
            while (search_area_half_size < m_quadtree_->GetArea().half_sizes[0]) {
                geometry::Aabb2D search_area(kPosition, search_area_half_size);
                clusters.clear();
                square_distances.clear();
                m_quadtree_->CollectClustersWithData(search_area, clusters, square_distances);
                if (!clusters.empty()) { break; }
                search_area_half_size *= 2.;
            }

            (*m_test_buffer_.distances)[i] = 0.;                                                    // init distance
            (*m_test_buffer_.distance_variances)[i] = 1. + m_setting_->gp_theta->sensor_range_var;  // init distance variance

            if (clusters.empty()) {
                static int warn_cnt = 0;
                ERL_WARN("[{}] Thread {}: no cluster found for position ({:f}, {:f})", warn_cnt, thread_idx, kPosition.x(), kPosition.y());
                warn_cnt++;
                continue;
            }  // no qualified cluster

            idx.resize(clusters.size());
            std::iota(idx.begin(), idx.end(), 0);
            if (clusters.size() > 1) {
                // closer cluster first
                std::stable_sort(idx.begin(), idx.end(), [&](size_t i_1, size_t i_2) -> bool { return square_distances[i_1] < square_distances[i_2]; });
            }

            tested_idx.clear();
            bool need_weighted_sum = false;

            long cnt = 0;
            for (auto &j: idx) {
                // call selected GPs for inference
                Eigen::Ref<Eigen::Vector3d> f = fs.col(cnt);      // distance, gradient_x, gradient_y
                Eigen::Ref<Eigen::Vector3d> var = vars.col(cnt);  // var_distance, var_gradient_x, var_gradient_y

                InferWithGpX(clusters[j]->GetData<void>(), kPosition, f, var);

                tested_idx.push_back(cnt++);
                if (m_setting_->test_query->use_nearest_only) { break; }
                if ((!need_weighted_sum) && (idx.size() > 1) && (var[0] > m_setting_->test_query->max_test_valid_distance_var)) { need_weighted_sum = true; }
                if ((!need_weighted_sum) || (cnt >= kMaxTries)) { break; }
            }

            // store the result
            if (need_weighted_sum) {
                // sort the results by distance variance
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](long j_1, long j_2) -> bool { return vars(0, j_1) < vars(0, j_2); });

                if (vars(0, tested_idx[0]) < m_setting_->test_query->max_test_valid_distance_var) {
                    auto j = tested_idx[0];
                    // column j is the result
                    (*m_test_buffer_.distances)[i] = fs(0, j);
                    m_test_buffer_.gradients->col(i) << fs(1, j), fs(2, j);
                    (*m_test_buffer_.distance_variances)[i] = vars(0, j);
                    m_test_buffer_.gradient_variances->col(i) << vars(1, j), vars(2, j);
                } else {
                    // pick the best two results to do weighted sum
                    auto j_1 = tested_idx[0];
                    auto j_2 = tested_idx[1];
                    auto w_1 = vars(0, j_1) - m_setting_->test_query->max_test_valid_distance_var;
                    auto w_2 = vars(0, j_2) - m_setting_->test_query->max_test_valid_distance_var;
                    double w_12 = w_1 + w_2;
                    // clang-format off
                    (*m_test_buffer_.distances)[i] = (fs(0, j_1) * w_2 + fs(0, j_2) * w_1) / w_12;
                    m_test_buffer_.gradients->col(i) << (fs(1, j_1) * w_2 + fs(1, j_2) * w_1) / w_12,
                                                        (fs(2, j_1) * w_2 + fs(2, j_2) * w_1) / w_12;
                    (*m_test_buffer_.distance_variances)[i] = (vars(0, j_1) * w_2 + vars(0, j_2) * w_1) / w_12;
                    m_test_buffer_.gradient_variances->col(i) << (vars(1, j_1) * w_2 + vars(1, j_2) * w_1) / w_12,
                                                                 (vars(2, j_1) * w_2 + vars(2, j_2) * w_1) / w_12;
                    // clang-format on
                }
            } else {
                // the first column is the result
                (*m_test_buffer_.distances)[i] = fs(0, 0);
                m_test_buffer_.gradients->col(i) << fs(1, 0), fs(2, 0);
                (*m_test_buffer_.distance_variances)[i] = vars(0, 0);
                m_test_buffer_.gradient_variances->col(i) << vars(1, 0), vars(2, 0);
            }

            if (!m_setting_->update_gp_sdf->add_offset_points) { (*m_test_buffer_.distances)[i] -= m_setting_->update_gp_sdf->offset_distance; }
            m_test_buffer_.gradients->col(i).normalize();
        }
    }

    Eigen::Matrix2Xd
    GpisMapBase2D::DumpSurfacePoints() const {

        std::vector<std::shared_ptr<geometry::Node>> quadtree_nodes;
        m_quadtree_->CollectNodes(quadtree_nodes);

        Eigen::Matrix2Xd points(2, quadtree_nodes.size());
        for (ssize_t i = 0; i < (ssize_t) quadtree_nodes.size(); i++) { points.col(i) = quadtree_nodes[i]->position; }

        return points;
    }

    Eigen::Matrix2Xd
    GpisMapBase2D::DumpSurfaceNormals() const {

        std::vector<std::shared_ptr<geometry::Node>> quadtree_nodes;
        m_quadtree_->CollectNodes(quadtree_nodes);

        Eigen::Matrix2Xd normals(2, quadtree_nodes.size());
        for (ssize_t i = 0; i < (ssize_t) quadtree_nodes.size(); i++) { normals.col(i) = quadtree_nodes[i]->GetData<GpisData2D>()->gradient; }

        return normals;
    }

    void
    GpisMapBase2D::DumpSurfaceData(
        Eigen::Matrix2Xd &surface_points,
        Eigen::Matrix2Xd &surface_normals,
        Eigen::VectorXd &points_variance,
        Eigen::VectorXd &normals_variance) const {

        std::vector<std::shared_ptr<geometry::Node>> quadtree_nodes;
        m_quadtree_->CollectNodes(quadtree_nodes);

        auto n = (ssize_t) quadtree_nodes.size();
        surface_points.resize(2, n);
        surface_normals.resize(2, n);
        points_variance.resize(n);
        normals_variance.resize(n);

        for (ssize_t i = 0; i < (ssize_t) quadtree_nodes.size(); ++i) {
            auto &node = quadtree_nodes[i];
            surface_points.col(i) = node->position;
            auto gpis_data = node->GetData<GpisData2D>();
            surface_normals.col(i) = gpis_data->gradient;
            points_variance[i] = gpis_data->var_position;
            normals_variance[i] = gpis_data->var_gradient;
        }
    }
}  // namespace erl::sdf_mapping::gpis
