#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
#include "erl_common/clip.hpp"

namespace erl::sdf_mapping {

    static inline void
    Cartesian2Polar(const Eigen::Ref<const Eigen::Vector2d> &xy, double &r, double &angle) {
        r = xy.norm();
        angle = std::atan2(xy.y(), xy.x());
    }

    bool
    GpOccSurfaceMapping2D::Update(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const Eigen::Ref<const Eigen::VectorXd> &distances,
        const Eigen::Ref<const Eigen::Matrix23d> &pose) {

        m_changed_keys_.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        m_gp_theta_->Train(angles, distances, pose);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("GP theta training time: %f ms.", dt);
        if (m_gp_theta_->IsTrained()) {

            // clang-format off
            m_xy_perturb_ << m_setting_->perturb_delta, -m_setting_->perturb_delta, 0., 0.,
                             0., 0., m_setting_->perturb_delta, -m_setting_->perturb_delta;
            // clang-format on

            // perform surface mapping
            t0 = std::chrono::high_resolution_clock::now();
            UpdateMapPoints();
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ERL_INFO("Update map points time: %f ms.", dt);

            if (m_setting_->update_occupancy) {
                t0 = std::chrono::high_resolution_clock::now();
                UpdateOccupancy(angles, distances, pose);
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                ERL_INFO("Update occupancy time: %f ms.", dt);
            }

            t0 = std::chrono::high_resolution_clock::now();
            AddNewMeasurement();
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            ERL_INFO("Add new measurement time: %f us.", dt);

            return true;
        }
        return false;
    }

    void
    GpOccSurfaceMapping2D::UpdateMapPoints() {
        if (m_quadtree_ == nullptr || !m_gp_theta_->IsTrained()) { return; }

        auto &kTrainBuffer = m_gp_theta_->GetTrainBuffer();
        std::vector<std::shared_ptr<geometry::QuadtreeKey>> keys_to_update;
        geometry::Aabb2D observed_area(kTrainBuffer.position, kTrainBuffer.max_distance + 3.0);
        Eigen::Vector2d &min_corner = observed_area.min();
        Eigen::Vector2d &max_corner = observed_area.max();

        auto it = m_quadtree_->BeginLeafInAabb(min_corner.x(), min_corner.y(), max_corner.x(), max_corner.y());
        auto end = m_quadtree_->EndLeafInAabb();
        double sensor_x = kTrainBuffer.position.x();
        double sensor_y = kTrainBuffer.position.y();
        double cluster_half_size = m_setting_->quadtree_resolution * std::pow(2, m_setting_->cluster_level - 1);
        double squared_dist_max = kTrainBuffer.max_distance * kTrainBuffer.max_distance + cluster_half_size * cluster_half_size * 2;
        for (; it != end; ++it) {
            double cluster_x, cluster_y;
            m_quadtree_->KeyToCoord(it.GetKey(), m_quadtree_->GetTreeDepth() - m_setting_->cluster_level, cluster_x, cluster_y);
            double squared_dist = (sensor_x - cluster_x) * (sensor_x - cluster_x) + (sensor_y - cluster_y) * (sensor_y - cluster_y);
            if (squared_dist > squared_dist_max) { continue; }  // out of range

            std::shared_ptr<SurfaceMappingQuadtreeNode::SurfaceData> surface_data = it->GetSurfaceData();
            if (surface_data == nullptr) { continue; }  // no surface data

            double distance_old, occ, occ_abs;
            Eigen::Scalard angle, distance_pred, var;
            Eigen::Vector2d &xy_global_old = surface_data->position;
            Eigen::Vector2d xy_local_old = m_gp_theta_->GlobalToLocalSe2(xy_global_old);
            Cartesian2Polar(xy_local_old, distance_old, angle[0]);
            if (angle[0] < m_setting_->gp_theta->train_buffer->valid_angle_min || angle[0] > m_setting_->gp_theta->train_buffer->valid_angle_max) { continue; }
            if (!m_gp_theta_->ComputeOcc(angle, distance_old, distance_pred, var, occ)) { continue; }
            if (occ < m_setting_->update_map_points->min_observable_occ) { continue; }
            Eigen::Vector2d grad_local_old = m_gp_theta_->GlobalToLocalSo2(surface_data->normal);

            // compute a new position for the point
            occ_abs = std::fabs(occ);
            double delta = m_setting_->perturb_delta;
            Eigen::Vector2d xy_local_new = xy_local_old;
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
                surface_data->var_position *= 2.;
                surface_data->var_normal *= 2.;
                continue;
            }

            Eigen::Vector2d xy_global_new = m_gp_theta_->LocalToGlobalSe2(xy_local_new);
            double &var_position_old = surface_data->var_position;
            double &var_gradient_old = surface_data->var_normal;
            if (var_gradient_old <= m_setting_->update_map_points->max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad
                double var_position_sum = var_position_new + var_position_old;
                double var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                xy_global_new = (xy_global_new * var_position_old + xy_global_old * var_position_new) / var_position_sum;
                double distance = (xy_global_new - xy_global_old).norm() * double(0.5);

                // gradient Update
                double &old_x = surface_data->normal.x();
                double &old_y = surface_data->normal.y();
                double &new_x = grad_global_new.x();
                double &new_y = grad_global_new.y();
                double angle_dist = std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_position_new / var_position_sum;
                double sin = std::sin(angle_dist);
                double cos = std::cos(angle_dist);
                // rotate grad_global_old by angle_dist
                grad_global_new.x() = cos * old_x - sin * old_y;
                grad_global_new.y() = sin * old_x + cos * old_y;

                // variance Update
                var_position_new = std::max((var_position_new * var_position_old) / var_position_sum + distance, m_setting_->gp_theta->sensor_range_var);
                var_gradient_new = common::ClipRange(
                    (var_gradient_new * var_gradient_old) / var_gradient_sum + distance,
                    m_setting_->compute_variance->min_gradient_var,
                    m_setting_->compute_variance->max_gradient_var);
            }

            // Update the surface data
            if ((var_position_new > m_setting_->update_map_points->max_bayes_position_var) &&
                (var_gradient_new > m_setting_->update_map_points->max_bayes_gradient_var)) {
                it->ResetSurfaceData();
                RecordChangedKey(it.GetKey());
                continue;  // too bad, skip
            }
            geometry::QuadtreeKey new_key = m_quadtree_->CoordToKey(xy_global_new.x(), xy_global_new.y());
            if (new_key != it.GetKey()) {
                it->ResetSurfaceData();
                RecordChangedKey(it.GetKey());
                std::shared_ptr<SurfaceMappingQuadtreeNode> new_node;
                // move the surface data to the new node before updating it
                if (m_setting_->update_occupancy) {
                    // get new node via occupancy update
                    constexpr bool kOccupied = true;
                    constexpr bool kLazyEval = true;
                    new_node = std::static_pointer_cast<SurfaceMappingQuadtreeNode>(m_quadtree_->UpdateNode(new_key, kOccupied, kLazyEval));
                } else {
                    // get new node via insert
                    new_node = std::static_pointer_cast<SurfaceMappingQuadtreeNode>(m_quadtree_->InsertNode(new_key));
                }
                ERL_ASSERTM(new_node != nullptr, "Failed to get the node");
                if (new_node->GetSurfaceData() != nullptr) { continue; }  // the new node is already occupied
                new_node->SetSurfaceData(surface_data);
                RecordChangedKey(new_key);
            }
            surface_data->position = xy_global_new;
            surface_data->normal = grad_global_new;
            surface_data->var_position = var_position_new;
            surface_data->var_normal = var_gradient_new;
        }
    }

    void
    GpOccSurfaceMapping2D::UpdateOccupancy(
        const Eigen::Ref<const Eigen::VectorXd> &angles,
        const Eigen::Ref<const Eigen::VectorXd> &distances,
        const Eigen::Ref<const Eigen::Matrix23d> &pose) {

        if (m_quadtree_ == nullptr) { m_quadtree_ = std::make_shared<SurfaceMappingQuadtree>(m_setting_->quadtree_resolution); }

        auto n = angles.size();
        auto rotation = pose.topLeftCorner<2, 2>();
        auto translation = pose.col(2);
        Eigen::Matrix2Xd points(2, n);
        for (long i = 0; i < n; ++i) {
            double x = distances[i] * std::cos(angles[i]);
            double y = distances[i] * std::sin(angles[i]);
            points(0, i) = rotation(0, 0) * x + rotation(0, 1) * y + translation[0];
            points(1, i) = rotation(1, 0) * x + rotation(1, 1) * y + translation[1];
        }
        constexpr bool kParallel = false;
        constexpr bool kLazyEval = false;
        constexpr bool kDiscrete = false;
        m_quadtree_->InsertPointCloud(points, translation, m_setting_->gp_theta->train_buffer->valid_range_max, kParallel, kLazyEval, kDiscrete);
    }

    void
    GpOccSurfaceMapping2D::AddNewMeasurement() {

        if (m_quadtree_ == nullptr) { m_quadtree_ = std::make_shared<SurfaceMappingQuadtree>(m_setting_->quadtree_resolution); }

        auto &kTrainBuffer = m_gp_theta_->GetTrainBuffer();

        auto n = kTrainBuffer.Size();
        Eigen::Scalard angle, f, var;
        for (ssize_t i = 0; i < n; ++i) {
            angle[0] = kTrainBuffer.vec_angles[i];
            m_gp_theta_->Test(angle, f, var, true);

            if (var[0] > m_setting_->gp_theta->max_valid_distance_var) { continue; }  // uncertain point, drop it

            constexpr bool kOccupied = true;
            constexpr bool kLazyEval = true;  // lazy evaluation to make sure we get the deepest node
            geometry::QuadtreeKey key = m_quadtree_->CoordToKey(kTrainBuffer.mat_xy_global(0, i), kTrainBuffer.mat_xy_global(1, i));
            std::shared_ptr<SurfaceMappingQuadtreeNode> leaf;
            if (m_setting_->update_occupancy) {
                // get the leaf node via occupancy update
                leaf = std::static_pointer_cast<SurfaceMappingQuadtreeNode>(m_quadtree_->UpdateNode(key, kOccupied, kLazyEval));
            } else {
                // get the leaf node via insert
                leaf = std::static_pointer_cast<SurfaceMappingQuadtreeNode>(m_quadtree_->InsertNode(key));
            }
            ERL_ASSERTM(leaf != nullptr, "Failed to insert a new point to the quadtree.");
            if (leaf->GetSurfaceData() != nullptr) { continue; }  // the leaf already has surface data, skip it

            double occ_mean;
            Eigen::Vector2d grad_local;
            if (!ComputeGradient2(kTrainBuffer.mat_xy_local.col(i), grad_local, occ_mean)) {
                if (m_setting_->update_occupancy) {
                    // undo the occupancy update
                    m_quadtree_->UpdateNode(key, !kOccupied, !kLazyEval);
                } else {
                    // undo the insert
                    m_quadtree_->DeleteNode(key);
                }
                continue;
            }

            Eigen::Vector2d grad_global;
            double var_position, var_gradient;
            ComputeVariance(
                kTrainBuffer.mat_xy_local.col(i),
                kTrainBuffer.vec_ranges[i],
                0.,  // distance_var is not used for a new point
                std::fabs(occ_mean),
                0.,  // occ_abs is not used for a new point
                true,
                grad_local,
                grad_global,
                var_position,
                var_gradient);

            // Insert the point to the tree and mark the key as changed
            leaf->SetSurfaceData(Eigen::Vector2d(kTrainBuffer.mat_xy_global.col(i)), grad_global, var_position, var_gradient);
            RecordChangedKey(key);
        }
    }

    bool
    GpOccSurfaceMapping2D::ComputeGradient1(
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
    GpOccSurfaceMapping2D::ComputeGradient2(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Ref<Eigen::Vector2d> gradient, double &occ_mean) {
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

    bool
    GpOccSurfaceMapping2D::ComputeVariance(
        const Eigen::Ref<const Eigen::Vector2d> &xy_local,
        const double &distance,
        const double &distance_var,
        const double &occ_mean_abs,
        const double &occ_abs,
        bool new_point,
        Eigen::Ref<Eigen::Vector2d> grad_local,
        Eigen::Ref<Eigen::Vector2d> grad_global,
        double &var_position,
        double &var_gradient) const {

        double gradient_norm = grad_local.norm();

        if (gradient_norm <= double(1.e-6)) {
            grad_global << grad_local;
            var_position = m_setting_->compute_variance->zero_gradient_position_var;
            var_gradient = m_setting_->compute_variance->zero_gradient_gradient_var;
            return false;
        }

        grad_local.x() /= gradient_norm;
        grad_local.y() /= gradient_norm;
        // grad_global << LocalToGlobalSo2(m_gp_theta_->m_train_buffer_.rotation, grad_local);
        grad_global << m_gp_theta_->LocalToGlobalSo2(grad_local);

        double &min_distance_var = m_setting_->compute_variance->min_distance_var;
        double &max_distance_var = m_setting_->compute_variance->max_distance_var;
        double &min_gradient_var = m_setting_->compute_variance->min_gradient_var;
        double &max_gradient_var = m_setting_->compute_variance->max_gradient_var;

        double var_distance = common::ClipRange(distance * distance, min_distance_var, max_distance_var);
        double cos_view_angle = -xy_local.dot(grad_local) / xy_local.norm();
        double cos2_view_angle = std::max(cos_view_angle * cos_view_angle, double(1.e-2));  // avoid zero division
        double var_direction = (1. - cos2_view_angle) / cos2_view_angle;

        if (new_point) {
            var_position = m_setting_->compute_variance->position_var_alpha * (var_distance + var_direction);
            var_gradient = common::ClipRange(occ_mean_abs, min_gradient_var, max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = m_setting_->compute_variance->position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient = common::ClipRange(occ_mean_abs + distance_var, min_gradient_var, max_gradient_var) + double(0.1) * var_direction;
        }

        return true;
    }
}  // namespace erl::mapping
