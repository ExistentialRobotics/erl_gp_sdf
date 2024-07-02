#include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"

#include "erl_common/clip.hpp"

namespace erl::sdf_mapping {

    static void
    Cartesian2Polar(const Eigen::Ref<const Eigen::Vector2d> &xy, double &r, double &angle) {
        r = xy.norm();
        angle = std::atan2(xy.y(), xy.x());
    }

    bool
    GpOccSurfaceMapping2D::Update(
        const Eigen::Ref<const Eigen::Matrix2d> &rotation,
        const Eigen::Ref<const Eigen::Vector2d> &translation,
        const Eigen::Ref<const Eigen::MatrixXd> &ranges) {

        m_changed_keys_.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        (void) m_sensor_gp_->Train(rotation, translation, ranges, false);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("GP theta training time: {:f} ms.", dt);
        if (!m_sensor_gp_->IsTrained()) { return false; }

        const double d = m_setting_->perturb_delta;
        // clang-format off
        m_xy_perturb_ << d, -d, 0., 0.,
                         0., 0., d, -d;
        // clang-format on

        if (m_setting_->update_occupancy) {
            t0 = std::chrono::high_resolution_clock::now();
            UpdateOccupancy();
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ERL_INFO("Update occupancy time: {:f} ms.", dt);
        }

        // perform surface mapping
        t0 = std::chrono::high_resolution_clock::now();
        UpdateMapPoints();
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update map points time: {:f} ms.", dt);

        t0 = std::chrono::high_resolution_clock::now();
        AddNewMeasurement();
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Add new measurement time: {:f} us.", dt);

        return true;
    }

    void
    GpOccSurfaceMapping2D::UpdateMapPoints() {
        if (m_quadtree_ == nullptr || !m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetLidarFrame();
        const Eigen::Vector2d &sensor_pos = sensor_frame->GetTranslationVector();
        const double max_sensor_range = sensor_frame->GetMaxValidRange();
        const geometry::Aabb2D observed_area(sensor_pos, max_sensor_range);
        const Eigen::Vector2d &min_corner = observed_area.min();
        const Eigen::Vector2d &max_corner = observed_area.max();

        std::vector<std::tuple<geometry::QuadtreeKey, geometry::SurfaceMappingQuadtreeNode *, std::optional<geometry::QuadtreeKey>>> nodes_in_aabb;
        for (auto it = m_quadtree_->BeginLeafInAabb(min_corner.x(), min_corner.y(), max_corner.x(), max_corner.y()), end = m_quadtree_->EndLeafInAabb();
             it != end;
             ++it) {
            nodes_in_aabb.emplace_back(it.GetKey(), *it, std::nullopt);
        }

#pragma omp parallel for default(none) shared(nodes_in_aabb, max_sensor_range, sensor_pos, sensor_frame, g_print_mutex)
        for (auto &[node_key, node, new_key]: nodes_in_aabb) {
            const uint32_t cluster_level = m_setting_->cluster_level;
            const double sensor_range_var = m_setting_->sensor_gp->sensor_range_var;
            const double min_comp_gradient_var = m_setting_->compute_variance.min_gradient_var;
            const double max_comp_gradient_var = m_setting_->compute_variance.max_gradient_var;
            const int max_adjust_tries = m_setting_->update_map_points.max_adjust_tries;
            const double min_observable_occ = m_setting_->update_map_points.min_observable_occ;
            const double max_surface_abs_occ = m_setting_->update_map_points.max_surface_abs_occ;
            const double max_valid_gradient_var = m_setting_->update_map_points.max_valid_gradient_var;
            const double min_position_var = m_setting_->update_map_points.min_position_var;
            const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;
            const double max_bayes_position_var = m_setting_->update_map_points.max_bayes_position_var;
            const double max_bayes_gradient_var = m_setting_->update_map_points.max_bayes_gradient_var;

            const double cluster_half_size = m_setting_->quadtree->resolution * std::pow(2, cluster_level - 1);
            const double squared_dist_max = max_sensor_range * max_sensor_range + cluster_half_size * cluster_half_size * 2.0;

            Eigen::Vector2d cluster_position;
            m_quadtree_->KeyToCoord(node_key, m_quadtree_->GetTreeDepth() - cluster_level, cluster_position.x(), cluster_position.y());
            if ((cluster_position - sensor_pos).squaredNorm() > squared_dist_max) { continue; }

            std::shared_ptr<geometry::SurfaceMappingQuadtreeNode::SurfaceData> surface_data = node->GetSurfaceData();
            if (surface_data == nullptr) { continue; }

            const Eigen::Vector2d &xy_global_old = surface_data->position;
            Eigen::Vector2d xy_local_old = sensor_frame->WorldToFrameSe2(xy_global_old);

            if (!sensor_frame->PointIsInFrame(xy_local_old)) { continue; }

            double occ, distance_old;
            Eigen::Scalard distance_pred, distance_pred_var, angle_local;
            Cartesian2Polar(xy_local_old, distance_old, angle_local[0]);
            if (!m_sensor_gp_->ComputeOcc(angle_local, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
            if (occ < min_observable_occ) { continue; }

            const Eigen::Vector2d &grad_global_old = surface_data->normal;
            Eigen::Vector2d grad_local_old = sensor_frame->WorldToFrameSo2(grad_global_old);

            // compute a new position for the point
            Eigen::Vector2d xy_local_new = xy_local_old;
            double delta = m_setting_->perturb_delta;
            int num_adjust_tries = 0;
            double occ_abs = std::fabs(occ);
            double distance_new = distance_old;
            while (num_adjust_tries < max_adjust_tries && occ_abs > max_surface_abs_occ) {
                // move one step
                // the direction is determined by the occupancy sign, the step size is heuristically determined according to iteration.
                if (occ < 0.) {
                    xy_local_new += grad_local_old * delta;  // point is inside the obstacle
                } else if (occ > 0.) {
                    xy_local_new -= grad_local_old * delta;  // point is outside the obstacle
                }

                // test the new point
                Cartesian2Polar(xy_local_new, distance_pred[0], angle_local[0]);
                if (double occ_new; m_sensor_gp_->ComputeOcc(angle_local, distance_pred[0], distance_pred, distance_pred_var, occ_new)) {
                    occ_abs = std::fabs(occ_new);
                    distance_new = distance_pred[0];
                    if (occ_abs < max_surface_abs_occ) { break; }
                    if (occ * occ_new < 0.) {
                        delta *= 0.5;  // too big, make it smaller
                    } else {
                        delta *= 1.1;
                    }
                    occ = occ_new;
                } else {
                    break;  // fail to estimate occ
                }
                ++num_adjust_tries;
            }

            // compute new gradient and uncertainty
            double occ_mean, var_distance;
            Eigen::Vector2d grad_local_new;
            if (!ComputeGradient1(xy_local_new, grad_local_new, occ_mean, var_distance)) { continue; }
            Eigen::Vector2d grad_global_new = sensor_frame->FrameToWorldSo2(grad_local_new);
            double var_position_new, var_gradient_new;
            ComputeVariance(xy_local_new, grad_local_new, distance_new, var_distance, std::fabs(occ_mean), occ_abs, false, var_position_new, var_gradient_new);

            Eigen::Vector2d xy_global_new = sensor_frame->FrameToWorldSe2(xy_local_new);
            if (const double var_position_old = surface_data->var_position, var_gradient_old = surface_data->var_normal;
                var_gradient_old <= max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad, otherwise, just replace it
                const double var_position_sum = var_position_new + var_position_old;
                const double var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                xy_global_new = (xy_global_new * var_position_old + xy_global_old * var_position_new) / var_position_sum;
                // gradient Update
                const double &old_x = grad_global_old.x();
                const double &old_y = grad_global_old.y();
                const double &new_x = grad_global_new.x();
                const double &new_y = grad_global_new.y();
                const double angle_dist = std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_position_new / var_position_sum;
                const double sin = std::sin(angle_dist);
                const double cos = std::cos(angle_dist);
                // rotate grad_global_old by angle_dist
                grad_global_new.x() = cos * old_x - sin * old_y;
                grad_global_new.y() = sin * old_x + cos * old_y;
                ERL_DEBUG_ASSERT(std::abs(grad_global_new.norm() - 1.0) < 1.e-6, "grad_global_new.norm() = {:.6f}", grad_global_new.norm());

                // variance Update
                const double distance = (xy_global_new - xy_global_old).norm() * 0.5;
                var_position_new = std::max((var_position_new * var_position_old) / var_position_sum + distance, sensor_range_var);
                var_gradient_new = common::ClipRange(  //
                    (var_gradient_new * var_gradient_old) / var_gradient_sum + distance,
                    min_comp_gradient_var,
                    max_comp_gradient_var);
            }
            var_position_new = std::max(var_position_new, min_position_var);
            var_gradient_new = std::max(var_gradient_new, min_gradient_var);

            // Update the surface data
            if ((var_position_new > max_bayes_position_var) && (var_gradient_new > max_bayes_gradient_var)) {
                node->ResetSurfaceData();
                continue;  // too bad, skip
            }
            if (new_key = m_quadtree_->CoordToKey(xy_global_new.x(), xy_global_new.y()); new_key.value() == node_key) { new_key = std::nullopt; }
            surface_data->position = xy_global_new;
            surface_data->normal = grad_global_new;
            surface_data->var_position = var_position_new;
            surface_data->var_normal = var_gradient_new;
            ERL_DEBUG_ASSERT(std::abs(surface_data->normal.norm() - 1.0) < 1.e-6, "surface_data->normal.norm() = {:.6f}", surface_data->normal.norm());
        }

        for (auto &[key, node, new_key]: nodes_in_aabb) {
            if (node->GetSurfaceData() == nullptr) {  // too bad, surface data is removed
                RecordChangedKey(key);
                continue;
            }
            if (new_key.has_value()) {  // the node is moved to a new position
                auto surface_data = node->GetSurfaceData();
                node->ResetSurfaceData();
                RecordChangedKey(key);

                geometry::SurfaceMappingQuadtreeNode *new_node = m_quadtree_->InsertNode(new_key.value());
                ERL_DEBUG_ASSERT(new_node != nullptr, "Failed to get the node");
                if (new_node->GetSurfaceData() != nullptr) { continue; }  // the new node is already occupied
                new_node->SetSurfaceData(surface_data);
                RecordChangedKey(new_key.value());
            }
        }

        /*const double valid_angle_min = m_setting_->sensor_gp->train_buffer->valid_angle_min;
        const double valid_angle_max = m_setting_->sensor_gp->train_buffer->valid_angle_max;
        const double valid_range_min = m_setting_->sensor_gp->train_buffer->valid_range_min;
        const double valid_range_max = m_setting_->sensor_gp->train_buffer->valid_range_max;

        const uint32_t cluster_level = m_setting_->cluster_level;
        const double sensor_range_var = m_setting_->sensor_gp->sensor_range_var;
        const double min_comp_gradient_var = m_setting_->compute_variance.min_gradient_var;
        const double max_comp_gradient_var = m_setting_->compute_variance.max_gradient_var;
        const int max_adjust_tries = m_setting_->update_map_points.max_adjust_tries;
        const double min_observable_occ = m_setting_->update_map_points.min_observable_occ;
        const double max_surface_abs_occ = m_setting_->update_map_points.max_surface_abs_occ;
        const double max_valid_gradient_var = m_setting_->update_map_points.max_valid_gradient_var;
        const double min_position_var = m_setting_->update_map_points.min_position_var;
        const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        const double max_bayes_position_var = m_setting_->update_map_points.max_bayes_position_var;
        const double max_bayes_gradient_var = m_setting_->update_map_points.max_bayes_gradient_var;

        const double cluster_half_size = m_setting_->quadtree->resolution * std::pow(2, cluster_level - 1);
        double squared_dist_max = train_buffer.max_distance * train_buffer.max_distance + cluster_half_size * cluster_half_size * 2;

        for (auto it = m_quadtree_->BeginLeafInAabb(min_corner.x(), min_corner.y(), max_corner.x(), max_corner.y()), end = m_quadtree_->EndLeafInAabb();
             it != end;
             ++it) {
            Eigen::Vector2d cluster_position;
            m_quadtree_->KeyToCoord(it.GetKey(), m_quadtree_->GetTreeDepth() - cluster_level, cluster_position.x(), cluster_position.y());
            if ((cluster_position - sensor_position).squaredNorm() > squared_dist_max) { continue; }  // out of range

            std::shared_ptr<geometry::SurfaceMappingQuadtreeNode::SurfaceData> surface_data = it->GetSurfaceData();
            if (surface_data == nullptr) { continue; }  // no surface data

            const Eigen::Vector2d &xy_global_old = surface_data->position;

            double distance_old;
            Eigen::Scalard angle;
            Eigen::Vector2d xy_local_old = m_sensor_gp_->GlobalToLocalSe2(xy_global_old);
            Cartesian2Polar(xy_local_old, distance_old, angle[0]);
            if (angle[0] < valid_angle_min || angle[0] > valid_angle_max || distance_old < valid_range_min || distance_old > valid_range_max) { continue; }

            double occ;
            Eigen::Scalard distance_pred, distance_pred_var;
            if (!m_sensor_gp_->ComputeOcc(angle, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
            if (occ < min_observable_occ) { continue; }

            const Eigen::Vector2d &grad_global_old = surface_data->normal;
            Eigen::Vector2d grad_local_old = m_sensor_gp_->GlobalToLocalSo2(grad_global_old);

            // compute a new position for the point
            Eigen::Vector2d xy_local_new = xy_local_old;
            double delta = m_setting_->perturb_delta;
            int num_adjust_tries = 0;
            double occ_abs = std::fabs(occ);
            double distance_new = distance_old;
            while (num_adjust_tries < max_adjust_tries && occ_abs > max_surface_abs_occ) {
                // move one step
                // the direction is determined by the occupancy sign, the step size is heuristically determined according to iteration.
                if (occ < 0.) {
                    xy_local_new += grad_local_old * delta;  // point is inside the obstacle
                } else if (occ > 0.) {
                    xy_local_new -= grad_local_old * delta;  // point is outside the obstacle
                }

                // test the new point
                Cartesian2Polar(xy_local_new, distance_pred[0], angle[0]);
                if (double occ_new; m_sensor_gp_->ComputeOcc(angle, distance_pred[0], distance_pred, distance_pred_var, occ_new)) {
                    occ_abs = std::fabs(occ_new);
                    distance_new = distance_pred[0];
                    if (occ_abs < max_surface_abs_occ) { break; }
                    if (occ * occ_new < 0.) {
                        delta *= 0.5;  // too big, make it smaller
                    } else {
                        delta *= 1.1;
                    }
                    occ = occ_new;
                } else {
                    break;  // fail to estimate occ
                }
                ++num_adjust_tries;
            }

            // compute new gradient and uncertainty
            double occ_mean, var_distance;
            Eigen::Vector2d grad_local_new;
            if (!ComputeGradient1(xy_local_new, grad_local_new, occ_mean, var_distance)) { continue; }

            Eigen::Vector2d grad_global_new = m_sensor_gp_->LocalToGlobalSo2(grad_local_new);
            double var_position_new, var_gradient_new;
            ComputeVariance(xy_local_new, grad_local_new, distance_new, var_distance, std::fabs(occ_mean), occ_abs, false, var_position_new, var_gradient_new);

            Eigen::Vector2d xy_global_new = m_sensor_gp_->LocalToGlobalSe2(xy_local_new);
            if (const double var_position_old = surface_data->var_position, var_gradient_old = surface_data->var_normal;
                var_gradient_old <= max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad, otherwise, just replace it
                const double var_position_sum = var_position_new + var_position_old;
                const double var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                xy_global_new = (xy_global_new * var_position_old + xy_global_old * var_position_new) / var_position_sum;

                // gradient Update
                const double &old_x = grad_global_old.x();
                const double &old_y = grad_global_old.y();
                const double &new_x = grad_global_new.x();
                const double &new_y = grad_global_new.y();
                const double angle_dist = std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_position_new / var_position_sum;
                const double sin = std::sin(angle_dist);
                const double cos = std::cos(angle_dist);
                // rotate grad_global_old by angle_dist
                grad_global_new.x() = cos * old_x - sin * old_y;
                grad_global_new.y() = sin * old_x + cos * old_y;

                // variance Update
                const double distance = (xy_global_new - xy_global_old).norm() * 0.5;
                var_position_new = std::max((var_position_new * var_position_old) / var_position_sum + distance, sensor_range_var);
                var_gradient_new = common::ClipRange(  //
                    (var_gradient_new * var_gradient_old) / var_gradient_sum + distance,
                    min_comp_gradient_var,
                    max_comp_gradient_var);
            }
            var_position_new = std::max(var_position_new, min_position_var);
            var_gradient_new = std::max(var_gradient_new, min_gradient_var);

            // Update the surface data
            if ((var_position_new > max_bayes_position_var) && (var_gradient_new > max_bayes_gradient_var)) {
                it->ResetSurfaceData();
                RecordChangedKey(it.GetKey());
                continue;  // too bad, skip
            }
            if (geometry::QuadtreeKey new_key = m_quadtree_->CoordToKey(xy_global_new.x(), xy_global_new.y()); new_key != it.GetKey()) {
                it->ResetSurfaceData();
                RecordChangedKey(it.GetKey());
                geometry::SurfaceMappingQuadtreeNode *new_node = m_quadtree_->InsertNode(new_key);
                ERL_ASSERTM(new_node != nullptr, "Failed to get the node");
                if (new_node->GetSurfaceData() != nullptr) { continue; }  // the new node is already occupied
                new_node->SetSurfaceData(surface_data);
                RecordChangedKey(new_key);
            }
            surface_data->position = xy_global_new;
            surface_data->normal = grad_global_new;
            surface_data->var_position = var_position_new;
            surface_data->var_normal = var_gradient_new;
            ERL_DEBUG_ASSERT(std::abs(surface_data->normal.norm() - 1.0) < 1.e-6, "surface_data->normal.norm() = {:.6f}", surface_data->normal.norm());
        }*/
    }

    void
    GpOccSurfaceMapping2D::UpdateOccupancy() {

        if (m_quadtree_ == nullptr) { m_quadtree_ = std::make_shared<geometry::SurfaceMappingQuadtree>(m_setting_->quadtree); }

        const auto sensor_frame = m_sensor_gp_->GetLidarFrame();
        const Eigen::Map<const Eigen::Matrix2Xd> map_points(sensor_frame->GetEndPointsInWorld().data()->data(), 2, sensor_frame->GetNumRays());
        constexpr bool parallel = false;
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        m_quadtree_->InsertPointCloud(map_points, sensor_frame->GetTranslationVector(), sensor_frame->GetMaxValidRange(), parallel, lazy_eval, discrete);

        /*const Eigen::Index n = angles.size();
        const auto rotation = pose.topLeftCorner<2, 2>();
        const auto translation = pose.col(2);
        Eigen::Matrix2Xd points(2, n);
        for (long i = 0; i < n; ++i) {
            const double x = distances[i] * std::cos(angles[i]);
            const double y = distances[i] * std::sin(angles[i]);
            points(0, i) = rotation(0, 0) * x + rotation(0, 1) * y + translation[0];
            points(1, i) = rotation(1, 0) * x + rotation(1, 1) * y + translation[1];
        }
        constexpr bool parallel = false;   // no improvement
        constexpr bool lazy_eval = false;  // no improvement
        constexpr bool discrete = false;
        m_quadtree_->InsertPointCloud(points, translation, m_setting_->sensor_gp->train_buffer->valid_range_max, parallel, lazy_eval, discrete);*/
    }

    void
    GpOccSurfaceMapping2D::AddNewMeasurement() {

        if (m_quadtree_ == nullptr) { m_quadtree_ = std::make_shared<geometry::SurfaceMappingQuadtree>(m_setting_->quadtree); }

        const auto sensor_frame = m_sensor_gp_->GetLidarFrame();

        const std::vector<long> &hit_ray_indices = sensor_frame->GetHitRayIndices();
        const std::vector<Eigen::Vector2d> &points_local = sensor_frame->GetEndPointsInFrame();
        const std::vector<Eigen::Vector2d> &directions_frame = sensor_frame->GetRayDirectionsInFrame();

        const long num_hit_rays = sensor_frame->GetNumHitRays();
        Eigen::VectorXd angles_local(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Eigen::Vector2d &direction = directions_frame[hit_ray_indices[i]];
            angles_local[i] = std::atan2(direction[1], direction[0]);
        }
        Eigen::VectorXd predicted_ranges(num_hit_rays);
        Eigen::VectorXd predicted_ranges_var(num_hit_rays);
        if (!m_sensor_gp_->Test(angles_local, true, predicted_ranges, predicted_ranges_var, true, true)) { return; }
        Eigen::VectorXb invalid = (predicted_ranges_var.array() > m_setting_->sensor_gp->max_valid_range_var) ||       //
                                  (predicted_ranges.array() < m_setting_->sensor_gp->lidar_frame->valid_range_min) ||  //
                                  (predicted_ranges.array() > m_setting_->sensor_gp->lidar_frame->valid_range_max);

        Eigen::VectorXd occ_mean_values(num_hit_rays);
        std::vector<Eigen::Vector2d> gradients_local(num_hit_rays);
#pragma omp parallel for default(none) shared(num_hit_rays, hit_ray_indices, points_local, invalid, occ_mean_values, gradients_local)
        for (long i = 0; i < num_hit_rays; ++i) {
            if (invalid[i]) { continue; }
            if (!ComputeGradient2(points_local[hit_ray_indices[i]], gradients_local[i], occ_mean_values[i])) { invalid[i] = true; }
        }

        const std::vector<Eigen::Vector2d> &hit_points = sensor_frame->GetHitPointsWorld();
        const Eigen::VectorXd &ranges = sensor_frame->GetRanges();
        const double min_position_var = m_setting_->update_map_points.min_position_var;
        const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        for (long i = 0; i < num_hit_rays; ++i) {
            if (invalid[i]) { continue; }

            const Eigen::Vector2d &hit_point = hit_points[i];
            geometry::QuadtreeKey key = m_quadtree_->CoordToKey(hit_point.x(), hit_point.y());
            geometry::SurfaceMappingQuadtreeNode *node = m_quadtree_->InsertNode(key);  // insert the node
            if (node == nullptr) { continue; }                                          // failed to insert the node
            if (node->GetSurfaceData() != nullptr) { continue; }                        // the node is already occupied

            double var_position, var_gradient;
            const long idx = hit_ray_indices[i];
            const Eigen::Vector2d &xy_local = points_local[idx];
            ComputeVariance(xy_local, gradients_local[i], ranges[idx], 0, std::fabs(occ_mean_values[i]), 0, true, var_position, var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);

            Eigen::Vector2d grad_global = sensor_frame->FrameToWorldSo2(gradients_local[i]);
            node->SetSurfaceData(hit_point, std::move(grad_global), var_position, var_gradient);
            RecordChangedKey(key);
        }

        /*auto &train_buffer = m_sensor_gp_->GetTrainBuffer();

        const auto n = train_buffer.Size();
        const double valid_range_min = m_setting_->sensor_gp->train_buffer->valid_range_min;
        const double valid_range_max = m_setting_->sensor_gp->train_buffer->valid_range_max;
        const double max_valid_range_var = m_setting_->sensor_gp->max_valid_range_var;
        const double min_position_var = m_setting_->update_map_points.min_position_var;
        const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        Eigen::Scalard angle, predicted_range, predicted_range_var;
        for (long i = 0; i < n; ++i) {
            angle[0] = train_buffer.vec_angles[i];
            m_sensor_gp_->Test(angle, predicted_range, predicted_range_var, true);
            // uncertain point, drop it
            if (!(predicted_range[0] >= valid_range_min && predicted_range[0] <= valid_range_max && predicted_range_var[0] < max_valid_range_var)) { continue; }

            geometry::QuadtreeKey key = m_quadtree_->CoordToKey(train_buffer.mat_xy_global(0, i), train_buffer.mat_xy_global(1, i));
            geometry::SurfaceMappingQuadtreeNode *leaf = m_quadtree_->InsertNode(key);             // insert the point to the tree
            if (leaf == nullptr) { continue; }                                                     // failed to insert the point, skip it
            if (m_setting_->update_occupancy && !m_quadtree_->IsNodeOccupied(leaf)) { continue; }  // the leaf is not marked as occupied, skip it
            if (leaf->GetSurfaceData() != nullptr) { continue; }                                   // the leaf already has surface data, skip it

            double occ_mean;
            Eigen::Vector2d grad_local;
            if (!ComputeGradient2(train_buffer.mat_xy_local.col(i), grad_local, occ_mean)) { continue; }  // uncertain point, drop it

            const Eigen::Vector2d grad_global = m_sensor_gp_->LocalToGlobalSo2(grad_local);
            double var_position, var_gradient;
            ComputeVariance(
                train_buffer.mat_xy_local.col(i),
                grad_local,
                train_buffer.vec_ranges[i],  // distance
                0.,                          // var_distance is not used for a new point
                std::fabs(occ_mean),         // occ_mean_abs
                0.,                          // occ_abs is not used for a new point
                true,                        // new point
                var_position,
                var_gradient);

            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);

            // Insert the point to the tree and mark the key as changed
            leaf->SetSurfaceData(Eigen::Vector2d(train_buffer.mat_xy_global.col(i)), grad_global, var_position, var_gradient);
            RecordChangedKey(key);
        }*/
    }

    bool
    GpOccSurfaceMapping2D::ComputeGradient1(const Eigen::Vector2d &xy_local, Eigen::Vector2d &gradient, double &occ_mean, double &distance_var) {

        double occ[4];
        occ_mean = 0.;
        double distance_sum = 0.;
        double distance_square_mean = 0.;
        gradient.setZero();

        for (int j = 0; j < 4; ++j) {
            double distance;
            Eigen::Scalard angle;
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            Eigen::Scalard distance_pred;
            if (Eigen::Scalard var; !m_sensor_gp_->ComputeOcc(angle, distance, distance_pred, var, occ[j])) { return false; }
            occ_mean += occ[j];
            distance_sum += distance_pred[0];
            distance_square_mean += distance_pred[0] * distance_pred[0];
        }

        occ_mean *= 0.25;
        // 4 samples in total, to calculate the unbiased variance
        // var(r) = sum((r_i - mean(r))^2) / 3. = (mean(r^2) - mean(r) * mean(r)) * 4. / 3. = (sum(r^2) - sum(r) * sum(r) * 0.25) / 3.
        // to remove the numerical approximation's influence, let var(r) = var(r) / delta
        distance_var = (distance_square_mean - distance_sum * distance_sum * 0.25) / (3. * m_setting_->perturb_delta);

        gradient << (occ[0] - occ[1]) / m_setting_->perturb_delta, (occ[2] - occ[3]) / m_setting_->perturb_delta;

        const double gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // uncertain point, drop it
        gradient /= gradient_norm;

        return true;
    }

    bool
    GpOccSurfaceMapping2D::ComputeGradient2(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Vector2d &gradient, double &occ_mean) {
        double occ[4];
        occ_mean = 0.;
        gradient.setZero();

        for (int j = 0; j < 4; ++j) {
            double distance;
            Eigen::Scalard angle;
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            if (Eigen::Scalard distance_pred, var; !m_sensor_gp_->ComputeOcc(angle, distance, distance_pred, var, occ[j])) { return false; }
            occ_mean += occ[j];
        }

        occ_mean *= 0.25;
        gradient << (occ[0] - occ[1]) / m_setting_->perturb_delta, (occ[2] - occ[3]) / m_setting_->perturb_delta;

        const double gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // uncertain point, drop it
        gradient /= gradient_norm;

        return true;
    }

    void
    GpOccSurfaceMapping2D::ComputeVariance(
        const Eigen::Ref<const Eigen::Vector2d> &xy_local,
        const Eigen::Vector2d &grad_local,
        const double &distance,
        const double &distance_var,
        const double &occ_mean_abs,
        const double &occ_abs,
        const bool new_point,
        double &var_position,
        double &var_gradient) const {

        const double min_distance_var = m_setting_->compute_variance.min_distance_var;
        const double max_distance_var = m_setting_->compute_variance.max_distance_var;
        const double min_gradient_var = m_setting_->compute_variance.min_gradient_var;
        const double max_gradient_var = m_setting_->compute_variance.max_gradient_var;

        const double var_distance = common::ClipRange(distance * distance, min_distance_var, max_distance_var);
        const double cos_view_angle = -xy_local.dot(grad_local) / xy_local.norm();
        const double cos2_view_angle = std::max(cos_view_angle * cos_view_angle, 1.e-2);  // avoid zero division
        const double var_direction = (1. - cos2_view_angle) / cos2_view_angle;

        if (new_point) {
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction);
            var_gradient = common::ClipRange(occ_mean_abs, min_gradient_var, max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient = common::ClipRange(occ_mean_abs + distance_var, min_gradient_var, max_gradient_var) + 0.1 * var_direction;
        }
    }
}  // namespace erl::sdf_mapping
