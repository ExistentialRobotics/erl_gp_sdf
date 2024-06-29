#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"

#include "erl_common/clip.hpp"

#include <chrono>

namespace erl::sdf_mapping {

    bool
    GpOccSurfaceMapping3D::Update(
        const Eigen::Ref<const Eigen::Matrix3d> &rotation,
        const Eigen::Ref<const Eigen::Vector3d> &translation,
        const Eigen::Ref<const Eigen::MatrixXd> &ranges) {

        m_changed_keys_.clear();
        auto t0 = std::chrono::high_resolution_clock::now();
        (void) m_sensor_gp_->Train(rotation, translation, ranges);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Sensor GP training time: {} ms.", dt);

        if (!m_sensor_gp_->IsTrained()) { return false; }

        const double d = m_setting_->perturb_delta;
        // clang-format off
        m_xyz_perturb_ << d, -d, 0,  0, 0,  0,
                          0,  0, d, -d, 0,  0,
                          0,  0, 0,  0, d, -d;
        // clang-format on

        if (m_setting_->update_occupancy) {
            t0 = std::chrono::high_resolution_clock::now();
            UpdateOccupancy();
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ERL_INFO("Occupancy update time: {} ms.", dt);
        }

        // perform surface mapping
        t0 = std::chrono::high_resolution_clock::now();
        UpdateMapPoints();
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Map points update time: {} ms.", dt);

        t0 = std::chrono::high_resolution_clock::now();
        AddNewMeasurement();
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Add new measurement time: {} ms.", dt);

        return true;
    }

    void
    GpOccSurfaceMapping3D::UpdateMapPoints() {
        if (m_octree_ == nullptr || !m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const Eigen::Vector3d &sensor_pos = sensor_frame->GetTranslationVector();
        const double max_sensor_range = sensor_frame->GetMaxValidRange();
        const geometry::Aabb3D observed_area(sensor_pos, max_sensor_range);
        const Eigen::Vector3d &min_corner = observed_area.min();
        const Eigen::Vector3d &max_corner = observed_area.max();

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

        const double cluster_half_size = m_setting_->octree->resolution * std::pow(2, cluster_level - 1);
        const double squared_dist_max = max_sensor_range * max_sensor_range + cluster_half_size * cluster_half_size * 2.0;

        // TODO: parallelize this loop if possible
        for (auto it = m_octree_->BeginLeafInAabb(min_corner.x(), min_corner.y(), min_corner.z(), max_corner.x(), max_corner.y(), max_corner.z()),
                  end = m_octree_->EndLeafInAabb();
             it != end;
             ++it) {
            Eigen::Vector3d cluster_position;
            m_octree_->KeyToCoord(it.GetKey(), m_octree_->GetTreeDepth() - cluster_level, cluster_position.x(), cluster_position.y(), cluster_position.z());
            if ((cluster_position - sensor_pos).squaredNorm() > squared_dist_max) { continue; }

            std::shared_ptr<SurfaceMappingOctreeNode::SurfaceData> surface_data = it->GetSurfaceData();
            if (surface_data == nullptr) { continue; }

            const Eigen::Vector3d &xyz_global_old = surface_data->position;
            Eigen::Vector3d xyz_local_old = sensor_frame->WorldToFrameSe3(xyz_global_old);
            double distance_old = xyz_local_old.norm();
            Eigen::Vector2d frame_coords = sensor_frame->ComputeFrameCoords(xyz_local_old);

            if (!sensor_frame->IsInFrame(frame_coords)) { continue; }

            double occ;
            Eigen::Scalard distance_pred, distance_pred_var;
            if (!m_sensor_gp_->ComputeOcc(frame_coords, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
            if (occ < min_observable_occ) { continue; }

            const Eigen::Vector3d &grad_global_old = surface_data->normal;
            Eigen::Vector3d grad_local_old = sensor_frame->WorldToFrameSo3(grad_global_old);

            // compute a new position for the point
            Eigen::Vector3d xyz_local_new = xyz_local_old;
            double delta = m_setting_->perturb_delta;
            int num_adjust_tries = 0;
            double occ_abs = std::fabs(occ);
            double distance_new = distance_old;
            while (num_adjust_tries < max_adjust_tries && occ_abs > max_surface_abs_occ) {
                // move one step
                // the direction is determined by the occupancy sign, the step size is heuristically determined according to iteration.
                if (occ < 0.) {
                    xyz_local_new += grad_local_old * delta;  // point is inside the obstacle
                } else if (occ > 0.) {
                    xyz_local_new -= grad_local_old * delta;  // point is outside the obstacle
                }

                // test the new point
                distance_pred[0] = xyz_local_new.norm();
                frame_coords = sensor_frame->ComputeFrameCoords(xyz_local_new);
                if (double occ_new; m_sensor_gp_->ComputeOcc(frame_coords, distance_pred[0], distance_pred, distance_pred_var, occ_new)) {
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
            Eigen::Vector3d grad_local_new;
            if (!ComputeGradient1(xyz_local_new, grad_local_new, occ_mean, var_distance)) { continue; }
            Eigen::Vector3d grad_global_new = sensor_frame->FrameToWorldSe3(grad_local_new);
            double var_position_new, var_gradient_new;
            ComputeVariance(xyz_local_new, grad_local_new, distance_new, var_distance, std::fabs(occ_mean), occ_abs, false, var_position_new, var_gradient_new);

            Eigen::Vector3d xyz_global_new = sensor_frame->FrameToWorldSe3(xyz_local_new);
            if (const double var_position_old = surface_data->var_position, var_gradient_old = surface_data->var_normal;
                var_gradient_old <= max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad, otherwise, just replace it
                const double var_position_sum = var_position_new + var_position_old;
                const double var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                xyz_global_new = (xyz_global_new * var_position_old + xyz_global_old * var_position_new) / var_position_sum;
                // gradient Update
                Eigen::Vector3d rot_axis = grad_global_old.cross(grad_global_new);
                const double axis_norm = rot_axis.norm();
                rot_axis /= axis_norm;
                const double angle_dist = std::atan2(axis_norm, grad_global_old.dot(grad_global_new)) * var_position_new / var_position_sum;
                Eigen::AngleAxisd rot(angle_dist, rot_axis);
                grad_global_new = rot * grad_global_old;
                // variance Update
                const double distance = (xyz_global_new - xyz_global_old).norm() * 0.5;
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
            if (geometry::OctreeKey new_key = m_octree_->CoordToKey(xyz_global_new.x(), xyz_global_new.y(), xyz_global_new.z()); new_key != it.GetKey()) {
                it->ResetSurfaceData();
                RecordChangedKey(it.GetKey());
                SurfaceMappingOctreeNode *new_node = m_octree_->InsertNode(new_key);
                ERL_ASSERTM(new_node != nullptr, "Failed to get the node");
                if (new_node->GetSurfaceData() != nullptr) { continue; }  // the new node is already occupied
                new_node->SetSurfaceData(surface_data);
                RecordChangedKey(new_key);
            }
            surface_data->position = xyz_global_new;
            surface_data->normal = grad_global_new;
            surface_data->var_position = var_position_new;
            surface_data->var_normal = var_gradient_new;
        }
    }

    void
    GpOccSurfaceMapping3D::UpdateOccupancy() {

        if (m_octree_ == nullptr) { m_octree_ = std::make_shared<SurfaceMappingOctree>(m_setting_->octree); }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const Eigen::Map<const Eigen::Matrix3Xd> map_points(sensor_frame->GetEndPointsInWorld().data()->data(), 3, sensor_frame->GetNumRays());
        constexpr bool parallel = false;
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        m_octree_->InsertPointCloud(map_points, sensor_frame->GetTranslationVector(), sensor_frame->GetMaxValidRange(), parallel, lazy_eval, discrete);
    }

    void
    GpOccSurfaceMapping3D::AddNewMeasurement() {

        if (m_octree_ == nullptr) { m_octree_ = std::make_shared<SurfaceMappingOctree>(m_setting_->octree); }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const std::vector<Eigen::Vector3d> &hit_points = sensor_frame->GetHitPointsWorld();
        const std::vector<std::pair<long, long>> &hit_ray_indices = sensor_frame->GetHitRayIndices();
        const Eigen::MatrixX<Eigen::Vector3d> points_frame = sensor_frame->GetEndPointsInFrame();
        const Eigen::MatrixXd &ranges = sensor_frame->GetRanges();
        const double min_position_var = m_setting_->update_map_points.min_position_var;
        const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;

        for (long i = 0; i < num_hit_rays; ++i) {
            const auto [row, col] = hit_ray_indices[i];
            Eigen::Vector3d xyz_frame = points_frame(row, col);
            Eigen::Vector2d frame_coords = sensor_frame->ComputeFrameCoords(xyz_frame);
            Eigen::Scalard predicted_range, predicted_range_var;
            if (!m_sensor_gp_->Test(frame_coords, predicted_range, predicted_range_var, true)) { continue; }

            geometry::OctreeKey key = m_octree_->CoordToKey(hit_points[i].x(), hit_points[i].y(), hit_points[i].z());
            SurfaceMappingOctreeNode *node = m_octree_->InsertNode(key);  // insert the node
            if (node == nullptr) { continue; }                            // failed to insert the node
            if (node->GetSurfaceData() != nullptr) { continue; }          // the node is already occupied

            double occ_mean;
            Eigen::Vector3d grad_local;
            if (!ComputeGradient2(xyz_frame, grad_local, occ_mean)) { continue; }

            const Eigen::Vector3d grad_global = sensor_frame->FrameToWorldSo3(grad_local);
            double var_position, var_gradient;
            ComputeVariance(xyz_frame, grad_local, ranges(row, col), 0, std::fabs(occ_mean), 0, true, var_position, var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);

            node->SetSurfaceData(hit_points[i], grad_global, var_position, var_gradient);
            RecordChangedKey(key);
        }
    }

    bool
    GpOccSurfaceMapping3D::ComputeGradient1(
        const Eigen::Ref<const Eigen::Vector3d> &xyz_local,
        Eigen::Ref<Eigen::Vector3d> gradient,
        double &occ_mean,
        double &distance_var) {

        double occ[6];
        occ_mean = 0;
        double distance_sum = 0.0;
        double distance_square_mean = 0.0;
        gradient.setZero();

        for (int i = 0; i < 6; ++i) {
            const Eigen::Vector3d xyz_local_perturbed = xyz_local + m_xyz_perturb_.col(i);
            const double distance = xyz_local_perturbed.norm();
            Eigen::Scalard distance_pred, distance_pred_var;
            if (!m_sensor_gp_->ComputeOcc(m_sensor_gp_->ComputeFrameCoords(xyz_local_perturbed), distance, distance_pred, distance_pred_var, occ[i])) {
                return false;
            }
            occ_mean += occ[i];
            distance_sum += distance_pred[0];
            distance_square_mean += distance_pred[0] * distance_pred[0];
        }

        occ_mean *= 1.0 / 6.0;
        const double delta = m_setting_->perturb_delta;
        // 6 samples in total, to calculate the unbiased variance
        // var(r) = sum((r_i - mean(r))^2) / 5 = (mean(r^2) - mean(r)^2) * 6 / 5 = (sum(r^2) - sum(r)^2 / 6) / 5
        // to remove the numerical approximation's influence, let var(r) = var(r) / delta
        distance_var = (distance_square_mean - distance_sum * distance_sum / 6.0) / (5.0 * delta);

        gradient << (occ[0] - occ[1]) / delta, (occ[2] - occ[3]) / delta, (occ[4] - occ[5]) / delta;
        const double gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // zero gradient
        gradient /= gradient_norm;
        return true;
    }

    bool
    GpOccSurfaceMapping3D::ComputeGradient2(const Eigen::Ref<const Eigen::Vector3d> &xyz_local, Eigen::Ref<Eigen::Vector3d> gradient, double &occ_mean) {
        //
        double occ[6];
        occ_mean = 0;
        gradient.setZero();

        for (int i = 0; i < 6; ++i) {
            const Eigen::Vector3d xyz_local_perturbed = xyz_local + m_xyz_perturb_.col(i);
            const double distance = xyz_local_perturbed.norm();
            Eigen::Scalard distance_pred, distance_pred_var;
            if (!m_sensor_gp_->ComputeOcc(m_sensor_gp_->ComputeFrameCoords(xyz_local_perturbed), distance, distance_pred, distance_pred_var, occ[i])) {
                return false;
            }
            occ_mean += occ[i];
        }

        occ_mean *= 1.0 / 6.0;
        const double delta = m_setting_->perturb_delta;

        gradient << (occ[0] - occ[1]) / delta, (occ[2] - occ[3]) / delta, (occ[4] - occ[5]) / delta;
        const double gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // zero gradient
        gradient /= gradient_norm;
        return true;
    }

    void
    GpOccSurfaceMapping3D::ComputeVariance(
        const Eigen::Ref<const Eigen::Vector3d> &xyz_local,
        const Eigen::Vector3d &grad_local,
        const double &distance,
        const double &distance_var,
        const double &occ_mean_abs,
        const double &occ_abs,
        bool new_point,
        double &var_position,
        double &var_gradient) const {

        const double min_distance_var = m_setting_->compute_variance.min_distance_var;
        const double max_distance_var = m_setting_->compute_variance.max_distance_var;
        const double min_gradient_var = m_setting_->compute_variance.min_gradient_var;
        const double max_gradient_var = m_setting_->compute_variance.max_gradient_var;

        const double var_distance = common::ClipRange(distance * distance, min_distance_var, max_distance_var);
        const double cos_view_angle = -xyz_local.dot(grad_local) / xyz_local.norm();
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
