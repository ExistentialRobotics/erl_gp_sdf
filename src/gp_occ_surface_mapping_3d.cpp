#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/clip.hpp"

namespace erl::sdf_mapping {

    bool
    GpOccSurfaceMapping3D::Update(
        const Eigen::Ref<const Eigen::Matrix3d> &rotation,
        const Eigen::Ref<const Eigen::Vector3d> &translation,
        const Eigen::Ref<const Eigen::MatrixXd> &ranges) {

        m_changed_keys_.clear();

        bool sensor_gp_trained = false;
        const bool verbose_timer = common::Logging::GetLevel() <= common::Logging::Level::kInfo;
        {
            common::BlockTimer<std::chrono::milliseconds> timer("Sensor GP training", nullptr, verbose_timer);
            sensor_gp_trained = m_sensor_gp_->Train(rotation, translation, ranges);
        }
        if (!sensor_gp_trained) { return false; }

        const double d = m_setting_->perturb_delta;
        // clang-format off
        m_xyz_perturb_ << d, -d, 0,  0, 0,  0,
                          0,  0, d, -d, 0,  0,
                          0,  0, 0,  0, d, -d;
        // clang-format on

        if (m_setting_->update_occupancy) {
            common::BlockTimer<std::chrono::milliseconds> timer("Update occupancy", nullptr, verbose_timer);
            UpdateOccupancy();
        }

        // perform surface mapping
        {
            common::BlockTimer<std::chrono::milliseconds> timer("Update map points", nullptr, verbose_timer);
            UpdateMapPoints();
        }

        {
            common::BlockTimer<std::chrono::milliseconds> timer("Add new measurement time", nullptr, verbose_timer);
            AddNewMeasurement();
        }

        return true;
    }

    bool
    GpOccSurfaceMapping3D::operator==(const AbstractSurfaceMapping3D &other) const {
        const auto *other_ptr = dynamic_cast<const GpOccSurfaceMapping3D *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) { return false; }
        if (m_sensor_gp_ == nullptr && other_ptr->m_sensor_gp_ != nullptr) { return false; }
        if (m_sensor_gp_ != nullptr && (other_ptr->m_sensor_gp_ == nullptr || *m_sensor_gp_ != *other_ptr->m_sensor_gp_)) { return false; }
        if (m_octree_ == nullptr && other_ptr->m_octree_ != nullptr) { return false; }
        if (m_octree_ != nullptr && (other_ptr->m_octree_ == nullptr || *m_octree_ != *other_ptr->m_octree_)) { return false; }
        if (m_xyz_perturb_ != other_ptr->m_xyz_perturb_) { return false; }
        if (m_changed_keys_ != other_ptr->m_changed_keys_) { return false; }
        return true;
    }

    bool
    GpOccSurfaceMapping3D::Write(const std::string &filename) const {
        ERL_INFO("Writing GpOccSurfaceMapping3D to file: {}", filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Write(file);
        file.close();
        return success;
    }

    static const std::string kFileHeader = "# erl::sdf_mapping::GpOccSurfaceMapping3D";

    bool
    GpOccSurfaceMapping3D::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        s << "sensor_gp" << std::endl;
        if (!m_sensor_gp_->Write(s)) {
            ERL_WARN("Failed to write sensor_gp.");
            return false;
        }
        s << "octree " << (m_octree_ != nullptr) << std::endl;
        if (m_octree_ != nullptr) {
            if (!m_octree_->Write(s)) {
                ERL_WARN("Failed to write octree.");
                return false;
            }
        }
        s << "xyz_perturb" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_xyz_perturb_)) {
            ERL_WARN("Failed to write xyz_perturb.");
            return false;
        }
        s << "changed_keys " << m_changed_keys_.size() << std::endl;
        for (const geometry::OctreeKey &key: m_changed_keys_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
            s.write(reinterpret_cast<const char *>(&key[2]), sizeof(key[2]));
        }
        s << "end_of_GpOccSurfaceMapping3D" << std::endl;
        return s.good();
    }

    bool
    GpOccSurfaceMapping3D::Read(const std::string &filename) {
        ERL_INFO("Reading GpOccSurfaceMapping3D from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    GpOccSurfaceMapping3D::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "sensor_gp",
            "octree",
            "xyz_perturb",
            "changed_keys",
            "end_of_GpOccSurfaceMapping3D",
        };

        // read data
        std::string token;
        int token_idx = 0;
        while (s.good()) {
            s >> token;
            if (token.compare(0, 1, "#") == 0) {
                skip_line();  // comment line, skip forward until end of line
                continue;
            }
            // non-comment line
            if (token != tokens[token_idx]) {
                ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
                return false;
            }
            // reading state machine
            switch (token_idx) {
                case 0: {  // setting
                    skip_line();
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // sensor_gp
                    skip_line();
                    m_sensor_gp_ = std::make_shared<gaussian_process::RangeSensorGaussianProcess3D>(m_setting_->sensor_gp);
                    if (!m_sensor_gp_->Read(s)) {
                        ERL_WARN("Failed to read sensor_gp.");
                        return false;
                    }
                    break;
                }
                case 2: {  // octree
                    bool has_octree;
                    s >> has_octree;
                    skip_line();
                    if (has_octree) {
                        m_octree_ = std::make_shared<geometry::SurfaceMappingOctree>(m_setting_->octree);
                        if (!m_octree_->LoadData(s)) {
                            ERL_WARN("Failed to read octree.");
                            return false;
                        }
                    }
                    break;
                }
                case 3: {  // xyz_perturb
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_xyz_perturb_)) {
                        ERL_WARN("Failed to read xyz_perturb.");
                        return false;
                    }
                    break;
                }
                case 4: {  // changed_keys
                    size_t num_keys;
                    s >> num_keys;
                    skip_line();
                    m_changed_keys_.reserve(num_keys);
                    for (size_t i = 0; i < num_keys; ++i) {
                        geometry::OctreeKey key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        s.read(reinterpret_cast<char *>(&key[2]), sizeof(key[2]));
                        m_changed_keys_.insert(key);
                    }
                    break;
                }
                case 5: {  // end_of_GpOccSurfaceMapping3D
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read GpOccSurfaceMapping3D. Truncated file?");
        return false;  // should not reach here
    }

    void
    GpOccSurfaceMapping3D::UpdateMapPoints() {
        if (m_octree_ == nullptr || !m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const Eigen::Vector3d &sensor_pos = sensor_frame->GetTranslationVector();
        const double max_sensor_range = sensor_frame->GetMaxValidRange();
        const geometry::Aabb3D observed_area(sensor_pos, max_sensor_range);

        std::vector<std::tuple<geometry::OctreeKey, geometry::SurfaceMappingOctreeNode *, std::optional<geometry::OctreeKey>>> nodes_in_aabb;
        for (auto it = m_octree_->BeginLeafInAabb(observed_area), end = m_octree_->EndLeafInAabb(); it != end; ++it) {
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

            const double cluster_half_size = m_setting_->octree->resolution * std::pow(2, cluster_level - 1);
            const double squared_dist_max = max_sensor_range * max_sensor_range + cluster_half_size * cluster_half_size * 3.0;

            if (const Eigen::Vector3d cluster_position = m_octree_->KeyToCoord(node_key, m_octree_->GetTreeDepth() - cluster_level);
                (cluster_position - sensor_pos).squaredNorm() > squared_dist_max) {
                continue;
            }

            std::shared_ptr<geometry::SurfaceMappingOctreeNode::SurfaceData> surface_data = node->GetSurfaceData();
            if (surface_data == nullptr) { continue; }

            const Eigen::Vector3d &xyz_global_old = surface_data->position;
            Eigen::Vector3d xyz_local_old = sensor_frame->WorldToFrameSe3(xyz_global_old);

            if (!sensor_frame->PointIsInFrame(xyz_local_old)) { continue; }

            double occ;
            Eigen::Scalard distance_pred, distance_pred_var;
            const double distance_old = xyz_local_old.norm();
            if (!m_sensor_gp_->ComputeOcc(xyz_local_old / distance_old, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
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
                if (double occ_new; m_sensor_gp_->ComputeOcc(xyz_local_new / distance_pred[0], distance_pred[0], distance_pred, distance_pred_var, occ_new)) {
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
            Eigen::Vector3d grad_global_new = sensor_frame->FrameToWorldSo3(grad_local_new);
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
                if (axis_norm < 1.e-6) {
                    rot_axis = grad_global_old;  // parallel
                } else {
                    rot_axis /= axis_norm;
                }
                const double angle_dist = std::atan2(axis_norm, grad_global_old.dot(grad_global_new)) * var_position_new / var_position_sum;
                Eigen::AngleAxisd rot(angle_dist, rot_axis);
                grad_global_new = rot * grad_global_old;
                ERL_DEBUG_ASSERT(std::abs(grad_global_new.norm() - 1.0) < 1.e-6, "grad_global_new.norm() = {:.6f}", grad_global_new.norm());

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
                node->ResetSurfaceData();
                continue;  // too bad, skip
            }
            if (new_key = m_octree_->CoordToKey(xyz_global_new); new_key.value() == node_key) { new_key = std::nullopt; }
            surface_data->position = xyz_global_new;
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

                geometry::SurfaceMappingOctreeNode *new_node = m_octree_->InsertNode(new_key.value());
                ERL_DEBUG_ASSERT(new_node != nullptr, "Failed to get the node");
                if (new_node->GetSurfaceData() != nullptr) { continue; }  // the new node is already occupied
                new_node->SetSurfaceData(surface_data);
                RecordChangedKey(new_key.value());
            }
        }
    }

    void
    GpOccSurfaceMapping3D::UpdateOccupancy() {

        if (m_octree_ == nullptr) { m_octree_ = std::make_shared<geometry::SurfaceMappingOctree>(m_setting_->octree); }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const Eigen::Map<const Eigen::Matrix3Xd> map_points(sensor_frame->GetHitPointsWorld().data()->data(), 3, sensor_frame->GetNumHitRays());
        constexpr bool parallel = false;
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        m_octree_->InsertPointCloud(map_points, sensor_frame->GetTranslationVector(), sensor_frame->GetMaxValidRange(), parallel, lazy_eval, discrete);
    }

    void
    GpOccSurfaceMapping3D::AddNewMeasurement() {

        if (m_octree_ == nullptr) { m_octree_ = std::make_shared<geometry::SurfaceMappingOctree>(m_setting_->octree); }

        const auto sensor_frame = m_sensor_gp_->GetRangeSensorFrame();
        const std::vector<std::pair<long, long>> &hit_ray_indices = sensor_frame->GetHitRayIndices();
        const Eigen::MatrixX<Eigen::Vector3d> &points_local = sensor_frame->GetEndPointsInFrame();
        const std::vector<Eigen::Vector3d> &hit_points = sensor_frame->GetHitPointsWorld();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const Eigen::MatrixXd &ranges = sensor_frame->GetRanges();
        const double min_position_var = m_setting_->update_map_points.min_position_var;
        const double min_gradient_var = m_setting_->update_map_points.min_gradient_var;

        // collect new measurements
        // if we iterate over the hit rays directly, some computations are unnecessary
        std::vector<std::tuple<geometry::OctreeKey, long, long, long, geometry::SurfaceMappingOctreeNode *, bool, double, Eigen::Vector3d>> new_measurements;
        new_measurements.reserve(num_hit_rays);
        geometry::OctreeKeySet new_measurement_keys;
        new_measurement_keys.reserve(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Eigen::Vector3d &hit_point = hit_points[i];
            geometry::OctreeKey key = m_octree_->CoordToKey(hit_point);
            if (!new_measurement_keys.insert(key).second) { continue; }             // the key is already in the set
            geometry::SurfaceMappingOctreeNode *node = m_octree_->InsertNode(key);  // insert the node
            if (node == nullptr) { continue; }                                      // failed to insert the node
            if (node->GetSurfaceData() != nullptr) { continue; }                    // the node is already occupied
            auto [row, col] = hit_ray_indices[i];
            new_measurements.emplace_back(key, i, row, col, node, false, 0.0, Eigen::Vector3d::Zero());
        }

        const auto num_new_measurements = static_cast<long>(new_measurements.size());
#pragma omp parallel for default(none) shared(num_new_measurements, new_measurements, points_local)
        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, row, col, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            invalid_flag = !ComputeGradient2(points_local(row, col), gradient_local, occ_mean);
        }

        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, row, col, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            if (invalid_flag) { continue; }
            if (node->GetSurfaceData() != nullptr) { continue; }  // the node is already occupied by previous new measurement

            double var_position, var_gradient;
            ComputeVariance(points_local(row, col), gradient_local, ranges(row, col), 0, std::fabs(occ_mean), 0, true, var_position, var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);

            node->SetSurfaceData(hit_points[hit_idx], sensor_frame->FrameToWorldSo3(gradient_local), var_position, var_gradient);
            RecordChangedKey(key);
        }
    }

    bool
    GpOccSurfaceMapping3D::ComputeGradient1(const Eigen::Vector3d &xyz_local, Eigen::Vector3d &gradient, double &occ_mean, double &distance_var) {

        double occ[6];
        occ_mean = 0;
        double distance_sum = 0.0;
        double distance_square_mean = 0.0;
        gradient.setZero();

        for (int i = 0; i < 6; ++i) {
            const Eigen::Vector3d xyz_local_perturbed = xyz_local + m_xyz_perturb_.col(i);
            const double distance = xyz_local_perturbed.norm();
            Eigen::Scalard distance_pred;
            if (Eigen::Scalard distance_pred_var;
                !m_sensor_gp_->ComputeOcc(xyz_local_perturbed / distance, distance, distance_pred, distance_pred_var, occ[i])) {
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
    GpOccSurfaceMapping3D::ComputeGradient2(const Eigen::Vector3d &xyz_local, Eigen::Vector3d &gradient, double &occ_mean) {
        double occ[6];
        occ_mean = 0;
        gradient.setZero();

        const double valid_range_min = m_setting_->sensor_gp->range_sensor_frame->valid_range_min;
        const double valid_range_max = m_setting_->sensor_gp->range_sensor_frame->valid_range_max;
        for (int i = 0; i < 6; ++i) {
            const Eigen::Vector3d xyz_local_perturbed = xyz_local + m_xyz_perturb_.col(i);
            const double distance = xyz_local_perturbed.norm();
            if (Eigen::Scalard distance_pred, distance_pred_var;
                !m_sensor_gp_->ComputeOcc(xyz_local_perturbed / distance, distance, distance_pred, distance_pred_var, occ[i]) ||
                distance_pred[0] < valid_range_min || distance_pred[0] > valid_range_max) {
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
        const bool new_point,
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
