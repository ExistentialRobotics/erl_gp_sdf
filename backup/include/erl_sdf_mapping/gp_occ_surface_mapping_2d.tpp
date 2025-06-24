#pragma once

#include "erl_common/block_timer.hpp"
#include "erl_common/clip.hpp"

namespace erl::gp_sdf {

    template<typename Dtype>
    YAML::Node
    GpOccSurfaceMapping2D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node = YAML::convert<GpOccSurfaceMappingBaseSetting>::encode(setting);
        node["sensor_gp"] = setting.sensor_gp;
        node["quadtree"] = setting.quadtree;
        return node;
    }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!YAML::convert<GpOccSurfaceMappingBaseSetting>::decode(node, setting)) { return false; }
        setting.sensor_gp = node["sensor_gp"].as<decltype(setting.sensor_gp)>();
        setting.quadtree = node["quadtree"].as<decltype(setting.quadtree)>();
        return true;
    }

    static void
    Cartesian2Polar(const Eigen::Ref<const Vector2> &xy, Dtype &r, Dtype &angle) {
        r = xy.norm();
        angle = std::atan2(xy.y(), xy.x());
    }

    template<typename Dtype>
    std::shared_ptr<const typename GpOccSurfaceMapping2D<Dtype>::Setting>
    GpOccSurfaceMapping2D<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    std::shared_ptr<const typename GpOccSurfaceMapping2D<Dtype>::SensorGp>
    GpOccSurfaceMapping2D<Dtype>::GetSensorGp() const {
        return m_sensor_gp_;
    }

    template<typename Dtype>
    geometry::QuadtreeKeySet
    GpOccSurfaceMapping2D<Dtype>::GetChangedClusters() {
        return m_changed_keys_;
    }

    template<typename Dtype>
    unsigned int
    GpOccSurfaceMapping2D<Dtype>::GetClusterLevel() const {
        return m_setting_->cluster_level;
    }

    template<typename Dtype>
    std::shared_ptr<typename GpOccSurfaceMapping2D<Dtype>::Tree>
    GpOccSurfaceMapping2D<Dtype>::GetQuadtree() {
        return m_tree_;
    }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::Update(
        const Eigen::Ref<const Matrix2> &rotation,
        const Eigen::Ref<const Vector2> &translation,
        const Eigen::Ref<const MatrixX> &ranges) {

        m_changed_keys_.clear();
        if (!m_sensor_gp_->Train(rotation, translation, ranges)) { return false; }
        if (m_setting_->update_occupancy) { UpdateOccupancy(); }
        UpdateMapPoints();
        AddNewMeasurement();

        return true;
    }

    template<typename Dtype>
    const typename GpOccSurfaceMapping2D<Dtype>::SurfaceDataManager2D &
    GpOccSurfaceMapping2D<Dtype>::GetSurfaceDataManager() const {
        return m_surface_data_manager_;
    }

    template<typename Dtype>
    Dtype
    GpOccSurfaceMapping2D<Dtype>::GetSensorNoise() const {
        return m_setting_->sensor_gp->sensor_range_var;
    }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::operator==(const Super &other) const {
        const auto *other_ptr = dynamic_cast<const GpOccSurfaceMapping2D *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) { return false; }
        if (m_sensor_gp_ == nullptr && other_ptr->m_sensor_gp_ != nullptr) { return false; }
        if (m_sensor_gp_ != nullptr && (other_ptr->m_sensor_gp_ == nullptr || *m_sensor_gp_ != *other_ptr->m_sensor_gp_)) { return false; }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr && (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) { return false; }
        if (m_xy_perturb_ != other_ptr->m_xy_perturb_) { return false; }
        if (m_changed_keys_ != other_ptr->m_changed_keys_) { return false; }
        return true;
    }

    // template<typename Dtype>
    // bool
    // GpOccSurfaceMapping2D<Dtype>::Write(const std::string &filename) const {
    //     ERL_INFO("Writing GpOccSurfaceMapping2D to file: {}", filename);
    //     std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    //     std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
    //     if (!file.is_open()) {
    //         ERL_WARN("Failed to open file: {}", filename);
    //         return false;
    //     }
    //
    //     const bool success = Write(file);
    //     file.close();
    //     return success;
    // }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::Write(std::ostream &s) const {
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
        s << "quadtree " << (m_tree_ != nullptr) << std::endl;
        if (m_tree_ != nullptr) {
            if (!m_tree_->Write(s)) {
                ERL_WARN("Failed to write quadtree.");
                return false;
            }
        }
        s << "xy_perturb" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_xy_perturb_)) {
            ERL_WARN("Failed to write xy_perturb.");
            return false;
        }
        s << "changed_keys " << m_changed_keys_.size() << std::endl;
        for (const geometry::QuadtreeKey &key: m_changed_keys_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
        }
        s << "end_of_GpOccSurfaceMapping2D" << std::endl;
        return s.good();
    }

    // template<typename Dtype>
    // bool
    // GpOccSurfaceMapping2D<Dtype>::Read(const std::string &filename) {
    //     ERL_INFO("Reading GpOccSurfaceMapping2D from file: {}", std::filesystem::absolute(filename));
    //     std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    //     if (!file.is_open()) {
    //         ERL_WARN("Failed to open file: {}", filename.c_str());
    //         return false;
    //     }
    //
    //     const bool success = Read(file);
    //     file.close();
    //     return success;
    // }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::Read(std::istream &s) {
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

        auto skip_line = [&s] {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "sensor_gp",
            "quadtree",
            "xy_perturb",
            "changed_keys",
            "end_of_GpOccSurfaceMapping2D",
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
                    m_sensor_gp_ = std::make_shared<gaussian_process::LidarGaussianProcess2D>(m_setting_->sensor_gp);
                    if (!m_sensor_gp_->Read(s)) {
                        ERL_WARN("Failed to read sensor_gp.");
                        return false;
                    }
                    break;
                }
                case 2: {  // quadtree
                    bool has_quadtree;
                    s >> has_quadtree;
                    skip_line();
                    if (has_quadtree) {
                        m_tree_ = std::make_shared<SurfaceMappingQuadtree>(m_setting_->quadtree);
                        if (!m_tree_->LoadData(s)) {
                            ERL_WARN("Failed to read quadtree.");
                            return false;
                        }
                    }
                    break;
                }
                case 3: {  // xy_perturb
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_xy_perturb_)) {
                        ERL_WARN("Failed to read xy_perturb.");
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
                        geometry::QuadtreeKey key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        m_changed_keys_.insert(key);
                    }
                    break;
                }
                case 5: {  // end_of_GpOccSurfaceMapping2D
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read GpOccSurfaceMapping2D. Truncated file?");
        return false;  // should not reach here
    }

    template<typename Dtype>
    void
    GpOccSurfaceMapping2D<Dtype>::UpdateMapPoints() {
        ERL_BLOCK_TIMER();

        if (m_tree_ == nullptr || !m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Vector2 &sensor_pos = sensor_frame->GetTranslationVector();
        const Dtype max_sensor_range = sensor_frame->GetMaxValidRange();
        const geometry::Aabb2D observed_area(sensor_pos, max_sensor_range);

        std::vector<std::tuple<geometry::QuadtreeKey, SurfaceMappingQuadtreeNode *, bool, std::optional<geometry::QuadtreeKey>>> nodes_in_aabb;
        for (auto it = m_tree_->BeginLeafInAabb(observed_area), end = m_tree_->EndLeafInAabb(); it != end; ++it) {
            nodes_in_aabb.emplace_back(it.GetKey(), *it, false, std::nullopt);
        }

#pragma omp parallel for default(none) shared(nodes_in_aabb, max_sensor_range, sensor_pos, sensor_frame, g_print_mutex)
        for (auto &[node_key, node, bad_update, new_key]: nodes_in_aabb) {
            const uint32_t cluster_level = m_setting_->cluster_level;
            const Dtype sensor_range_var = m_setting_->sensor_gp->sensor_range_var;
            const Dtype min_comp_gradient_var = m_setting_->compute_variance.min_gradient_var;
            const Dtype max_comp_gradient_var = m_setting_->compute_variance.max_gradient_var;
            const int max_adjust_tries = m_setting_->update_map_points.max_adjust_tries;
            const Dtype min_observable_occ = m_setting_->update_map_points.min_observable_occ;
            const Dtype max_surface_abs_occ = m_setting_->update_map_points.max_surface_abs_occ;
            const Dtype max_valid_gradient_var = m_setting_->update_map_points.max_valid_gradient_var;
            const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
            const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;
            const Dtype max_bayes_position_var = m_setting_->update_map_points.max_bayes_position_var;
            const Dtype max_bayes_gradient_var = m_setting_->update_map_points.max_bayes_gradient_var;

            const Dtype cluster_half_size = m_setting_->quadtree->resolution * std::pow(2, cluster_level - 1);
            const Dtype squared_dist_max = max_sensor_range * max_sensor_range + cluster_half_size * cluster_half_size * 2.0;

            if (const Vector2 cluster_position = m_tree_->KeyToCoord(node_key, m_tree_->GetTreeDepth() - cluster_level);
                (cluster_position - sensor_pos).squaredNorm() > squared_dist_max) {
                continue;
            }

            if (!node->HasSurfaceData()) { continue; }
            auto &surface_data = m_surface_data_manager_[node->surface_data_index];

            const Vector2 &pos_global_old = surface_data.position;
            Vector2 pos_local_old = sensor_frame->PosWorldToFrame(pos_global_old);

            if (!sensor_frame->PointIsInFrame(pos_local_old)) { continue; }

            Dtype occ, distance_old;
            Scalar distance_pred, distance_pred_var, angle_local;
            Cartesian2Polar(pos_local_old, distance_old, angle_local[0]);
            if (!m_sensor_gp_->ComputeOcc(angle_local, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
            if (occ < min_observable_occ) { continue; }

            const Vector2 &grad_global_old = surface_data.normal;
            Vector2 grad_local_old = sensor_frame->DirWorldToFrame(grad_global_old);

            // compute a new position for the point
            Vector2 pos_local_new = pos_local_old;
            Dtype delta = m_setting_->perturb_delta;
            int num_adjust_tries = 0;
            Dtype occ_abs = std::fabs(occ);
            Dtype distance_new = distance_old;
            while (num_adjust_tries < max_adjust_tries && occ_abs > max_surface_abs_occ) {
                // move one step
                // the direction is determined by the occupancy sign, the step size is heuristically determined according to iteration.
                if (occ < 0.) {
                    pos_local_new += grad_local_old * delta;  // point is inside the obstacle
                } else if (occ > 0.) {
                    pos_local_new -= grad_local_old * delta;  // point is outside the obstacle
                }

                // test the new point
                Cartesian2Polar(pos_local_new, distance_pred[0], angle_local[0]);
                if (Dtype occ_new; m_sensor_gp_->ComputeOcc(angle_local, distance_pred[0], distance_pred, distance_pred_var, occ_new)) {
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
            Dtype occ_mean, var_distance;
            Vector2 grad_local_new;
            if (!ComputeGradient1(pos_local_new, grad_local_new, occ_mean, var_distance)) { continue; }
            Vector2 grad_global_new = sensor_frame->DirFrameToWorld(grad_local_new);
            Dtype var_position_new, var_gradient_new;
            ComputeVariance(pos_local_new, grad_local_new, distance_new, var_distance, std::fabs(occ_mean), occ_abs, false, var_position_new, var_gradient_new);

            Vector2 pos_global_new = sensor_frame->PosFrameToWorld(pos_local_new);
            if (const Dtype var_position_old = surface_data.var_position, var_gradient_old = surface_data.var_normal;
                var_gradient_old <= max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad, otherwise, just replace it
                const Dtype var_position_sum = var_position_new + var_position_old;
                const Dtype var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                pos_global_new = (pos_global_new * var_position_old + pos_global_old * var_position_new) / var_position_sum;
                // gradient Update
                const Dtype &old_x = grad_global_old.x();
                const Dtype &old_y = grad_global_old.y();
                const Dtype &new_x = grad_global_new.x();
                const Dtype &new_y = grad_global_new.y();
                const Dtype angle_dist = std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_gradient_new / var_gradient_sum;
                const Dtype sin = std::sin(angle_dist);
                const Dtype cos = std::cos(angle_dist);
                // rotate grad_global_old by angle_dist
                grad_global_new.x() = cos * old_x - sin * old_y;
                grad_global_new.y() = sin * old_x + cos * old_y;
                ERL_DEBUG_ASSERT(std::abs(grad_global_new.norm() - 1.0) < 1.e-6, "grad_global_new.norm() = {:.6f}", grad_global_new.norm());

                // variance Update
                const Dtype distance = (pos_global_new - pos_global_old).norm() * 0.5;
                var_position_new = std::max(var_position_new * var_position_old / var_position_sum + distance, sensor_range_var);
                var_gradient_new = common::ClipRange(  //
                    var_gradient_new * var_gradient_old / var_gradient_sum + distance,
                    min_comp_gradient_var,
                    max_comp_gradient_var);
            }
            var_position_new = std::max(var_position_new, min_position_var);
            var_gradient_new = std::max(var_gradient_new, min_gradient_var);

            // Update the surface data
            if (var_position_new > max_bayes_position_var && var_gradient_new > max_bayes_gradient_var) {
                bad_update = true;
                continue;  // too bad, skip
            }
            if (new_key = m_tree_->CoordToKey(pos_global_new); new_key.value() == node_key) { new_key = std::nullopt; }
            surface_data.position = pos_global_new;
            surface_data.normal = grad_global_new;
            surface_data.var_position = var_position_new;
            surface_data.var_normal = var_gradient_new;
            ERL_DEBUG_ASSERT(std::abs(surface_data.normal.norm() - 1.0) < 1.e-5, "surface_data->normal.norm() = {:.6f}", surface_data.normal.norm());
        }

        for (auto &[key, node, bad_update, new_key]: nodes_in_aabb) {
            if (bad_update) {                                                   // too bad, the surface data should be removed
                m_surface_data_manager_.RemoveEntry(node->surface_data_index);  // this surface data is not used anymore
                node->ResetSurfaceDataIndex();                                  // the node has no surface data now
            }
            if (!node->HasSurfaceData()) {  // surface data is removed
                RecordChangedKey(key);
                continue;
            }
            if (new_key.has_value()) {  // the node is moved to a new position
                RecordChangedKey(key);

                SurfaceMappingQuadtreeNode *new_node = m_tree_->InsertNode(new_key.value());
                ERL_DEBUG_ASSERT(new_node != nullptr, "Failed to get the node");
                if (new_node->HasSurfaceData()) {                                   // the new node is already occupied
                    m_surface_data_manager_.RemoveEntry(node->surface_data_index);  // this surface data is not used anymore
                    node->ResetSurfaceDataIndex();                                  // the old node is empty now
                    continue;
                }
                new_node->surface_data_index = node->surface_data_index;  // move the surface data to the new node
                node->ResetSurfaceDataIndex();                            // the old node is empty now

                RecordChangedKey(new_key.value());
            }
        }
    }

    template<typename Dtype>
    void
    GpOccSurfaceMapping2D<Dtype>::UpdateOccupancy() {
        ERL_BLOCK_TIMER();
        if (m_tree_ == nullptr) { m_tree_ = std::make_shared<SurfaceMappingQuadtree>(m_setting_->quadtree); }
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        // In AddNewMeasurement(), only rays classified as hit are used. So, we use the same here to avoid inconsistency.
        // Experiments show that this achieves higher fps and better results.
        const Eigen::Map<const Matrix2X> map_points(sensor_frame->GetHitPointsWorld().data()->data(), 2, sensor_frame->GetNumHitRays());
        constexpr bool parallel = false;
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        m_tree_->InsertPointCloud(map_points, sensor_frame->GetTranslationVector(), sensor_frame->GetMaxValidRange(), parallel, lazy_eval, discrete);
    }

    template<typename Dtype>
    void
    GpOccSurfaceMapping2D<Dtype>::AddNewMeasurement() {
        ERL_BLOCK_TIMER();

        if (m_tree_ == nullptr) { m_tree_ = std::make_shared<SurfaceMappingQuadtree>(m_setting_->quadtree); }
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const std::vector<long> &hit_ray_indices = sensor_frame->GetHitRayIndices();
        const std::vector<Vector2> &points_local = sensor_frame->GetEndPointsInFrame();
        const std::vector<Vector2> &hit_points = sensor_frame->GetHitPointsWorld();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const VectorX &ranges = sensor_frame->GetRanges();
        const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
        const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;

        // collect new measurements
        // if we iterate over the hit rays directly, some computations are unnecessary
        std::vector<std::tuple<geometry::QuadtreeKey, long, long, SurfaceMappingQuadtreeNode *, bool, Dtype, Vector2>> new_measurements;
        new_measurements.reserve(num_hit_rays);
        geometry::QuadtreeKeySet new_measurement_keys;
        new_measurement_keys.reserve(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Vector2 &hit_point = hit_points[i];
            geometry::QuadtreeKey key = m_tree_->CoordToKey(hit_point);
            if (!new_measurement_keys.insert(key).second) { continue; }   // the key is already in the set
            SurfaceMappingQuadtreeNode *node = m_tree_->InsertNode(key);  // insert the node
            if (node == nullptr) { continue; }                            // failed to insert the node
            if (node->HasSurfaceData()) { continue; }                     // the node is already occupied
            new_measurements.emplace_back(key, i, hit_ray_indices[i], node, false, 0.0, Vector2::Zero());
        }

        const auto num_new_measurements = static_cast<long>(new_measurements.size());
#pragma omp parallel for default(none) shared(num_new_measurements, new_measurements, points_local)
        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, idx, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            invalid_flag = !ComputeGradient2(points_local[idx], gradient_local, occ_mean);
        }

        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, idx, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            if (invalid_flag) { continue; }            // invalid measurement
            if (node->HasSurfaceData()) { continue; }  // the node is already occupied by previous new measurement

            Dtype var_position, var_gradient;
            ComputeVariance(points_local[idx], gradient_local, ranges[idx], 0, std::fabs(occ_mean), 0, true, var_position, var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);
            node->surface_data_index = m_surface_data_manager_.AddEntry(SurfaceDataManager<2>::SurfaceData{
                hit_points[hit_idx],
                sensor_frame->DirFrameToWorld(gradient_local),
                var_position,
                var_gradient,
            });
            RecordChangedKey(key);
        }
    }

    template<typename Dtype>
    void
    GpOccSurfaceMapping2D<Dtype>::RecordChangedKey(const geometry::QuadtreeKey &key) {
        ERL_DEBUG_ASSERT(m_tree_ != nullptr, "Quadtree is not initialized.");
        m_changed_keys_.insert(m_tree_->AdjustKeyToDepth(key, m_tree_->GetTreeDepth() - m_setting_->cluster_level));
    }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::ComputeGradient1(const Vector2 &xy_local, Vector2 &gradient, Dtype &occ_mean, Dtype &distance_var) {

        Dtype occ[4];
        occ_mean = 0.;
        Dtype distance_sum = 0.;
        Dtype distance_square_mean = 0.;
        gradient.setZero();

        for (int j = 0; j < 4; ++j) {
            Dtype distance;
            Scalar angle;
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            Scalar distance_pred;
            if (Scalar var; !m_sensor_gp_->ComputeOcc(angle, distance, distance_pred, var, occ[j])) { return false; }
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

        const Dtype gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // uncertain point, drop it
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype>
    bool
    GpOccSurfaceMapping2D<Dtype>::ComputeGradient2(const Eigen::Ref<const Vector2> &xy_local, Vector2 &gradient, Dtype &occ_mean) {
        Dtype occ[4];
        occ_mean = 0.;
        gradient.setZero();

        const Dtype valid_range_min = m_setting_->sensor_gp->lidar_frame->valid_range_min;
        const Dtype valid_range_max = m_setting_->sensor_gp->lidar_frame->valid_range_max;
        for (int j = 0; j < 4; ++j) {
            Dtype distance;
            Scalar angle;
            Cartesian2Polar(xy_local + m_xy_perturb_.col(j), distance, angle[0]);
            if (Scalar distance_pred, var;                                                 //
                !m_sensor_gp_->ComputeOcc(angle, distance, distance_pred, var, occ[j]) ||  //
                distance_pred[0] < valid_range_min || distance_pred[0] > valid_range_max) {
                return false;
            }
            occ_mean += occ[j];
        }

        occ_mean *= 0.25;
        gradient << (occ[0] - occ[1]) / m_setting_->perturb_delta, (occ[2] - occ[3]) / m_setting_->perturb_delta;

        const Dtype gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // uncertain point, drop it
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype>
    void
    GpOccSurfaceMapping2D<Dtype>::ComputeVariance(
        const Eigen::Ref<const Vector2> &xy_local,
        const Vector2 &grad_local,
        const Dtype &distance,
        const Dtype &distance_var,
        const Dtype &occ_mean_abs,
        const Dtype &occ_abs,
        const bool new_point,
        Dtype &var_position,
        Dtype &var_gradient) const {

        const Dtype min_distance_var = m_setting_->compute_variance.min_distance_var;
        const Dtype max_distance_var = m_setting_->compute_variance.max_distance_var;
        const Dtype min_gradient_var = m_setting_->compute_variance.min_gradient_var;
        const Dtype max_gradient_var = m_setting_->compute_variance.max_gradient_var;

        const Dtype var_distance = common::ClipRange(distance * distance, min_distance_var, max_distance_var);
        const Dtype cos_view_angle = -xy_local.dot(grad_local) / xy_local.norm();
        const Dtype cos2_view_angle = std::max(cos_view_angle * cos_view_angle, 1.e-2);  // avoid zero division
        const Dtype var_direction = (1. - cos2_view_angle) / cos2_view_angle;

        if (new_point) {
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction);
            var_gradient = common::ClipRange(occ_mean_abs, min_gradient_var, max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient = common::ClipRange(occ_mean_abs + distance_var, min_gradient_var, max_gradient_var) + 0.1 * var_direction;
        }
    }
}  // namespace erl::gp_sdf
