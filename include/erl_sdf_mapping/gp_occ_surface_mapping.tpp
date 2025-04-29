#pragma once
#include "erl_common/clip.hpp"
#include "erl_common/template_helper.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    YAML::Node
    GpOccSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;

        YAML::Node compute_variance;
        compute_variance["zero_gradient_position_var"] = setting.compute_variance.zero_gradient_position_var;
        compute_variance["zero_gradient_gradient_var"] = setting.compute_variance.zero_gradient_gradient_var;
        compute_variance["position_var_alpha"] = setting.compute_variance.position_var_alpha;
        compute_variance["min_distance_var"] = setting.compute_variance.min_distance_var;
        compute_variance["max_distance_var"] = setting.compute_variance.max_distance_var;
        compute_variance["min_gradient_var"] = setting.compute_variance.min_gradient_var;
        compute_variance["max_gradient_var"] = setting.compute_variance.max_gradient_var;
        node["compute_variance"] = compute_variance;

        YAML::Node update_map_points;
        update_map_points["max_adjust_tries"] = setting.update_map_points.max_adjust_tries;
        update_map_points["min_observable_occ"] = setting.update_map_points.min_observable_occ;
        update_map_points["min_position_var"] = setting.update_map_points.min_position_var;
        update_map_points["min_gradient_var"] = setting.update_map_points.min_gradient_var;
        update_map_points["max_surface_abs_occ"] = setting.update_map_points.max_surface_abs_occ;
        update_map_points["max_valid_gradient_var"] = setting.update_map_points.max_valid_gradient_var;
        update_map_points["max_bayes_position_var"] = setting.update_map_points.max_bayes_position_var;
        update_map_points["max_bayes_gradient_var"] = setting.update_map_points.max_bayes_gradient_var;
        node["update_map_points"] = update_map_points;

        node["sensor_gp"] = setting.sensor_gp;
        node["tree"] = setting.tree;
        node["scaling"] = setting.scaling;
        node["perturb_delta"] = setting.perturb_delta;
        node["zero_gradient_threshold"] = setting.zero_gradient_threshold;
        node["update_occupancy"] = setting.update_occupancy;
        node["cluster_depth"] = setting.cluster_depth;
        return node;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!node.IsMap()) { return false; }

        const YAML::Node compute_variance = node["compute_variance"];
        setting.compute_variance.zero_gradient_position_var = compute_variance["zero_gradient_position_var"].as<Dtype>();
        setting.compute_variance.zero_gradient_gradient_var = compute_variance["zero_gradient_gradient_var"].as<Dtype>();
        setting.compute_variance.position_var_alpha = compute_variance["position_var_alpha"].as<Dtype>();
        setting.compute_variance.min_distance_var = compute_variance["min_distance_var"].as<Dtype>();
        setting.compute_variance.max_distance_var = compute_variance["max_distance_var"].as<Dtype>();
        setting.compute_variance.min_gradient_var = compute_variance["min_gradient_var"].as<Dtype>();
        setting.compute_variance.max_gradient_var = compute_variance["max_gradient_var"].as<Dtype>();

        const YAML::Node update_map_points = node["update_map_points"];
        setting.update_map_points.max_adjust_tries = update_map_points["max_adjust_tries"].as<int>();
        setting.update_map_points.min_observable_occ = update_map_points["min_observable_occ"].as<Dtype>();
        setting.update_map_points.min_position_var = update_map_points["min_position_var"].as<Dtype>();
        setting.update_map_points.min_gradient_var = update_map_points["min_gradient_var"].as<Dtype>();
        setting.update_map_points.max_surface_abs_occ = update_map_points["max_surface_abs_occ"].as<Dtype>();
        setting.update_map_points.max_valid_gradient_var = update_map_points["max_valid_gradient_var"].as<Dtype>();
        setting.update_map_points.max_bayes_position_var = update_map_points["max_bayes_position_var"].as<Dtype>();
        setting.update_map_points.max_bayes_gradient_var = update_map_points["max_bayes_gradient_var"].as<Dtype>();

        setting.sensor_gp = node["sensor_gp"].as<decltype(setting.sensor_gp)>();
        setting.tree = node["tree"].as<decltype(setting.tree)>();
        setting.scaling = node["scaling"].as<Dtype>();
        setting.perturb_delta = node["perturb_delta"].as<Dtype>();
        setting.zero_gradient_threshold = node["zero_gradient_threshold"].as<Dtype>();
        setting.update_occupancy = node["update_occupancy"].as<bool>();
        setting.cluster_depth = node["cluster_depth"].as<int>();
        return true;
    }

    template<typename Dtype, int Dim>
    GpOccSurfaceMapping<Dtype, Dim>::GpOccSurfaceMapping(std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), true, "setting is nullptr")),
          m_sensor_gp_(std::make_shared<SensorGp>(m_setting_->sensor_gp)) {

        const Dtype d = m_setting_->perturb_delta;
        m_pos_perturb_.setZero();
        for (int i = 0; i < Dim; i++) {
            m_pos_perturb_(i, 2 * i) = d;
            m_pos_perturb_(i, 2 * i + 1) = -d;
        }
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::Setting>
    GpOccSurfaceMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::SensorGp>
    GpOccSurfaceMapping<Dtype, Dim>::GetSensorGp() const {
        return m_sensor_gp_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::Tree>
    GpOccSurfaceMapping<Dtype, Dim>::GetTree() const {
        return m_tree_;
    }

    template<typename Dtype, int Dim>
    const typename GpOccSurfaceMapping<Dtype, Dim>::SurfDataManager &
    GpOccSurfaceMapping<Dtype, Dim>::GetSurfaceDataManager() const {
        return m_surf_data_manager_;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &ranges) {

        m_changed_keys_.clear();
        if (const Dtype s = m_setting_->scaling; s != 1.0f) {
            if (!m_sensor_gp_->Train(rotation, translation.array() * s, ranges.array() * s)) { return false; }
        } else {
            if (!m_sensor_gp_->Train(rotation, translation, ranges)) { return false; }
        }

        {
            auto lock_guard = GetLockGuard();  // CRITICAL SECTION
            if (m_setting_->update_occupancy) { UpdateOccupancy(); }
            UpdateMapPoints();
            AddNewMeasurement();
        }

        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Ready() const {
        return m_tree_ != nullptr;
    }

    template<typename Dtype, int Dim>
    std::lock_guard<std::mutex>
    GpOccSurfaceMapping<Dtype, Dim>::GetLockGuard() {
        return std::lock_guard(m_mutex_);
    }

    template<typename Dtype, int Dim>
    Dtype
    GpOccSurfaceMapping<Dtype, Dim>::GetScaling() const {
        return m_setting_->scaling;
    }

    template<typename Dtype, int Dim>
    Dtype
    GpOccSurfaceMapping<Dtype, Dim>::GetClusterSize() const {
        return m_tree_->GetNodeSize(m_setting_->cluster_depth);
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::Position
    GpOccSurfaceMapping<Dtype, Dim>::GetClusterCenter(const Key &key) const {
        return m_tree_->KeyToCoord(key, m_setting_->cluster_depth);
    }

    template<typename Dtype, int Dim>
    const typename GpOccSurfaceMapping<Dtype, Dim>::KeySet &
    GpOccSurfaceMapping<Dtype, Dim>::GetChangedClusters() const {
        return m_changed_keys_;
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback) const {
        ERL_DEBUG_ASSERT(m_tree_ != nullptr, "Tree is not ready.");
        const uint32_t cluster_depth = m_setting_->cluster_depth;
        for (auto it = m_tree_->BeginTreeInAabb(aabb, cluster_depth), end = m_tree_->EndTreeInAabb(); it != end; ++it) {
            if (it->GetDepth() != cluster_depth) { continue; }
            callback(m_tree_->AdjustKeyToDepth(it.GetKey(), cluster_depth));
        }
    }

    template<typename Dtype, int Dim>
    const std::vector<typename GpOccSurfaceMapping<Dtype, Dim>::SurfData> &
    GpOccSurfaceMapping<Dtype, Dim>::GetSurfaceDataBuffer() const {
        return m_surf_data_manager_.GetEntries();
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::CollectSurfaceDataInAabb(const Aabb &aabb, std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const {
        ERL_DEBUG_ASSERT(m_tree_ != nullptr, "Tree is not ready.");
        surface_data_indices.clear();
        for (auto it = m_tree_->BeginLeafInAabb(aabb), end = m_tree_->EndLeafInAabb(); it != end; ++it) {
            if (!it->HasSurfaceData()) { continue; }
            const auto &surface_data = m_surf_data_manager_[it->surface_data_index];
            surface_data_indices.emplace_back((aabb.center - surface_data.position).norm(), it->surface_data_index);
        }
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::Aabb
    GpOccSurfaceMapping<Dtype, Dim>::GetMapBoundary() const {
        ERL_DEBUG_ASSERT(m_tree_ != nullptr, "Tree is not ready.");
        Position min, max;
        m_tree_->GetMetricMinMax(min, max);
        return Aabb(min, max);
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::IsInFreeSpace(const Positions &positions, VectorX &in_free_space) const {
        if (m_tree_ == nullptr) { return false; }  // tree is not ready
        const long num_positions = positions.cols();
        if (in_free_space.size() < num_positions) { in_free_space.resize(num_positions); }
        const Dtype s = m_setting_->scaling;

#pragma omp parallel for default(none) shared(positions, in_free_space, num_positions, s)
        for (long i = 0; i < num_positions; ++i) {
            const auto node = m_tree_->Search(positions.col(i) * s);
            if (node == nullptr || m_tree_->IsNodeOccupied(node)) {
                in_free_space[i] = -1.0f;
            } else {
                in_free_space[i] = 1.0f;
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::operator==(const AbstractSurfaceMapping &other) const {
        const auto *other_ptr = dynamic_cast<const GpOccSurfaceMapping *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) { return false; }
        if (m_sensor_gp_ == nullptr && other_ptr->m_sensor_gp_ != nullptr) { return false; }
        if (m_sensor_gp_ != nullptr && (other_ptr->m_sensor_gp_ == nullptr || *m_sensor_gp_ != *other_ptr->m_sensor_gp_)) { return false; }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr && (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) { return false; }
        if (m_pos_perturb_ != other_ptr->m_pos_perturb_) { return false; }
        if (m_changed_keys_ != other_ptr->m_changed_keys_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Write(std::ostream &s) const {
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
        s << "tree " << (m_tree_ != nullptr) << std::endl;
        if (m_tree_ != nullptr) {
            if (!m_tree_->Write(s)) {
                ERL_WARN("Failed to write the tree.");
                return false;
            }
        }
        s << "pos_perturb" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, m_pos_perturb_)) {
            ERL_WARN("Failed to write xyz_perturb.");
            return false;
        }
        s << "changed_keys " << m_changed_keys_.size() << std::endl;
        for (const Key &key: m_changed_keys_) {
            for (int i = 0; i < Dim; ++i) { s.write(reinterpret_cast<const char *>(&key[i]), sizeof(key[i])); }
        }
        s << kFileFooter << std::endl;
        return s.good();
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Read(std::istream &s) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader);
            return false;
        }

        auto skip_line = [&s] {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "sensor_gp",
            "tree",
            "pos_perturb",
            "changed_keys",
            kFileFooter,
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
                    m_sensor_gp_ = std::make_shared<SensorGp>(m_setting_->sensor_gp);
                    if (!m_sensor_gp_->Read(s)) {
                        ERL_WARN("Failed to read sensor_gp.");
                        return false;
                    }
                    break;
                }
                case 2: {  // tree
                    bool has_tree;
                    s >> has_tree;
                    skip_line();
                    if (has_tree) {
                        m_tree_ = std::make_shared<Tree>(m_setting_->tree);
                        if (!m_tree_->LoadData(s)) {
                            ERL_WARN("Failed to read tree.");
                            return false;
                        }
                    }
                    break;
                }
                case 3: {  // pos_perturb
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, m_pos_perturb_)) {
                        ERL_WARN("Failed to read pos_perturb.");
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
                        Key key;
                        for (int j = 0; j < Dim; ++j) { s.read(reinterpret_cast<char *>(&key[j]), sizeof(key[j])); }
                        m_changed_keys_.insert(key);
                    }
                    break;
                }
                case 5: {  // footer
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read {}. Truncated file?", kClassName);
        return false;  // should not reach here
    }

    template<typename Dtype, int Dim>
    std::pair<Dtype, Dtype>
    GpOccSurfaceMapping<Dtype, Dim>::Cartesian2Polar(const Dtype x, const Dtype y) {
        Dtype r = std::sqrt(x * x + y * y);
        Dtype angle = std::atan2(y, x);
        return {r, angle};
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateMapPoints() {
        ERL_BLOCK_TIMER();

        if (m_tree_ == nullptr || !m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Position &sensor_pos = sensor_frame->GetTranslationVector();
        const Dtype max_sensor_range = sensor_frame->GetMaxValidRange();
        const Aabb observed_area(sensor_pos, max_sensor_range);

        std::vector<std::tuple<Key, TreeNode *, bool, std::optional<Key>>> nodes_in_aabb;  // key, node, bad_update, new_key
        for (auto it = m_tree_->BeginLeafInAabb(observed_area), end = m_tree_->EndLeafInAabb(); it != end; ++it) {
            nodes_in_aabb.emplace_back(it.GetKey(), *it, false, std::nullopt);
        }

#pragma omp parallel for default(none) shared(nodes_in_aabb, max_sensor_range, sensor_pos, sensor_frame)
        for (auto &[node_key, node, bad_update, new_key]: nodes_in_aabb) {
            const uint32_t cluster_depth = m_setting_->cluster_depth;
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

            const Dtype cluster_half_size = m_tree_->GetNodeSize(cluster_depth) * 0.5f;
            const Dtype squared_dist_max = max_sensor_range * max_sensor_range + cluster_half_size * cluster_half_size * 3.0f;

            if (const Position cluster_position = m_tree_->KeyToCoord(node_key, cluster_depth);
                (cluster_position - sensor_pos).squaredNorm() > squared_dist_max) {
                continue;
            }

            if (!node->HasSurfaceData()) { continue; }
            auto &surface_data = m_surf_data_manager_[node->surface_data_index];

            const Position &pos_global_old = surface_data.position;
            Position pos_local_old = sensor_frame->PosWorldToFrame(pos_global_old);

            if (!sensor_frame->PointIsInFrame(pos_local_old)) { continue; }

            Dtype occ, distance_old;
            Scalar distance_pred, distance_pred_var;
            if (!ComputeOcc(pos_local_old, distance_old, distance_pred, distance_pred_var, occ)) { continue; }
            if (occ < min_observable_occ) { continue; }

            const Gradient &grad_global_old = surface_data.normal;
            Gradient grad_local_old = sensor_frame->DirWorldToFrame(grad_global_old);

            // compute a new position for the point
            Position pos_local_new = pos_local_old;
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
                Dtype occ_new;
                if (!ComputeOcc(pos_local_new, distance_new, distance_pred, distance_pred_var, occ_new)) { break; }  // fail to estimate occ
                occ_abs = std::fabs(occ_new);
                distance_new = distance_pred[0];
                if (occ_abs < max_surface_abs_occ) { break; }
                if (occ * occ_new < 0.) {
                    delta *= 0.5f;  // too big, make it smaller
                } else {
                    delta *= 1.1f;
                }
                occ = occ_new;
                ++num_adjust_tries;
            }

            // compute new gradient and uncertainty
            Dtype occ_mean, var_distance;
            Gradient grad_local_new;
            if (!ComputeGradient1(pos_local_new, grad_local_new, occ_mean, var_distance)) { continue; }
            Gradient grad_global_new = sensor_frame->DirFrameToWorld(grad_local_new);
            Dtype var_position_new, var_gradient_new;
            ComputeVariance(pos_local_new, grad_local_new, distance_new, var_distance, std::fabs(occ_mean), occ_abs, false, var_position_new, var_gradient_new);

            Position pos_global_new = sensor_frame->PosFrameToWorld(pos_local_new);
            if (const Dtype var_position_old = surface_data.var_position, var_gradient_old = surface_data.var_normal;
                var_gradient_old <= max_valid_gradient_var) {
                // do bayes Update only when the old result is not too bad, otherwise, just replace it
                const Dtype var_position_sum = var_position_new + var_position_old;
                const Dtype var_gradient_sum = var_gradient_new + var_gradient_old;

                // position Update
                pos_global_new = (pos_global_new * var_position_old + pos_global_old * var_position_new) / var_position_sum;
                // gradient Update
                UpdateGradient(var_gradient_new, var_gradient_sum, grad_global_old, grad_global_new);

                // variance Update
                const Dtype distance = (pos_global_new - pos_global_old).norm() * 0.5f;
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
            ERL_DEBUG_ASSERT(std::abs(surface_data.normal.norm() - 1.0f) < 1.e-5f, "surface_data->normal.norm() = {:.6f}", surface_data.normal.norm());
        }

        for (auto &[key, node, bad_update, new_key]: nodes_in_aabb) {
            if (bad_update) {                                                // too bad, the surface data should be removed
                m_surf_data_manager_.RemoveEntry(node->surface_data_index);  // this surface data is not used anymore
                node->ResetSurfaceDataIndex();                               // the node has no surface data now
            }
            if (!node->HasSurfaceData()) {  // surface data is removed
                RecordChangedKey(key);
                continue;
            }
            if (new_key.has_value()) {  // the node is moved to a new position
                RecordChangedKey(key);

                TreeNode *new_node = m_tree_->InsertNode(new_key.value());
                ERL_DEBUG_ASSERT(new_node != nullptr, "Failed to get the node");
                if (new_node->HasSurfaceData()) {                                // the new node is already occupied
                    m_surf_data_manager_.RemoveEntry(node->surface_data_index);  // this surface data is not used anymore
                    node->ResetSurfaceDataIndex();                               // the old node is empty now
                    continue;
                }
                new_node->surface_data_index = node->surface_data_index;  // move the surface data to the new node
                node->ResetSurfaceDataIndex();                            // the old node is empty now

                RecordChangedKey(new_key.value());
            }
        }
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 2, bool>
    GpOccSurfaceMapping<Dtype, Dim>::ComputeOcc(
        const Position &pos_local,
        Dtype &distance_local,
        Eigen::Ref<Scalar> distance_pred,
        Eigen::Ref<Scalar> distance_pred_var,
        Dtype &occ) const {

        Scalar angle_local;
        std::tie(distance_local, angle_local[0]) = Cartesian2Polar(pos_local.x(), pos_local.y());
        bool success = m_sensor_gp_->ComputeOcc(angle_local, distance_local, distance_pred, distance_pred_var, occ);
        return success;
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3, bool>
    GpOccSurfaceMapping<Dtype, Dim>::ComputeOcc(
        const Position &pos_local,
        Dtype &distance_local,
        Eigen::Ref<Scalar> distance_pred,
        Eigen::Ref<Scalar> distance_pred_var,
        Dtype &occ) const {

        distance_local = pos_local.norm();
        bool success = m_sensor_gp_->ComputeOcc(pos_local / distance_local, distance_local, distance_pred, distance_pred_var, occ);
        return success;
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 2>
    GpOccSurfaceMapping<Dtype, Dim>::UpdateGradient(const Dtype var_new, const Dtype var_sum, const Gradient &grad_old, Gradient &grad_new) {
        const Dtype &old_x = grad_old.x();
        const Dtype &old_y = grad_old.y();
        Dtype &new_x = grad_new.x();
        Dtype &new_y = grad_new.y();
        const Dtype angle_dist = std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_new / var_sum;
        const Dtype sin = std::sin(angle_dist);
        const Dtype cos = std::cos(angle_dist);
        // rotate grad_old by angle_dist
        new_x = cos * old_x - sin * old_y;
        new_y = sin * old_x + cos * old_y;
        ERL_DEBUG_ASSERT(std::abs(grad_new.norm() - 1.0f) < 1.e-5f, "grad_new.norm() = {:.6f}, diff = {}.", grad_new.norm(), std::abs(grad_new.norm() - 1.0f));
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3>
    GpOccSurfaceMapping<Dtype, Dim>::UpdateGradient(const Dtype var_new, const Dtype var_sum, const Gradient &grad_old, Gradient &grad_new) {
        Gradient rot_axis = grad_old.cross(grad_new);
        const Dtype axis_norm = rot_axis.norm();
        if (axis_norm < 1.e-6) {
            rot_axis = grad_old;  // parallel
        } else {
            rot_axis /= axis_norm;
        }
        const Dtype angle_dist = std::atan2(axis_norm, grad_old.dot(grad_new)) * var_new / var_sum;
        Eigen::AngleAxis<Dtype> rot(angle_dist, rot_axis);
        grad_new = rot * grad_old;
        ERL_DEBUG_ASSERT(std::abs(grad_new.norm() - 1.0f) < 1.e-5f, "grad_new.norm() = {:.6f}", grad_new.norm());
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateOccupancy() {
        ERL_BLOCK_TIMER();
        if (m_tree_ == nullptr) { m_tree_ = std::make_shared<Tree>(m_setting_->tree); }
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        // In AddNewMeasurement(), only rays classified as hit are used. So, we use the same here to avoid inconsistency.
        // Experiments show that this achieves higher fps and better results.
        const Eigen::Map<const Positions> map_points(sensor_frame->GetHitPointsWorld().data()->data(), Dim, sensor_frame->GetNumHitRays());
        constexpr bool parallel = false;
        constexpr bool lazy_eval = false;
        constexpr bool discrete = true;
        m_tree_->InsertPointCloud(map_points, sensor_frame->GetTranslationVector(), sensor_frame->GetMaxValidRange(), parallel, lazy_eval, discrete);
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::AddNewMeasurement() {
        ERL_BLOCK_TIMER();

        if (m_tree_ == nullptr) { m_tree_ = std::make_shared<Tree>(m_setting_->tree); }
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Position sensor_pos = sensor_frame->GetTranslationVector();
        const std::vector<Position> &hit_points_local = sensor_frame->GetHitPointsFrame();
        const std::vector<Position> &hit_points_global = sensor_frame->GetHitPointsWorld();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
        const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;

        // collect new measurements
        // if we iterate over the hit rays directly, some computations are unnecessary
        // [key, hit_idx, distance, node, invalid_flag, occ_mean, gradient_local]
        ERL_DEBUG("Collecting new measurements");
        std::vector<std::tuple<Key, long, Dtype, TreeNode *, bool, Dtype, Gradient>> new_measurements;
        new_measurements.reserve(num_hit_rays);
        KeySet new_measurement_keys;
        new_measurement_keys.reserve(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Position &hit_point_global = hit_points_global[i];
            Key key = m_tree_->CoordToKey(hit_point_global);
            if (!new_measurement_keys.insert(key).second) { continue; }  // the key is already in the set
            TreeNode *node = m_tree_->InsertNode(key);                   // insert the node
            if (node == nullptr) { continue; }                           // failed to insert the node
            if (node->HasSurfaceData()) { continue; }                    // the node is already occupied
            new_measurements.emplace_back(key, i, (hit_point_global - sensor_pos).norm(), node, false, 0.0f, Gradient::Zero());
        }

        ERL_DEBUG("Check validity of new measurements");
        const auto num_new_measurements = static_cast<long>(new_measurements.size());
#pragma omp parallel for default(none) shared(num_new_measurements, new_measurements, hit_points_local)
        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, range, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            invalid_flag = !ComputeGradient2(hit_points_local[hit_idx], gradient_local, occ_mean);
        }

        ERL_DEBUG("Add new measurements");
        for (long i = 0; i < num_new_measurements; ++i) {
            auto &[key, hit_idx, distance, node, invalid_flag, occ_mean, gradient_local] = new_measurements[i];
            if (invalid_flag) { continue; }            // invalid measurement
            if (node->HasSurfaceData()) { continue; }  // the node is already occupied by previous new measurement

            Dtype var_position, var_gradient;
            ComputeVariance(hit_points_local[hit_idx], gradient_local, distance, 0, std::fabs(occ_mean), 0, true, var_position, var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);
            node->surface_data_index =
                m_surf_data_manager_.AddEntry(hit_points_global[hit_idx], sensor_frame->DirFrameToWorld(gradient_local), var_position, var_gradient);
            RecordChangedKey(key);
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::RecordChangedKey(const Key &key) {
        ERL_DEBUG_ASSERT(m_tree_ != nullptr, "m_tree_ is nullptr.");
        m_changed_keys_.insert(m_tree_->AdjustKeyToDepth(key, m_setting_->cluster_depth));
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::ComputeGradient1(const Position &pos_local, Gradient &gradient, Dtype &occ_mean, Dtype &distance_var) {

        gradient.setZero();
        occ_mean = 0;
        distance_var = std::numeric_limits<Dtype>::infinity();

        Dtype occ[Dim << 1];
        Dtype distance_sum = 0.0f;
        Dtype distance_square_mean = 0.0f;
        const Dtype delta = m_setting_->perturb_delta;

        for (int i = 0; i < Dim; ++i) {

            for (int j: {i << 1, (i << 1) + 1}) {
                const Position pos_perturbed = pos_local + m_pos_perturb_.col(j);
                Scalar distance_pred, distance_pred_var;
                if (Dtype distance; !ComputeOcc(pos_perturbed, distance, distance_pred, distance_pred_var, occ[j])) { return false; }
                occ_mean += occ[j];
                distance_sum += distance_pred[0];
                distance_square_mean += distance_pred[0] * distance_pred[0];
            }
            gradient[i] = (occ[i << 1] - occ[(i << 1) + 1]) / delta;
        }

        occ_mean /= static_cast<Dtype>(Dim << 1);  // occ_mean = sum(occ) / size(occ)
        // 2*Dim samples in total, to calculate the unbiased variance
        // var(r) = sum((r_i - mean(r))^2) / (2*Dim-1) = (mean(r^2) - mean(r) * mean(r)) * (2*Dim) / (2*Dim-1)
        //        = (sum(r^2) - sum(r) * sum(r) / (2*Dim)) / (2*Dim-1)
        // to remove the numerical approximation's influence, let var(r) = var(r) / delta
        distance_var = (distance_square_mean - distance_sum * distance_sum / static_cast<Dtype>(Dim << 1)) / (static_cast<Dtype>((Dim << 1) - 1) * delta);
        const Dtype gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // zero gradient
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::ComputeGradient2(const Eigen::Ref<const Position> &pos_local, Gradient &gradient, Dtype &occ_mean) {
        occ_mean = 0;
        gradient.setZero();

        const Dtype valid_range_min = m_setting_->sensor_gp->sensor_frame->valid_range_min;
        const Dtype valid_range_max = m_setting_->sensor_gp->sensor_frame->valid_range_max;
        const Dtype delta = m_setting_->perturb_delta;

        Dtype occ[Dim << 1];

        for (int i = 0; i < Dim; ++i) {
            for (int j: {i << 1, (i << 1) + 1}) {
                const Position pos_perturbed = pos_local + m_pos_perturb_.col(j);
                Scalar distance_pred, distance_pred_var;
                if (Dtype distance; !ComputeOcc(pos_perturbed, distance, distance_pred, distance_pred_var, occ[j]) ||  //
                                    distance_pred[0] < valid_range_min || distance_pred[0] > valid_range_max) {
                    return false;
                }
                occ_mean += occ[j];
            }
            gradient[i] = (occ[i << 1] - occ[(i << 1) + 1]) / delta;
        }

        occ_mean /= static_cast<Dtype>(Dim << 1);
        const Dtype gradient_norm = gradient.norm();
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }  // zero gradient
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::ComputeVariance(
        const Eigen::Ref<const Position> &pos_local,
        const Gradient &grad_local,
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
        const Dtype cos_view_angle = -pos_local.dot(grad_local) / pos_local.norm();
        const Dtype cos2_view_angle = std::max(cos_view_angle * cos_view_angle, static_cast<Dtype>(1.e-2));  // avoid zero division
        const Dtype var_direction = (1. - cos2_view_angle) / cos2_view_angle;

        if (new_point) {
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction);
            var_gradient = common::ClipRange(occ_mean_abs, min_gradient_var, max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = m_setting_->compute_variance.position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient = common::ClipRange(occ_mean_abs + distance_var, min_gradient_var, max_gradient_var) + 0.1f * var_direction;
        }
    }

}  // namespace erl::sdf_mapping
