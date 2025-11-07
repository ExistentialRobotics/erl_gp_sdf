#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

#include "erl_common/angle_utils.hpp"
#include "erl_common/block_timer.hpp"
#include "erl_geometry/bayesian_hilbert_map_torch.hpp"

#include <utility>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BayesianHilbertSurfaceMapping(
        std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), true, "setting is nullptr")),
          m_tree_(std::make_shared<Tree>(m_setting_->tree)),
          m_ray_selector_(m_setting_->ray_selector) {

        InitConstants();
        GenerateHingedPoints();
        GenerateWeightAddress();

        // do it on the main thread only
#pragma omp parallel default(none)
#pragma omp critical
        {
            if (omp_get_thread_num() == 0) {
                m_ray_indices_.resize(omp_get_num_threads());
                for (auto &indices: m_ray_indices_) { indices.reserve(512); }
            }
        }
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Tree>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetTree() const {
        return m_tree_;
    }

    template<typename Dtype, int Dim>
    const absl::flat_hash_map<
        typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Key,
        std::shared_ptr<typename BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBhm>> &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetLocalBhms() const {
        return m_key_bhm_dict_;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &scan,
        const bool are_points,
        const bool are_local) {
        ERL_ASSERTM(are_points, "scan must be points, not range data.");

        if (scan.cols() == 0) {
            ERL_WARN("No points in the scan, nothing to update.");
            return false;
        }

        MatrixDX points;
        if (are_local) {
            points.resize(Dim, scan.cols());
#pragma omp parallel for default(none) shared(scan, points, rotation, translation) schedule(static)
            for (long i = 0; i < scan.cols(); ++i) {
                points.col(i) = rotation * scan.col(i) + translation;
            }
        } else {
            points = scan;
        }
        {
            ERL_BLOCK_TIMER_MSG("BHSM Update");
            return Update(rotation, translation, points, true /*parallel*/);
        }
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &sensor_rotation,
        const Eigen::Ref<const VectorD> &sensor_origin,
        const Eigen::Ref<const MatrixDX> &points,
        const bool parallel) {

        if (points.cols() == 0) {
            ERL_WARN("No points in the scan, nothing to update.");
            return false;
        }

        VectorD sensor_origin_s = sensor_origin;
        MatrixDX points_s = points;
        if (m_setting_->scaling != 1.0f) {
            sensor_origin_s.array() *= m_setting_->scaling;
            points_s.array() *= m_setting_->scaling;
        }

        // to update the occupancy tree first, the resolution of the tree should not be too high so
        // that we will not spend too much time on the tree update. the tree helps us find where to
        // place local Bayesian Hilbert maps.
        {
            ERL_BLOCK_TIMER_MSG("tree update");
            m_tree_->InsertPointCloud(
                points_s,
                sensor_origin_s,
                m_setting_->local_bhm->bhm->min_distance,
                m_setting_->local_bhm->bhm->max_distance,
                m_setting_->update_tree.with_count,
                m_setting_->update_tree.parallel,
                m_setting_->update_tree.lazy_eval,
                m_setting_->update_tree.discrete);
            if (m_setting_->update_tree.lazy_eval) {
                m_tree_->UpdateInnerOccupancy();
                m_tree_->Prune();
            }
        }

        // find the local Bayesian Hilbert maps to build or update
        m_changed_clusters_.clear();
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        const auto &end_point_maps = m_tree_->GetEndPointMaps();
        if (m_setting_->build_bhm_on_hit) {
            // any hit point will trigger building the corresponding local Bayesian Hilbert map
            for (const auto &[key, hit_indices]: end_point_maps) {
                if (hit_indices.empty()) { continue; }
                Key bhm_key = m_tree_->AdjustKeyToDepth(key, bhm_depth);
                m_changed_clusters_.insert(bhm_key);
                if (!m_setting_->update_map.include_neighbor_bhm) { continue; }
                // also add neighboring bhm keys if their local bhm exists
                typename Key::KeyType key_offset = 1 << (m_tree_->GetTreeDepth() - bhm_depth);
                for (int i = 0; i < Dim; ++i) {
                    Key neighbor_key = bhm_key;
                    neighbor_key[i] += key_offset;
                    auto it = m_key_bhm_dict_.find(neighbor_key);
                    if (it != m_key_bhm_dict_.end() && it->second->active) {
                        m_changed_clusters_.insert(neighbor_key);
                    }
                    neighbor_key[i] = bhm_key[i] - key_offset;
                    it = m_key_bhm_dict_.find(neighbor_key);
                    if (it != m_key_bhm_dict_.end() && it->second->active) {
                        m_changed_clusters_.insert(neighbor_key);
                    }
                }
            }
        } else {
            // only the occupied node will trigger building the corresponding local BHM
            for (const auto &[key, hit_indices]: end_point_maps) {
                if (const TreeNode *node = m_tree_->Search(key);
                    node != nullptr && m_tree_->IsNodeOccupied(node)) {
                    m_changed_clusters_.insert(m_tree_->AdjustKeyToDepth(key, bhm_depth));
                }
            }
        }

        // create bhm for new keys
        KeyVector &bhm_keys = m_clusters_to_update_;
        bhm_keys.clear();
        bhm_keys.insert(bhm_keys.end(), m_changed_clusters_.begin(), m_changed_clusters_.end());
        for (const Key &key: bhm_keys) { CreateBhm(key); }

        // some local BHMs may not use up all rays in the last update, we can use them to update
        // more local BHMs.
        const typename Setting::UpdateMap &update_map_setting = m_setting_->update_map;
        auto max_num_bhm = static_cast<std::size_t>(update_map_setting.max_num_bhm);
        max_num_bhm = std::min(max_num_bhm, m_key_bhm_positions_.size());
        std::size_t num_hit_bhms = bhm_keys.size();
        std::size_t cnt = 0;
        while (bhm_keys.size() < max_num_bhm && cnt < m_key_bhm_positions_.size()) {
            ++cnt;
            std::size_t index = ++m_local_bhm_head_ % m_key_bhm_positions_.size();
            m_local_bhm_head_ = index;
            const Key &key = m_key_bhm_positions_[index].first;
            const auto local_bhm = m_key_bhm_dict_.at(key);
            if (!local_bhm->active) { continue; }
            if (!local_bhm->HasRaysUnused()) { continue; }
            if (m_changed_clusters_.insert(key).second) { bhm_keys.push_back(key); }
        }

        // update the ray selector with the new sensor pose and points
        m_ray_selector_.UpdateRays(sensor_origin_s, sensor_rotation, points_s);

        // update the local Bayesian Hilbert maps
        (void) parallel;
        m_updated_flags_.resize(bhm_keys.size(), 0);
        ERL_INFO("{} local bhm(s) to update", bhm_keys.size());

        {
            ERL_BLOCK_TIMER_MSG("update local bhm");
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(bhm_keys, points_s, sensor_origin_s, sensor_rotation, num_hit_bhms)
            for (std::size_t i = 0; i < bhm_keys.size(); ++i) {
                auto &bhm_key = bhm_keys[i];
                std::vector<long> &ray_indices = m_ray_indices_[omp_get_thread_num()];
                LocalBhm &local_bhm = *m_key_bhm_dict_.at(bhm_key);
                if (i < num_hit_bhms) {
                    m_ray_selector_.SelectRays(
                        sensor_origin_s,
                        sensor_rotation,
                        local_bhm.bhm.GetMapBoundary().center,
                        m_ray_search_radius_,
                        ray_indices);
                    m_updated_flags_[i] = local_bhm.Update(
                        sensor_origin_s,
                        points_s,
                        ray_indices,
                        m_setting_->update_map.method == 2);
                } else {
                    m_updated_flags_[i] = local_bhm.Update(
                        sensor_origin_s,
                        MatrixDX(),  // no points
                        ray_indices,
                        m_setting_->update_map.method == 2);
                }
            }
        }

        // turn off local Bayesian Hilbert maps whose node in the tree is not occupied
        for (auto &[key, local_bhm]: m_key_bhm_dict_) {
            if (local_bhm->active) {
                const TreeNode *node = m_tree_->Search(key, bhm_depth);
                if (node == nullptr || !m_tree_->IsNodeOccupied(node)) {
                    local_bhm->active = false;
                    // mark the cluster as changed because its surface data will be removed.
                    m_changed_clusters_.insert(key);
                }
            }
            if (!local_bhm->active) {
                for (const auto &[grid_index, surf_index]: local_bhm->surface_indices) {
                    m_surf_data_manager_.RemoveEntry(surf_index);
                }
                local_bhm->surface_indices.clear();
            }
        }

        if (m_setting_->weight_sync) {
            // compute keys of BHMs that need to sync weights
            m_bhm_to_sync_vector_.clear();
            m_bhm_to_sync_.clear();
            m_bhm_to_sync_.insert(bhm_keys.begin(), bhm_keys.end());
            for (const Key &key: bhm_keys) {
                if (!m_key_bhm_dict_.at(key)->active) { continue; }
                m_bhm_to_sync_vector_.push_back(key);
                Key neighbor_key;
                for (const auto &offset: m_key_offsets_) {
                    for (int dim = 0; dim < Dim; ++dim) {
                        neighbor_key[dim] = key[dim] + offset[dim];
                    }
                    auto it = m_key_bhm_dict_.find(neighbor_key);
                    if (it == m_key_bhm_dict_.end() || !it->second->active) { continue; }
                    if (m_bhm_to_sync_.insert(neighbor_key).second) {
                        m_bhm_to_sync_vector_.push_back(neighbor_key);
                    }
                }
            }

            // sync weights
#pragma omp parallel for default(none) schedule(static)
            for (const Key &key: m_bhm_to_sync_vector_) { SyncBhmWeights(key); }
        }

        // check if any local bhm is updated.
        // remove active BHMs that are not updated from m_changed_clusters_.
        bool any_update = false;
        for (std::size_t i = 0; i < m_updated_flags_.size(); ++i) {
            if (m_updated_flags_[i] > 0) {
                any_update = true;
            } else if (m_key_bhm_dict_.at(bhm_keys[i])->active) {  // no update, but still active
                m_changed_clusters_.erase(bhm_keys[i]);            // remove from changed clusters
            }
        }

        if (any_update) {
            ERL_BLOCK_TIMER_MSG("bhm update map points");
            UpdateMapPoints(sensor_origin_s, points_s);
        }
        return any_update;
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::SurfDataManager::Iterator
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BeginSurfaceData() {
        return this->m_surf_data_manager_.begin();
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::SurfDataManager::Iterator
    BayesianHilbertSurfaceMapping<Dtype, Dim>::EndSurfaceData() {
        return this->m_surf_data_manager_.end();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Predict(
        const Eigen::Ref<const MatrixDX> &points,
        const bool logodd,
        const bool compute_free_space,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &prob_occupied,
        Eigen::VectorXb &in_free_space,
        MatrixDX &gradients) const {

        MatrixDX points_s = points;
        if (m_setting_->scaling != 1.0f) { points_s.array() *= m_setting_->scaling; }

        const long num_points = points_s.cols();
        Dtype init_prob_occupied = m_setting_->unknown_log_odds;
        const bool init_free_space = init_prob_occupied <= m_setting_->local_bhm->surface_log_odds;
        if (!logodd) { init_prob_occupied = geometry::logodd::Probability(init_prob_occupied); }

        if (prob_occupied.size() < num_points) { prob_occupied.resize(num_points); }
        if (compute_free_space) {
            if (in_free_space.size() < num_points) { in_free_space.resize(num_points); }
            in_free_space.fill(init_free_space);
        }
        if (compute_gradient) {
            if (gradients.cols() < num_points) { gradients.resize(Dim, num_points); }
            gradients.setZero();
        }

        prob_occupied.fill(init_prob_occupied);  // initialize to unknown
        BuildBhmKdtree();

        const int batch_size = m_setting_->test_batch_size;
        if (batch_size > num_points) {  // no need to run in parallel here
            PredictThread(
                points_s.data(),
                0,
                num_points,
                logodd,
                compute_free_space,
                compute_gradient,
                gradient_with_sigmoid,
                parallel,  // let the thread decide
                prob_occupied.data(),
                in_free_space.data(),
                gradients.data());
            return;
        }

        const uint32_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (long i = 0; i < num_points; i += batch_size) {
            long start = i;
            long end = std::min(i + batch_size, num_points);

            bool no_free = threads.size() >= num_threads;
            while (no_free) {
                for (auto it = threads.begin(); it != threads.end(); ++it) {
                    if (!it->joinable()) { continue; }
                    it->join();
                    threads.erase(it);
                    no_free = false;
                    break;
                }
                if (no_free) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            threads.emplace_back(
                &BayesianHilbertSurfaceMapping::PredictThread,
                this,
                points_s.data(),
                start,
                end,
                logodd,
                compute_free_space,
                compute_gradient,
                gradient_with_sigmoid,
                false /*parallel*/,  // no need to run in parallel within the thread
                prob_occupied.data(),
                in_free_space.data(),
                gradients.data());
        }

        for (auto &thread: threads) { thread.join(); }
        threads.clear();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictGradient(
        const Eigen::Ref<const MatrixDX> &points,
        const bool with_sigmoid,
        const bool parallel,
        MatrixDX &gradient) const {

        // we can only predict the gradient when bhm is available for the point

        if (gradient.cols() < points.cols()) { gradient.resize(Dim, points.cols()); }
        gradient.fill(0.0f);
        BuildBhmKdtree();

        absl::flat_hash_map<Key, std::vector<long>> key_to_point_indices;
        key_to_point_indices.reserve(points.size());
        std::vector<Key> bhm_keys_set;
        bhm_keys_set.reserve(points.size());
        long bhm_index = -1;
        Dtype bhm_distance = 0;
        for (long i = 0; i < points.size(); ++i) {
            m_bhm_kdtree_->Nearest(points.col(i), bhm_index, bhm_distance);  // find the nearest bhm
            bhm_distance = std::sqrt(bhm_distance);                          // distance is squared
            if (bhm_distance > m_half_bhm_test_size_) { continue; }          // too far from the bhm

            const Key bhm_key = m_key_bhm_positions_[bhm_index].first;
            auto [it, inserted] = key_to_point_indices.insert({bhm_key, std::vector<long>()});
            // if the key is new, add it to the set
            if (inserted) { bhm_keys_set.push_back(bhm_key); }
            it->second.push_back(i);
        }

        // dynamic scheduling because some local Bayesian Hilbert maps may have more points.
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(bhm_keys_set, key_to_point_indices, points, with_sigmoid, parallel, gradient)
        for (const Key &bhm_key: bhm_keys_set) {
            const auto &indices = key_to_point_indices[bhm_key];

            // copy the points of this key to a new matrix
            MatrixDX points_of_key(Dim, static_cast<long>(indices.size()));
            for (long i = 0; i < points_of_key.cols(); ++i) {
                points_of_key.col(i) = points.col(indices[i]);
            }

            // predict
            MatrixDX gradients_of_key;
            m_key_bhm_dict_.at(bhm_key)
                ->PredictGradient(points_of_key, with_sigmoid, !parallel, gradients_of_key);

            // copy the results back to the original matrix
            for (long i = 0; i < points_of_key.cols(); ++i) {
                gradient.col(indices[i]) = gradients_of_key.col(i);
            }
        }
    }

    template<typename Dtype, int Dim>
    Dtype
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetScaling() const {
        return m_setting_->scaling;
    }

    template<typename Dtype, int Dim>
    Dtype
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetClusterSize() const {
        return m_tree_->GetNodeSize(m_setting_->bhm_depth);
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::VectorD
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetClusterCenter(const Key &key) const {
        return m_tree_->KeyToCoord(key, m_setting_->bhm_depth);
    }

    template<typename Dtype, int Dim>
    const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::KeySet &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetChangedClusters() const {
        return m_changed_clusters_;
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::KeySet
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetAllClusters() const {
        KeySet cluster_keys;
        cluster_keys.reserve(m_key_bhm_dict_.size());
        for (const auto &[key, local_bhm]: m_key_bhm_dict_) {
            if (!local_bhm->active) { continue; }
            cluster_keys.insert(key);
        }
        return cluster_keys;
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Key
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetClusterKey(
        const Eigen::Ref<const VectorD> &pos) const {
        VectorD pos_s = pos.array() * m_setting_->scaling;
        return m_tree_->CoordToKey(pos_s, m_setting_->bhm_depth);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::IterateClustersInAabb(
        const Aabb &aabb,
        std::function<void(const Key &)> callback) const {
        const uint32_t cluster_depth = m_setting_->bhm_depth;
        for (auto it = m_tree_->BeginTreeInAabb(aabb, cluster_depth),
                  end = m_tree_->EndTreeInAabb();
             it != end;
             ++it) {
            if (it->GetDepth() != cluster_depth) { continue; }
            callback(m_tree_->AdjustKeyToDepth(it.GetKey(), cluster_depth));
        }
    }

    template<typename Dtype, int Dim>
    const std::vector<typename BayesianHilbertSurfaceMapping<Dtype, Dim>::SurfData> &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetSurfaceDataBuffer() const {
        return m_surf_data_manager_.GetBuffer();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::CollectSurfaceDataInAabb(
        const Aabb &aabb,
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const {
        surface_data_indices.clear();
        for (auto it = m_tree_->BeginTreeInAabb(aabb, m_setting_->bhm_depth),
                  end = m_tree_->EndTreeInAabb();
             it != end;
             ++it) {
            if (it.GetDepth() != m_setting_->bhm_depth) { continue; }
            Key key = it.GetIndexKey();
            auto bhm_it = m_key_bhm_dict_.find(key);
            if (bhm_it == m_key_bhm_dict_.end()) { continue; }
            const LocalBhm &local_bhm = *bhm_it->second;
            if (!local_bhm.active) { continue; }  // skip inactive local BHM
            for (const auto &[grid_idx, surf_idx]: local_bhm.surface_indices) {
                ERL_DEBUG_ASSERT(static_cast<long>(surf_idx) != -1l, "surf_idx should not be -1");
                const SurfData &surf_data = m_surf_data_manager_[surf_idx];
                if (surf_data.var_position >= 1.0e6f) { continue; }  // skip bad surface data
                surface_data_indices.emplace_back(
                    (aabb.center - surf_data.position).norm(),
                    surf_idx);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetMesh(
        std::vector<VectorD> &vertices,
        std::vector<Face> &faces) const {
        if (m_setting_->update_map.method != 2) {
            ERL_WARN(
                "GetMesh is only supported when update_map.method == 2 (i.e., using marching "
                "squares/cubes). Current method: {}",
                m_setting_->update_map.method);
            return;
        }

        const_cast<BayesianHilbertSurfaceMapping *>(this)->RunMarchingQueue(true);

        vertices.clear();
        faces.clear();
        absl::flat_hash_map<GridIndex, int> edge_to_vertex_map;
        edge_to_vertex_map.reserve(64);
        for (const auto &[key, bhm_ptr]: m_key_bhm_dict_) {
            if (bhm_ptr == nullptr) { continue; }
            const LocalBhm &local_bhm = *bhm_ptr;
            if (!local_bhm.active || local_bhm.surface_indices.empty()) { continue; }
            edge_to_vertex_map.clear();  // edge_idx is local, vertex_idx is global
            for (const auto &[edge_idx, surf_idx]: local_bhm.surface_indices) {
                VectorD position = GetUniqueVertex(key, edge_idx, surf_idx);
                // scale back
                for (int dim = 0; dim < Dim; ++dim) { position[dim] /= m_setting_->scaling; }
                edge_to_vertex_map[edge_idx] = static_cast<int>(vertices.size());
                vertices.push_back(position);
            }
            for (const auto &[voxel_idx, voxel]: local_bhm.surf_voxels) {
                if (!voxel.good) { continue; }
                for (Face face: voxel.faces) {
                    for (int dim = 0; dim < Dim; ++dim) {
                        face[dim] = edge_to_vertex_map.at(voxel.edges[face[dim]]);
                    }
                    faces.push_back(face);
                }
            }
        }
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Aabb
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetMapBoundary() const {
        VectorD min, max;
        m_tree_->GetMetricMinMax(min, max);
        return Aabb(min, max);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::IsInFreeSpace(
        const MatrixDX &positions,
        Eigen::VectorXb &in_free_space) const {
        if (positions.cols() == 0) {
            ERL_WARN("No points in the positions, nothing to check.");
            return false;
        }
        VectorX log_odds(positions.cols());
        in_free_space.resize(positions.cols());
        MatrixDX gradients;
        Predict(
            positions,
            true /*logodd*/,
            true /*compute_free_space*/,
            false /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            true /*parallel*/,
            log_odds,
            in_free_space,
            gradients);
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::operator==(const Super &other) const {
        const auto *other_ptr = dynamic_cast<const BayesianHilbertSurfaceMapping *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) {
            return false;
        }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr &&
            (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) {
            return false;
        }
        if (m_hinged_points_ != other_ptr->m_hinged_points_) { return false; }
        if (m_key_bhm_positions_ != other_ptr->m_key_bhm_positions_) { return false; }
        // because m_key_bhm_dict_ maps a key to a shared pointer,
        // we cannot use the operator!= directly.
        if (m_key_bhm_dict_.size() != other_ptr->m_key_bhm_dict_.size()) { return false; }
        for (const auto &[key, bhm_ptr]: m_key_bhm_dict_) {
            auto it = other_ptr->m_key_bhm_dict_.find(key);
            if (it == other_ptr->m_key_bhm_dict_.end()) { return false; }
            if (bhm_ptr == nullptr && it->second != nullptr) { return false; }
            if (bhm_ptr != nullptr && (it->second == nullptr || *bhm_ptr != *(it->second))) {
                return false;
            }
        }
        if (m_surf_data_manager_ != other_ptr->m_surf_data_manager_) { return false; }
        if (m_changed_clusters_ != other_ptr->m_changed_clusters_) { return false; }
        if (m_clusters_to_update_ != other_ptr->m_clusters_to_update_) { return false; }
        if (m_updated_flags_ != other_ptr->m_updated_flags_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Write(std::ostream &stream) const {
        using namespace common;
        using namespace common::serialization;
        static const TokenWriteFunctionPairs<BayesianHilbertSurfaceMapping> pairs = {
            {
                "setting",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    return self->m_setting_->Write(s) && s.good();
                },
            },
            {
                "tree",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    return self->m_tree_->Write(s) && s.good();
                },
            },
            // m_bhm_kdtree_
            // m_bhm_kdtree_needs_update_
            {
                "hinged_points",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    return SaveEigenMatrixToBinaryStream(s, self->m_hinged_points_) && s.good();
                },
            },
            // key_bhm_positions_ is constructed when reading key_bhm_dict
            {
                "key_bhm_dict",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_key_bhm_dict_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, center]: self->m_key_bhm_positions_) {  // keep order
                        auto bhm = self->m_key_bhm_dict_.at(key);
                        s.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        const bool has_bhm = bhm != nullptr;
                        s.write(reinterpret_cast<const char *>(&has_bhm), sizeof(bool));
                        if (has_bhm && !bhm->Write(s)) { return false; }
                    }
                    return s.good();
                },
            },
            {
                "surf_data_manager",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    return self->m_surf_data_manager_.Write(s) && s.good();
                },
            },
            {
                "changed_clusters",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_changed_clusters_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const Key &key: self->m_changed_clusters_) {
                        s.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                    }
                    return s.good();
                },
            },
            {
                "clusters_to_update",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_clusters_to_update_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    s.write(
                        reinterpret_cast<const char *>(self->m_clusters_to_update_.data()),
                        sizeof(Key) * n);
                    return s.good();
                },
            },
            {
                "updated_flags",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_updated_flags_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    s.write(
                        reinterpret_cast<const char *>(self->m_updated_flags_.data()),
                        sizeof(int) * n);
                    return s.good();
                },
            },
            {
                "marching_queue_keys",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_marching_queue_keys_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, handle]: self->m_marching_queue_keys_) {
                        s.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        s.write(reinterpret_cast<const char *>(&(*handle).priority), sizeof(long));
                    }
                    return s.good();
                },
            },
            // m_marching_queue_ can be reconstructed from m_marching_queue_keys_.
        };
        return WriteTokens(stream, this, pairs);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Read(std::istream &stream) {
        using namespace common;
        using namespace common::serialization;
        m_bhm_kdtree_needs_update_ = true;
        static const TokenReadFunctionPairs<BayesianHilbertSurfaceMapping> pairs = {
            {
                "setting",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    if (!self->m_setting_->Read(s)) { return false; }
                    self->m_tree_ = std::make_unique<Tree>(self->m_setting_->tree);
                    self->m_ray_selector_ = RaySelector(self->m_setting_->ray_selector);
                    self->InitConstants();
                    self->GenerateHingedPoints();
                    self->GenerateWeightAddress();
                    return s.good();
                },
            },
            {
                "tree",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    return self->m_tree_->Read(s) && s.good();
                },
            },

            {
                "hinged_points",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    return LoadEigenMatrixFromBinaryStream(s, self->m_hinged_points_) && s.good();
                },
            },
            // key_bhm_positions is constructed when reading key_bhm_dict
            {
                "key_bhm_dict",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_key_bhm_dict_.clear();
                    self->m_key_bhm_dict_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        auto [it, inserted] = self->CreateBhm(key);
                        if (!inserted) {
                            ERL_WARN("Duplicate key {} in key_bhm_dict", std::string(key));
                            return false;
                        }
                        bool has_bhm;
                        s.read(reinterpret_cast<char *>(&has_bhm), sizeof(bool));
                        if (!has_bhm) {
                            it->second = nullptr;
                            continue;
                        }
                        if (!it->second->Read(s)) {
                            ERL_WARN("Failed to read bhm for key {}", std::string(key));
                            return false;
                        }
                    }
                    return s.good();
                },
            },
            {
                "surf_data_manager",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    return self->m_surf_data_manager_.Read(s) && s.good();
                },
            },
            {
                "changed_clusters",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_changed_clusters_.clear();
                    self->m_changed_clusters_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        const auto [it, inserted] = self->m_changed_clusters_.insert(key);
                        if (!inserted) {
                            ERL_WARN("Duplicate key {} in changed_clusters", std::string(key));
                            return false;
                        }
                    }
                    return s.good();
                },
            },
            {
                "clusters_to_update",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_clusters_to_update_.clear();
                    self->m_clusters_to_update_.resize(n);
                    s.read(
                        reinterpret_cast<char *>(self->m_clusters_to_update_.data()),
                        sizeof(Key) * n);
                    return s.good();
                },
            },
            {
                "updated_flags",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_updated_flags_.clear();
                    self->m_updated_flags_.resize(n);
                    s.read(
                        reinterpret_cast<char *>(self->m_updated_flags_.data()),
                        static_cast<std::streamsize>(sizeof(int) * n));
                    return s.good();
                },
            },
            // m_ray_selector_ is set when m_setting_ is loaded
            // m_ray_indices_, m_hit_points are temporary
            {
                "marching_queue_keys",
                [](BayesianHilbertSurfaceMapping *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_marching_queue_keys_.clear();
                    self->m_marching_queue_keys_.reserve(n);
                    self->m_marching_queue_.clear();
                    self->m_marching_queue_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        long priority;
                        s.read(reinterpret_cast<char *>(&priority), sizeof(long));
                        auto [it, inserted] = self->m_marching_queue_keys_.try_emplace(
                            key,
                            self->m_marching_queue_.push({priority, key}));
                        if (!inserted) {
                            ERL_WARN("Duplicate BHM key: {}.", static_cast<std::string>(key));
                            return false;
                        }
                    }
                    return s.good();
                },
            },
            // m_marching_queue_ is recovered when reading marching_queue_keys
            // m_core_indices_, m_managed_share_indices_, m_unmanaged_share_indices,
            // m_key_offsets_, m_hinged_point_order_, m_hinged_point_order_reverse_ are set when
            // m_setting_ is loaded.
            // m_bhm_to_sync_ and m_bhm_to_sync_vector_ are temporary.
            // m_block_size_, m_sync_method_ and other frequently used intermediate values are set
            // when m_setting_ is loaded.
        };
        return ReadTokens(stream, this, pairs);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::ResetMarchingResults() {
        m_marching_queue_.clear();
        m_marching_queue_keys_.clear();

        for (auto &[key, bhm_ptr]: m_key_bhm_dict_) {
            if (bhm_ptr == nullptr) { continue; }
            if (!bhm_ptr->active) { continue; }  // skip inactive local BHM
            const long time_stamp =
                std::chrono::high_resolution_clock::now().time_since_epoch().count();
            m_marching_queue_keys_.insert({
                key,
                m_marching_queue_.push({time_stamp, key}),
            });
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::InitConstants() {
        const int hinged_grid_size = m_setting_->hinged_grid_size;
        m_block_size_ = hinged_grid_size - 2 * m_setting_->bhm_overlap_sync;
        if (m_setting_->sync_method == "copy") {
            m_sync_method_ = 0;
        } else if (m_setting_->sync_method == "mean") {
            m_sync_method_ = 1;
        } else if (m_setting_->sync_method == "bayesian") {
            m_sync_method_ = 2;
        } else {
            ERL_WARN("Unknown sync method '{}', defaulting to 'copy'", m_setting_->sync_method);
            m_sync_method_ = 0;
        }
        m_bhm_node_size_ = m_tree_->GetNodeSize(m_setting_->bhm_depth);
        m_half_bhm_node_size_ = m_bhm_node_size_ * 0.5f;
        if (m_setting_->weight_sync) {
            m_hinged_grid_res_ = m_bhm_node_size_ / static_cast<Dtype>(m_block_size_);
            m_half_bhm_size_ = static_cast<Dtype>(hinged_grid_size) * 0.5f * m_hinged_grid_res_;
        } else {
            m_half_bhm_size_ = m_half_bhm_node_size_ + m_setting_->bhm_overlap;
            m_hinged_grid_res_ = m_half_bhm_size_ * 2.0f / static_cast<Dtype>(hinged_grid_size);
        }
        const Dtype r = m_half_bhm_size_ * m_setting_->local_bhm->bhm->sampling_area_scale;
        m_ray_search_radius_ = std::sqrt(Dim * r * r);
        m_half_bhm_test_size_ = m_half_bhm_size_ + m_setting_->bhm_test_margin;
        const long surface_grid_size = m_setting_->local_bhm->surface_grid_size;
        m_surface_res_ = m_bhm_node_size_ / static_cast<Dtype>(surface_grid_size);
        m_surf_point_capacity_ = 1;
        for (int dim = 0; dim < Dim; ++dim) { m_surf_point_capacity_ *= surface_grid_size; }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GenerateHingedPoints() {
        const Eigen::Vector<int, Dim> grid_shape =
            Eigen::Vector<int, Dim>::Constant(m_setting_->hinged_grid_size);
        const Eigen::Vector<Dtype, Dim> grid_half_size =
            Eigen::Vector<Dtype, Dim>::Constant(m_half_bhm_size_);
        common::GridMapInfo<Dtype, Dim> grid_map_info(grid_shape, -grid_half_size, grid_half_size);
        m_hinged_points_ = grid_map_info.GenerateMeterCoordinates(true);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GenerateWeightAddress() {
        if (!m_setting_->weight_sync) { return; }

        const int hinged_grid_size = m_setting_->hinged_grid_size;
        const int overlap_size = m_setting_->bhm_overlap_sync;
        const int shared_size = 2 * overlap_size;
        const int core_size = m_block_size_ - shared_size;  // not shared
        const int key_offset = 1 << (m_setting_->tree->tree_depth - m_setting_->bhm_depth);
        const int n_core = std::pow(core_size, Dim);
        const int n_managed = std::pow(m_block_size_, Dim) - n_core;
        const int n_unmanaged = std::pow(hinged_grid_size, Dim) - n_core - n_managed;
        const Dtype grid_min = -m_hinged_grid_res_ * static_cast<Dtype>(overlap_size);

        ERL_ASSERTM(core_size > 0, "Core size must be positive");

        // | overlap-size    | overlap-size  | core | overlap-size  | overlap-size    |
        // | unmanaged share | managed share | core | managed share | unmanaged share |

        m_core_indices_.clear();
        m_core_indices_.reserve(n_core);
        m_managed_share_indices_.clear();
        m_managed_share_indices_.reserve(n_managed);
        m_unmanaged_share_indices_.clear();
        m_unmanaged_share_indices_.reserve(n_unmanaged);

        GridShape grid_shape;
        for (int i = 0; i < Dim; ++i) { grid_shape[i] = hinged_grid_size; }
        GridShape grid_stride = common::ComputeCStrides<long, Dim>(grid_shape, 1);
        const int n_weights = grid_shape.prod();
        enum class WeightType { Core = 0, ManagedShare = 1, UnmanagedShare = 2 };
        absl::flat_hash_set<Eigen::Vector<int, Dim>> unique_offsets;
        unique_offsets.reserve(std::pow(3, Dim) - 1);
        m_key_offsets_.clear();
        for (int i = 0; i < n_weights; ++i) {
            WeightAddr addr;
            auto grid_coord = common::IndexToCoordsWithStrides<long, Dim>(grid_stride, i, true);
            WeightType weight_type = WeightType::Core;
            for (int dim = 0; dim < Dim; ++dim) {
                if (grid_coord[dim] < overlap_size ||
                    grid_coord[dim] >= hinged_grid_size - overlap_size) {
                    weight_type = WeightType::UnmanagedShare;
                    break;
                }
                if (grid_coord[dim] < shared_size ||
                    grid_coord[dim] >= hinged_grid_size - shared_size) {
                    weight_type = WeightType::ManagedShare;
                }
            }
            for (int dim = 0; dim < Dim; ++dim) {
                if (grid_coord[dim] < overlap_size) {
                    addr[dim] = -key_offset;
                    Dtype p = common::GridToMeter(grid_coord[dim], grid_min, m_hinged_grid_res_);
                    grid_coord[dim] =
                        common::MeterToGrid(p, grid_min - m_bhm_node_size_, m_hinged_grid_res_);
                } else if (grid_coord[dim] >= hinged_grid_size - overlap_size) {
                    addr[dim] = key_offset;
                    Dtype p = common::GridToMeter(grid_coord[dim], grid_min, m_hinged_grid_res_);
                    grid_coord[dim] =
                        common::MeterToGrid(p, grid_min + m_bhm_node_size_, m_hinged_grid_res_);
                } else {
                    addr[dim] = 0;
                }
            }
            addr[Dim] = static_cast<int>(common::CoordsToIndex<long, Dim>(grid_stride, grid_coord));
            addr[Dim + 1] = i;
            switch (weight_type) {
                case WeightType::Core: {
                    m_core_indices_.push_back(addr);
                    break;
                }
                case WeightType::ManagedShare: {
                    m_managed_share_indices_.push_back(addr);
                    break;
                }
                case WeightType::UnmanagedShare: {
                    m_unmanaged_share_indices_.push_back(addr);
                    break;
                }
            }
            Eigen::Vector<int, Dim> key = addr.template head<Dim>();
            if (unique_offsets.insert(key).second) { m_key_offsets_.push_back(key); }
        }

        // rearrange hinged points so that points of different weight types are grouped together.
        // this is important to avoid false sharing when synchronizing weights in parallel.
        // if using float, hinged_grid_size=5, overlap_size=1,
        // for 2D, n_unmanaged=16, n_core=1, n_managed=8 (false sharing is resolved);
        // for 3D, n_unmanaged=98, n_core=1, n_managed=26 (false sharing still exists).
        // the size of cache line is 64 bytes (16 floats or 8 doubles).
        m_hinged_point_order_.resize(n_weights);
        m_hinged_point_order_reverse_.resize(n_weights);
        long idx = 0;
        for (const WeightAddr &addr: m_unmanaged_share_indices_) {
            m_hinged_point_order_[idx] = addr[Dim + 1];
            m_hinged_point_order_reverse_[addr[Dim + 1]] = idx;
            ++idx;
        }
        for (const WeightAddr &addr: m_core_indices_) {
            m_hinged_point_order_[idx] = addr[Dim + 1];
            m_hinged_point_order_reverse_[addr[Dim + 1]] = idx;
            ++idx;
        }
        for (const WeightAddr &addr: m_managed_share_indices_) {
            m_hinged_point_order_[idx] = addr[Dim + 1];
            m_hinged_point_order_reverse_[addr[Dim + 1]] = idx;
            ++idx;
        }
        // update src & dst indices to the sorted indices
        for (WeightAddr &addr: m_unmanaged_share_indices_) {
            addr[Dim] = m_hinged_point_order_reverse_[addr[Dim]];
            addr[Dim + 1] = m_hinged_point_order_reverse_[addr[Dim + 1]];
        }
        for (WeightAddr &addr: m_core_indices_) {
            addr[Dim] = m_hinged_point_order_reverse_[addr[Dim]];
            addr[Dim + 1] = m_hinged_point_order_reverse_[addr[Dim + 1]];
        }
        for (WeightAddr &addr: m_managed_share_indices_) {
            addr[Dim] = m_hinged_point_order_reverse_[addr[Dim]];
            addr[Dim + 1] = m_hinged_point_order_reverse_[addr[Dim + 1]];
        }
        // update m_hinged_points_
        MatrixDX ordered_hinged_points(Dim, m_hinged_points_.cols());
        for (long i = 0; i < m_hinged_points_.cols(); ++i) {
            ordered_hinged_points.col(i) = m_hinged_points_.col(m_hinged_point_order_[i]);
        }
        m_hinged_points_ = std::move(ordered_hinged_points);
    }

    template<typename Dtype, int Dim>
    std::pair<
        typename absl::flat_hash_map<
            typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Key,
            std::shared_ptr<typename BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBhm>>::
            iterator,
        bool>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::CreateBhm(const Key &key) {
        auto it0 = m_key_bhm_dict_.find(key);
        if (it0 != m_key_bhm_dict_.end()) { return {it0, false}; }  // already exist

        VectorD map_center = m_tree_->KeyToCoord(key, m_setting_->bhm_depth);
        MatrixDX hinged_points = m_hinged_points_.colwise() + map_center;
        m_key_bhm_positions_.emplace_back(key, map_center);
        m_bhm_kdtree_needs_update_ = true;  // need to update the kdtree after adding new bhm
        auto local_bhm = std::make_shared<LocalBhm>(
            m_setting_->local_bhm,
            hinged_points,
            Aabb(map_center, m_half_bhm_size_) /*map_boundary*/,
            typename Key::KeyHash()(key) /*seed*/,
            Aabb(map_center, m_half_bhm_node_size_) /*track_surface_boundary*/);
        return m_key_bhm_dict_.try_emplace(key, local_bhm);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::SyncBhmWeights(const Key &key) {
        std::shared_ptr<LocalBhm> local_bhm = m_key_bhm_dict_.at(key);
        if (local_bhm == nullptr) { return; }
        auto &bhm = local_bhm->bhm;
        auto mu = bhm.GetWeights();
        Key other_key;
        if (m_sync_method_ == 0) {  // copy
            for (const WeightAddr &addr: m_unmanaged_share_indices_) {
                for (int dim = 0; dim < Dim; ++dim) { other_key[dim] = key[dim] + addr[dim]; }
                auto it = m_key_bhm_dict_.find(other_key);
                if (it == m_key_bhm_dict_.end() || !it->second->active) { continue; }
                auto &other_bhm = it->second->bhm;
                mu[addr[Dim + 1]] = other_bhm.GetWeights()[addr[Dim]];
            }
            bhm.SetWeights(mu);
            return;
        }
        if (m_sync_method_ == 1) {  // mean
            for (const WeightAddr &addr: m_unmanaged_share_indices_) {
                for (int dim = 0; dim < Dim; ++dim) { other_key[dim] = key[dim] + addr[dim]; }
                auto it = m_key_bhm_dict_.find(other_key);
                if (it == m_key_bhm_dict_.end() || !it->second->active) { continue; }
                auto &other_bhm = it->second->bhm;
                mu[addr[Dim + 1]] = 0.5f * (mu[addr[Dim + 1]] + other_bhm.GetWeights()[addr[Dim]]);
            }
            bhm.SetWeights(mu);
            return;
        }
        if (m_sync_method_ == 2) {  // bayesian
            for (const WeightAddr &addr: m_unmanaged_share_indices_) {
                for (int dim = 0; dim < Dim; ++dim) { other_key[dim] = key[dim] + addr[dim]; }
                auto it = m_key_bhm_dict_.find(other_key);
                if (it == m_key_bhm_dict_.end() || !it->second->active) { continue; }
                auto &other_bhm = it->second->bhm;
                Dtype &mu1 = mu[addr[Dim + 1]];
                Dtype sigma1 = bhm.GetWeightVariance(addr[Dim + 1]);
                Dtype mu2 = other_bhm.GetWeights()[addr[Dim]];
                Dtype sigma2 = other_bhm.GetWeightVariance(addr[Dim]);
                mu1 = (sigma1 * mu2 + sigma2 * mu1) / (sigma1 + sigma2 + 1e-8f);
            }
            bhm.SetWeights(mu);
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BuildBhmKdtree() const {
        if (!m_bhm_kdtree_needs_update_ && m_bhm_kdtree_ != nullptr &&
            m_key_bhm_positions_.empty()) {
            return;
        }
        MatrixDX bhm_positions(Dim, m_key_bhm_positions_.size());
        long i = 0;
        for (const auto &[key, center]: m_key_bhm_positions_) { bhm_positions.col(i++) = center; }
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_ =
            std::make_shared<Kdtree>(bhm_positions);
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_needs_update_ = false;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictThread(
        const Dtype *points_ptr,
        const long start,
        const long end,
        const bool logodd,
        const bool compute_free_space,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        Dtype *prob_occupied_ptr,
        bool *in_free_space_ptr,
        Dtype *gradient_ptr) const {

        ERL_DEBUG_ASSERT(points_ptr != nullptr, "points_ptr is nullptr.");
        ERL_DEBUG_ASSERT(prob_occupied_ptr != nullptr, "prob_occupied_ptr is nullptr.");
        ERL_DEBUG_ASSERT(!compute_gradient || gradient_ptr != nullptr, "gradient_ptr is nullptr.");
        ERL_DEBUG_ASSERT(start >= 0, "start is negative.");
        ERL_DEBUG_ASSERT(end > start, "end is not greater than start.");

        points_ptr += start * Dim;
        prob_occupied_ptr += start;
        if (compute_free_space) { in_free_space_ptr += start; }
        if (compute_gradient) { gradient_ptr += start * Dim; }

        const long num_points = end - start;
        const long knn = m_setting_->test_knn;
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        Dtype half_size = 0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;

        (void) parallel;
        // dynamic scheduling because some points may cause `continue`.
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(num_points,                                                 \
               knn,                                                    \
               half_size,                                              \
               logodd,                                                 \
               compute_free_space,                                     \
               compute_gradient,                                       \
               gradient_with_sigmoid,                                  \
               points_ptr,                                             \
               prob_occupied_ptr,                                      \
               in_free_space_ptr,                                      \
               gradient_ptr)
        for (long i = 0; i < num_points; ++i) {

            Eigen::Map<const VectorD> point(points_ptr + i * Dim, Dim);

            // is there a free node that covers the corresponding bhm completely?
            Key key;  // use the tree to predict the occupancy
            if (!m_tree_->CoordToKeyChecked(point, key)) { continue; }  // outside the map
            const TreeNode *node = m_tree_->Search(key);
            if (node == nullptr) { continue; }  // no observation of this point at all

            if (knn == 1) {
                long bhm_index = -1;
                Dtype bhm_distance = 0.0f;
                m_bhm_kdtree_->Nearest(point, bhm_index, bhm_distance);      // find the nearest bhm
                bhm_distance = std::sqrt(bhm_distance);                      // distance is squared
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;  // the key of the bhm
                const auto &local_bhm = m_key_bhm_dict_.at(bhm_key);         // obtain the bhm

                if (bhm_index < 0 || bhm_distance > half_size || !local_bhm->active) {
                    if (node == nullptr) { continue; }  // unknown
                    // get the occupancy from the tree
                    prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();
                    if (compute_free_space) {
                        in_free_space_ptr[i] = !m_tree_->IsNodeOccupied(node);
                    }
                    continue;
                }

                VectorD grad;
                local_bhm->PredictAt(
                    point,
                    logodd,
                    compute_free_space,
                    compute_gradient,
                    gradient_with_sigmoid,
                    prob_occupied_ptr[i],
                    in_free_space_ptr[i],
                    grad);
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) { grad_ptr[dim] = grad[dim]; }
                }
                continue;
            }

            std::vector<nanoflann::ResultItem<long, Dtype>> bhm_indices_dists;
            m_bhm_kdtree_->RadiusSearch(point, half_size, bhm_indices_dists, true /*sorted*/);
            Dtype weight_sum = 0.0f;
            Dtype prob_sum = 0.0f;
            Dtype in_free_space_sum = 0.0f;
            VectorD gradient_sum = VectorD::Zero();
            long cnt = 0;
            for (auto &idx_dist: bhm_indices_dists) {  // iterate over the neighbors
                if (cnt >= knn) { break; }             // only use the first knn active neighbors
                const long &bhm_index = idx_dist.first;
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;    // obtain the bhm key
                const auto &local_bhm = m_key_bhm_dict_.at(bhm_key);           // obtain the bhm
                if (local_bhm == nullptr || !local_bhm->active) { continue; }  // not active bhm

                ++cnt;
                Dtype prob;
                bool in_free_space;
                VectorD grad;
                local_bhm->PredictAt(
                    point,
                    logodd,
                    compute_free_space,
                    compute_gradient,
                    gradient_with_sigmoid,
                    prob,
                    in_free_space,
                    grad);
                Dtype weight = (local_bhm->bhm.GetMapBoundary().center - point).cwiseAbs().prod();
                weight = 1.0f / (weight + 1e-6f);
                weight_sum += weight;
                prob_sum += prob * weight;
                if (compute_free_space) { in_free_space_sum += static_cast<Dtype>(in_free_space); }
                if (compute_gradient) {
                    for (int dim = 0; dim < Dim; ++dim) { gradient_sum[dim] += grad[dim] * weight; }
                }
            }
            if (cnt == 0) {                         // no neighboring bhm
                if (node == nullptr) { continue; }  // unknown
                // get the occupancy from the tree, gradient is not available
                prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();
                if (compute_free_space) { in_free_space_ptr[i] = in_free_space_sum > 0.0f; }
            } else {
                prob_occupied_ptr[i] = prob_sum / weight_sum;
                if (compute_free_space) {
                    in_free_space_ptr[i] = in_free_space_sum * 2.0f > weight_sum;
                }
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) {
                        grad_ptr[dim] = gradient_sum[dim] / weight_sum;
                    }
                }
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoints(
        const VectorD &sensor_origin,
        const Eigen::Ref<const MatrixDX> &points) {
        switch (m_setting_->update_map.method) {
            case 1:
                UpdateMapPoints1(sensor_origin, points);
                break;
            case 2:
                UpdateMapPoints2();
                break;
            default:
                ERL_DEBUG_ASSERT(false, "Unknown update map method.");
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoints1(
        const VectorD &sensor_origin,
        const Eigen::Ref<const MatrixDX> &points) {
        if (m_changed_clusters_.empty()) { return; }

        // sequential:
        // collect hit points from the local Bayesian Hilbert maps
        // collect the map pointer and the local index of the hit points
        // may need to ask the surface data manager to allocate new points
        m_hit_points_.clear();
        m_hit_points_.reserve(m_changed_clusters_.size());
        long cnt_new_points = 0;
        long cnt_existing_points = 0;
        const int max_num_points = m_setting_->update_map.max_num_points;

        {
            ERL_BLOCK_TIMER_MSG("collect hit points from local BHMs");

            std::vector<std::tuple<Key, Dtype, std::shared_ptr<LocalBhm>>> clusters;
            clusters.reserve(m_changed_clusters_.size());
            for (const Key &key: m_changed_clusters_) {
                auto local_bhm = m_key_bhm_dict_.at(key);
                if (!local_bhm->active) { continue; }  // skip inactive local BHM
                Dtype dist = (local_bhm->tracked_surface_boundary.center - sensor_origin).norm();
                clusters.emplace_back(key, dist, local_bhm);
            }
            std::sort(clusters.begin(), clusters.end(), [](const auto &a, const auto &b) {
                return std::get<1>(a) < std::get<1>(b);
            });

            for (const auto &[key, dist, local_bhm_ptr]: clusters) {
                LocalBhm &local_bhm = *local_bhm_ptr;

                std::vector<PointInfo> existing_points;
                if (max_num_points <= 0 || cnt_existing_points < max_num_points) {
                    /// collect existing hit points
                    existing_points.reserve(local_bhm.surface_indices.size());
                    for (const auto &[grid_idx, surf_idx]: local_bhm.surface_indices) {
                        existing_points.emplace_back(grid_idx, surf_idx);
                    }
                    cnt_existing_points += existing_points.size();
                }

                /// add new hit points
                std::vector<PointInfo> new_points;
                if (local_bhm.surface_indices.size() >= m_surf_point_capacity_) {  // no new points
                    m_hit_points_.emplace_back(
                        key,
                        std::move(new_points),
                        std::move(existing_points));
                    continue;
                }
                new_points.reserve(m_surf_point_capacity_ - local_bhm.surface_indices.size());
                GridIndex grid_coords;
                grid_coords[Dim] = 0;  // edge coord, not used here
                for (const long &hit_index: local_bhm.hit_indices) {
                    auto point = points.col(hit_index);
                    if (!local_bhm.GetGridCoords(point, true, grid_coords)) { continue; }
                    auto [it, inserted] = local_bhm.surface_indices.try_emplace(grid_coords, -1);
                    if (!inserted) { continue; }

                    it->second = m_surf_data_manager_.AddEntry(point, VectorD::Zero(), 0.0f, 0.0f);
                    ERL_DEBUG_ASSERT(
                        static_cast<long>(it->second) != -1l,
                        "Failed to add entry to the surface data manager.");
                    new_points.emplace_back(grid_coords, it->second);
                    if (new_points.size() >= new_points.capacity()) { break; }  // no more needed
                }
                cnt_new_points += new_points.size();
                m_hit_points_.emplace_back(key, std::move(new_points), std::move(existing_points));
            }
        }
        ERL_INFO(
            "Collected {} new points and {} existing points from {} local BHMs.",
            cnt_new_points,
            cnt_existing_points,
            m_hit_points_.size());

        // parallel:
        // for each point to update, compute the logodd and the gradient
        // adjust the point to make |logodd| close to 0 as much as possible
        // if |logodd| is too large, the point should be removed.
        // after the move, the local index may change

        {
            ERL_BLOCK_TIMER_MSG("compute logodd and gradient for points");
            // need dynamic scheduling here because the workload is not evenly distributed.
            // some local BHMs may have more points than others.
            // some points need to be removed so that the computation stops earlier.
#pragma omp parallel for default(none) schedule(dynamic)
            for (auto &[key, new_points, existing_points]: m_hit_points_) {
                // abs(logodd) may be larger than the threshold, but for new points, we don't remove
                // them immediately, we will check when we try to update the point again.
                LocalBhm &local_bhm = *m_key_bhm_dict_.at(key);
                for (PointInfo &new_point: new_points) {
                    SurfData &surf_data = m_surf_data_manager_[new_point.surf_idx];
                    InitMapPoint1(local_bhm, surf_data, new_point.to_remove);
                }
                for (PointInfo &existing_point: existing_points) {
                    SurfData &surf_data = m_surf_data_manager_[existing_point.surf_idx];
                    UpdateMapPoint1(local_bhm, surf_data, existing_point.to_remove);
                    if (!local_bhm.tracked_surface_boundary.contains(surf_data.position)) {
                        existing_point.to_remove = true;
                    }
                    if (existing_point.to_remove) { continue; }
                    // if the point is not removed, we need to update the local index
                    const auto &map_min = local_bhm.tracked_surface_boundary.min();
                    for (long dim = 0; dim < Dim; ++dim) {
                        existing_point.new_grid_idx[dim] = common::MeterToGrid<Dtype>(
                            surf_data.position[dim],
                            map_min[dim],
                            m_surface_res_);
                    }
                    existing_point.new_grid_idx[Dim] = 0;  // edge coord, not used here
                    existing_point.to_move =
                        (existing_point.new_grid_idx != existing_point.grid_idx);
                }
            }
        }

#ifndef NDEBUG
        Dtype init_logodd_abs = 0.0f;
        Dtype init_logodd_abs_min = std::numeric_limits<Dtype>::max();
        Dtype init_logodd_abs_max = std::numeric_limits<Dtype>::lowest();
        Dtype adjust_logodd_abs = 0.0f;
        Dtype adjust_logodd_abs_min = std::numeric_limits<Dtype>::max();
        Dtype adjust_logodd_abs_max = std::numeric_limits<Dtype>::lowest();
        for (const auto &[key, new_points, existing_points]: m_hit_points_) {
            for (const PointInfo &new_point: new_points) {
                const SurfData &surf_data = m_surf_data_manager_[new_point.surf_idx];
                init_logodd_abs += surf_data.var_position;
                init_logodd_abs_min = std::min(init_logodd_abs_min, surf_data.var_position);
                init_logodd_abs_max = std::max(init_logodd_abs_max, surf_data.var_position);
                ++cnt_new_points;
            }
            for (const PointInfo &existing_point: existing_points) {
                const SurfData &surf_data = m_surf_data_manager_[existing_point.surf_idx];
                adjust_logodd_abs += surf_data.var_position;
                adjust_logodd_abs_min = std::min(adjust_logodd_abs_min, surf_data.var_position);
                adjust_logodd_abs_max = std::max(adjust_logodd_abs_max, surf_data.var_position);
                ++cnt_existing_points;
            }
        }
        const typename Setting::UpdateMap &update_map_setting = m_setting_->update_map;
        init_logodd_abs /= static_cast<Dtype>(cnt_new_points) * update_map_setting.var_scale;
        init_logodd_abs_min /= update_map_setting.var_scale;
        init_logodd_abs_max /= update_map_setting.var_scale;
        ERL_INFO(
            "Initial logodd abs: {} (min: {}, max: {})",
            init_logodd_abs,
            init_logodd_abs_min,
            init_logodd_abs_max);
        adjust_logodd_abs /= static_cast<Dtype>(cnt_existing_points) * update_map_setting.var_scale;
        adjust_logodd_abs_min /= update_map_setting.var_scale;
        adjust_logodd_abs_max /= update_map_setting.var_scale;
        ERL_INFO(
            "Adjusted logodd abs: {} (min: {}, max: {})",
            adjust_logodd_abs,
            adjust_logodd_abs_min,
            adjust_logodd_abs_max);
#endif

        UpdateSurfaceManager1();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateSurfaceManager1() {
        // sequential:
        // update the local Bayesian Hilbert maps with the new points
        for (auto &[key, new_points, existing_points]: m_hit_points_) {
            LocalBhm &local_bhm = *m_key_bhm_dict_.at(key);
            for (PointInfo &new_point: new_points) {
                if (!new_point.to_remove) { continue; }
                local_bhm.surface_indices.erase(new_point.grid_idx);
                m_surf_data_manager_.RemoveEntry(new_point.surf_idx);
            }
            for (PointInfo &existing_point: existing_points) {
                if (existing_point.to_remove) {
                    m_surf_data_manager_.RemoveEntry(existing_point.surf_idx);
                    local_bhm.surface_indices.erase(existing_point.grid_idx);
                    continue;
                }
                if (!existing_point.to_move) { continue; }  // no change in local index

                auto new_surf_it = local_bhm.surface_indices.find(existing_point.new_grid_idx);
                if (new_surf_it == local_bhm.surface_indices.end()) {
                    local_bhm.surface_indices.emplace(
                        existing_point.new_grid_idx,
                        existing_point.surf_idx);
                } else {
                    m_surf_data_manager_.RemoveEntry(existing_point.surf_idx);
                }
                local_bhm.surface_indices.erase(existing_point.grid_idx);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::InitMapPoint1(
        LocalBhm &local_bhm,
        SurfData &surf_data,
        bool &to_remove) const {
        Dtype logodd;
        bool in_free_space;
        local_bhm.PredictAt(
            surf_data.position,
            true /*logodd*/,
            false /*compute_free_space*/,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            logodd,
            in_free_space,
            surf_data.normal);
        Dtype norm = surf_data.normal.norm();
        if (norm < 1e-6f) {
            to_remove = true;  // if the normal is too small, remove the point
            return;
        }
        surf_data.normal = surf_data.normal / -norm;  // normal = -gradient
        surf_data.var_position = std::min(
            m_setting_->update_map.var_scale * std::abs(logodd),
            m_setting_->update_map.var_max);
        surf_data.var_normal = surf_data.var_position;  // use the same variance for normal
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoint1(
        LocalBhm &local_bhm,
        SurfData &surf_data,
        bool &to_remove) const {
        const typename Setting::UpdateMap &update_map_setting = m_setting_->update_map;
        const Dtype max_logodd_abs = update_map_setting.surface_max_abs_logodd;
        const int max_num_adjusts = update_map_setting.max_adjust_tries;
        int num_adjusts = 0;
        Dtype logodd;
        bool in_free_space;
        VectorD &gradient = surf_data.normal;  // reuse the normal as the gradient
        local_bhm.PredictAt(
            surf_data.position,
            true /*logodd*/,
            false /*compute_free_space*/,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            logodd,
            in_free_space,
            gradient);
        Dtype norm = gradient.norm();
        if (norm < 1e-6f) {
            to_remove = true;  // if the gradient is too small, remove the point
            return;
        }
        Dtype logodd_abs = std::abs(logodd);
#ifndef NDEBUG
        Dtype logodd_init = logodd;
        Dtype logodd_abs_init = logodd_abs;
        Dtype norm_init = norm;
#endif

        Dtype delta = update_map_setting.surface_step_size;
        while (num_adjusts < max_num_adjusts && logodd_abs > max_logodd_abs) {
            // logodd > 0, prob(occupied) > 0.5, move the point along -gradient
            // logodd < 0, prob(occupied) < 0.5, move the point along gradient
            Dtype step = -logodd * delta / (norm * norm);
            surf_data.position += step * gradient;
            Dtype logodd_new;
            local_bhm.PredictAt(
                surf_data.position,
                true /*logodd*/,
                false /*compute_free_space*/,
                true /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                logodd_new,
                in_free_space,
                gradient);
            norm = gradient.norm();
            if (norm < 1e-6f) {
                to_remove = true;  // if the gradient is too small, remove the point
                break;
            }
            logodd_abs = std::abs(logodd_new);
            if (logodd_abs <= max_logodd_abs) { break; }
            if (logodd_new * logodd < 0) {  // logodd changed sign, reduce the step size
                delta *= 0.5f;
            } else {
                delta *= 1.1f;  // increase the step size
            }
            logodd = logodd_new;
            ++num_adjusts;
        }
        if (logodd_abs >= update_map_setting.surface_bad_abs_logodd) {
            to_remove = true;
            return;
        }
        ERL_DEBUG_WARN_COND(
            logodd_abs > logodd_abs_init,
            "logodd_abs {} is larger than initial {} after {} adjustments. logodd: {} (initial: "
            "{}), norm: {} (initial: {}).",
            logodd_abs,
            logodd_abs_init,
            num_adjusts,
            logodd,
            logodd_init,
            norm,
            norm_init);
        surf_data.normal = gradient / -norm;  // normal = -gradient
        surf_data.var_position =
            std::min(update_map_setting.var_scale * logodd_abs, update_map_setting.var_max);
        surf_data.var_normal = surf_data.var_position;  // use the same variance for normal
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoints2() {

        if (m_changed_clusters_.empty()) { return; }

        // 1. update the queue
        {
            ERL_BLOCK_TIMER_MSG("update marching queue");
            for (const Key &key: m_changed_clusters_) {
                if (!m_key_bhm_dict_.at(key)->active) { continue; }  // skip inactive local BHM
                long time_stamp =
                    std::chrono::high_resolution_clock::now().time_since_epoch().count();
                if (auto it = m_marching_queue_keys_.find(key);
                    it == m_marching_queue_keys_.end()) {
                    m_marching_queue_keys_.insert({
                        key,
                        m_marching_queue_.push({time_stamp, key}),
                    });
                } else {
                    auto &queue_key = it->second;
                    (*queue_key).priority = time_stamp;
                    m_marching_queue_.increase(queue_key);
                }
            }
        }

        // 2. collect local BHMs
        m_local_bhms_.clear();
        {
            ERL_BLOCK_TIMER_MSG("collect local BHMs from marching queue");
            const int max_num_voxels = m_setting_->update_map.max_num_voxels;
            int cnt_voxels = 0;
            while (!m_marching_queue_.empty()) {
                Key key = m_marching_queue_.top().key;
                m_marching_queue_.pop();
                m_marching_queue_keys_.erase(key);
                auto local_bhm = m_key_bhm_dict_.at(key);
                if (!local_bhm->active) { continue; }        // skip inactive local BHM
                m_local_bhms_.emplace_back(key, local_bhm);  // collect local BHMs
                cnt_voxels += static_cast<int>(local_bhm->surf_voxels.size());
                if (max_num_voxels > 0 && cnt_voxels >= max_num_voxels) { break; }
            }
        }
        ERL_INFO("{} local BHMs in the marching queue.", m_marching_queue_.size());

        // 3. run marching squares/cubes for each surface voxel (edge)
        ERL_INFO("Marching {} local BHMs.", m_local_bhms_.size());
        {
            ERL_BLOCK_TIMER_MSG("marching BHMs");
#pragma omp parallel for schedule(dynamic) default(none)
            for (auto &[key, local_bhm_ptr]: m_local_bhms_) { MarchingBhm(key, *local_bhm_ptr); }
        }

        // 4. remove voxels that do not contain the surface
        UpdateSurfaceManager2();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::MarchingBhm(const Key &key, LocalBhm &local_bhm)
        const {

        const Dtype var_scale = m_setting_->update_map.var_scale;
        const Dtype var_max = m_setting_->update_map.var_max;
        const VectorD &map_min = local_bhm.tracked_surface_boundary.min();
        const VectorD &map_max = local_bhm.tracked_surface_boundary.max();
        constexpr int n_vertices = (1 << Dim);
        const int key_offset = 1 << (m_tree_->GetTreeDepth() - m_setting_->bhm_depth);
        const int surf_grid_size = m_setting_->local_bhm->surface_grid_size;

        auto &query_results = local_bhm.surf_data_cache;
        query_results.clear();
        query_results.reserve(local_bhm.hit_indices.size() * 12);
        for (auto &[voxel_coords, voxel]: local_bhm.surf_voxels) {
            // 1. iterate over the vertices
            VectorX vertex_values(n_vertices);
            GridIndex vertex_coords;
            vertex_coords[Dim] = 0;
            for (int i = 0; i < n_vertices; ++i) {
                const int *vertex_code = MC::GetVertexCode(i);
                // compute vertex coordinates
                bool on_max_boundary = false;
                Key bhm_key = key;
                for (int dim = 0; dim < Dim; ++dim) {
                    vertex_coords[dim] = voxel_coords[dim] + vertex_code[dim];
                    if (vertex_coords[dim] == surf_grid_size) {
                        bhm_key[dim] += key_offset;
                        on_max_boundary = true;
                    }
                }
                // check if the vertex is already queried
                auto it = query_results.find(vertex_coords);
                if (it != query_results.end()) {
                    vertex_values[i] = it->second.var_position;
                    continue;
                }
                // if not queried, run prediction
                SurfData surf_data;
                for (int dim = 0; dim < Dim; ++dim) {
                    surf_data.position[dim] = common::VertexIndexToMeter<Dtype>(
                        vertex_coords[dim],
                        map_min[dim],
                        m_surface_res_);
                    if (surf_data.position[dim] > map_max[dim]) {  // avoid numerical issue
                        surf_data.position[dim] = map_max[dim];
                    }
                }
                // pick the correct BHM
                auto *bhm = &local_bhm.bhm;
                if (on_max_boundary) {
                    auto it_bhm = m_key_bhm_dict_.find(bhm_key);
                    if (it_bhm != m_key_bhm_dict_.end() && it_bhm->second->active) {
                        bhm = &it_bhm->second->bhm;
                    }
                }
                // query the BHM
                bhm->Predict(
                    surf_data.position,
                    true /* logodd */,
                    m_setting_->local_bhm->faster_prediction,
                    false /*compute_gradient*/,
                    false /*gradient_with_sigmoid*/,
                    surf_data.var_position,  // use this field to store logodd temporarily
                    surf_data.normal);
                query_results[vertex_coords] = surf_data;
                vertex_values[i] = surf_data.var_position;
            }

            // 2. run marching squares/cubes
            const Dtype surf_log_odds = local_bhm.surface_log_odds;
            int new_surf_cfg = MC::CalculateVertexConfigIndex(vertex_values.data(), surf_log_odds);
            const int *unique_edge_indices = MC::GetUniqueEdgeIndices(new_surf_cfg);
            if (unique_edge_indices == nullptr) {  // not a surface voxel
                voxel.good = false;
                continue;
            }
            voxel.good = true;
            const bool config_changed = (voxel.surf_config != new_surf_cfg);
            GridIndex edge_coords;
            if (config_changed) {
                voxel.edges.clear();
                int col = 0;
                while (unique_edge_indices[col] != -1) {
                    const int edge_index = unique_edge_indices[col++];
                    const int *edge_code = MC::GetEdgeCode(edge_index);
                    for (int dim = 0; dim <= Dim; ++dim) {
                        edge_coords[dim] = voxel_coords[dim] + edge_code[dim];
                    }
                    voxel.edges.emplace_back(edge_coords);
                }
                voxel.faces.clear();
                const int *vertex_indices = MC::GetVertexIndices(new_surf_cfg);
                while (*vertex_indices != -1) {
                    Face face;
                    for (int i = 0; i < Dim; ++i) { face[i] = vertex_indices[i]; }
                    vertex_indices += Dim;
                    voxel.faces.emplace_back(face);
                }
            }
            voxel.surf_config = new_surf_cfg;  // update the surface configuration
            // interpolate edges
            int col = 0;
            while (unique_edge_indices[col] != -1) {
                const int edge_index = unique_edge_indices[col++];
                const int *edge_code = MC::GetEdgeCode(edge_index);
                for (int dim = 0; dim <= Dim; ++dim) {
                    edge_coords[dim] = voxel_coords[dim] + edge_code[dim];
                }
                // check if interpolation for the edge exists
                auto [it, inserted] = query_results.try_emplace(edge_coords, SurfData());
                if (!inserted) { continue; }  // interpolation for the edge exists.

                // do the interpolation

                // const int *vertex_index = MC::GetEdgeVertexIndices(edge_index);
                // const int v1 = vertex_index[0], v2 = vertex_index[1];
                // const int *v1_code = MC::GetVertexCode(v1);
                // const int *v2_code = MC::GetVertexCode(v2);
                // GridIndex v1_coords = GridIndex::Zero(), v2_coords = GridIndex::Zero();
                // for (int dim = 0; dim < Dim; ++dim) {
                //     v1_coords[dim] = voxel_coords[dim] + v1_code[dim];
                //     v2_coords[dim] = voxel_coords[dim] + v2_code[dim];
                // }
                // faster way to get the vertex coordinates
                GridIndex v1_coords = edge_coords, v2_coords = edge_coords;
                v1_coords[Dim] = 0;
                ++v2_coords[edge_coords[Dim] - 1];
                v2_coords[Dim] = 0;

                const SurfData &v1 = query_results[v1_coords];
                const SurfData &v2 = query_results[v2_coords];
                const Dtype val_diff = v1.var_position - v2.var_position;
                constexpr Dtype kEpsilon = 1e-6f;
                SurfData &surf_data = it->second;
                if (std::abs(val_diff) >= kEpsilon) {
                    const Dtype t = (v1.var_position - surf_log_odds) / val_diff;
                    surf_data.position = v1.position + t * (v2.position - v1.position);
                } else {
                    surf_data.position = 0.5f * (v1.position + v2.position);
                }
                ERL_DEBUG_ASSERT(!surf_data.position.hasNaN(), "NaN in surface position.");
                local_bhm.bhm.Predict(
                    surf_data.position,
                    true /* logodd */,
                    m_setting_->local_bhm->faster_prediction,
                    true /* compute_gradient */,
                    false /* gradient_with_sigmoid */,
                    surf_data.var_position,  // use this field to store logodd temporarily
                    surf_data.normal);
                Dtype norm = surf_data.normal.norm();
                if (norm < 1.0e-10f) {
                    surf_data.normal.setZero();
                    surf_data.var_position = 1e6f;  // set a large variance
                    surf_data.var_normal = surf_data.var_position;
                    continue;
                }
                surf_data.normal /= -norm;
                surf_data.var_position =
                    std::min(var_scale * std::abs(surf_data.var_position - surf_log_odds), var_max);
                surf_data.var_normal = surf_data.var_position;
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateSurfaceManager2() {
        for (auto &[key, local_bhm_ptr]: m_local_bhms_) {
            LocalBhm &local_bhm = *local_bhm_ptr;
            auto &query_results = local_bhm.surf_data_cache;
            SurfaceIndexMap new_surface_indices;
            new_surface_indices.reserve(local_bhm.surface_indices.size());
            for (const auto &[voxel_coords, voxel]: local_bhm.surf_voxels) {
                if (!voxel.good) { continue; }
                for (const GridIndex &edge_coords: voxel.edges) {
                    auto [it, inserted] = new_surface_indices.try_emplace(edge_coords, -1);
                    if (!inserted) { continue; }  // already exist.
                    auto old_it = local_bhm.surface_indices.find(edge_coords);
                    if (old_it == local_bhm.surface_indices.end()) {  // new entry
                        it->second = m_surf_data_manager_.AddEntry(query_results[edge_coords]);
                    } else {
                        it->second = old_it->second;  // reuse the index
                        m_surf_data_manager_[it->second] = query_results[edge_coords];
                        old_it->second = -1;  // mark as reused
                    }
                }
            }
            for (const auto &[edge_coords, surf_idx]: local_bhm.surface_indices) {
                if (surf_idx == static_cast<std::size_t>(-1)) { continue; }  // already reused
                m_surf_data_manager_.RemoveEntry(surf_idx);
            }
            local_bhm.surface_indices.swap(new_surface_indices);
            query_results.clear();  // clear the cache to save memory
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::RunMarchingQueue(const bool run_all) {
        // 1. collect local BHMs
        m_local_bhms_.clear();
        int cnt_voxels = 0;
        int max_num_voxels = m_setting_->update_map.max_num_voxels;
        if (run_all) { max_num_voxels = -1; }
        while (!m_marching_queue_.empty()) {
            Key key = m_marching_queue_.top().key;
            m_marching_queue_.pop();
            m_marching_queue_keys_.erase(key);
            auto local_bhm = m_key_bhm_dict_.at(key);
            if (!local_bhm->active) { continue; }  // skip inactive local BHM
            m_local_bhms_.emplace_back(key, local_bhm);
            cnt_voxels += static_cast<int>(local_bhm->surf_voxels.size());
            if (max_num_voxels > 0 && cnt_voxels >= max_num_voxels) { break; }
        }

        // 2. run marching squares/cubes for each surface voxel (edge)
#pragma omp parallel for schedule(dynamic) default(none)
        for (auto &[key, local_bhm_ptr]: m_local_bhms_) { MarchingBhm(key, *local_bhm_ptr); }

        // 3. update surface manager buffer
        UpdateSurfaceManager2();
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::VectorD
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetUniqueVertex(
        Key key,
        GridIndex edge_idx,
        int buffer_idx) const {

        // check if [key, edge_idx] has duplicates and the edge is on the max boundary.
        // for duplicates, we pick the one on the min boundary if it exists.
        bool has_duplicate = false;
        const int surf_grid_size = m_setting_->local_bhm->surface_grid_size;
        const uint32_t offset = 1 << (m_setting_->tree->tree_depth - m_setting_->bhm_depth);
        for (int dim = 0; dim < Dim; ++dim) {
            if (edge_idx[dim] == surf_grid_size) {
                has_duplicate = true;
                key[dim] += offset;
                edge_idx[dim] = 0;
            }
        }
        VectorD pos1 = m_surf_data_manager_[buffer_idx].position;
        if (!has_duplicate) { return pos1; }
        auto it = m_key_bhm_dict_.find(key);
        if (it == m_key_bhm_dict_.end()) { return pos1; }
        auto &neighbor_bhm = it->second;
        if (!neighbor_bhm->active) { return pos1; }
        auto surf_it = neighbor_bhm->surface_indices.find(edge_idx);
        if (surf_it == neighbor_bhm->surface_indices.end()) { return pos1; }
        VectorD pos2 = m_surf_data_manager_[surf_it->second].position;
        return pos2;
    }

    template class RaySelector2D<float>;
    template class RaySelector2D<double>;
    template class RaySelector3D<float>;
    template class RaySelector3D<double>;

    template class BayesianHilbertSurfaceMapping<float, 2>;
    template class BayesianHilbertSurfaceMapping<float, 3>;
    template class BayesianHilbertSurfaceMapping<double, 2>;
    template class BayesianHilbertSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf
