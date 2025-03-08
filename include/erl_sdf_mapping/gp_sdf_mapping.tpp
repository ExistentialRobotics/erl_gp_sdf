#pragma once

namespace erl::sdf_mapping {
    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestBuffer::ConnectBuffers(
        const Eigen::Ref<const Positions>& positions_in,
        Distances& distances_out,
        Gradients& gradients_out,
        Variances& variances_out,
        Covariances& covariances_out,
        const bool compute_covariance

    ) {

        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
        const long n = positions_in.cols();
        if (n == 0) return false;

        distances_out.resize(n);
        gradients_out.resize(Gradients::RowsAtCompileTime, n);
        variances_out.resize(Variances::RowsAtCompileTime, n);
        if (compute_covariance) { covariances_out.resize(Covariances::RowsAtCompileTime, n); }
        this->positions = std::make_unique<Eigen::Ref<const Positions>>(positions_in);
        this->distances = std::make_unique<Eigen::Ref<Distances>>(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<Gradients>>(gradients_out);
        this->variances = std::make_unique<Eigen::Ref<Variances>>(variances_out);
        this->covariances = std::make_unique<Eigen::Ref<Covariances>>(covariances_out);
        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::GpSdfMapping(std::shared_ptr<Setting> setting, std::shared_ptr<SurfaceMapping> surface_mapping)
        : m_setting_(std::move(setting)),
          m_surface_mapping_(std::move(surface_mapping)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1.");
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Update(
        const Eigen::Ref<const Rotation>& rotation,
        const Eigen::Ref<const Translation>& translation,
        const Eigen::Ref<const Eigen::MatrixX<Dtype>>& ranges) {

        ERL_TRACY_FRAME_MARK_START();

        bool success = false;
        double time_budget_us = 1e6 / m_setting_->update_hz;  // us
        double total_update_time = 0;
        double surf_mapping_time = 0;
        double sdf_gp_update_time = 0;

        {
            ERL_BLOCK_TIMER_TIME(total_update_time);
            {
                std::lock_guard lock(m_mutex_);
                ERL_BLOCK_TIMER_MSG_TIME("Surface mapping update", surf_mapping_time);
                success = m_surface_mapping_->Update(rotation, translation, ranges);
            }
            time_budget_us -= surf_mapping_time * 1000;  // us

            if (success) {
                std::lock_guard lock(m_mutex_);  // CRITICAL SECTION
                ERL_BLOCK_TIMER_MSG_TIME("Update SDF GPs", sdf_gp_update_time);
                UpdateGpSdf(time_budget_us);
            }
        }

        ERL_TRACY_PLOT("[update]surf_mapping_time (ms)", surf_mapping_time);
        ERL_TRACY_PLOT("[update]sdf_gp_update_time (ms)", sdf_gp_update_time);
        ERL_TRACY_PLOT("[update]total_update_time (ms)", total_update_time);
        ERL_TRACY_PLOT("m_gp_map_.size()", static_cast<long>(m_gp_map_.size()));
        ERL_TRACY_PLOT("m_cluster_queue_keys_.size()", static_cast<long>(m_cluster_queue_keys_.size()));
        ERL_TRACY_PLOT("m_cluster_queue_.size()", static_cast<long>(m_cluster_queue_.size()));
        ERL_TRACY_PLOT("m_clusters_to_train_.size()", static_cast<long>(m_clusters_to_train_.size()));
        ERL_TRACY_PLOT_CONFIG("m_gp_map_.memory_usage", tracy::PlotFormatType::Memory, true, true, 0);
        ERL_TRACY_PLOT("m_gp_map_.memory_usage", static_cast<long>([&]() {
                           std::size_t gps_memory_usage = 0;
                           for (const auto& [key, gp]: m_gp_map_) {
                               gps_memory_usage += sizeof(key);
                               gps_memory_usage += sizeof(gp);
                               if (gp != nullptr) { gps_memory_usage += gp->GetMemoryUsage(); }
                           }
                           return gps_memory_usage;
                       }()));
        ERL_TRACY_FRAME_MARK_END();

        return success;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::UpdateGpSdf(double time_budget_us) {
        ERL_BLOCK_TIMER();
        ERL_TRACY_FRAME_MARK_START();

        double collect_clusters_time = 0;
        double queue_clusters_time = 0;
        double train_gps_time = 0;

        const auto cluster_size = static_cast<Dtype>(m_surface_mapping_->GetClusterSize());
        const Dtype area_half_size = cluster_size * static_cast<Dtype>(m_setting_->gp_sdf_area_scale) / 2;

        // collect changed clusters
        {
            ERL_BLOCK_TIMER_MSG_TIME("Collect affected clusters", collect_clusters_time);
            const KeySet& changed_clusters = m_surface_mapping_->GetChangedClusters();
            KeySet affected_clusters(changed_clusters);
            for (const auto& cluster_key: changed_clusters) {
                const Aabb area(m_surface_mapping_->GetClusterCenter(cluster_key));
                KeyVector clusters_in_aabb = m_surface_mapping_->CollectClustersInAabb(area);
                affected_clusters.insert(clusters_in_aabb.begin(), clusters_in_aabb.end());
            }
            m_affected_clusters_.clear();
            m_affected_clusters_.insert(m_affected_clusters_.end(), affected_clusters.begin(), affected_clusters.end());
            ERL_INFO("Collect {} -> {} affected clusters", changed_clusters.size(), m_affected_clusters_.size());
        }

        // put affected clusters into m_cluster_queue_
        {
            std::lock_guard lock(m_mutex_);  // CRITICAL SECTION
            // we are going to access m_gp_map_, which is also touched in test.

            ERL_BLOCK_TIMER_MSG_TIME("Queue affected clusters", queue_clusters_time);
            long cnt_new_gps = 0;
            for (const auto& cluster_key: m_affected_clusters_) {
                if (auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, nullptr); inserted || it->second->locked_for_test) {
                    it->second = std::make_shared<SdfGp>();    // new GP is required
                    it->second->Activate(m_setting_->edf_gp);  // activate the GP
                    it->second->position = m_surface_mapping_->GetClusterCenter(cluster_key);
                    it->second->half_size = area_half_size;
                    ++cnt_new_gps;
                } else {
                    it->second->Activate(m_setting_->edf_gp);
                }
                auto time_stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                if (auto itr = m_cluster_queue_keys_.find(cluster_key); itr == m_cluster_queue_keys_.end()) {  // new cluster
                    m_cluster_queue_keys_.insert({cluster_key, m_cluster_queue_.push({time_stamp, cluster_key})});
                } else {
                    auto& heap_key = itr->second;
                    (*heap_key).time_stamp = time_stamp;
                    m_cluster_queue_.increase(heap_key);
                }
            }
            ERL_INFO("Create {} new GPs when queuing clusters", cnt_new_gps);
        }

        // train GPs if we still have time
        const double dt = timer.Elapsed<double, std::micro>();
        time_budget_us -= dt;
        ERL_INFO("Time spent: {} us, time budget: {} us", dt, time_budget_us);
        if (time_budget_us > 2.0 * m_train_gp_time_us_) {

            std::lock_guard lock(m_mutex_);  // CRITICAL SECTION
            // we are going to access m_clusters_to_train_ and m_gp_map_, which are also touched in test.

            ERL_BLOCK_TIMER_MSG_TIME("Train GPs", train_gps_time);

            auto max_num_clusters_to_train = static_cast<std::size_t>(std::floor(time_budget_us / m_train_gp_time_us_));
            max_num_clusters_to_train = std::min(max_num_clusters_to_train, m_cluster_queue_.size());
            m_clusters_to_train_.clear();
            while (!m_cluster_queue_.empty() && m_clusters_to_train_.size() < max_num_clusters_to_train) {
                Key cluster_key = m_cluster_queue_.top().key;
                m_cluster_queue_.pop();
                m_cluster_queue_keys_.erase(cluster_key);
                m_clusters_to_train_.emplace_back(cluster_key, m_gp_map_.at(cluster_key));
            }
            TrainGps();
        }
        ERL_INFO("{} cluster(s) not trained yet due to time limit.", m_cluster_queue_.size());

        ERL_TRACY_PLOT("[update_gp_sdf]collect_clusters_time (ms)", collect_clusters_time);
        ERL_TRACY_PLOT("[update_gp_sdf]queue_clusters_time (ms)", queue_clusters_time);
        ERL_TRACY_PLOT("[update_gp_sdf]train_gps_time (ms)", train_gps_time);

        ERL_TRACY_FRAME_MARK_END();
        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TrainGps() {
        const std::size_t n = m_clusters_to_train_.size();
        if (n == 0) { return; }

        ERL_INFO("Training {} GPs ...", n);
        const auto t0 = std::chrono::high_resolution_clock::now();

        const uint32_t num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = n / num_threads;
        const std::size_t left_over = n - batch_size * num_threads;
        std::size_t end_idx = 0;
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            std::size_t start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (thread_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping::TrainGpThread, this, thread_idx, start_idx, end_idx);
        }
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }
        threads.clear();
        ERL_INFO("Trained {} GPs", n);

        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Function {} takes {} ms", __PRETTY_FUNCTION__, time / 1e3);
        time /= static_cast<double>(n);
        m_train_gp_time_us_ = m_train_gp_time_us_ * 0.1 + time * 0.9;
        ERL_INFO("Per GP training time: {} us.", m_train_gp_time_us_);
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TrainGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (!m_surface_mapping_->Ready()) { return; }  // surface mapping is not ready

        const Dtype sensor_noise = m_surface_mapping_->GetSensorNoise();
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        const Dtype area_half_size = cluster_size * static_cast<Dtype>(m_setting_->gp_sdf_area_scale) / 2;
        const auto offset_distance = static_cast<Dtype>(m_setting_->offset_distance);
        const auto max_valid_gradient_var = static_cast<Dtype>(m_setting_->max_valid_gradient_var);
        const auto invalid_position_var = static_cast<Dtype>(m_setting_->invalid_position_var);
        const auto& surface_data_buffer = m_surface_mapping_->GetSurfaceDataBuffer();

        std::vector<std::pair<Dtype, std::size_t>> surface_data_indices;
        const auto max_num_samples = static_cast<std::size_t>(m_setting_->edf_gp->max_num_samples);
        surface_data_indices.reserve(max_num_samples);
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto& [cluster_key, gp] = m_clusters_to_train_[i];
            ERL_DEBUG_ASSERT(gp->active, "GP is not active");
            // collect surface data in the area
            const Aabb area(gp->position, area_half_size);
            surface_data_indices.clear();
            m_surface_mapping_->CollectSurfaceDataInAabb(area, surface_data_indices);
            if (surface_data_indices.empty()) {  // no surface data in the area
                gp->Deactivate();                // deactivate the GP if there is no training data
                continue;
            }
            gp->LoadSurfaceData(surface_data_indices, surface_data_buffer, offset_distance, sensor_noise, max_valid_gradient_var, invalid_position_var);
            gp->Train();
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    [[nodiscard]] bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Test(
        const Eigen::Ref<const Positions>& positions_in,
        Distances& distances_out,
        Gradients& gradients_out,
        Variances& variances_out,
        Covariances& covariances_out) {

        ERL_BLOCK_TIMER();

        {
            std::lock_guard lock(m_mutex_);
            if (m_gp_map_.empty()) {
                ERL_WARN("No GPs available for testing.");
                return false;
            }
        }

        if (positions_in.cols() == 0) {
            ERL_WARN("No query positions provided.");
            return false;
        }
        if (!m_surface_mapping_->Ready()) {
            ERL_WARN("Surface mapping is not ready.");
            return false;
        }
        if (!m_test_buffer_.ConnectBuffers(  // allocate memory for test results
                positions_in,
                distances_out,
                gradients_out,
                variances_out,
                covariances_out,
                m_setting_->test_query->compute_covariance)) {
            ERL_WARN("Failed to connect test buffers.");
            return false;
        }

        const uint32_t num_queries = positions_in.cols();
        const uint32_t num_threads = std::min(std::min(m_setting_->num_threads, std::thread::hardware_concurrency()), num_queries);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = num_queries / num_threads;
        const std::size_t leftover = num_queries - batch_size * num_threads;
        std::size_t start_idx, end_idx;

        double gp_search_time = 0;
        double gp_train_time = 0;
        double gp_test_time = 0;
        {
            std::lock_guard lock(m_mutex_);               // CRITICAL SECTION
            (void) m_surface_mapping_->GetMapBoundary();  // trigger the octree to update its metric min/max

            // If we iterate through the octree for each query position separately, it takes too much CPU time.
            // Instead, we collect all GPs in the area of all query positions and then assign them to the query positions.
            // Some query positions may not have any GPs from m_candidate_gps_. We need to search for them separately.
            // We are sure that m_candidate_gps_ are not empty because m_gp_map_ is not empty.
            // Experiments show that this can reduce the search time by at most 50%.
            // For 15k crowded query positions, the search time is reduced from ~60ms to ~30ms.
            // Another important trick is to use a KdTree to search for candidate GPs for each query position.
            // Knn search is much faster and the result is sorted by distance.
            // Experiments show that this knn search can reduce the search time further to 2ms.

            // Search GPs for each query position
            {
                ERL_BLOCK_TIMER_MSG_TIME("Search GPs", gp_search_time);
                SearchCandidateGps(positions_in);     // search for candidate GPs, collected GPs will be locked for testing
                m_query_to_gps_.clear();              // clear the previous query to GPs
                m_query_to_gps_.resize(num_queries);  // allocate memory for n threads
                if (num_queries == 1) {
                    SearchGpThread(0, 0, 1);  // save time on thread creation
                } else {
                    start_idx = 0;
                    for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                        end_idx = start_idx + batch_size;
                        if (thread_idx < leftover) { end_idx++; }
                        threads.emplace_back(&GpSdfMapping::SearchGpThread, this, thread_idx, start_idx, end_idx);
                        start_idx = end_idx;
                    }
                    for (auto& thread: threads) { thread.join(); }
                    threads.clear();
                }
            }

            // Train any updated GPs
            if (!m_cluster_queue_.empty()) {
                ERL_BLOCK_TIMER_MSG_TIME("Train GPs", gp_train_time);
                KeySet keys;
                m_clusters_to_train_.clear();
                for (auto& gps: m_query_to_gps_) {
                    for (auto& [distance, key_and_gp]: gps) {
                        if (!keys.insert(key_and_gp.first).second) { continue; }
                        if (const auto& gp = key_and_gp.second; !gp->active || gp->edf_gp->IsTrained()) { continue; }
                        m_clusters_to_train_.emplace_back(key_and_gp);
                    }
                }
                TrainGps();
            }

            if (m_setting_->use_occ_sign) {
                // collect the sign of query positions since the quadtree is not thread-safe
                const auto tree = m_surface_mapping_->GetOctree();
                if (m_query_signs_.size() < num_queries) { m_query_signs_.resize(num_queries); }
#pragma omp parallel for default(none) shared(tree, positions_in, num_queries)
                for (uint32_t i = 0; i < num_queries; i++) {
                    const Eigen::Vector3d& position = positions_in.col(i);
                    const auto node = tree->Search(position.x(), position.y(), position.z());
                    m_query_signs_[i] = node == nullptr ? -1.0 : 1.0;
                }
            }
        }

        // Compute the inference result for each query position
        {
            ERL_BLOCK_TIMER_MSG_TIME("Test GPs", gp_test_time);
            m_query_used_gps_.clear();
            m_query_used_gps_.resize(num_queries);
            if (num_queries == 1) {
                TestGpThread(0, 0, 1);  // save time on thread creation
            } else {
                start_idx = 0;
                for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                    end_idx = start_idx + batch_size;
                    if (thread_idx < leftover) { ++end_idx; }
                    threads.emplace_back(&GpSdfMapping::TestGpThread, this, thread_idx, start_idx, end_idx);
                    start_idx = end_idx;
                }
                for (auto& thread: threads) { thread.join(); }
                threads.clear();
            }
        }

        m_test_buffer_.DisconnectBuffers();
        for (const auto& gps: m_query_to_gps_) {
            for (const auto& [_, key_and_gp]: gps) { key_and_gp.second->locked_for_test = false; }  // unlock GPs
        }

        ERL_TRACY_PLOT("[test]gp_search_time (ms)", gp_search_time);
        ERL_TRACY_PLOT("[test]gp_train_time (ms)", gp_train_time);
        ERL_TRACY_PLOT("[test]gp_test_time (ms)", gp_test_time);

        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::SearchCandidateGps(const Eigen::Ref<const Positions>& positions_in) {
        ERL_BLOCK_TIMER();

        const double search_area_half_size = m_setting_->test_query->search_area_half_size;
        const Aabb map_boundary = m_surface_mapping_->GetMapBoundary();
        Aabb query_area(  // current queried search area
            positions_in.rowwise().minCoeff().array() - search_area_half_size,
            positions_in.rowwise().maxCoeff().array() + search_area_half_size);
        Aabb area = query_area.Intersection(map_boundary);                                 // current area of interest
        m_candidate_gps_.clear();                                                          // clear the buffer
        while (area.sizes().prod() > 0) {                                                  // search until the intersection is empty
            KeyVector clusters_in_aabb = m_surface_mapping_->CollectClustersInAabb(area);  // search for clusters in the area
            for (const auto& cluster_key: clusters_in_aabb) {
                if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                    const auto& gp = it->second;
                    if (!gp->active) { continue; }  // gp is inactive (e.g. due to no training data)
                    gp->locked_for_test = true;     // lock the GP for testing
                    m_candidate_gps_.emplace_back(it->first, gp);
                }
            }
            if (!m_candidate_gps_.empty()) { break; }                                         // found at least one GP
            area = Aabb(area.center, 2.0 * area.half_sizes);                                  // double the size of area
            Aabb new_area = area.Intersection(map_boundary);                                  // new area of interest
            if ((area.min() == new_area.min()) && (area.max() == new_area.max())) { break; }  // area did not change
            area = std::move(new_area);                                                       // update area
        }
        if (!m_candidate_gps_.empty()) {
            Positions positions(Positions::RowsAtCompileTime, m_candidate_gps_.size());
            for (std::size_t i = 0; i < m_candidate_gps_.size(); ++i) { positions.col(static_cast<long>(i)) = m_candidate_gps_[i].second->position; }
            m_kd_tree_candidate_gps_ = std::make_shared<KdTree>(std::move(positions));  // build kdtree of candidate GPs to allow fast search
        }
        ERL_INFO("{} candidate GPs found.", m_candidate_gps_.size());
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::SearchGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (!m_surface_mapping_->Ready()) { return; }
        const Aabb map_boundary = m_surface_mapping_->GetMapBoundary();

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            const Position test_position = m_test_buffer_.positions->col(i);
            std::vector<std::pair<Dtype, KeyGpPair>>& gps = m_query_to_gps_[i];
            gps.clear();
            constexpr int kMaxNumGps = 16;
            gps.reserve(kMaxNumGps);

            auto search_area_half_size = static_cast<Dtype>(m_setting_->test_query->search_area_half_size);
            Aabb query_area(test_position, search_area_half_size);
            Aabb search_area = map_boundary.Intersection(query_area);

            if (m_kd_tree_candidate_gps_ != nullptr) {  // search gps from the common buffer at first to save time
                constexpr long kMaxNumNeighbors = 32;   // use kdtree to search for 32 nearest GPs
                Eigen::VectorXl indices = Eigen::VectorXl::Constant(kMaxNumNeighbors, -1);
                Eigen::VectorXd squared_distances(kMaxNumNeighbors);
                m_kd_tree_candidate_gps_->Knn(kMaxNumNeighbors, test_position, indices, squared_distances);
                for (long j = 0; j < kMaxNumNeighbors; ++j) {
                    const long& index = indices[j];
                    if (index < 0) { break; }  // no more GPs
                    const auto& key_and_gp = m_candidate_gps_[index];
                    if (!key_and_gp.second->Intersects(search_area.center, search_area.half_sizes)) { continue; }
                    gps.emplace_back(std::sqrt(squared_distances[j]), key_and_gp);
                    if (gps.size() >= kMaxNumGps) { break; }  // found enough GPs
                }
            } else {
                // no kdtree, search all GPs in the common buffer, which is slow
                gps.reserve(m_candidate_gps_.size());  // request more memory
                for (const auto& key_and_gp: m_candidate_gps_) {
                    if (!key_and_gp.second->Intersects(search_area.center, search_area.half_sizes)) { continue; }
                    gps.emplace_back((key_and_gp.second->position - test_position).norm(), key_and_gp);
                }
            }

            if (gps.empty()) {               // no gp found
                search_area_half_size *= 2;  // double search area size
                query_area = Aabb(test_position, search_area_half_size);
                Aabb new_area = map_boundary.Intersection(query_area);
                if ((search_area.min() == new_area.min()) && (search_area.max() == new_area.max())) { continue; }  // no need to search again
                search_area = std::move(new_area);                                                                 // update area
            } else {
                continue;  // found at least one gp
            }

            while (gps.empty() && search_area.sizes().prod() > 0) {  // still no gp found, maybe the test position is on the query boundary
                KeyVector clusters_in_aabb = m_surface_mapping_->CollectClustersInAabb(search_area);  // search for clusters in the area
                for (const auto& cluster_key: clusters_in_aabb) {                                     // search for clusters in the area
                    if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                        const auto& gp = it->second;
                        if (!gp->active) { continue; }  // gp is inactive (e.g. due to no training data)
                        gp->locked_for_test = true;     // lock the GP for testing
                        gps.emplace_back((gp->position - test_position).norm(), std::make_pair(it->first, gp));
                    }
                }
                if (!gps.empty()) { break; }  // found at least one gp
                search_area_half_size *= 2;   // double search area size
                query_area = Aabb(test_position, search_area_half_size);
                Aabb new_area = map_boundary.Intersection(query_area);
                if ((search_area.min() == new_area.min()) && (search_area.max() == new_area.max())) { break; }  // no need to search again
                search_area = std::move(new_area);                                                              // update area
            }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (!m_surface_mapping_->Ready()) { return; }  // surface mapping is not ready

        const bool use_gp_covariance = m_setting_->test_query->use_gp_covariance;
        const bool compute_covariance = m_setting_->test_query->compute_covariance;
        const int num_neighbor_gps = m_setting_->test_query->num_neighbor_gps;
        const bool use_smallest = m_setting_->test_query->use_smallest;
        const auto max_test_valid_distance_var = static_cast<Dtype>(m_setting_->test_query->max_test_valid_distance_var);
        const auto softmin_temperature = static_cast<Dtype>(m_setting_->test_query->softmin_temperature);
        const auto offset_distance = static_cast<Dtype>(m_setting_->offset_distance);
        const bool use_occ_sign = m_setting_->use_occ_sign;

        std::vector<std::size_t> gp_indices;
        Eigen::Matrix<Dtype, Dim + 1, Eigen::Dynamic> fs(Dim + 1, num_neighbor_gps);  // f, fGrad1, fGrad2, fGrad3
        Variances variances(Variances::RowsAtCompileTime, num_neighbor_gps);          // variances of f, fGrad1, fGrad2, fGrad3
        Covariances covariances(Covariances::RowsAtCompileTime, num_neighbor_gps);    // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)
        Eigen::MatrixXd no_variance;
        Eigen::MatrixXd no_covariance;
        std::vector<std::pair<long, long>> tested_idx;
        tested_idx.reserve(num_neighbor_gps);

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            Dtype& distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);
            auto& used_gps = m_query_used_gps_[i];

            distance_out = 0.;
            gradient_out.setZero();
            variances.setConstant(1e6);
            if (compute_covariance) { covariances.setConstant(1e6); }
            used_gps.clear();

            auto& gps = m_query_to_gps_[i];
            if (gps.empty()) { continue; }

            const Position test_position = m_test_buffer_.positions->col(i);
            gp_indices.resize(gps.size());
            std::iota(gp_indices.begin(), gp_indices.end(), 0);
            if (gps.size() > 1) {  // sort GPs by distance to the test position
                std::stable_sort(gp_indices.begin(), gp_indices.end(), [&gps](const size_t i1, const size_t i2) { return gps[i1].first < gps[i2].first; });
            }
            gp_indices.resize(std::min(gps.size(), static_cast<std::size_t>(num_neighbor_gps)));

            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            for (std::size_t& j: gp_indices) {  // call selected GPs for inference
                const auto& gp = gps[j].second.second;
                if (!gp->active || !gp->edf_gp->IsTrained()) { continue; }  // skip inactive / untrained GPs
                if (!gp->Test(
                        test_position,
                        fs.col(cnt),
                        variances.col(cnt),
                        covariances.col(cnt),
                        offset_distance,
                        softmin_temperature,
                        use_gp_covariance,
                        compute_covariance)) {
                    continue;
                }

                tested_idx.emplace_back(cnt++, j);
                if (!use_smallest) {
                    if ((!need_weighted_sum) && (gp_indices.size() > 1) && (variances(0, cnt) > max_test_valid_distance_var)) { need_weighted_sum = true; }
                    if (!need_weighted_sum) { break; }
                }
            }
            if (tested_idx.empty()) { continue; }

            if (use_smallest && tested_idx.size() > 1) {
                std::sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return fs(0, a.first) < fs(0, b.first); });
                need_weighted_sum = false;
                fs.col(0) = fs.col(tested_idx[0].first);
                variances.col(0) = variances.col(tested_idx[0].first);
                if (compute_covariance) { covariances.col(0) = covariances.col(tested_idx[0].first); }
            }

            // sort the results by distance variance
            if (need_weighted_sum && tested_idx.size() > 1) {
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return variances(0, a.first) < variances(0, b.first); });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, tested_idx[0].first) * fs(0, tested_idx[1].first) < 0) { need_weighted_sum = false; }
            }

            // store the result
            if (need_weighted_sum) {
                if (variances(0, tested_idx[0].first) < max_test_valid_distance_var) {  // the first result is good enough
                    auto j = tested_idx[0].first;                                       // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs.col(j).template tail<Dim>();
                    variance_out << variances.col(j);
                    if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariances.col(j); }
                    used_gps.emplace_back(gps[tested_idx[0].second].second.second);
                } else {  // do weighted sum
                    ComputeWeightedSum<Dim>(i, tested_idx, fs, variances, covariances);
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs.col(0).template tail<Dim>();
                variance_out << variances.col(0);
                if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariances.col(0); }
                used_gps.emplace_back(gps[tested_idx[0].second].second.second);
            }
            gradient_out.normalize();
            if (use_occ_sign && distance_out > 0 && m_query_signs_[i] < 0) { distance_out = -distance_out; }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    template<int D>
    std::enable_if_t<D == 3, void>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::ComputeWeightedSum(
        const uint32_t i,
        const std::vector<std::pair<long, long>>& tested_idx,
        Eigen::Matrix<Dtype, 4, Eigen::Dynamic>& fs,
        Variances& variances,
        Covariances& covariances) {

        const Dtype max_test_valid_distance_var = m_setting_->test_query->max_test_valid_distance_var;
        const bool compute_covariance = m_setting_->test_query->compute_covariance;
        auto& gps = m_query_to_gps_[i];

#if defined(ERL_GP_SDF_MAPPING_TRACK_QUERY_GPS)
        auto& used_gps = m_query_used_gps_[i];
        used_gps.clear();
#endif

        // pick the best <= 4 results to do weighted sum
        const std::size_t m = std::min(tested_idx.size(), 4ul);
        Dtype w_sum = 0;
        Eigen::Vector4d f = Eigen::Vector4d::Zero();
        Eigen::Vector4d variance_f = Eigen::Vector4d::Zero();
        Eigen::Vector6d covariance_f = Eigen::Vector6d::Zero();
        for (std::size_t k = 0; k < m; ++k) {
            const long jk = tested_idx[k].first;
            const Dtype w = 1.0 / (variances(0, jk) - max_test_valid_distance_var);
            w_sum += w;
            f += fs.col(jk) * w;
            variance_f += variances.col(jk) * w;

            used_gps.emplace_back(gps[tested_idx[k].second].second.second);

            if (compute_covariance) { covariance_f += covariances.col(jk) * w; }
        }
        f /= w_sum;

        (*m_test_buffer_.distances)[i] = f[0];                   // distance
        m_test_buffer_.gradients->col(i) << f.tail<3>();         // gradient
        m_test_buffer_.variances->col(i) << variance_f / w_sum;  // variance
        if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance_f / w_sum; }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    template<int D>
    std::enable_if_t<D == 2, void>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::ComputeWeightedSum(
        const uint32_t i,
        const std::vector<std::pair<long, long>>& tested_idx,
        Eigen::Matrix<Dtype, 3, Eigen::Dynamic>& fs,
        Variances& variances,
        Covariances& covariances) {

        const bool compute_covariance = m_setting_->test_query->compute_covariance;
        auto& gps = m_query_to_gps_[i];

        // pick the best two results to do weighted sum
        const long j1 = tested_idx[0].first;
        const long j2 = tested_idx[1].first;
        const Dtype w1 = variances(0, j1) - m_setting_->test_query->max_test_valid_distance_var;
        const Dtype w2 = variances(0, j2) - m_setting_->test_query->max_test_valid_distance_var;
        const Dtype w12 = w1 + w2;
        // clang-format off
        (*m_test_buffer_.distances)[i] = (fs(0, j1) * w2 + fs(0, j2) * w1) / w12;     // distance
        m_test_buffer_.gradients->col(i) << (fs(1, j1) * w2 + fs(1, j2) * w1) / w12,  // gradient
                                            (fs(2, j1) * w2 + fs(2, j2) * w1) / w12;
        m_test_buffer_.variances->col(i) <<                                           // variance
            (variances(0, j1) * w2 + variances(0, j2) * w1) / w12,
            (variances(1, j1) * w2 + variances(1, j2) * w1) / w12,
            (variances(2, j1) * w2 + variances(2, j2) * w1) / w12;
        if (compute_covariance) {
            m_test_buffer_.covariances->col(i) <<
                (covariances(0, j1) * w2 + covariances(0, j2) * w1) / w12,
                (covariances(1, j1) * w2 + covariances(1, j2) * w1) / w12,
                (covariances(2, j1) * w2 + covariances(2, j2) * w1) / w12;
        }
        // clang-format on

#if defined(ERL_GP_SDF_MAPPING_TRACK_QUERY_GPS)
        auto& used_gps = m_query_used_gps_[i];
        used_gps.clear();
        used_gps.emplace_back(gps[tested_idx[0].second].second.second);
        used_gps.emplace_back(gps[tested_idx[1].second].second.second);
#endif
    }
}  // namespace erl::sdf_mapping
