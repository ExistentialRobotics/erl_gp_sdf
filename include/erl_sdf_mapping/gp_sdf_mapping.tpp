#pragma once

#include "erl_common/tracy.hpp"

namespace erl::sdf_mapping {
    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestBuffer::ConnectBuffers(
        const Eigen::Ref<const Positions> &positions_in,
        Distances &distances_out,
        Gradients &gradients_out,
        Variances &variances_out,
        Covariances &covariances_out,
        const bool compute_covariance) {
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
        this->positions = std::make_unique<Eigen::Ref<const Positions> >(positions_in);
        this->distances = std::make_unique<Eigen::Ref<Distances> >(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<Gradients> >(gradients_out);
        this->variances = std::make_unique<Eigen::Ref<Variances> >(variances_out);
        this->covariances = std::make_unique<Eigen::Ref<Covariances> >(covariances_out);
        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestBuffer::DisconnectBuffers() {
        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestBuffer::PrepareGpBuffer(
        const long num_queries,
        const long num_neighbor_gps) {
        // (num_queries, 2 * Dim + 1, num_neighbor_gps)
        const long rows = num_neighbor_gps * (2 * Dim + 1);
        if (gp_buffer.rows() < rows || gp_buffer.cols() < num_queries) {
            gp_buffer.setConstant(rows, num_queries, 0.0f);
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::GpSdfMapping(
        std::shared_ptr<Setting> setting,
        std::shared_ptr<SurfaceMapping> surface_mapping)
        : m_setting_(std::move(setting)),
          m_surface_mapping_(std::move(surface_mapping)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1.");
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    std::shared_ptr<const typename GpSdfMapping<Dtype, Dim, SurfaceMapping>::Setting>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &ranges) {
        ERL_TRACY_FRAME_MARK_START();
        bool success = false;
        double time_budget_us = 1e6 / m_setting_->update_hz;  // us
        double total_update_time = 0;
        double surf_mapping_time = 0;
        double sdf_gp_update_time = 0;
        {
            ERL_BLOCK_TIMER_MSG_TIME("GpSdfMapping Update", total_update_time);
            {
                ERL_BLOCK_TIMER_MSG_TIME("Surface mapping update", surf_mapping_time);
                success = m_surface_mapping_->Update(rotation, translation, ranges);
            }
            time_budget_us -= surf_mapping_time * 1000;  // us

            if (success) {
                ERL_BLOCK_TIMER_MSG_TIME("Update SDF GPs", sdf_gp_update_time);
                UpdateGpSdf(time_budget_us);
            }
        }

        ERL_TRACY_PLOT("[update]surf_mapping_time (ms)", surf_mapping_time);
        ERL_TRACY_PLOT("[update]sdf_gp_update_time (ms)", sdf_gp_update_time);
        ERL_TRACY_PLOT("[update]total_update_time (ms)", total_update_time);
        ERL_TRACY_PLOT("#clusters", static_cast<long>(m_gp_map_.size()));
        ERL_TRACY_PLOT("#queued clusters", static_cast<long>(m_cluster_queue_keys_.size()));
        ERL_TRACY_PLOT("#training clusters.size()", static_cast<long>(m_clusters_to_train_.size()));
        ERL_TRACY_PLOT_CONFIG(
            "m_gp_map_.memory_usage",
            tracy::PlotFormatType::Memory,
            true,
            true,
            0);
        ERL_TRACY_PLOT("m_gp_map_.memory_usage", static_cast<long>([&] {
                           std::size_t gps_memory_usage = 0;
                           for (const auto &[key, gp]: m_gp_map_) {
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
        ERL_TRACY_FRAME_MARK_START();
        ERL_BLOCK_TIMER_MSG("UpdateGpSdf");  // start timer

        CollectChangedClusters();
        UpdateClusterQueue();

        // train GPs if we still have time
        const double dt = timer.Elapsed<double, std::micro>();
        time_budget_us -= dt;
        if (time_budget_us > 2.0f * m_train_gp_time_us_) {
            // CRITICAL SECTION: access m_clusters_to_train_ and m_gp_map_
            auto lock = GetLockGuard();
            auto max_num_clusters = static_cast<std::size_t>(  //
                std::floor(time_budget_us / m_train_gp_time_us_));
            max_num_clusters = std::min(max_num_clusters, m_cluster_queue_.size());
            m_clusters_to_train_.clear();
            while (!m_cluster_queue_.empty() && m_clusters_to_train_.size() < max_num_clusters) {
                Key cluster_key = m_cluster_queue_.top().key;
                m_cluster_queue_.pop();
                m_cluster_queue_keys_.erase(cluster_key);
                auto gp = m_gp_map_.at(cluster_key);
                if (!gp->active) { continue; }  // skip inactive GP
                m_clusters_to_train_.emplace_back(cluster_key, gp);
            }
            TrainGps();  // m_surface_mapping_ is locked in TrainGps
        }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    [[nodiscard]] bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Test(
        const Eigen::Ref<const Positions> &positions_in,
        Distances &distances_out,
        Gradients &gradients_out,
        Variances &variances_out,
        Covariances &covariances_out) {
        {
            auto lock = GetLockGuard();  // CRITICAL SECTION: access m_gp_map_
            if (m_gp_map_.empty()) {
                ERL_WARN("No GPs available for testing.");
                return false;
            }
        }

        if (positions_in.cols() == 0) {
            ERL_WARN("No query positions provided.");
            return false;
        }
        const Dtype scaling = m_surface_mapping_->GetScaling();
        const Positions positions_scaled =
            scaling == 1 ? positions_in : positions_in.array() * scaling;
        if (!m_test_buffer_.ConnectBuffers(  // allocate memory for test results
                positions_scaled,
                distances_out,
                gradients_out,
                variances_out,
                covariances_out,
                m_setting_->test_query.compute_covariance)) {
            ERL_WARN("Failed to connect test buffers.");
            return false;
        }

        const uint32_t num_queries = positions_scaled.cols();
        const uint32_t num_threads = std::min(
            std::min(m_setting_->num_threads, std::thread::hardware_concurrency()),
            num_queries);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = num_queries / num_threads;
        const std::size_t leftover = num_queries - batch_size * num_threads;
        std::size_t start_idx, end_idx;
        {
            // CRITICAL SECTION: access m_surface_mapping_
            auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
            m_map_boundary_ = m_surface_mapping_->GetMapBoundary();
        }

        // If we iterate through the clusters for each query position separately, it takes too much
        // CPU time. Instead, we collect all GPs in the area of all query positions and then assign
        // them to the query positions. Some query positions may not have any GPs from
        // m_candidate_gps_. We need to search for them separately. We are sure that
        // m_candidate_gps_ is not empty because m_gp_map_ is not empty. Experiments show that this
        // can reduce the search time by at most 50%. For 15k crowded query positions, the search
        // time is reduced from ~60 ms to ~30 ms. Another important trick is to use a KdTree to
        // search for candidate GPs for each query position. Knn search is much faster, and the
        // result is sorted by distance. Experiments show that this knn search can reduce the search
        // time further to 2 ms.
        // Search for candidate GPs, collected GPs will be locked for testing.
        SearchCandidateGps(positions_scaled);

        if (m_candidate_gps_.empty()) { return false; }  // no candidate GPs

        // Train any updated GPs
        // we need to train call candidate GPs before we select them for testing.
        if (!m_cluster_queue_.empty()) {
            auto lock = GetLockGuard();  // CRITICAL SECTION: access m_clusters_to_train_
            const bool retrain_outdated = m_setting_->test_query.retrain_outdated;
            m_clusters_to_train_.clear();
            for (auto &key_and_gp: m_candidate_gps_) {
                ERL_DEBUG_ASSERT(key_and_gp.second->active, "GP is not active");
                // If retrain_outdated is true, we train the GP if it is outdated or not trained.
                // If retrain_outdated is false, we train the GP only if it is not trained.
                if ((!retrain_outdated || !key_and_gp.second->outdated) &&
                    key_and_gp.second->IsTrained()) {
                    continue;
                }
                m_clusters_to_train_.push_back(key_and_gp);
            }
            TrainGps();  // m_surface_mapping_ is locked in TrainGps
        }

        m_kdtree_candidate_gps_.reset();
        if (!m_candidate_gps_.empty()) {
            // build kdtree of candidate GPs to allow fast search.
            // remove inactive GPs and collect GP positions
            Positions gp_positions(Dim, m_candidate_gps_.size());
            std::vector<KeyGpPair> new_candidate_gps;
            new_candidate_gps.reserve(m_candidate_gps_.size());
            for (auto &[key, gp]: m_candidate_gps_) {
                if (!gp->active || !gp->IsTrained()) {
                    gp->locked_for_test = false;  // unlock GPs
                    continue;
                }
                gp_positions.col(static_cast<long>(new_candidate_gps.size())) = gp->position;
                new_candidate_gps.emplace_back(key, gp);
            }
            m_candidate_gps_ = std::move(new_candidate_gps);
            gp_positions.conservativeResize(Dim, m_candidate_gps_.size());
            m_kdtree_candidate_gps_ = std::make_shared<KdTree>(std::move(gp_positions));
        }

        std::vector<std::vector<std::size_t> > no_gps_indices(num_threads);
        {
            m_query_to_gps_.clear();              // clear the previous query to GPs
            m_query_to_gps_.resize(num_queries);  // allocate memory for n threads
            if (num_queries == 1) {
                SearchGpThread(0, 0, 1, no_gps_indices[0]);  // save time on thread creation
            } else {
                start_idx = 0;
                for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                    end_idx = start_idx + batch_size;
                    if (thread_idx < leftover) { end_idx++; }
                    threads.emplace_back(
                        &GpSdfMapping::SearchGpThread,
                        this,
                        thread_idx,
                        start_idx,
                        end_idx,
                        std::ref(no_gps_indices[thread_idx]));
                    start_idx = end_idx;
                }
                for (auto &thread: threads) { thread.join(); }
                threads.clear();
                for (uint32_t i = 1; i < num_threads; i++) {
                    no_gps_indices[0].insert(
                        no_gps_indices[0].end(),
                        no_gps_indices[i].begin(),
                        no_gps_indices[i].end());
                }
            }
            // Some query positions may not have any GPs from m_candidate_gps_.
            // We need to search for them separately.
            SearchGpFallback(no_gps_indices[0]);
        }

        if (!no_gps_indices[0].empty()) {
            auto lock = GetLockGuard();  // CRITICAL SECTION: access m_clusters_to_train_
            m_clusters_to_train_.clear();
            KeySet keys;
            for (const std::size_t &i: no_gps_indices[0]) {
                for (auto &[distance, key_and_gp]: m_query_to_gps_[i]) {
                    // skip if the key is already in the set
                    if (!keys.insert(key_and_gp.first).second) { continue; }
                    if (const auto &gp = key_and_gp.second; !gp->active || gp->IsTrained()) {
                        continue;  // GP is inactive or already trained
                    }
                    m_clusters_to_train_.emplace_back(key_and_gp);
                }
            }
            TrainGps();  // m_surface_mapping_ is locked in TrainGps
        }

        bool surface_mapping_sign = false;
        const SignMethod sign_method = m_setting_->sdf_gp->sign_method;
        m_query_signs_.setConstant(num_queries, 1.0f);
        if (const auto &hybrid_sign_methods = m_setting_->sdf_gp->hybrid_sign_methods;
            sign_method == kExternal ||
            (sign_method == kHybrid &&
             (hybrid_sign_methods.first == kExternal || hybrid_sign_methods.second == kExternal))) {
            // collect the sign from the surface mapping, which is not thread-safe
            // CRITICAL SECTION: access m_surface_mapping_
            auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
            surface_mapping_sign = m_surface_mapping_->IsInFreeSpace(positions_in, m_query_signs_);
            ERL_WARN_COND(!surface_mapping_sign, "Failed to get sign from the surface mapping.");
        }

        if (m_setting_->test_query.use_global_buffer) {
            m_test_buffer_.PrepareGpBuffer(num_queries, m_setting_->test_query.num_neighbor_gps);
        }

        // Compute the inference result for each query position
        m_query_used_gps_.clear();
        m_query_used_gps_.resize(num_queries);
        if (num_queries == 1) {
            TestGpThread(0, 0, 1);  // save time on thread creation
        } else {
            start_idx = 0;
            for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                end_idx = start_idx + batch_size;
                if (thread_idx < leftover) { ++end_idx; }
                threads.emplace_back(
                    &GpSdfMapping::TestGpThread,
                    this,
                    thread_idx,
                    start_idx,
                    end_idx);
                start_idx = end_idx;
            }
            for (auto &thread: threads) { thread.join(); }
            threads.clear();
        }

        m_test_buffer_.DisconnectBuffers();

        // unlock GPs
        for (auto &[key, gp]: m_candidate_gps_) { gp->locked_for_test = false; }
        if (!no_gps_indices[0].empty()) {
            for (const std::size_t &i: no_gps_indices[0]) {
                for (auto &[distance, key_and_gp]: m_query_to_gps_[i]) {
                    key_and_gp.second->locked_for_test = false;
                }
            }
        }

        // scaling
        if (scaling != 1) {
            distances_out /= scaling;
            variances_out /= (scaling * scaling);
            covariances_out.template topRows<Dim>() /= scaling;  // cov(grad_x, d)
        }

        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<GpSdfMapping> token_function_pairs = {
            {
                "setting",
                [](const GpSdfMapping *self, std::ostream &stream) {
                    return self->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "surface_mapping",
                [](const GpSdfMapping *self, std::ostream &stream) {
                    return self->m_surface_mapping_->Write(stream) && stream.good();
                },
            },
            {
                "gp_map",
                [](const GpSdfMapping *self, std::ostream &stream) {
                    const std::size_t n = self->m_gp_map_.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (auto &[key, gp]: self->m_gp_map_) {
                        stream.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        bool has_gp = gp != nullptr;
                        stream.write(reinterpret_cast<const char *>(&has_gp), sizeof(bool));
                        if (has_gp && !gp->Write(stream)) { return false; }
                    }
                    return stream.good();
                },
            },
            // m_affected_clusters_ is temporary data.
            {
                "cluster_queue_keys",
                [](const GpSdfMapping *self, std::ostream &stream) {
                    const std::size_t n = self->m_cluster_queue_keys_.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, handle]: self->m_cluster_queue_keys_) {
                        stream.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        stream.write(
                            reinterpret_cast<const char *>(&(*handle).time_stamp),
                            sizeof(long));
                    }
                    return stream.good();
                },
            },
            // m_cluster_queue_ can be reconstructed from m_cluster_queue_keys_.
            // m_clusters_to_train_ is temporary data.
            {
                "train_gp_time_us",
                [](const GpSdfMapping *self, std::ostream &stream) {
                    return stream.write(
                               reinterpret_cast<const char *>(&self->m_train_gp_time_us_),
                               sizeof(double)) &&
                           stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<GpSdfMapping> token_function_pairs = {
            {
                "setting",
                [](GpSdfMapping *self, std::istream &stream) {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "surface_mapping",
                [](GpSdfMapping *self, std::istream &stream) {
                    return self->m_surface_mapping_->Read(stream) && stream.good();
                },
            },
            {
                "gp_map",
                [](GpSdfMapping *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_gp_map_.clear();
                    self->m_gp_map_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        stream.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        auto [it, inserted] = self->m_gp_map_.try_emplace(key, nullptr);
                        if (!inserted) {
                            ERL_WARN("Duplicate GP key: {}.", static_cast<std::string>(key));
                            return false;
                        }
                        bool has_gp;
                        stream.read(reinterpret_cast<char *>(&has_gp), sizeof(bool));
                        if (has_gp) {
                            it->second = std::make_shared<SdfGp>(self->m_setting_->sdf_gp);
                            if (!it->second->Read(stream)) {
                                ERL_WARN(
                                    "Failed to read GP of key {}.",
                                    static_cast<std::string>(key));
                                return false;
                            }
                        }
                    }
                    return stream.good();
                },
            },
            {
                "cluster_queue_keys",
                [](GpSdfMapping *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_cluster_queue_keys_.clear();
                    self->m_cluster_queue_keys_.reserve(n);
                    self->m_cluster_queue_.clear();
                    self->m_cluster_queue_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        stream.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        long time_stamp;
                        stream.read(reinterpret_cast<char *>(&time_stamp), sizeof(long));
                        auto [it, inserted] = self->m_cluster_queue_keys_.try_emplace(
                            key,
                            self->m_cluster_queue_.push({time_stamp, key}));
                        if (!inserted) {
                            ERL_WARN("Duplicate cluster key: {}.", static_cast<std::string>(key));
                            return false;
                        }
                    }
                    return stream.good();
                },
            },
            {
                "train_gp_time_us",
                [](GpSdfMapping *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->m_train_gp_time_us_),
                        sizeof(double));
                    return stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    bool
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::operator==(const GpSdfMapping &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) {
            return false;
        }
        if (m_surface_mapping_ == nullptr && other.m_surface_mapping_ != nullptr) { return false; }
        if (m_surface_mapping_ != nullptr && (other.m_surface_mapping_ == nullptr ||
                                              *m_surface_mapping_ != *other.m_surface_mapping_)) {
            return false;
        }
        if (m_gp_map_.size() != other.m_gp_map_.size()) { return false; }
        for (const auto &[key, gp]: m_gp_map_) {
            auto it = other.m_gp_map_.find(key);
            if (it == other.m_gp_map_.end()) { return false; }
            const auto &[other_key, other_gp] = *it;
            if (key != other_key) { return false; }
            if (gp == nullptr && other_gp != nullptr) { return false; }
            if (gp != nullptr && (other_gp == nullptr || *gp != *other_gp)) { return false; }
        }
        if (m_cluster_queue_keys_.size() != other.m_cluster_queue_keys_.size()) { return false; }
        for (const auto &[key, handle]: m_cluster_queue_keys_) {
            auto it = other.m_cluster_queue_keys_.find(key);
            if (it == other.m_cluster_queue_keys_.end()) { return false; }
            const auto &[other_key, other_handle] = *it;
            if (key != other_key) { return false; }
            if ((*handle).time_stamp != (*other_handle).time_stamp) { return false; }
        }
        if (m_train_gp_time_us_ != other.m_train_gp_time_us_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    std::lock_guard<std::mutex>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::GetLockGuard() {
        return std::lock_guard(m_mutex_);
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::CollectChangedClusters() {
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        const Dtype area_half_size = cluster_size * m_setting_->gp_sdf_area_scale * 0.5f;

        // CRITICAL SECTION: access m_surface_mapping_
        auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        const KeySet &changed_clusters = m_surface_mapping_->GetChangedClusters();
        KeySet clusters(changed_clusters);
        for (const auto &cluster_key: changed_clusters) {
            const Aabb area(m_surface_mapping_->GetClusterCenter(cluster_key), area_half_size);
            m_surface_mapping_->IterateClustersInAabb(area, [&clusters](const Key &key) {
                clusters.insert(key);
            });
        }
        m_affected_clusters_.clear();
        m_affected_clusters_.insert(m_affected_clusters_.end(), clusters.begin(), clusters.end());
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::UpdateClusterQueue() {
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        const Dtype area_half_size = cluster_size * m_setting_->gp_sdf_area_scale * 0.5f;

        auto lock = GetLockGuard();  // CRITICAL SECTION: access m_gp_map_
        long cnt_new_gps = 0;
        for (const auto &cluster_key: m_affected_clusters_) {
            if (auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, nullptr);
                inserted || it->second->locked_for_test) {
                it->second = std::make_shared<SdfGp>(m_setting_->sdf_gp);
                it->second->Activate();
                it->second->position = m_surface_mapping_->GetClusterCenter(cluster_key);
                it->second->half_size = area_half_size;
                ++cnt_new_gps;
            } else {
                it->second->Activate();
                it->second->MarkOutdated();
            }
            // add the cluster to the queue
            const long time_stamp =
                std::chrono::high_resolution_clock::now().time_since_epoch().count();
            if (auto itr = m_cluster_queue_keys_.find(cluster_key);
                itr == m_cluster_queue_keys_.end()) {
                // new cluster
                m_cluster_queue_keys_.insert(
                    {cluster_key, m_cluster_queue_.push({time_stamp, cluster_key})});
            } else {
                auto &heap_key = itr->second;
                (*heap_key).time_stamp = time_stamp;
                m_cluster_queue_.increase(heap_key);
            }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TrainGps() {
        const std::size_t n = m_clusters_to_train_.size();
        if (n == 0) { return; }
        // CRITICAL SECTION: access m_surface_mapping_ in TrainGpThread
        auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        const auto t0 = std::chrono::high_resolution_clock::now();
        const uint32_t num_threads =
            std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = n / num_threads;
        const std::size_t left_over = n - batch_size * num_threads;
        std::size_t end_idx = 0;
        for (uint32_t t_idx = 0; t_idx < num_threads; ++t_idx) {
            std::size_t start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (t_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping::TrainGpThread, this, t_idx, start_idx, end_idx);
        }
        for (uint32_t t_idx = 0; t_idx < num_threads; ++t_idx) { threads[t_idx].join(); }
        threads.clear();
        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        time /= static_cast<double>(n);
        m_train_gp_time_us_ = m_train_gp_time_us_ * 0.1f + time * 0.9f;
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TrainGpThread(
        const uint32_t thread_idx,
        const std::size_t start_idx,
        const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        const Dtype sensor_noise = m_setting_->sensor_noise;
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        const Dtype area_half_size = cluster_size * m_setting_->gp_sdf_area_scale * 0.5f;
        const Dtype max_valid_gradient_var = m_setting_->max_valid_gradient_var;
        const Dtype invalid_position_var = m_setting_->invalid_position_var;
        const auto &surface_data_buffer = m_surface_mapping_->GetSurfaceDataBuffer();

        std::vector<std::pair<Dtype, std::size_t> > surface_data_indices;
        const auto sdf_gp_setting = m_setting_->sdf_gp;
        const auto max_num_samples = std::max(
            sdf_gp_setting->edf_gp->max_num_samples,
            sdf_gp_setting->sign_gp->max_num_samples);
        surface_data_indices.reserve(max_num_samples);
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto &[cluster_key, gp] = m_clusters_to_train_[i];
            ERL_DEBUG_ASSERT(gp->active, "GP is not active");
            // collect surface data in the area
            const Aabb area(gp->position, area_half_size);
            surface_data_indices.clear();
            m_surface_mapping_->CollectSurfaceDataInAabb(area, surface_data_indices);
            if (surface_data_indices.empty()) {  // no surface data in the area
                gp->Deactivate();                // deactivate the GP if there is no training data
                continue;
            }
            gp->LoadSurfaceData(
                surface_data_indices,
                surface_data_buffer,
                sensor_noise,
                max_valid_gradient_var,
                invalid_position_var);
            gp->Train();
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::SearchCandidateGps(
        const Eigen::Ref<const Positions> &positions_in) {
        m_candidate_gps_.clear();

        Position query_area_min = positions_in.col(0);
        Position query_area_max = positions_in.col(0);
        for (long i = 1; i < positions_in.cols(); ++i) {
            query_area_min = query_area_min.cwiseMin(positions_in.col(i));
            query_area_max = query_area_max.cwiseMax(positions_in.col(i));
        }

        Dtype search_area_padding = m_setting_->test_query.search_area_half_size;
        Aabb area = m_map_boundary_.Intersection(
            {query_area_min.array() - search_area_padding,
             query_area_max.array() + search_area_padding});
        while (m_candidate_gps_.empty()) {
            // search until the intersection is empty
            if (area.IsValid()) {
                // valid area: min < max
                // CRITICAL SECTION: access m_surface_mapping_ and m_gp_map_
                auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
                auto lock = GetLockGuard();
                m_surface_mapping_->IterateClustersInAabb(area, [&](const Key &cluster_key) {
                    // search for clusters in the area
                    if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                        const auto &gp = it->second;
                        if (!gp->active) { return; }
                        gp->locked_for_test = true;  // lock the GP for testing
                        m_candidate_gps_.emplace_back(it->first, gp);
                    }
                });
            }
            if (!m_candidate_gps_.empty()) { break; }  // found at least one GP
            search_area_padding *= 2;                  // double search area size
            Aabb new_area = m_map_boundary_.Intersection(
                {query_area_min.array() - search_area_padding,
                 query_area_max.array() + search_area_padding});
            if (new_area.IsValid() && (area.min() == new_area.min()) &&
                (area.max() == new_area.max())) {
                break;  // the area did not change
            }
            area = std::move(new_area);  // update area
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::SearchGpThread(
        const uint32_t thread_idx,
        const std::size_t start_idx,
        const std::size_t end_idx,
        std::vector<std::size_t> &no_gps_indices) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (m_kdtree_candidate_gps_ == nullptr) { return; }  // no candidate GPs

        constexpr int kMaxNumGps = 16;
        constexpr long kMaxNumNeighbors = 32;  // use kdtree to search for 32 nearest GPs
        Eigen::VectorXl indices = Eigen::VectorXl::Constant(kMaxNumNeighbors, -1);
        Distances squared_distances(kMaxNumNeighbors);

        for (std::size_t i = start_idx; i < end_idx; ++i) {
            const Position test_position = m_test_buffer_.positions->col(i);
            std::vector<std::pair<Dtype, KeyGpPair> > &gps = m_query_to_gps_[i];

            Dtype search_area_half_size = m_setting_->test_query.search_area_half_size;

            gps.clear();
            gps.reserve(kMaxNumGps);
            indices.setConstant(-1);
            m_kdtree_candidate_gps_
                ->Knn(kMaxNumNeighbors, test_position, indices, squared_distances);
            for (long j = 0; j < kMaxNumNeighbors; ++j) {
                const long &index = indices[j];
                if (index < 0) { break; }  // no more GPs
                const auto &key_and_gp = m_candidate_gps_[index];
                if (!key_and_gp.second->Intersects(test_position, search_area_half_size)) {
                    continue;  // GP is not in the query area
                }
                key_and_gp.second->locked_for_test = true;  // lock the GP for testing
                gps.emplace_back(std::sqrt(squared_distances[j]), key_and_gp);
                if (gps.size() >= kMaxNumGps) { break; }  // found enough GPs
            }
            if (gps.empty()) { no_gps_indices.push_back(i); }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::SearchGpFallback(
        const std::vector<std::size_t> &no_gps_indices) {
        if (no_gps_indices.empty()) { return; }

        // CRITICAL SECTION: access m_surface_mapping_ and m_gp_map_
        auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        auto lock = GetLockGuard();

        ERL_WARN_COND(
            !no_gps_indices.empty(),
            "Run fallback search for {} query positions.",
            no_gps_indices.size());

#pragma omp parallel for default(none) shared(no_gps_indices)
        for (const std::size_t &i: no_gps_indices) {
            auto &gps = m_query_to_gps_[i];
            // failed to find GPs in the kd-tree, fall back to search clusters in the area
            // double search area size
            Dtype search_area_hs = 2 * m_setting_->test_query.search_area_half_size;
            const Position test_position = m_test_buffer_.positions->col(i);
            Aabb search_area = m_map_boundary_.Intersection({test_position, search_area_hs});
            while (gps.empty()) {
                // no gp found, maybe the test position is on the query boundary
                if (search_area.IsValid()) {
                    m_surface_mapping_->IterateClustersInAabb(
                        search_area,
                        [&](const Key &cluster_key) {
                            // search for clusters in the area
                            if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                                const auto &gp = it->second;
                                if (!gp->active) { return; }  // e.g., due to no training data
                                gp->locked_for_test = true;   // lock the GP for testing
                                gps.emplace_back(
                                    (gp->position - test_position).norm(),
                                    std::make_pair(it->first, gp));
                            }
                        });
                }
                if (!gps.empty()) { break; }  // found at least one gp
                search_area_hs *= 2;          // double search area size
                Aabb new_area = m_map_boundary_.Intersection({test_position, search_area_hs});
                if (new_area.IsValid() && (search_area.min() == new_area.min()) &&
                    (search_area.max() == new_area.max())) {
                    break;  // no need to search again
                }
                search_area = std::move(new_area);  // update area
            }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    void
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::TestGpThread(
        const uint32_t thread_idx,
        const std::size_t start_idx,
        const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        const auto &tq = m_setting_->test_query;
        const bool compute_gradient = tq.compute_gradient;
        const bool compute_gradient_variance = tq.compute_gradient_variance;
        const bool compute_covariance = tq.compute_covariance;
        const bool use_gp_covariance = tq.use_gp_covariance;
        const int num_neighbor_gps = tq.num_neighbor_gps;
        const bool use_smallest = tq.use_smallest;  // use the nearest GP
        const Dtype max_test_valid_distance_var = tq.max_test_valid_distance_var;

        // f, grad_f (by logGP), scaled normal (by normal GP)
        using FsType = Eigen::Matrix<Dtype, 2 * Dim + 1, Eigen::Dynamic>;
        FsType fs_local;
        Dtype *fs_ptr = nullptr;

        // variances of f, fGrad1, fGrad2, fGrad3
        Variances variances(Dim + 1, num_neighbor_gps);
        // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)
        Covariances covariances((Dim + 1) * Dim / 2, num_neighbor_gps);
        std::vector<std::pair<long, long> > tested_idx;  // (column index, gps index)
        tested_idx.reserve(num_neighbor_gps);

        std::vector<std::size_t> gp_indices;
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            // set up buffer
            if (m_test_buffer_.gp_buffer.size() == 0) {
                // use the local buffer
                if (fs_local.size() == 0) {
                    fs_local.resize(2 * Dim + 1, num_neighbor_gps);  // set up the local buffer
                    fs_ptr = fs_local.data();
                }
            } else {
                fs_ptr = m_test_buffer_.gp_buffer.col(i).data();  // use the global buffer
            }
            Eigen::Map<FsType> fs(fs_ptr, 2 * Dim + 1, num_neighbor_gps);

            // set up output
            Dtype &distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);
            auto &used_gps = m_query_used_gps_[i];

            // initialization
            distance_out = 0.;
            gradient_out.setZero();
            variances.setConstant(1e6);
            if (compute_covariance) { covariances.setConstant(1e6); }
            used_gps.fill(nullptr);

            auto &gps = m_query_to_gps_[i];  // [distance, (key, gp)]
            if (gps.empty()) { continue; }   // no GPs found for this query position

            // sort GPs by distance
            const Position test_position = m_test_buffer_.positions->col(i);
            gp_indices.resize(gps.size());
            std::iota(gp_indices.begin(), gp_indices.end(), 0);
            if (gps.size() > 1) {
                // sort GPs by distance to the test position
                std::stable_sort(
                    gp_indices.begin(),
                    gp_indices.end(),
                    [&gps](const size_t i1, const size_t i2) {
                        return gps[i1].first < gps[i2].first;
                    });
            }
            gp_indices.resize(std::min(gps.size(), static_cast<std::size_t>(num_neighbor_gps)));

            // test GPs
            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            long pos_cnt = 0;  // count the number of positive GPs
            for (const std::size_t &j: gp_indices) {
                // call selected GPs for inference
                const auto &gp = gps[j].second.second;              // (distance, (key, gp))
                if (!gp->active || !gp->IsTrained()) { continue; }  // skip inactive / untrained GPs
                if (!gp->Test(
                        test_position,
                        fs.col(cnt),
                        variances.col(cnt),
                        covariances.col(cnt),
                        m_query_signs_[i],
                        compute_gradient,
                        compute_gradient_variance,
                        compute_covariance,
                        use_gp_covariance)) {
                    continue;
                }
                if (fs(0, cnt) > 0) { ++pos_cnt; }  // count the number of positive GPs
                tested_idx.emplace_back(cnt++, j);
                if (use_smallest) { continue; }
                // the current gp prediction is not good enough,
                // we use more GPs to compute the result.
                if (!need_weighted_sum && gp_indices.size() > 1 &&
                    variances(0, cnt) > max_test_valid_distance_var) {
                    need_weighted_sum = true;
                }
                if (!need_weighted_sum) { break; }
            }
            if (tested_idx.empty()) { continue; }  // no successful GP test

            if (use_smallest && tested_idx.size() > 1) {
                // sort the results by distance
                std::sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool {
                    return fs(0, a.first) < fs(0, b.first);
                });
                need_weighted_sum = false;
                fs.col(0) = fs.col(tested_idx[0].first);
                variances.col(0) = variances.col(tested_idx[0].first);
                if (compute_covariance) {
                    covariances.col(0) = covariances.col(tested_idx[0].first);
                }
            }

            if (need_weighted_sum && tested_idx.size() > 1) {
                // sort the results by distance variance
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool {
                    return variances(0, a.first) < variances(0, b.first);
                });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, tested_idx[0].first) * fs(0, tested_idx[1].first) < 0) {
                    need_weighted_sum = false;
                }
            }

            // store the result
            if (need_weighted_sum) {
                if (variances(0, tested_idx[0].first) < max_test_valid_distance_var) {
                    // the first result is good enough
                    auto j = tested_idx[0].first;  // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs.col(j).template segment<Dim>(1);
                    variance_out << variances.col(j);
                    if (compute_covariance) {
                        m_test_buffer_.covariances->col(i) = covariances.col(j);
                    }
                    used_gps[0] = gps[tested_idx[0].second].second.second;
                } else {
                    // compute a weighted sum
                    ComputeWeightedSum<Dim>(i, tested_idx, fs, variances, covariances);
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs.col(0).template segment<Dim>(1);
                variance_out << variances.col(0);
                if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariances.col(0); }
                used_gps[0] = gps[tested_idx[0].second].second.second;
            }
            if (compute_gradient) { gradient_out.normalize(); }

            // flip the sign of the distance and gradient if necessary
            bool flip = false;
            if ((pos_cnt << 1) > cnt) {
                if (distance_out < 0) { flip = true; }
            } else if ((pos_cnt << 1) < cnt) {
                if (distance_out > 0) { flip = true; }
            }
            if (flip) {
                // flip the sign of the distance and gradient
                distance_out = -distance_out;
                if (compute_gradient) {
                    for (long j = 0; j < Dim; ++j) { gradient_out[j] = -gradient_out[j]; }
                }
            }
        }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    template<int D>
    std::enable_if_t<D == 3, void>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::ComputeWeightedSum(
        const uint32_t i,
        const std::vector<std::pair<long, long> > &tested_idx,
        Eigen::Ref<Eigen::Matrix<Dtype, 7, Eigen::Dynamic> > fs,
        Variances &variances,
        Covariances &covariances) {
        const Dtype max_test_valid_distance_var =
            m_setting_->test_query.max_test_valid_distance_var;
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        auto &gps = m_query_to_gps_[i];
        auto &used_gps = m_query_used_gps_[i];
        used_gps.fill(nullptr);

        // pick the best <= 4 results to compute the weighted sum.
        const std::size_t m = std::min(tested_idx.size(), 4ul);
        Dtype w_sum = 0;
        Eigen::Vector4<Dtype> f = Eigen::Vector4<Dtype>::Zero();
        Eigen::Vector4<Dtype> variance_f = Eigen::Vector4<Dtype>::Zero();
        Eigen::Vector<Dtype, 6> covariance_f = Eigen::Vector<Dtype, 6>::Zero();
        for (std::size_t k = 0; k < m; ++k) {
            const long jk = tested_idx[k].first;
            const Dtype w = 1.0f / (variances(0, jk) - max_test_valid_distance_var);
            w_sum += w;
            f += fs.col(jk).template head<4>() * w;
            variance_f += variances.col(jk) * w;
            used_gps[k] = gps[tested_idx[k].second].second.second;
            if (compute_covariance) { covariance_f += covariances.col(jk) * w; }
        }
        f /= w_sum;

        (*m_test_buffer_.distances)[i] = f[0];                     // distance
        m_test_buffer_.gradients->col(i) << f.template tail<3>();  // gradient
        m_test_buffer_.variances->col(i) << variance_f / w_sum;    // variance
        if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance_f / w_sum; }
    }

    template<typename Dtype, int Dim, typename SurfaceMapping>
    template<int D>
    std::enable_if_t<D == 2, void>
    GpSdfMapping<Dtype, Dim, SurfaceMapping>::ComputeWeightedSum(
        const uint32_t i,
        const std::vector<std::pair<long, long> > &tested_idx,
        Eigen::Ref<Eigen::Matrix<Dtype, 5, Eigen::Dynamic> > fs,
        Variances &variances,
        Covariances &covariances) {
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        auto &gps = m_query_to_gps_[i];

        // pick the best two results to do the weighted sum
        const long j1 = tested_idx[0].first;
        const long j2 = tested_idx[1].first;
        const Dtype w1 = variances(0, j1) - m_setting_->test_query.max_test_valid_distance_var;
        const Dtype w2 = variances(0, j2) - m_setting_->test_query.max_test_valid_distance_var;
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

        auto &used_gps = m_query_used_gps_[i];
        used_gps.fill(nullptr);
        used_gps[0] = gps[tested_idx[0].second].second.second;
        used_gps[1] = gps[tested_idx[1].second].second.second;
    }
}  // namespace erl::sdf_mapping
