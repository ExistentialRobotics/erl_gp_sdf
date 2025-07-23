#pragma once

#include "erl_common/block_timer.hpp"
#include "erl_common/template_helper.hpp"
#include "erl_common/tracy.hpp"
#include "erl_gp_sdf/gp_sdf_mapping_3d.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace erl::gp_sdf {

    template<typename Dtype>
    YAML::Node
    GpSdfMapping3D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node = GpSdfMappingBaseSetting<Dtype>::YamlConvertImpl::encode(setting);
        node["surface_mapping_type"] = setting.surface_mapping_type;
        node["surface_mapping_setting_type"] = setting.surface_mapping_setting_type;
        node["surface_mapping"] = setting.surface_mapping->AsYamlNode();
        return node;
    }

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::Setting::YamlConvertImpl::decode(const YAML::Node &node, Setting &setting) {
        if (!GpSdfMappingBaseSetting<Dtype>::YamlConvertImpl::decode(node, setting)) { return false; }
        setting.surface_mapping_type = node["surface_mapping_type"].as<std::string>();
        setting.surface_mapping_setting_type = node["surface_mapping_setting_type"].as<std::string>();
        using SettingBase = typename AbstractSurfaceMapping3D<Dtype>::Setting;
        setting.surface_mapping = common::YamlableBase::Create<SettingBase>(setting.surface_mapping_setting_type);
        if (setting.surface_mapping == nullptr) {
            ERL_WARN("Failed to decode surface_mapping of type: {}", setting.surface_mapping_setting_type);
            return false;
        }
        return setting.surface_mapping->FromYamlNode(node["surface_mapping"]);
    }

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::TestBuffer::ConnectBuffers(
        const Eigen::Ref<const Matrix3X> &positions_in,
        VectorX &distances_out,
        Matrix3X &gradients_out,
        Matrix4X &variances_out,
        Matrix6X &covariances_out,
        const bool compute_covariance) {

        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
        const long n = positions_in.cols();
        if (n == 0) return false;

        distances_out.resize(n);
        gradients_out.resize(3, n);
        variances_out.resize(4, n);
        if (compute_covariance) { covariances_out.resize(6, n); }
        this->positions = std::make_unique<Eigen::Ref<const Matrix3X>>(positions_in);
        this->distances = std::make_unique<Eigen::Ref<VectorX>>(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<Matrix3X>>(gradients_out);
        this->variances = std::make_unique<Eigen::Ref<Matrix4X>>(variances_out);
        this->covariances = std::make_unique<Eigen::Ref<Matrix6X>>(covariances_out);
        return true;
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::TestBuffer::DisconnectBuffers() {
        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
    }

    template<typename Dtype>
    GpSdfMapping3D<Dtype>::GpSdfMapping3D(std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), "setting is nullptr.")),
          m_surface_mapping_(
              AbstractSurfaceMapping::CreateSurfaceMapping<AbstractSurfaceMapping3D<Dtype>>(m_setting_->surface_mapping_type, m_setting_->surface_mapping)) {
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
    }

    template<typename Dtype>
    std::shared_ptr<const typename GpSdfMapping3D<Dtype>::Setting>
    GpSdfMapping3D<Dtype>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype>
    std::shared_ptr<AbstractSurfaceMapping3D<Dtype>>
    GpSdfMapping3D<Dtype>::GetSurfaceMapping() const {
        return m_surface_mapping_;
    }

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::Update(
        const Eigen::Ref<const Matrix3> &rotation,
        const Eigen::Ref<const Vector3> &translation,
        const Eigen::Ref<const MatrixX> &ranges) {

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

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::UpdateGpSdf(double time_budget_us) {
        ERL_BLOCK_TIMER();
        ERL_DEBUG_ASSERT(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1");

        double collect_clusters_time = 0;
        double queue_clusters_time = 0;
        double train_gps_time = 0;

        const uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        const std::shared_ptr<SurfaceMappingOctree<Dtype>> tree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_depth = tree->GetTreeDepth() - cluster_level;
        const Dtype cluster_size = tree->GetNodeSize(cluster_depth);
        const Dtype area_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2;

        // add affected clusters
        {
            ERL_BLOCK_TIMER_MSG_TIME("Collect affected clusters", collect_clusters_time);
            const KeySet &changed_clusters = m_surface_mapping_->GetChangedClusters();
            KeySet affected_clusters(changed_clusters);
            for (const auto &cluster_key: changed_clusters) {
                const Aabb area(tree->KeyToCoord(cluster_key, cluster_depth), area_half_size);
                for (auto it = tree->BeginTreeInAabb(area, cluster_depth), end = tree->EndTreeInAabb(); it != end; ++it) {
                    if (it->GetDepth() != cluster_depth) { continue; }
                    affected_clusters.insert(tree->AdjustKeyToDepth(it.GetKey(), cluster_depth));
                }
            }
            m_affected_clusters_.clear();
            m_affected_clusters_.insert(m_affected_clusters_.end(), affected_clusters.begin(), affected_clusters.end());
            ERL_INFO("Collect {} -> {} affected clusters", changed_clusters.size(), affected_clusters.size());
        }

        // put affected clusters into m_cluster_queue_
        {
            ERL_BLOCK_TIMER_MSG_TIME("Queue affected clusters", queue_clusters_time);

            long cnt_new_gps = 0;
            for (const auto &cluster_key: m_affected_clusters_) {
                if (auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, nullptr); inserted || it->second->locked_for_test) {
                    it->second = std::make_shared<Gp>();       // new GP is required
                    it->second->Activate(m_setting_->edf_gp);  // activate the GP
                    tree->KeyToCoord(cluster_key, cluster_depth, it->second->position);
                    it->second->half_size = area_half_size;
                    ++cnt_new_gps;
                } else {
                    it->second->Activate(m_setting_->edf_gp);
                }
                auto time_stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                if (auto itr = m_cluster_queue_keys_.find(cluster_key); itr == m_cluster_queue_keys_.end()) {  // new cluster
                    m_cluster_queue_keys_.insert({cluster_key, m_cluster_queue_.push({time_stamp, cluster_key})});
                } else {
                    auto &heap_key = itr->second;
                    (*heap_key).time_stamp = time_stamp;
                    m_cluster_queue_.increase(heap_key);
                }
            }
            ERL_INFO("Create {} new GPs when queuing clusters", cnt_new_gps);
        }

        // train GPs if we still have time
        const auto dt = timer.Elapsed<double, std::micro>();
        time_budget_us -= dt;
        ERL_INFO("Time spent: {} us, time budget: {} us", dt, time_budget_us);
        if (time_budget_us > 2.0 * m_train_gp_time_) {
            ERL_BLOCK_TIMER_MSG_TIME("Train GPs", train_gps_time);

            auto max_num_clusters_to_train = static_cast<std::size_t>(std::floor(time_budget_us / m_train_gp_time_));
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

        ERL_TRACY_PLOT("[update_sdf_gp]collect_clusters_time (ms)", collect_clusters_time);
        ERL_TRACY_PLOT("[update_sdf_gp]queue_clusters_time (ms)", queue_clusters_time);
        ERL_TRACY_PLOT("[update_sdf_gp]train_gps_time (ms)", train_gps_time);
    }

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::Test(
        const Eigen::Ref<const Matrix3X> &positions_in,
        VectorX &distances_out,
        Matrix3X &gradients_out,
        Matrix4X &variances_out,
        Matrix6X &covariances_out) {

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
        if (m_surface_mapping_ == nullptr) {
            ERL_WARN("Surface mapping is not initialized.");
            return false;
        }
        if (m_surface_mapping_->GetOctree() == nullptr) {
            ERL_WARN("Octree is not initialized.");
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
            std::lock_guard lock(m_mutex_);  // CRITICAL SECTION
            {
                Dtype x, y, z;
                m_surface_mapping_->GetOctree()->GetMetricMin(x, y, z);  // trigger the octree to update its metric min/max
            }

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
                        threads.emplace_back(&GpSdfMapping3D::SearchGpThread, this, thread_idx, start_idx, end_idx);
                        start_idx = end_idx;
                    }
                    for (auto &thread: threads) { thread.join(); }
                    threads.clear();
                }
            }

            // Train any updated GPs
            if (!m_cluster_queue_.empty()) {
                ERL_BLOCK_TIMER_MSG_TIME("Train GPs", gp_train_time);
                KeySet keys;
                m_clusters_to_train_.clear();
                for (auto &gps: m_query_to_gps_) {
                    for (auto &[distance, key_and_gp]: gps) {
                        if (!keys.insert(key_and_gp.first).second) { continue; }
                        if (const auto &gp = key_and_gp.second; !gp->active || gp->edf_gp->IsTrained()) { continue; }
                        m_clusters_to_train_.emplace_back(key_and_gp);
                    }
                }
                TrainGps();
            }

            if (m_setting_->use_occ_sign) {
                // collect the sign of query positions since the quadtree is not thread-safe
                const auto tree = m_surface_mapping_->GetOctree();
                if (m_query_signs_.size() < num_queries) { m_query_signs_.resize(num_queries); }
#pragma omp parallel for default(none) shared(tree, positions_in, num_queries) schedule(static)
                for (uint32_t i = 0; i < num_queries; i++) {
                    const Vector3 &position = positions_in.col(i);
                    const auto node = tree->Search(position.x(), position.y(), position.z());
                    if (node == nullptr || tree->IsNodeOccupied(node)) {
                        m_query_signs_[i] = -1.0;
                    } else {
                        m_query_signs_[i] = 1.0;
                    }
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
                    threads.emplace_back(&GpSdfMapping3D::TestGpThread, this, thread_idx, start_idx, end_idx);
                    start_idx = end_idx;
                }
                for (auto &thread: threads) { thread.join(); }
                threads.clear();
            }
        }

        m_test_buffer_.DisconnectBuffers();
        for (const auto &gps: m_query_to_gps_) {
            for (const auto &[_, key_and_gp]: gps) { key_and_gp.second->locked_for_test = false; }  // unlock GPs
        }

        ERL_TRACY_PLOT("[test]gp_search_time (ms)", gp_search_time);
        ERL_TRACY_PLOT("[test]gp_train_time (ms)", gp_train_time);
        ERL_TRACY_PLOT("[test]gp_test_time (ms)", gp_test_time);

        return true;
    }

    template<typename Dtype>
    const std::vector<std::array<std::shared_ptr<typename GpSdfMapping3D<Dtype>::Gp>, 4>> &
    GpSdfMapping3D<Dtype>::GetUsedGps() const {
        return m_query_used_gps_;
    }

    template<typename Dtype>
    const typename GpSdfMapping3D<Dtype>::KeyGpMap &
    GpSdfMapping3D<Dtype>::GetGpMap() const {
        return m_gp_map_;
    }

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::operator==(const GpSdfMapping3D &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_surface_mapping_ == nullptr && other.m_surface_mapping_ != nullptr) { return false; }
        if (m_surface_mapping_ != nullptr && (other.m_surface_mapping_ == nullptr || *m_surface_mapping_ != *other.m_surface_mapping_)) { return false; }
        if (m_gp_map_.size() != other.m_gp_map_.size()) { return false; }
        for (const auto &[key, gp]: m_gp_map_) {
            auto it = other.m_gp_map_.find(key);
            if (it == other.m_gp_map_.end()) { return false; }
            const auto &[other_key, other_gp] = *it;
            if (gp == nullptr && other_gp != nullptr) { return false; }
            if (gp != nullptr && (other_gp == nullptr || *gp != *other_gp)) { return false; }
        }
        if (m_cluster_queue_keys_.size() != other.m_cluster_queue_keys_.size()) { return false; }
        for (const auto &[key, handle]: m_cluster_queue_keys_) {
            const auto it = other.m_cluster_queue_keys_.find(key);
            if (it == other.m_cluster_queue_keys_.end()) { return false; }
            if ((*handle).time_stamp != (*it->second).time_stamp) { return false; }
        }
        if (m_train_gp_time_ != other.m_train_gp_time_) { return false; }
        return true;
    }

    // template<typename Dtype>
    // bool
    // GpSdfMapping3D<Dtype>::Write(const std::string &filename) const {
    //     ERL_INFO("Writing GpSdfMapping3D to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::gp_sdf::GpSdfMapping3D";

    template<typename Dtype>
    bool
    GpSdfMapping3D<Dtype>::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        s << "surface_mapping " << (m_surface_mapping_ != nullptr) << std::endl;
        if (m_surface_mapping_ != nullptr && !m_surface_mapping_->Write(s)) {
            ERL_WARN("Failed to write surface mapping.");
            return false;
        }
        s << "gp_map " << m_gp_map_.size() << std::endl;
        for (const auto &[key, gp]: m_gp_map_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
            s.write(reinterpret_cast<const char *>(&key[2]), sizeof(key[2]));
            const bool gp_exists = gp != nullptr;
            s.write(reinterpret_cast<const char *>(&gp_exists), sizeof(gp_exists));
            if (gp_exists && !gp->Write(s)) {
                ERL_WARN("Failed to write GP.");
                return false;
            }
        }
        // m_affected_clusters_ is temporary data.
        s << "cluster_queue_keys " << m_cluster_queue_keys_.size() << std::endl;
        for (const auto &[key, handle]: m_cluster_queue_keys_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
            s.write(reinterpret_cast<const char *>(&key[2]), sizeof(key[2]));
            s.write(reinterpret_cast<const char *>(&(*handle).time_stamp), sizeof((*handle).time_stamp));
        }
        // m_cluster_queue_ can be reconstructed from m_cluster_queue_keys_.
        // m_clusters_to_train_ is temporary data.
        s << "train_gp_time" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_train_gp_time_), sizeof(m_train_gp_time_));
        s << "end_of_GpSdfMapping3D" << std::endl;
        return s.good();
    }

    // template<typename Dtype>
    // bool
    // GpSdfMapping3D<Dtype>::Read(const std::string &filename) {
    //     ERL_INFO("Reading GpSdfMapping3D from file: {}", std::filesystem::absolute(filename));
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
    GpSdfMapping3D<Dtype>::Read(std::istream &s) {
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
            "surface_mapping",
            "gp_map",
            "cluster_queue_keys",
            "train_gp_time",
            "end_of_GpSdfMapping2D",
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
                case 0: {         // setting
                    skip_line();  // skip the line to read the binary data section
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // surface_mapping
                    bool has_surface_mapping;
                    s >> has_surface_mapping;
                    skip_line();
                    if (has_surface_mapping) {
                        if (m_surface_mapping_ == nullptr) {
                            m_surface_mapping_ = AbstractSurfaceMapping::CreateSurfaceMapping<AbstractSurfaceMapping3D<Dtype>>(
                                m_setting_->surface_mapping_type,
                                m_setting_->surface_mapping);
                        }
                        if (!m_surface_mapping_->Read(s)) {
                            ERL_WARN("Failed to read surface mapping.");
                            return false;
                        }
                    }
                    break;
                }
                case 2: {  // gp_map
                    uint32_t num_gps;
                    s >> num_gps;
                    skip_line();
                    for (uint32_t i = 0; i < num_gps; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        s.read(reinterpret_cast<char *>(&key[2]), sizeof(key[2]));
                        auto [it, inserted] = m_gp_map_.try_emplace(key, nullptr);
                        if (!inserted) {
                            ERL_WARN("Duplicate GP key: ({}, {}).", key[0], key[1]);
                            return false;
                        }
                        bool has_gp;
                        s.read(reinterpret_cast<char *>(&has_gp), sizeof(has_gp));
                        if (has_gp) {
                            it->second = std::make_shared<Gp>();
                            if (!it->second->Read(s, m_setting_->edf_gp)) {
                                ERL_WARN("Failed to read GP.");
                                return false;
                            }
                        }
                    }
                    break;
                }
                case 3: {  // cluster_queue_keys
                    uint32_t num_cluster_keys;
                    s >> num_cluster_keys;
                    skip_line();
                    for (uint32_t i = 0; i < num_cluster_keys; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        s.read(reinterpret_cast<char *>(&key[2]), sizeof(key[2]));
                        long time_stamp;
                        s.read(reinterpret_cast<char *>(&time_stamp), sizeof(time_stamp));
                        m_cluster_queue_keys_.insert({key, m_cluster_queue_.push({time_stamp, key})});
                    }
                    break;
                }
                case 4: {  // train_gp_time
                    skip_line();
                    s.read(reinterpret_cast<char *>(&m_train_gp_time_), sizeof(m_train_gp_time_));
                    break;
                }
                case 5: {  // end_of_GpSdfMapping3D
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read GpSdfMapping3D. Truncated file?");
        return false;  // should not reach here
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::TrainGps() {
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
            threads.emplace_back(&GpSdfMapping3D::TrainGpThread, this, thread_idx, start_idx, end_idx);
        }
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }
        threads.clear();
        ERL_INFO("Trained {} GPs", n);

        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Function {} takes {} ms", __PRETTY_FUNCTION__, time / 1e3);
        time /= static_cast<double>(n);
        m_train_gp_time_ = m_train_gp_time_ * 0.1 + time * 0.9;
        ERL_INFO("Per GP training time: {} us.", m_train_gp_time_);
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::TrainGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        const auto tree = m_surface_mapping_->GetOctree();
        if (tree == nullptr) { return; }
        const Dtype sensor_noise = m_surface_mapping_->GetSensorNoise();
        const uint32_t cluster_depth = tree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel();
        const Dtype cluster_size = tree->GetNodeSize(cluster_depth);
        const Dtype aabb_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2.;
        const Dtype offset_distance = m_setting_->offset_distance;
        const Dtype max_valid_gradient_var = m_setting_->max_valid_gradient_var;
        const Dtype invalid_position_var = m_setting_->invalid_position_var;
        const auto &surface_data_vec = m_surface_mapping_->GetSurfaceDataManager().GetEntries();

        std::vector<std::pair<Dtype, std::size_t>> surface_data_indices;
        const auto max_num_samples = static_cast<std::size_t>(m_setting_->edf_gp->max_num_samples);
        surface_data_indices.reserve(max_num_samples);
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto &[cluster_key, gp] = m_clusters_to_train_[i];
            ERL_DEBUG_ASSERT(gp->active, "GP is not active");

            // collect surface data in the area
            surface_data_indices.clear();
            const Aabb area(tree->KeyToCoord(cluster_key, cluster_depth), aabb_half_size);
            for (auto it = tree->BeginLeafInAabb(area), end = tree->EndLeafInAabb(); it != end; ++it) {
                if (!it->HasSurfaceData()) { continue; }  // no surface data in the node
                const auto &surface_data = surface_data_vec[it->surface_data_index];
                surface_data_indices.emplace_back((gp->position - surface_data.position).norm(), it->surface_data_index);
            }
            if (surface_data_indices.empty()) {  // no surface data in the area
                gp->Deactivate();                // deactivate the GP if there is no training data
                continue;
            }
            gp->LoadSurfaceData(surface_data_indices, surface_data_vec, offset_distance, sensor_noise, max_valid_gradient_var, invalid_position_var);
            gp->Train();
        }
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::SearchCandidateGps(const Eigen::Ref<const Matrix3X> &positions_in) {
        ERL_BLOCK_TIMER();
        // collect some numbers needed for searching
        const Dtype search_area_half_size = m_setting_->test_query->search_area_half_size;
        const auto tree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_depth = tree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel();
        const Aabb tree_aabb = tree->GetMetricAabb();  // biggest AABB of the tree
        Aabb query_aabb(                               // current queried search area
            positions_in.rowwise().minCoeff().array() - search_area_half_size,
            positions_in.rowwise().maxCoeff().array() + search_area_half_size);
        Aabb region = query_aabb.Intersection(tree_aabb);  // current region of interest
        m_candidate_gps_.clear();                          // clear the buffer
        while (region.sizes().prod() > 0) {                // search until the intersection is empty
            // search the octree for clusters in the search area
            for (auto it = tree->BeginTreeInAabb(region, cluster_depth), end = tree->EndTreeInAabb(); it != end; ++it) {
                if (it->GetDepth() != cluster_depth) { continue; }  // not a cluster node
                auto it_gp = m_gp_map_.find(it.GetIndexKey());      // get the GP of the cluster
                if (it_gp == m_gp_map_.end()) { continue; }         // no gp for this cluster
                const auto gp = it_gp->second;                      // get the GP of the cluster
                if (!gp->active) { continue; }                      // gp is inactive (e.g. due to no training data)
                gp->locked_for_test = true;                         // lock the GP for testing
                m_candidate_gps_.emplace_back(it_gp->first, gp);
            }
            if (!m_candidate_gps_.empty()) { break; }  // found at least one GP
            // double the size of query_aabb
            query_aabb = Aabb(query_aabb.center, 2.0 * query_aabb.half_sizes);
            Aabb new_region = query_aabb.Intersection(tree_aabb);
            if ((region.min() == new_region.min()) && (region.max() == new_region.max())) { break; }  // region did not change
            region = std::move(new_region);                                                           // update region
        }
        // build kdtree of candidate GPs to allow fast search
        if (!m_candidate_gps_.empty()) {
            Matrix3X positions(3, m_candidate_gps_.size());
            for (std::size_t i = 0; i < m_candidate_gps_.size(); ++i) { positions.col(static_cast<long>(i)) = m_candidate_gps_[i].second->position; }
            m_kd_tree_candidate_gps_ = std::make_shared<Kdtree>(std::move(positions));
        }
        ERL_INFO("{} candidate GPs found.", m_candidate_gps_.size());
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (m_surface_mapping_ == nullptr) { return; }
        const auto tree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        const uint32_t cluster_depth = tree->GetTreeDepth() - cluster_level;
        const Aabb tree_aabb = tree->GetMetricAabb();

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            const Vector3 test_position = m_test_buffer_.positions->col(i);
            std::vector<std::pair<Dtype, KeyGpPair>> &gps = m_query_to_gps_[i];
            gps.clear();
            constexpr int kMaxNumGps = 16;
            gps.reserve(kMaxNumGps);

            Dtype search_area_half_size = m_setting_->test_query->search_area_half_size;
            Aabb search_aabb(test_position, search_area_half_size);
            Aabb region = tree_aabb.Intersection(search_aabb);

            // search gps from the common buffer at first to save time
            if (m_kd_tree_candidate_gps_ != nullptr) {
                // use kdtree to search for 32 nearest GPs
                constexpr long kMaxNumNeighbors = 32;
                Eigen::VectorXl indices = Eigen::VectorXl::Constant(kMaxNumNeighbors, -1);
                VectorX squared_distances(kMaxNumNeighbors);
                m_kd_tree_candidate_gps_->Knn(kMaxNumNeighbors, test_position, indices, squared_distances);
                for (long j = 0; j < kMaxNumNeighbors; ++j) {
                    const long &index = indices[j];
                    if (index < 0) { break; }  // no more GPs
                    const auto &key_and_gp = m_candidate_gps_[index];
                    if (!key_and_gp.second->Intersects(region.center, region.half_sizes)) { continue; }
                    gps.emplace_back(std::sqrt(squared_distances[j]), key_and_gp);
                    if (gps.size() >= kMaxNumGps) { break; }  // found enough GPs
                }
            } else {
                // no kdtree, search all GPs in the common buffer, which is slow
                gps.reserve(m_candidate_gps_.size());  // request more memory
                for (const auto &key_and_gp: m_candidate_gps_) {
                    if (!key_and_gp.second->Intersects(region.center, region.half_sizes)) { continue; }
                    gps.emplace_back((key_and_gp.second->position - test_position).norm(), key_and_gp);
                }
            }

            if (gps.empty()) {               // no gp found
                search_area_half_size *= 2;  // double search area size
                search_aabb = Aabb(test_position, search_area_half_size);
                Aabb new_region = tree_aabb.Intersection(search_aabb);
                if ((region.min() == new_region.min()) && (region.max() == new_region.max())) { continue; }  // no need to search again
                region = std::move(new_region);                                                              // update intersection
            } else {
                continue;  // found at least one gp
            }

            while (gps.empty() && region.sizes().prod() > 0) {
                // search the tree for clusters in the search area
                for (auto it = tree->BeginTreeInAabb(region, cluster_depth), end = tree->EndTreeInAabb(); it != end; ++it) {
                    if (it->GetDepth() != cluster_depth) { continue; }  // not a cluster node
                    auto it_gp = m_gp_map_.find(it.GetIndexKey());
                    if (it_gp == m_gp_map_.end()) { continue; }  // no gp for this cluster
                    const auto gp = it_gp->second;
                    if (!gp->active) { continue; }  // gp is inactive (e.g. due to no training data)
                    gp->locked_for_test = true;     // lock the GP for testing
                    gps.emplace_back((gp->position - test_position).norm(), std::make_pair(it_gp->first, gp));
                }
                if (!gps.empty()) { break; }  // found at least one gp
                search_area_half_size *= 2;   // double search area size
                search_aabb = Aabb(test_position, search_area_half_size);
                Aabb new_region = tree_aabb.Intersection(search_aabb);
                if ((region.min() == new_region.min()) && (region.max() == new_region.max())) { break; }  // no need to search again
                region = std::move(new_region);                                                           // update region
            }
        }
    }

    template<typename Dtype>
    void
    GpSdfMapping3D<Dtype>::TestGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (m_surface_mapping_ == nullptr) { return; }

        const bool use_gp_covariance = m_setting_->test_query->use_gp_covariance;
        const bool compute_covariance = m_setting_->test_query->compute_covariance;
        const int num_neighbor_gps = m_setting_->test_query->num_neighbor_gps;
        const bool use_smallest = m_setting_->test_query->use_smallest;
        const Dtype max_test_valid_distance_var = m_setting_->test_query->max_test_valid_distance_var;
        const Dtype softmin_temperature = m_setting_->test_query->softmin_temperature;
        const Dtype offset_distance = m_setting_->offset_distance;
        const bool use_occ_sign = m_setting_->use_occ_sign;

        std::vector<std::size_t> gp_indices;
        Matrix4X fs(4, num_neighbor_gps);          // f, fGrad1, fGrad2, fGrad3
        Matrix4X variances(4, num_neighbor_gps);   // variances of f, fGrad1, fGrad2, fGrad3
        Matrix6X covariance(6, num_neighbor_gps);  // cov (gx, d), (gy, d), (gz, d), (gy, gx), (gz, gx), (gz, gy)
        MatrixX no_variance;
        MatrixX no_covariance;
        std::vector<std::pair<long, long>> tested_idx;
        tested_idx.reserve(num_neighbor_gps);

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            Dtype &distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);

            distance_out = 0.;
            gradient_out.setZero();
            variances.setConstant(1e6);
            if (compute_covariance) { covariance.setConstant(1e6); }

            auto &gps = m_query_to_gps_[i];
            if (gps.empty()) { continue; }

            const Vector3 test_position = m_test_buffer_.positions->col(i);
            gp_indices.resize(gps.size());
            std::iota(gp_indices.begin(), gp_indices.end(), 0);
            if (gps.size() > 1) {  // sort GPs by distance to the test position
                std::stable_sort(gp_indices.begin(), gp_indices.end(), [&gps](const size_t i1, const size_t i2) { return gps[i1].first < gps[i2].first; });
            }
            gp_indices.resize(std::min(gps.size(), static_cast<std::size_t>(num_neighbor_gps)));

            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            for (std::size_t &j: gp_indices) {
                // call selected GPs for inference
                const auto &gp = gps[j].second.second;
                if (!gp->active || !gp->edf_gp->IsTrained()) { continue; }  // skip inactive GPs
                if (!gp->Test(
                        test_position,
                        fs.col(cnt),
                        variances.col(cnt),
                        covariance.col(cnt),
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
                // std::sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return std::abs(fs(0, a.first)) < std::abs(fs(0, b.first)); });
                std::sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return fs(0, a.first) < fs(0, b.first); });
                need_weighted_sum = false;
                fs.col(0) = fs.col(tested_idx[0].first);
                variances.col(0) = variances.col(tested_idx[0].first);
                if (compute_covariance) { covariance.col(0) = covariance.col(tested_idx[0].first); }
            }

            // sort the results by distance variance
            if (need_weighted_sum && tested_idx.size() > 1) {
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return variances(0, a.first) < variances(0, b.first); });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, tested_idx[0].first) * fs(0, tested_idx[1].first) < 0) { need_weighted_sum = false; }
            }

            // store the result
            if (need_weighted_sum) {
                if (variances(0, tested_idx[0].first) < max_test_valid_distance_var) {
                    auto j = tested_idx[0].first;
                    // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs.col(j).template tail<3>();
                    variance_out << variances.col(j);
                    if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(j); }
                    m_query_used_gps_[i][0] = gps[tested_idx[0].second].second.second;
                    m_query_used_gps_[i][1] = nullptr;
                } else {
                    // pick the best <= 4 results to do weighted sum
                    const std::size_t m = std::min(tested_idx.size(), 4ul);
                    Dtype w_sum = 0;
                    Vector4 f = Vector4::Zero();
                    Vector4 variance_f = Vector4::Zero();
                    Vector6 covariance_f = Vector6::Zero();
                    for (std::size_t k = 0; k < m; ++k) {
                        const long jk = tested_idx[k].first;
                        const Dtype w = 1.0 / (variances(0, jk) - max_test_valid_distance_var);
                        w_sum += w;
                        f += fs.col(jk) * w;
                        variance_f += variances.col(jk) * w;
                        m_query_used_gps_[i][k] = gps[tested_idx[k].second].second.second;
                        if (compute_covariance) { covariance_f += covariance.col(jk) * w; }
                    }
                    f /= w_sum;
                    distance_out = f[0];
                    gradient_out << f.template tail<3>();
                    variance_out << variance_f / w_sum;
                    if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance_f / w_sum; }
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs.col(0).template tail<3>();
                variance_out << variances.col(0);
                if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(0); }
                m_query_used_gps_[i][0] = gps[tested_idx[0].second].second.second;
                m_query_used_gps_[i][1] = nullptr;
            }
            gradient_out.normalize();
            if (use_occ_sign && distance_out > 0 && m_query_signs_[i] < 0) { distance_out = -distance_out; }
        }
    }

}  // namespace erl::gp_sdf
