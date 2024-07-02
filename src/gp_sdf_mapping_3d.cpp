#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

#include "erl_common/angle_utils.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace erl::sdf_mapping {

    bool
    GpSdfMapping3D::Update(
        const Eigen::Ref<const Eigen::Matrix3d> &rotation,
        const Eigen::Ref<const Eigen::Vector3d> &translation,
        const Eigen::Ref<const Eigen::MatrixXd> &ranges) {

        if (m_setting_->log_timing) {
            std::lock_guard<std::mutex> lock(m_log_mutex_);
            if (m_last_position_.has_value()) {
                const Eigen::Vector3d delta = translation - m_last_position_.value();
                m_travel_distance_ += delta.norm();
            } else {
                m_travel_distance_ = 0;
            }
            m_last_position_ = translation;
            m_train_log_file_ << m_travel_distance_;
        }

        const std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

        double time_budget = 1e6 / m_setting_->update_hz;  // us
        bool success;
        std::chrono::high_resolution_clock::time_point t0, t1;
        double dt;
        {
            std::lock_guard<std::mutex> lock(m_mutex_);
            t0 = std::chrono::high_resolution_clock::now();
            success = m_surface_mapping_->Update(rotation, translation, ranges);
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (m_setting_->log_timing) { m_train_log_file_ << "," << dt; }  // surface_mapping_time
            ERL_INFO("Surface mapping update time: {:f} us.", dt);
        }
        time_budget -= dt;

        if (success) {
            std::lock_guard<std::mutex> lock(m_mutex_);
            t0 = std::chrono::high_resolution_clock::now();
            UpdateGps(time_budget);
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ERL_INFO("GP update time: {:f} ms.", dt);

            if (m_setting_->log_timing) {
                m_train_log_file_ << "," << dt;  // total_gp_update_time
                dt = std::chrono::duration<double, std::milli>(t1 - t_start).count();
                m_train_log_file_ << "," << dt << std::endl << std::flush;  // total_update_time
            }

            return true;
        }

        if (m_setting_->log_timing) { m_train_log_file_ << std::endl; }
        return false;
    }

    bool
    GpSdfMapping3D::Test(
        const Eigen::Ref<const Eigen::Matrix3Xd> &positions_in,
        Eigen::VectorXd &distances_out,
        Eigen::Matrix3Xd &gradients_out,
        Eigen::Matrix4Xd &variances_out,
        Eigen::Matrix6Xd &covariances_out) {

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

        const std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

        const uint32_t num_queries = positions_in.cols();
        const uint32_t num_threads = std::min(std::min(m_setting_->num_threads, std::thread::hardware_concurrency()), num_queries);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = num_queries / num_threads;
        const std::size_t leftover = num_queries - batch_size * num_threads;
        std::size_t start_idx, end_idx;

        if (m_setting_->log_timing) {
            std::lock_guard<std::mutex> lock(m_log_mutex_);
            m_test_log_file_ << m_travel_distance_;
        }

        std::chrono::high_resolution_clock::time_point t0, t1;
        double dt;
        {
            std::lock_guard<std::mutex> lock(m_mutex_);  // CRITICAL SECTION
            // Search GPs for each query position
            t0 = std::chrono::high_resolution_clock::now();
            m_query_to_gps_.clear();
            m_query_to_gps_.resize(num_queries);  // allocate memory for n threads, collected GPs will be locked for testing
            {
                double x, y, z;
                m_surface_mapping_->GetOctree()->GetMetricMin(x, y, z);  // trigger the octree to update its metric min/max
            }

            // if we iterate through the octree for each query position separately, it takes too much CPU time
            Eigen::Vector3d query_aabb_min = positions_in.rowwise().minCoeff();
            Eigen::Vector3d query_aabb_max = positions_in.rowwise().maxCoeff();
            query_aabb_min.array() -= m_setting_->test_query->search_area_half_size;
            query_aabb_max.array() += m_setting_->test_query->search_area_half_size;
            geometry::Aabb3D query_aabb(query_aabb_min, query_aabb_max);

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
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_search_time
            ERL_INFO("Search GPs: {:f} us", dt);

            // Train any updated GPs
            if (!m_new_gps_.empty()) {
                t0 = std::chrono::high_resolution_clock::now();
                std::unordered_set<std::shared_ptr<Gp>> new_gps;
                for (auto &gps: m_query_to_gps_) {
                    for (auto &[distance, gp]: gps) {
                        if (gp->active && !gp->gp->IsTrained()) { new_gps.insert(gp); }
                    }
                }
                m_gps_to_train_.clear();
                m_gps_to_train_.insert(m_gps_to_train_.end(), new_gps.begin(), new_gps.end());
                TrainGps();
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
                if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_train_time
                ERL_INFO("Train GPs: {:f} us", dt);
            } else {
                if (m_setting_->log_timing) { m_test_log_file_ << ",0"; }
            }
        }

        // Compute the inference result for each query position
        m_query_used_gps_.resize(num_queries);
        t0 = std::chrono::high_resolution_clock::now();
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
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_test_time
        ERL_INFO("Test GPs: {:f} us", dt);

        m_test_buffer_.DisconnectBuffers();

        for (const auto &gps: m_query_to_gps_) {
            for (const auto &[distance, gp]: gps) { gp->locked_for_test = false; }  // unlock GPs
        }

        if (m_setting_->log_timing) {
            const std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            m_test_log_file_ << "," << dt << std::endl << std::flush;  // total_test_time
        }

        return true;
    }

    void
    GpSdfMapping3D::UpdateGps(double time_budget) {
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1");

        auto t0 = std::chrono::high_resolution_clock::now();
        // add affected clusters
        const geometry::OctreeKeySet changed_clusters = m_surface_mapping_->GetChangedClusters();
        const uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        const std::shared_ptr<geometry::SurfaceMappingOctree> octree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_depth = octree->GetTreeDepth() - cluster_level;
        const double cluster_size = octree->GetNodeSize(cluster_depth);
        const double area_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2;
        geometry::OctreeKeySet affected_clusters(changed_clusters);
        for (const auto &cluster_key: changed_clusters) {
            double x, y, z;
            octree->KeyToCoord(cluster_key, cluster_depth, x, y, z);
            for (auto it = octree->BeginLeafInAabb(
                          x - area_half_size,
                          y - area_half_size,
                          z - area_half_size,
                          x + area_half_size,
                          y + area_half_size,
                          z + area_half_size),
                      end = octree->EndLeafInAabb();
                 it != end;
                 ++it) {
                affected_clusters.insert(octree->AdjustKeyToDepth(it.GetKey(), cluster_depth));
            }
        }

        // update GPs in affected clusters
        uint32_t num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        m_clusters_to_update_.clear();
        m_clusters_to_update_.insert(m_clusters_to_update_.end(), affected_clusters.begin(), affected_clusters.end());
        for (auto &cluster_key: m_clusters_to_update_) {
            auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, std::make_shared<Gp>());
            Eigen::Vector3d &gp_position = it->second->position;
            octree->KeyToCoord(cluster_key, cluster_depth, gp_position);
            it->second->half_size = area_half_size;
        }
        std::size_t batch_size = m_clusters_to_update_.size() / num_threads;
        std::size_t left_over = m_clusters_to_update_.size() - batch_size * num_threads;
        std::size_t end_idx = 0;
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            std::size_t start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (thread_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping3D::UpdateGpThread, this, thread_idx, start_idx, end_idx);
        }
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }

        if (m_setting_->train_gp_immediately) {  // new GPs are already trained in UpdateGpThread
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            ERL_INFO("Update GPs' training data: {:f} us.", dt);
            return;
        }

        for (auto &cluster_key: m_clusters_to_update_) {
            auto it = m_gp_map_.find(cluster_key);
            if (it == m_gp_map_.end() || !it->second->active) { continue; }  // GP does not exist or deactivated (e.g. due to no training data)
            if (it->second->gp != nullptr && !it->second->gp->IsTrained()) { m_new_gps_.push_front(it->second); }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Update GPs' training data: {:f} us.", dt);
        time_budget -= dt;

        // train as many new GPs as possible within the time limit
        if (time_budget <= m_train_gp_time_) {  // no time left for training
            ERL_INFO("{} GP(s) not trained yet due to time limit.", m_new_gps_.size());
            return;
        }

        auto max_num_gps_to_train = static_cast<std::size_t>(std::floor(time_budget / m_train_gp_time_));
        max_num_gps_to_train = std::min(max_num_gps_to_train, m_new_gps_.size());
        std::unordered_set<std::shared_ptr<Gp>> gps_to_train;
        while (!m_new_gps_.empty() && gps_to_train.size() < max_num_gps_to_train) {
            if (auto maybe_new_gp = m_new_gps_.front(); maybe_new_gp->active && maybe_new_gp->gp != nullptr && !maybe_new_gp->gp->IsTrained()) {
                gps_to_train.insert(maybe_new_gp);
            }
            m_new_gps_.pop_front();
        }
        m_gps_to_train_.clear();
        m_gps_to_train_.insert(m_gps_to_train_.end(), gps_to_train.begin(), gps_to_train.end());
        TrainGps();
        ERL_INFO("{} GP(s) not trained yet due to time limit.", m_new_gps_.size());

        m_train_log_file_ << "," << dt << "," << m_new_gps_.size() << "," << m_train_gp_time_;
    }

    void
    GpSdfMapping3D::UpdateGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        (void) thread_idx;
        const auto octree = m_surface_mapping_->GetOctree();
        if (octree == nullptr) { return; }
        const double sensor_noise = m_surface_mapping_->GetSensorNoise();

        const uint32_t cluster_depth = octree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel();
        const double cluster_size = octree->GetNodeSize(cluster_depth);
        const double aabb_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2.;

        std::vector<std::pair<double, std::shared_ptr<geometry::SurfaceMappingOctreeNode::SurfaceData>>> surface_data_vec;
        surface_data_vec.reserve(1024);
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto &cluster_key = m_clusters_to_update_[i];
            // get the GP of the cluster
            std::shared_ptr<Gp> &gp = m_gp_map_.at(cluster_key);
            // testing thread may unlock the GP, but it is impossible to lock it here due to the mutex
            if (gp->locked_for_test) {  // create a new GP if the old one is locked for testing
                const auto new_gp = std::make_shared<Gp>();
                new_gp->position = gp->position;
                new_gp->half_size = gp->half_size;
                gp = new_gp;
            }
            gp->active = true;  // activate the GP
            if (gp->gp == nullptr) { gp->gp = std::make_shared<LogSdfGaussianProcess>(m_setting_->gp_sdf); }

            // collect surface data in the area
            // double x, y, z;
            const Eigen::Vector3d cluster_center = octree->KeyToCoord(cluster_key, cluster_depth);
            surface_data_vec.clear();
            for (auto it = octree->BeginLeafInAabb(geometry::Aabb3D(cluster_center.array() - aabb_half_size, cluster_center.array() + aabb_half_size)),
                      end = octree->EndLeafInAabb();
                 it != end;
                 ++it) {  // iterate through all leaf nodes in the area
                auto surface_data = it->GetSurfaceData();
                if (surface_data == nullptr) { continue; }  // no surface data in the node
                surface_data_vec.emplace_back((cluster_center - surface_data->position).norm(), surface_data);
            }

            if (surface_data_vec.empty()) {
                gp->active = false;  // deactivate the GP if there is no training data
                gp->num_train_samples = 0;
                continue;
            }

            // sort surface data by distance
            std::sort(surface_data_vec.begin(), surface_data_vec.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            if (surface_data_vec.size() > static_cast<std::size_t>(m_setting_->gp_sdf->max_num_samples)) {
                surface_data_vec.resize(m_setting_->gp_sdf->max_num_samples);
            }

            // prepare data for GP training
            gp->gp->Reset(static_cast<long>(surface_data_vec.size()), 3);
            Eigen::MatrixXd &mat_x = gp->gp->GetTrainInputSamplesBuffer();
            Eigen::VectorXd &vec_y = gp->gp->GetTrainOutputSamplesBuffer();
            Eigen::MatrixXd &mat_grad = gp->gp->GetTrainOutputGradientSamplesBuffer();
            Eigen::VectorXd &vec_var_x = gp->gp->GetTrainInputSamplesVarianceBuffer();
            Eigen::VectorXd &vec_var_y = gp->gp->GetTrainOutputValueSamplesVarianceBuffer();
            Eigen::VectorXd &vec_var_grad = gp->gp->GetTrainOutputGradientSamplesVarianceBuffer();
            Eigen::VectorXb &vec_grad_flag = gp->gp->GetTrainGradientFlagsBuffer();
            long count = 0;
            for (auto &[distance, surface_data]: surface_data_vec) {
                mat_x.col(count) = surface_data->position;
                vec_y[count] = m_setting_->offset_distance;
                vec_var_y[count] = sensor_noise;
                vec_var_grad[count] = surface_data->var_normal;
                if ((surface_data->var_normal > m_setting_->max_valid_gradient_var) ||  // invalid gradient
                    (surface_data->normal.norm() < 0.9)) {                              // invalid normal
                    vec_var_x[count] = m_setting_->invalid_position_var;                // position is unreliable
                    vec_grad_flag[count] = false;
                    mat_grad.col(count++).setZero();
                    continue;
                }
                vec_var_x[count] = surface_data->var_position;
                vec_grad_flag[count] = true;
                mat_grad.col(count++) = surface_data->normal;
                if (count >= mat_x.cols()) { break; }  // reached max_num_samples
            }
            gp->num_train_samples = count;
            if (m_setting_->train_gp_immediately) { gp->Train(); }
        }
    }

    void
    GpSdfMapping3D::TrainGps() {
        ERL_INFO("Training {} GPs ...", m_gps_to_train_.size());
        const auto t0 = std::chrono::high_resolution_clock::now();
        const uint32_t n = m_gps_to_train_.size();
        if (n == 0) { return; }
        uint32_t num_threads = std::min(n, std::thread::hardware_concurrency());
        num_threads = std::min(num_threads, m_setting_->num_threads);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = n / num_threads;
        const std::size_t leftover = n - batch_size * num_threads;
        std::size_t start_idx = 0;
        for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
            std::size_t end_idx = start_idx + batch_size;
            if (thread_idx < leftover) { ++end_idx; }
            threads.emplace_back(&GpSdfMapping3D::TrainGpThread, this, thread_idx, start_idx, end_idx);
            start_idx = end_idx;
        }
        for (auto &thread: threads) { thread.join(); }
        m_gps_to_train_.clear();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double time = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(n);
        m_train_gp_time_ = m_train_gp_time_ * 0.1 + time * 0.9;
        ERL_INFO("Per GP training time: {:f} us.", m_train_gp_time_);
    }

    void
    GpSdfMapping3D::TrainGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) const {
        (void) thread_idx;
        for (std::size_t i = start_idx; i < end_idx; ++i) { m_gps_to_train_[i]->Train(); }
    }

    void
    GpSdfMapping3D::SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        if (m_surface_mapping_ == nullptr) { return; }
        auto octree = m_surface_mapping_->GetOctree();
        uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        uint32_t cluster_depth = octree->GetTreeDepth() - cluster_level;
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            const Eigen::Vector3d position = m_test_buffer_.positions->col(i);
            std::vector<std::pair<double, std::shared_ptr<Gp>>> &gps = m_query_to_gps_[i];
            gps.reserve(16);
            double search_area_half_size = m_setting_->test_query->search_area_half_size;
            Eigen::Vector3d tree_min, tree_max;
            octree->GetMetricMinMax(tree_min.x(), tree_min.y(), tree_min.z(), tree_max.x(), tree_max.y(), tree_max.z());
            geometry::Aabb3D octree_aabb(tree_min, tree_max);
            geometry::Aabb3D search_aabb(position, search_area_half_size);
            geometry::Aabb3D intersection = octree_aabb.Intersection(search_aabb);
            while (intersection.sizes().prod() > 0) {
                // search the octree for clusters in the search area
                for (auto it = octree->BeginTreeInAabb(intersection, cluster_depth), end = octree->EndTreeInAabb(); it != end; ++it) {
                    if (it->GetDepth() != cluster_depth) { continue; }  // not a cluster node
                    geometry::OctreeKey cluster_key = it.GetIndexKey();
                    auto it_gp = m_gp_map_.find(cluster_key);
                    if (it_gp == m_gp_map_.end()) { continue; }  // no gp for this cluster
                    if (!it_gp->second->active) { continue; }    // gp is inactive (e.g. due to no training data)
                    const Eigen::Vector3d cluster_center = octree->KeyToCoord(cluster_key, cluster_depth);
                    it_gp->second->locked_for_test = true;  // lock the GP for testing
                    gps.emplace_back((cluster_center - position).norm(), it_gp->second);
                }
                if (!gps.empty()) { break; }  // found at least one gp
                search_area_half_size *= 2;   // double search area size
                search_aabb = geometry::Aabb3D(position, search_area_half_size);
                geometry::Aabb3D new_intersection = octree_aabb.Intersection(search_aabb);
                if ((intersection.min() == new_intersection.min()) && (intersection.max() == new_intersection.max())) { break; }  // intersection did not change
                intersection = new_intersection;
            }
        }
    }

    void
    GpSdfMapping3D::TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        if (m_surface_mapping_ == nullptr) { return; }

        std::vector<std::size_t> gp_indices;
        constexpr int max_tries = 4;                // we ask at most 4 GPs for each query position
        Eigen::Matrix4Xd fs(4, max_tries);          // f, fGrad1, fGrad2, fGrad3
        Eigen::Matrix4Xd variances(4, max_tries);   // variances of f, fGrad1, fGrad2, fGrad3
        Eigen::Matrix6Xd covariance(6, max_tries);  // covariances of (fGrad1,f), (fGrad2,f), (fGrad2, fGrad1), (fGrad3, f), (fGrad3, fGrad1), (fGrad3, fGrad2)
        Eigen::MatrixXd no_variance;
        Eigen::MatrixXd no_covariance;
        std::vector<std::pair<long, long>> tested_idx;
        tested_idx.reserve(max_tries);

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            double &distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);

            distance_out = 0.;
            gradient_out.setZero();

            variances.setConstant(1e6);
            if (m_setting_->test_query->compute_covariance) { covariance.setConstant(1e6); }

            auto &gps = m_query_to_gps_[i];
            if (gps.empty()) { continue; }

            const Eigen::Vector3d test_position = m_test_buffer_.positions->col(i);
            gp_indices.resize(gps.size());
            std::iota(gp_indices.begin(), gp_indices.end(), 0);
            if (gps.size() > 1) {
                std::stable_sort(gp_indices.begin(), gp_indices.end(), [&gps](const size_t i1, const size_t i2) { return gps[i1].first < gps[i2].first; });
            }

            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            for (std::size_t &j: gp_indices) {
                // call selected GPs for inference
                Eigen::Ref<Eigen::Vector4d> f = fs.col(cnt);           // distance, gradient_x, gradient_y, gradient_z
                Eigen::Ref<Eigen::VectorXd> var = variances.col(cnt);  // var_distance, var_gradient_x, var_gradient_y, var_gradient_z
                auto &gp = gps[j].second->gp;

                if (m_setting_->test_query->recompute_variance) {
                    if (m_setting_->test_query->compute_covariance) {
                        gp->Test(test_position, f, no_variance, covariance.col(cnt));
                    } else {
                        gp->Test(test_position, f, no_variance, no_covariance);
                    }
                    Eigen::Vector3d grad(f[1], f[2], f[3]);
                    if (grad.norm() < 1.e-15) { continue; }  // invalid gradient, skip this GP
                    grad.normalize();
                    double grad_azimuth, grad_elevation;
                    common::DirectionToAzimuthElevation(grad, grad_azimuth, grad_elevation);

                    auto &mat_x = gp->GetTrainInputSamplesBuffer();
                    auto &vec_x_var = gp->GetTrainInputSamplesVarianceBuffer();
                    const Eigen::Vector3d predicted_surf_position = test_position - grad * f[0];
                    long num_samples = gp->GetNumTrainSamples();
                    Eigen::VectorXd weight(num_samples);
                    double weight_sum = 0;
                    double &var_sdf = var[0];
                    double &var_grad_x = var[1];
                    double &var_grad_y = var[2];
                    double &var_grad_z = var[3];
                    var_sdf = 0;
                    var_grad_x = 0;
                    var_grad_y = 0;
                    var_grad_z = 0;
                    double var_azimuth = 0;
                    double var_elevation = 0;
                    for (long k = 0; k < num_samples; ++k) {
                        // difference between training surface position and predicted surface position
                        const double d = (mat_x.col(k) - predicted_surf_position).norm();
                        weight[k] = std::max(1.e-6, std::exp(-d * m_setting_->test_query->softmax_temperature));
                        weight_sum += weight[k];

                        var_sdf += weight[k] * vec_x_var[k];

                        Eigen::Vector3d v = test_position - mat_x.col(k);
                        v.normalize();                             // warning: unchanged if v's norm is very small
                        if (v.squaredNorm() < 0.81) { continue; }  // invalid gradient, skip this sample
                        double v_azimuth, v_elevation;
                        common::DirectionToAzimuthElevation(v, v_azimuth, v_elevation);
                        const double diff_azimuth = std::fabs(v_azimuth - grad_azimuth);
                        const double diff_elevation = std::fabs(v_elevation - grad_elevation);
                        var_azimuth += weight[k] * diff_azimuth * diff_azimuth;
                        var_elevation += weight[k] * diff_elevation * diff_elevation;
                    }

                    var_sdf /= weight_sum;
                    var_azimuth /= weight_sum;
                    var_elevation /= weight_sum;

                    const double sin_azimuth = std::sin(grad_azimuth);
                    const double cos_azimuth = std::cos(grad_azimuth);
                    const double sin_elevation = std::sin(grad_elevation);
                    const double cos_elevation = std::cos(grad_elevation);
                    const double std_azimuth = std::sqrt(var_azimuth);
                    const double std_elevation = std::sqrt(var_elevation);
                    var_grad_x = sin_azimuth * cos_elevation * std_azimuth + cos_azimuth * sin_elevation * std_elevation;
                    var_grad_x *= var_grad_x;
                    var_grad_y = cos_azimuth * cos_elevation * std_azimuth - sin_azimuth * sin_elevation * std_elevation;
                    var_grad_y *= var_grad_y;
                    var_grad_z = cos_elevation * std_elevation;
                    var_grad_z *= var_grad_z;
                } else {
                    if (m_setting_->test_query->compute_covariance) {
                        gp->Test(test_position, f, var, covariance.col(cnt));
                    } else {
                        gp->Test(test_position, f, var, no_covariance);
                    }
                    if (f.tail<3>().norm() < 1.e-15) { continue; }  // invalid gradient, skip this GP
                }

                tested_idx.emplace_back(cnt++, j);
                if (m_setting_->test_query->use_nearest_only) { break; }
                if ((!need_weighted_sum) && (gp_indices.size() > 1) && (var[0] > m_setting_->test_query->max_test_valid_distance_var)) {
                    need_weighted_sum = true;
                }
                if ((!need_weighted_sum) || (cnt >= max_tries)) { break; }
            }

            // sort the results by distance variance
            if (tested_idx.size() > 1 && need_weighted_sum) {
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return variances(0, a.first) < variances(0, b.first); });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, tested_idx[0].first) * fs(0, tested_idx[1].first) < 0) { need_weighted_sum = false; }
            }

            // store the result
            if (need_weighted_sum) {
                if (m_test_buffer_.Size() == 1) { ERL_INFO("SDF1: {:f}, SDF2: {:f}", fs(0, tested_idx[0].first), fs(0, tested_idx[1].first)); }

                if (variances(0, tested_idx[0].first) < m_setting_->test_query->max_test_valid_distance_var) {
                    auto j = tested_idx[0].first;
                    // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs.col(j).tail<3>();
                    variance_out << variances.col(j);
                    if (m_setting_->test_query->compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(j); }
                    m_query_used_gps_[i][0] = gps[tested_idx[0].second].second;
                    m_query_used_gps_[i][1] = nullptr;
                } else {
                    // pick the best two results to do weighted sum
                    long j1 = tested_idx[0].first;
                    long j2 = tested_idx[1].first;
                    double w1 = variances(0, j1) - m_setting_->test_query->max_test_valid_distance_var;
                    double w2 = variances(0, j2) - m_setting_->test_query->max_test_valid_distance_var;
                    double w12 = w1 + w2;
                    // clang-format off
                    Eigen::Vector4d f = (fs.col(j1) * w2 + fs.col(j2) * w1) / w12;
                    distance_out = f[0];
                    gradient_out << f.tail<3>();
                    variance_out << (variances.col(j1) * w2 + variances.col(j2) * w1) / w12;
                    if (m_setting_->test_query->compute_covariance) {
                        m_test_buffer_.covariances->col(i) << (covariance.col(j1) * w2 + covariance.col(j2) * w1) / w12;
                    }
                    // clang-format on
                    m_query_used_gps_[i][0] = gps[tested_idx[0].second].second;
                    m_query_used_gps_[i][1] = gps[tested_idx[1].second].second;
                }
            } else {
                if (m_test_buffer_.Size() == 1) { ERL_INFO("SDF1: {:f}", fs(0, tested_idx[0].first)); }

                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs.col(0).tail<3>();
                variance_out << variances.col(0);
                if (m_setting_->test_query->compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(0); }
                m_query_used_gps_[i][0] = gps[tested_idx[0].second].second;
                m_query_used_gps_[i][1] = nullptr;
            }

            distance_out -= m_setting_->offset_distance;
            gradient_out.normalize();
        }
    }

}  // namespace erl::sdf_mapping
