#include "erl_sdf_mapping/gp_sdf_mapping_2d.hpp"
#include <vector>
#include <thread>
#include <algorithm>

namespace erl::sdf_mapping {

    bool
    GpSdfMapping2D::Test(
        const Eigen::Ref<const Eigen::Matrix2Xd> &positions_in,
        Eigen::VectorXd &distances_out,
        Eigen::Matrix2Xd &gradients_out,
        Eigen::VectorXd &distance_variances_out,
        Eigen::Matrix2Xd &gradient_variances_out) {

        if (positions_in.cols() == 0) { return false; }
        if (m_test_buffer_.ConnectBuffers(positions_in, distances_out, gradients_out, distance_variances_out, gradient_variances_out)) {
            unsigned int n = positions_in.cols();
            unsigned int num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
            num_threads = std::min(num_threads, n);
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            std::size_t batch_size = n / num_threads;
            std::size_t leftover = n - batch_size * num_threads;
            std::size_t start_idx, end_idx;

            std::chrono::high_resolution_clock::time_point t0, t1;
            long dt;
            {
                std::lock_guard<std::mutex> lock(m_mutex_);
                // Search GPs for each query position
                t0 = std::chrono::high_resolution_clock::now();
                m_query_to_gps_.clear();
                m_query_to_gps_.resize(n);
                {
                    double x, y;
                    m_surface_mapping_->GetQuadtree()->GetMetricMin(x, y);  // trigger the quadtree to update its metric min/max
                }
                if (n == 1) {
                    SearchGpThread(0, 0, 1);  // save time on thread creation
                } else {
                    start_idx = 0;
                    for (unsigned int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                        end_idx = start_idx + batch_size;
                        if (thread_idx < leftover) { end_idx++; }
                        threads.emplace_back(&GpSdfMapping2D::SearchGpThread, this, thread_idx, start_idx, end_idx);
                        start_idx = end_idx;
                    }
                    for (auto &thread: threads) { thread.join(); }
                    threads.clear();
                }
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                ERL_INFO("Search GPs: %ld us", dt);

                // Train any new needed GPs
                if (m_might_need_training_) {
                    t0 = std::chrono::high_resolution_clock::now();
                    std::unordered_set<std::shared_ptr<GP>> new_gps;
                    for (auto &gps: m_query_to_gps_) {
                        for (auto &[distance, gp]: gps) {
                            if (gp->gp == nullptr) { new_gps.insert(gp); }
                        }
                    }
                    m_gps_to_train_.clear();
                    m_gps_to_train_.insert(m_gps_to_train_.end(), new_gps.begin(), new_gps.end());
                    TrainGps();
                    t1 = std::chrono::high_resolution_clock::now();
                    dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                    ERL_INFO("Train GPs: %ld us", dt);
                }

                // move gps from m_query_to_gps_ to m_query_to_test_gps_
                m_query_to_test_gps_.clear();
                m_query_to_test_gps_.resize(n);
                for (unsigned int i = 0; i < n; i++) {
                    for (auto &[distance, gp]: m_query_to_gps_[i]) {
                        if (gp->gp != nullptr) { m_query_to_test_gps_[i].emplace_back(distance, gp->gp); }
                    }
                }
            }

            // Compute the inference result for each query position
            t0 = std::chrono::high_resolution_clock::now();
            if (n == 1) {
                TestGpThread(0, 0, 1);  // save time on thread creation
            } else {
                start_idx = 0;
                for (unsigned int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                    end_idx = start_idx + batch_size;
                    if (thread_idx < leftover) end_idx++;
                    threads.emplace_back(&GpSdfMapping2D::TestGpThread, this, thread_idx, start_idx, end_idx);
                    start_idx = end_idx;
                }
                for (auto &thread: threads) { thread.join(); }
                threads.clear();
            }
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            ERL_INFO("Test GPs: %ld us", dt);

            m_test_buffer_.DisconnectBuffers();
            return true;
        }
        return false;
    }

    void
    GpSdfMapping2D::UpdateGps(double time_budget) {
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1");

        auto t0 = std::chrono::high_resolution_clock::now();
        // add affected clusters
        geometry::QuadtreeKeySet changed_clusters = m_surface_mapping_->GetChangedClusters();
        unsigned int cluster_level = m_surface_mapping_->GetClusterLevel();
        std::shared_ptr<SurfaceMappingQuadtree> quadtree = m_surface_mapping_->GetQuadtree();
        unsigned int cluster_depth = quadtree->GetTreeDepth() - cluster_level;
        double cluster_size = quadtree->GetNodeSize(cluster_depth);
        double area_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2;
        geometry::QuadtreeKeySet affected_clusters(changed_clusters);
        for (const auto &kClusterKey: changed_clusters) {
            double x, y;
            quadtree->KeyToCoord(kClusterKey, cluster_depth, x, y);
            auto it = quadtree->BeginLeafInAabb(x - area_half_size, y - area_half_size, x + area_half_size, y + area_half_size);
            auto end = quadtree->EndLeafInAabb();
            for (; it != end; ++it) { affected_clusters.insert(quadtree->AdjustKeyToDepth(it.GetKey(), cluster_depth)); }
        }

        // update GPs in affected clusters
        unsigned int num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        m_clusters_to_update_.clear();
        m_clusters_to_update_.insert(m_clusters_to_update_.end(), affected_clusters.begin(), affected_clusters.end());
        for (auto &cluster_key: m_clusters_to_update_) { m_gp_map_.try_emplace(cluster_key, std::make_shared<GP>()); }
        m_clusters_not_updated_.clear();
        m_clusters_not_updated_.resize(num_threads);
        std::size_t batch_size = m_clusters_to_update_.size() / num_threads;
        std::size_t left_over = m_clusters_to_update_.size() - batch_size * num_threads;
        std::size_t start_idx = 0;
        std::size_t end_idx = 0;
        for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (thread_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping2D::UpdateGpThread, this, thread_idx, start_idx, end_idx);
        }
        for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }

        for (auto &cluster_keys: m_clusters_not_updated_) {
            for (auto &cluster_key: cluster_keys) { m_gp_map_.erase(cluster_key); }
        }

        if (m_setting_->train_gp_immediately) {  // new GPs are already trained in UpdateGpThread
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = double(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
            ERL_INFO("Update GPs' training data: %f us.", dt);
            return;
        }

        std::unordered_set<std::shared_ptr<GP>> new_gps;
        for (auto &cluster_key: m_clusters_to_update_) {
            auto it = m_gp_map_.find(cluster_key);
            if (it == m_gp_map_.end()) { continue; }  // GP has been removed (e.g. due to no training data)
            if (it->second->gp == nullptr) { new_gps.insert(it->second); }
        }
        m_gps_to_train_.clear();
        m_gps_to_train_.insert(m_gps_to_train_.end(), new_gps.begin(), new_gps.end());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = double(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        ERL_INFO("Update GPs' training data: %f us.", dt);
        time_budget -= dt;

        // train as many new GPs as possible within the time limit
        if (time_budget <= m_train_gp_time_) {  // no time left for training
            m_might_need_training_ = true;
            return;
        }
        unsigned long max_num_gps_to_train = std::floor(time_budget / m_train_gp_time_);
        if (m_gps_to_train_.size() > max_num_gps_to_train) {
            m_gps_to_train_.resize(max_num_gps_to_train);
            m_might_need_training_ = true;
        }
        TrainGps();
    }

    void
    GpSdfMapping2D::UpdateGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        if (m_surface_mapping_ == nullptr) { return; }
        auto quadtree = m_surface_mapping_->GetQuadtree();
        if (quadtree == nullptr) { return; }

        double cluster_size = quadtree->GetNodeSize(quadtree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel());
        double aabb_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2.;

        std::vector<std::shared_ptr<SurfaceMappingQuadtreeNode::SurfaceData>> surface_data_vec;
        surface_data_vec.reserve(1024);
        for (unsigned int i = start_idx; i < end_idx; ++i) {
            auto &cluster_key = m_clusters_to_update_[i];
            // get the GP of the cluster
            std::shared_ptr<GP> gp = m_gp_map_.at(cluster_key);
            ERL_ASSERTM(gp != nullptr, "GP must exist");

            // collect surface data in the area
            double x, y;
            quadtree->KeyToCoord(cluster_key, x, y);
            double aabb_min_x = x - aabb_half_size;
            double aabb_min_y = y - aabb_half_size;
            double aabb_max_x = x + aabb_half_size;
            double aabb_max_y = y + aabb_half_size;
            auto it = quadtree->BeginLeafInAabb(aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y);
            auto end = quadtree->EndLeafInAabb();
            unsigned int count = 0;
            surface_data_vec.clear();
            for (; it != end; ++it) {
                auto surface_data = it->GetSurfaceData();
                if (surface_data == nullptr) { continue; }
                surface_data_vec.push_back(surface_data);
                count++;
            }

            if (count == 0) {
                m_clusters_not_updated_[thread_idx].push_back(cluster_key);
                continue;
            }

            // prepare data for GP training
            gp->mat_x.resize(2, count);
            gp->vec_sigma_x.resize(count);
            gp->vec_sigma_grad.resize(count);
            gp->vec_grad_flag.resize(count);
            Eigen::Matrix2Xd mat_valid_grad(2, count);
            count = 0;
            unsigned int valid_grad_count = 0;
            for (auto &surface_data: surface_data_vec) {
                gp->mat_x.col(count) = surface_data->position;
                gp->vec_sigma_grad[count] = surface_data->var_normal;
                if ((surface_data->var_normal > m_setting_->max_valid_gradient_var) ||
                    ((std::fabs(surface_data->normal.x()) < m_setting_->zero_gradient_threshold) &&
                     (std::fabs(surface_data->normal.y()) < m_setting_->zero_gradient_threshold))) {
                    gp->vec_sigma_x[count] = m_setting_->invalid_position_var;
                    gp->vec_grad_flag[count++] = false;
                    continue;
                }
                gp->vec_sigma_x[count] = surface_data->var_position;
                gp->vec_grad_flag[count++] = true;
                mat_valid_grad.col(valid_grad_count++) = surface_data->normal;
            }
            // ERL_ASSERTM(count == gp->mat_x.cols(), "count: %d, gp.mat_x.cols(): %ld", count, gp->mat_x.cols());
            gp->vec_y.resize(count + 2 * valid_grad_count);
            gp->vec_y.head(count).array() = m_setting_->offset_distance;
            mat_valid_grad.conservativeResize(2, valid_grad_count);
            gp->vec_y.tail(2 * valid_grad_count) = mat_valid_grad.reshaped<Eigen::RowMajor>(2 * valid_grad_count, 1);
            gp->gp = nullptr;
            if (m_setting_->train_gp_immediately) { TrainGp(gp); }
        }
    }

    void
    GpSdfMapping2D::TrainGp(const std::shared_ptr<GP> &gp) {
        if (gp->gp != nullptr) { return; }
        Eigen::VectorXd vec_sigma_dist = Eigen::VectorXd::Constant(gp->mat_x.cols(), m_surface_mapping_->GetSensorNoise());
        gp->gp = GpSdf::Create(m_setting_->gp_sdf);
        gp->gp->Train(gp->mat_x, gp->vec_grad_flag, gp->vec_y, gp->vec_sigma_x, vec_sigma_dist, gp->vec_sigma_grad);
    }

    void
    GpSdfMapping2D::TrainGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        for (unsigned int i = start_idx; i < end_idx; ++i) { TrainGp(m_gps_to_train_[i]); }
    }

    void
    GpSdfMapping2D::SearchGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        if (m_surface_mapping_ == nullptr) { return; }
        auto quadtree = m_surface_mapping_->GetQuadtree();
        unsigned int cluster_level = m_surface_mapping_->GetClusterLevel();
        unsigned int cluster_depth = quadtree->GetTreeDepth() - cluster_level;
        for (unsigned int i = start_idx; i < end_idx; ++i) {
            const auto &kPosition = m_test_buffer_.positions->col(i);
            std::vector<std::pair<double, std::shared_ptr<GP>>> &gps = m_query_to_gps_[i];
            gps.reserve(16);
            double search_area_half_size = m_setting_->test_query->search_area_half_size;
            double tree_min_x, tree_min_y, tree_max_x, tree_max_y;
            quadtree->GetMetricMinMax(tree_min_x, tree_min_y, tree_max_x, tree_max_y);
            Eigen::AlignedBox2d quadtree_aabb(Eigen::Vector2d(tree_min_x, tree_min_y), Eigen::Vector2d(tree_max_x, tree_max_y));
            Eigen::AlignedBox2d search_aabb(
                Eigen::Vector2d(kPosition.x() - search_area_half_size, kPosition.y() - search_area_half_size),
                Eigen::Vector2d(kPosition.x() + search_area_half_size, kPosition.y() + search_area_half_size));
            Eigen::AlignedBox2d intersection = quadtree_aabb.intersection(search_aabb);
            while (intersection.sizes().prod() > 0) {
                auto it =
                    quadtree->BeginTreeInAabb(intersection.min().x(), intersection.min().y(), intersection.max().x(), intersection.max().y(), cluster_depth);
                auto end = quadtree->EndTreeInAabb();
                for (; it != end; ++it) {
                    if (it.GetDepth() != cluster_depth) { continue; }  // not a cluster node
                    auto cluster_key = it.GetIndexKey();
                    auto it_gp = m_gp_map_.find(cluster_key);
                    if (it_gp == m_gp_map_.end()) { continue; }  // no gp for this cluster
                    Eigen::Vector2d cluster_center;
                    quadtree->KeyToCoord(cluster_key, cluster_center.x(), cluster_center.y());
                    double dx = cluster_center.x() - kPosition.x();
                    double dy = cluster_center.y() - kPosition.y();
                    gps.emplace_back(dx * dx + dy * dy, it_gp->second);
                }
                if (!gps.empty()) { break; }  // found at least one gp
                search_area_half_size *= 2;   // double search area size
                search_aabb = Eigen::AlignedBox2d(
                    Eigen::Vector2d(kPosition.x() - search_area_half_size, kPosition.y() - search_area_half_size),
                    Eigen::Vector2d(kPosition.x() + search_area_half_size, kPosition.y() + search_area_half_size));
                auto new_intersection = quadtree_aabb.intersection(search_aabb);
                if ((intersection.min() == new_intersection.min()) && (intersection.max() == new_intersection.max())) { break; }  // intersection did not change
                intersection = new_intersection;
            }
        }
    }

    void
    GpSdfMapping2D::TestGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        if (m_surface_mapping_ == nullptr) { return; }

        std::vector<size_t> idx;
        const int kMaxTries = 4;
        Eigen::Matrix3Xd fs(3, kMaxTries);    // f, fGrad1, fGrad2
        Eigen::Matrix3Xd vars(3, kMaxTries);  // variances of f, fGrad1, fGrad2
        std::vector<long> tested_idx;
        tested_idx.reserve(kMaxTries);

        for (unsigned int i = start_idx; i < end_idx; ++i) {
            double &distance_out = (*m_test_buffer_.distances)[i];
            double &distance_variance_out = (*m_test_buffer_.distance_variances)[i];
            auto gradient_out = (*m_test_buffer_.gradients).col(i);
            auto gradient_variance_out = (*m_test_buffer_.gradient_variances).col(i);

            distance_out = 0.;
            distance_variance_out = 1e6;
            gradient_out.setZero();
            gradient_variance_out.setConstant(1e6);

            auto &gps = m_query_to_test_gps_[i];
            if (gps.empty()) { continue; }

            const auto &kPosition = m_test_buffer_.positions->col(i);
            idx.resize(gps.size());
            std::iota(idx.begin(), idx.end(), 0);
            if (gps.size() > 1) {
                std::stable_sort(idx.begin(), idx.end(), [&gps](size_t i1, size_t i2) { return gps[i1].first < gps[i2].first; });
            }

            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            for (auto &j: idx) {
                // call selected GPs for inference
                Eigen::Ref<Eigen::Vector3d> f = fs.col(cnt);      // distance, gradient_x, gradient_y
                Eigen::Ref<Eigen::Vector3d> var = vars.col(cnt);  // var_distance, var_gradient_x, var_gradient_y
                gps[j].second->Test(kPosition, f, var);
                tested_idx.push_back(cnt++);
                if (m_setting_->test_query->use_nearest_only) { break; }
                if ((!need_weighted_sum) && (idx.size() > 1) && (var[0] > m_setting_->test_query->max_test_valid_distance_var)) { need_weighted_sum = true; }
                if ((!need_weighted_sum) || (cnt >= kMaxTries)) { break; }
            }

            // store the result
            if (need_weighted_sum) {
                // sort the results by distance variance
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](long j_1, long j_2) -> bool { return vars(0, j_1) < vars(0, j_2); });

                if (vars(0, tested_idx[0]) < m_setting_->test_query->max_test_valid_distance_var) {
                    auto j = tested_idx[0];
                    // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs(1, j), fs(2, j);
                    distance_variance_out = vars(0, j);
                    gradient_variance_out << vars(1, j), vars(2, j);
                } else {
                    // pick the best two results to do weighted sum
                    auto j_1 = tested_idx[0];
                    auto j_2 = tested_idx[1];
                    auto w_1 = vars(0, j_1) - m_setting_->test_query->max_test_valid_distance_var;
                    auto w_2 = vars(0, j_2) - m_setting_->test_query->max_test_valid_distance_var;
                    double w_12 = w_1 + w_2;
                    // clang-format off
                    distance_out = (fs(0, j_1) * w_2 + fs(0, j_2) * w_1) / w_12;
                    gradient_out << (fs(1, j_1) * w_2 + fs(1, j_2) * w_1) / w_12,
                                    (fs(2, j_1) * w_2 + fs(2, j_2) * w_1) / w_12;
                    distance_variance_out = (vars(0, j_1) * w_2 + vars(0, j_2) * w_1) / w_12;
                    gradient_variance_out << (vars(1, j_1) * w_2 + vars(1, j_2) * w_1) / w_12,
                                             (vars(2, j_1) * w_2 + vars(2, j_2) * w_1) / w_12;
                    // clang-format on
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs(1, 0), fs(2, 0);
                distance_variance_out = vars(0, 0);
                gradient_variance_out << vars(1, 0), vars(2, 0);
            }

            distance_out -= m_setting_->offset_distance;
            gradient_out.normalize();
        }
    }

}  // namespace erl::sdf_mapping
