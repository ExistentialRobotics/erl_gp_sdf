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
        Eigen::Matrix3Xd &variances_out,
        Eigen::Matrix3Xd &covariances_out) {

        if (positions_in.cols() == 0) { return false; }
        if (m_surface_mapping_ == nullptr) { return false; }
        if (m_surface_mapping_->GetQuadtree() == nullptr) { return false; }
        if (!m_test_buffer_.ConnectBuffers(  // allocate memory for test results
                positions_in,
                distances_out,
                gradients_out,
                variances_out,
                covariances_out,
                m_setting_->test_query->compute_covariance)) {
            return false;
        }

        unsigned int n = positions_in.cols();
        unsigned int num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        num_threads = std::min(num_threads, n);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        std::size_t batch_size = n / num_threads;
        std::size_t leftover = n - batch_size * num_threads;
        std::size_t start_idx, end_idx;

        std::chrono::high_resolution_clock::time_point t0, t1;
        double dt;
        {
            std::lock_guard<std::mutex> lock(m_mutex_);  // CRITICAL SECTION
            // Search GPs for each query position
            t0 = std::chrono::high_resolution_clock::now();
            m_query_to_gps_.clear();
            m_query_to_gps_.resize(n);  // allocate memory for n threads, collected GPs will be locked for testing
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
            dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
            ERL_INFO("Search GPs: %f us", dt);

            // Train any updated GPs
            if (!m_new_gps_.empty()) {
                t0 = std::chrono::high_resolution_clock::now();
                std::unordered_set<std::shared_ptr<GP>> new_gps;
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
                ERL_INFO("Train GPs: %f us", dt);
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
        dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Test GPs: %f us", dt);

        m_test_buffer_.DisconnectBuffers();

        for (auto &gps: m_query_to_gps_) {
            for (auto &gp: gps) { gp.second->locked_for_test = false; }  // unlock GPs
        }

        return true;
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

        if (m_setting_->train_gp_immediately) {  // new GPs are already trained in UpdateGpThread
            auto t1 = std::chrono::high_resolution_clock::now();
            auto dt = double(std::chrono::duration<double, std::micro>(t1 - t0).count());
            ERL_INFO("Update GPs' training data: %f us.", dt);
            return;
        }

        for (auto &cluster_key: m_clusters_to_update_) {
            auto it = m_gp_map_.find(cluster_key);
            if (it == m_gp_map_.end() || !it->second->active) { continue; }  // GP does not exist or deactivated (e.g. due to no training data)
            if (it->second->gp != nullptr && !it->second->gp->IsTrained()) { m_new_gps_.push(it->second); }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = double(std::chrono::duration<double, std::micro>(t1 - t0).count());
        ERL_INFO("Update GPs' training data: %f us.", dt);
        time_budget -= dt;

        // train as many new GPs as possible within the time limit
        if (time_budget <= m_train_gp_time_) {// no time left for training
            ERL_INFO("%zu GP(s) not trained yet due to time limit.", m_new_gps_.size());
            return;
        }

        auto max_num_gps_to_train = std::size_t(std::floor(time_budget / m_train_gp_time_));
        max_num_gps_to_train = std::min(max_num_gps_to_train, m_new_gps_.size());
        std::unordered_set<std::shared_ptr<GP>> gps_to_train;
        while (!m_new_gps_.empty() && gps_to_train.size() < max_num_gps_to_train) {
            auto maybe_new_gp = m_new_gps_.front();
            if (maybe_new_gp->active && maybe_new_gp->gp != nullptr && !maybe_new_gp->gp->IsTrained()) {
                gps_to_train.insert(maybe_new_gp);
            }
            m_new_gps_.pop();
        }
        m_gps_to_train_.clear();
        m_gps_to_train_.insert(m_gps_to_train_.end(), gps_to_train.begin(), gps_to_train.end());
        TrainGps();
        ERL_INFO("%zu GP(s) not trained yet due to time limit.", m_new_gps_.size());
    }

    void
    GpSdfMapping2D::UpdateGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        auto quadtree = m_surface_mapping_->GetQuadtree();
        if (quadtree == nullptr) { return; }
        double sensor_noise = m_surface_mapping_->GetSensorNoise();

        double cluster_size = quadtree->GetNodeSize(quadtree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel());
        double aabb_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2.;

        std::vector<std::pair<double, std::shared_ptr<SurfaceMappingQuadtreeNode::SurfaceData>>> surface_data_vec;
        surface_data_vec.reserve(1024);
        for (unsigned int i = start_idx; i < end_idx; ++i) {
            auto &cluster_key = m_clusters_to_update_[i];
            // get the GP of the cluster
            std::shared_ptr<GP> &gp = m_gp_map_.at(cluster_key);
            // testing thread may unlock the GP, but it is impossible to lock it here due to the mutex
            if (gp->locked_for_test) { gp = std::make_shared<GP>(); }  // create a new GP if the old one is locked for testing
            gp->active = true;                                         // activate the GP
            if (gp->gp == nullptr) { gp->gp = std::make_shared<LogSdfGaussianProcess>(m_setting_->gp_sdf); }

            // collect surface data in the area
            double x, y;
            quadtree->KeyToCoord(cluster_key, x, y);
            double aabb_min_x = x - aabb_half_size;
            double aabb_min_y = y - aabb_half_size;
            double aabb_max_x = x + aabb_half_size;
            double aabb_max_y = y + aabb_half_size;
            auto it = quadtree->BeginLeafInAabb(aabb_min_x, aabb_min_y, aabb_max_x, aabb_max_y);
            auto end = quadtree->EndLeafInAabb();
            long count = 0;
            surface_data_vec.clear();
            for (; it != end; ++it) {
                auto surface_data = it->GetSurfaceData();
                if (surface_data == nullptr) { continue; }
                double dx = surface_data->position.x() - x;
                double dy = surface_data->position.y() - y;
                double distance = std::sqrt(dx * dx + dy * dy);
                surface_data_vec.emplace_back(distance, surface_data);
                count++;
            }

            if (count == 0) {
                gp->active = false;  // deactivate the GP if there is no training data
                gp->num_train_samples = 0;
                gp->num_train_samples_with_grad = 0;
                continue;
            }

            // sort surface data by distance
            std::sort(surface_data_vec.begin(), surface_data_vec.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            if (surface_data_vec.size() > std::size_t(m_setting_->gp_sdf->max_num_samples)) { surface_data_vec.resize(m_setting_->gp_sdf->max_num_samples); }

            // prepare data for GP training
            gp->gp->Reset(long(surface_data_vec.size()), 2);
            auto &mat_x = gp->gp->GetTrainInputSamplesBuffer();
            auto &vec_y = gp->gp->GetTrainOutputSamplesBuffer();
            auto &vec_var_y = gp->gp->GetTrainOutputValueSamplesVarianceBuffer();
            auto &vec_var_grad = gp->gp->GetTrainOutputGradientSamplesVarianceBuffer();
            auto &vec_var_x = gp->gp->GetTrainInputSamplesVarianceBuffer();
            auto &vec_grad_flag = gp->gp->GetTrainGradientFlagsBuffer();
            Eigen::Matrix2Xd mat_valid_grad(2, count);
            count = 0;
            long valid_grad_count = 0;
            for (auto &[distance, surface_data]: surface_data_vec) {
                mat_x.col(count) = surface_data->position;
                vec_var_y[count] = sensor_noise;
                vec_var_grad[count] = surface_data->var_normal;
                if ((surface_data->var_normal > m_setting_->max_valid_gradient_var) ||  // invalid gradient
                    ((std::fabs(surface_data->normal.x()) < m_setting_->zero_gradient_threshold) &&
                     (std::fabs(surface_data->normal.y()) < m_setting_->zero_gradient_threshold))) {
                    vec_var_x[count] = m_setting_->invalid_position_var;  // position is unreliable
                    vec_grad_flag[count++] = false;
                    continue;
                }
                vec_var_x[count] = surface_data->var_position;
                vec_grad_flag[count++] = true;
                mat_valid_grad.col(valid_grad_count++) = surface_data->normal;
                if (count >= mat_x.cols()) { break; }  // reached max_num_samples
            }
            vec_y.head(count).setConstant(m_setting_->offset_distance);
            for (long j = 0; j < valid_grad_count; ++j) {
                long j1 = j + count;
                long j2 = j1 + valid_grad_count;
                vec_y[j1] = mat_valid_grad(0, j);
                vec_y[j2] = mat_valid_grad(1, j);
            }
            gp->num_train_samples = count;
            gp->num_train_samples_with_grad = valid_grad_count;
            if (m_setting_->train_gp_immediately) { gp->Train(); }
        }
    }

    void GpSdfMapping2D::TrainGps() {
            ERL_INFO("Training %zu GPs ...", m_gps_to_train_.size());
            auto t0 = std::chrono::high_resolution_clock::now();
            unsigned int n = m_gps_to_train_.size();
            if (n == 0) return;
            unsigned int num_threads = std::min(n, std::thread::hardware_concurrency());
            num_threads = std::min(num_threads, m_setting_->num_threads);
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            std::size_t batch_size = n / num_threads;
            std::size_t leftover = n - batch_size * num_threads;
            std::size_t start_idx = 0, end_idx;
            for (unsigned int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                end_idx = start_idx + batch_size;
                if (thread_idx < leftover) { end_idx++; }
                threads.emplace_back(&GpSdfMapping2D::TrainGpThread, this, thread_idx, start_idx, end_idx);
                start_idx = end_idx;
            }
            for (auto& thread: threads) { thread.join(); }
            m_gps_to_train_.clear();
            auto t1 = std::chrono::high_resolution_clock::now();
            double time = double(std::chrono::duration<double, std::micro>(t1 - t0).count()) / double(n);
            m_train_gp_time_ = m_train_gp_time_ * 0.4 + time * 0.6;
            ERL_INFO("Per GP training time: %f us.", m_train_gp_time_);
        }

    void
    GpSdfMapping2D::TrainGpThread(unsigned int thread_idx, std::size_t start_idx, std::size_t end_idx) {
        (void) thread_idx;
        for (std::size_t i = start_idx; i < end_idx; ++i) { m_gps_to_train_[i]->Train(); }
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
                auto it = quadtree->BeginTreeInAabb(  // search the quadtree for clusters in the search area
                    intersection.min().x(),
                    intersection.min().y(),
                    intersection.max().x(),
                    intersection.max().y(),
                    cluster_depth);
                auto end = quadtree->EndTreeInAabb();
                for (; it != end; ++it) {
                    if (it.GetDepth() != cluster_depth) { continue; }  // not a cluster node
                    geometry::QuadtreeKey cluster_key = it.GetIndexKey();
                    auto it_gp = m_gp_map_.find(cluster_key);
                    if (it_gp == m_gp_map_.end()) { continue; }  // no gp for this cluster
                    if (!it_gp->second->active) { continue; }    // gp is inactive (e.g. due to no training data)
                    Eigen::Vector2d cluster_center;
                    quadtree->KeyToCoord(cluster_key, cluster_center.x(), cluster_center.y());
                    double dx = cluster_center.x() - kPosition.x();
                    double dy = cluster_center.y() - kPosition.y();
                    it_gp->second->locked_for_test = true;  // lock the GP for testing
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
        Eigen::Matrix3Xd fs(3, kMaxTries);          // f, fGrad1, fGrad2
        Eigen::Matrix3Xd variances(3, kMaxTries);   // variances of f, fGrad1, fGrad2
        Eigen::Matrix3Xd covariance(3, kMaxTries);  // covariances of (fGrad1,f), (fGrad2,f), (fGrad2, fGrad1)
        Eigen::MatrixXd no_covariance;
        std::vector<long> tested_idx;
        tested_idx.reserve(kMaxTries);

        for (unsigned int i = start_idx; i < end_idx; ++i) {
            double &distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);

            distance_out = 0.;
            gradient_out.setZero();

            variances.setConstant(1e6);
            if (m_setting_->test_query->compute_covariance) { covariance.setConstant(1e6); }

            auto &gps = m_query_to_gps_[i];
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
                Eigen::Ref<Eigen::Vector3d> f = fs.col(cnt);           // distance, gradient_x, gradient_y
                Eigen::Ref<Eigen::VectorXd> var = variances.col(cnt);  // var_distance, var_gradient_x, var_gradient_y
                auto &gp = gps[j].second->gp;
                if (m_setting_->test_query->compute_covariance) {
                    gp->Test(kPosition, f, var, covariance.col(cnt));
                } else {
                    gp->Test(kPosition, f, var, no_covariance);
                }
                tested_idx.push_back(cnt++);
                if (m_setting_->test_query->use_nearest_only) { break; }
                if (m_setting_->test_query->use_surface_variance) {
                    auto &mat_x = gp->GetTrainInputSamplesBuffer();
                    auto &vec_x_var = gp->GetTrainInputSamplesVarianceBuffer();
                    Eigen::Vector2d pos(kPosition[0] - f[1] * f[0], kPosition[1] - f[2] * f[0]);
                    long num_samples = gp->GetNumTrainSamples();
                    Eigen::VectorXd weight(num_samples);
                    double weight_sum = 0;
                    double &var_sdf = var[0];
                    var_sdf = 0;
                    for (long k = 0; k < num_samples; ++k) {
                        double dx = mat_x(0, k) - pos.x();
                        double dy = mat_x(1, k) - pos.y();
                        double d = std::sqrt(dx * dx + dy * dy);
                        weight[k] = std::max(1.e-6, std::exp(-d * m_setting_->test_query->softmax_temperature));
                        weight_sum += weight[k];
                        var_sdf += weight[k] * vec_x_var[k];
                    }
                    var_sdf /= weight_sum;
                }
                if ((!need_weighted_sum) && (idx.size() > 1) && (var[0] > m_setting_->test_query->max_test_valid_distance_var)) { need_weighted_sum = true; }
                if ((!need_weighted_sum) || (cnt >= kMaxTries)) { break; }
            }

            // store the result
            if (need_weighted_sum) {
                // sort the results by distance variance
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](long j_1, long j_2) -> bool { return variances(0, j_1) < variances(0, j_2); });

                if (variances(0, tested_idx[0]) < m_setting_->test_query->max_test_valid_distance_var) {
                    auto j = tested_idx[0];
                    // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs(1, j), fs(2, j);
                    variance_out << variances.col(j);
                    if (m_setting_->test_query->compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(j); }
                } else {
                    // pick the best two results to do weighted sum
                    auto j_1 = tested_idx[0];
                    auto j_2 = tested_idx[1];
                    auto w_1 = variances(0, j_1) - m_setting_->test_query->max_test_valid_distance_var;
                    auto w_2 = variances(0, j_2) - m_setting_->test_query->max_test_valid_distance_var;
                    double w_12 = w_1 + w_2;
                    // clang-format off
                    distance_out = (fs(0, j_1) * w_2 + fs(0, j_2) * w_1) / w_12;
                    gradient_out << (fs(1, j_1) * w_2 + fs(1, j_2) * w_1) / w_12,
                                    (fs(2, j_1) * w_2 + fs(2, j_2) * w_1) / w_12;
                    variance_out <<
                        (variances(0, j_1) * w_2 + variances(0, j_2) * w_1) / w_12,
                        (variances(1, j_1) * w_2 + variances(1, j_2) * w_1) / w_12,
                        (variances(2, j_1) * w_2 + variances(2, j_2) * w_1) / w_12;
                    if (m_setting_->test_query->compute_covariance) {
                        m_test_buffer_.covariances->col(i) <<
                            (covariance(0, j_1) * w_2 + covariance(0, j_2) * w_1) / w_12,
                            (covariance(1, j_1) * w_2 + covariance(1, j_2) * w_1) / w_12,
                            (covariance(2, j_1) * w_2 + covariance(2, j_2) * w_1) / w_12;
                    }
                    // clang-format on
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs(1, 0), fs(2, 0);
                variance_out << variances.col(0);
                if (m_setting_->test_query->compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(0); }
            }

            distance_out -= m_setting_->offset_distance;
            gradient_out.normalize();
        }
    }

}  // namespace erl::sdf_mapping
