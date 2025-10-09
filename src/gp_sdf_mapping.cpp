#include "erl_gp_sdf/gp_sdf_mapping.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/tracy.hpp"
#include "erl_geometry/marching_cubes.hpp"
#include "erl_geometry/marching_squares.hpp"

#include <utility>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::TestBuffer::ConnectBuffers(
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
        this->positions = std::make_unique<Eigen::Ref<const Positions>>(positions_in);
        this->distances = std::make_unique<Eigen::Ref<Distances>>(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<Gradients>>(gradients_out);
        this->variances = std::make_unique<Eigen::Ref<Variances>>(variances_out);
        this->covariances = std::make_unique<Eigen::Ref<Covariances>>(covariances_out);
        return true;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TestBuffer::DisconnectBuffers() {
        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TestBuffer::PrepareGpBuffer(
        const long num_queries,
        const long num_neighbor_gps) {
        // (num_queries, 2 * Dim + 1, num_neighbor_gps)
        const long rows = num_neighbor_gps * (2 * Dim + 1);
        if (gp_buffer.rows() < rows || gp_buffer.cols() < num_queries) {
            gp_buffer.setConstant(rows, num_queries, 0.0f);
        }
    }

    template<typename Dtype, int Dim>
    GpSdfMapping<Dtype, Dim>::GpSdfMapping(
        std::shared_ptr<Setting> setting,
        std::shared_ptr<SurfaceMapping> surface_mapping)
        : m_setting_(std::move(setting)),
          m_surface_mapping_(std::move(surface_mapping)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1.");
    }

    template<typename Dtype, int Dim>
    std::lock_guard<std::mutex>
    GpSdfMapping<Dtype, Dim>::GetLockGuard() {
        return std::lock_guard<std::mutex>(m_mutex_);
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpSdfMapping<Dtype, Dim>::Setting>
    GpSdfMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<AbstractSurfaceMapping<Dtype, Dim>>
    GpSdfMapping<Dtype, Dim>::GetSurfaceMapping() const {
        return m_surface_mapping_;
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &scan,
        bool are_points,
        bool are_local) {

        double surf_mapping_time;
        bool ok;
        {
            ERL_BLOCK_TIMER_MSG_TIME("Surface mapping update", surf_mapping_time);
            ok = m_surface_mapping_->Update(rotation, translation, scan, are_points, are_local);
        }

        if (ok) {
            const double time_budget_us = 1e6 / m_setting_->update_hz;  // us
            ERL_BLOCK_TIMER_MSG("Update SDF GPs");
            UpdateGpSdf(time_budget_us - surf_mapping_time * 1000);
        }

        return ok;
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::UpdateGpSdf(double time_budget_us) {
        ERL_TRACY_FRAME_MARK_START();
        ERL_BLOCK_TIMER_MSG("UpdateGpSdf");  // start timer

        CollectChangedClusters();
        UpdateClusterQueue();

        // train GPs if we still have time
        const auto dt = timer.Elapsed<double, std::micro>();
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

    template<typename Dtype, int Dim>
    [[nodiscard]] bool
    GpSdfMapping<Dtype, Dim>::Test(
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

#pragma region test_prepare_gps

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
#pragma endregion

#pragma region test_search_gps
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
            if (new_candidate_gps.empty()) {
                ERL_WARN("No active and trained GPs available for testing.");
                m_candidate_gps_.clear();
                return false;
            }
            m_candidate_gps_ = std::move(new_candidate_gps);
            gp_positions.conservativeResize(Dim, m_candidate_gps_.size());
            m_kdtree_candidate_gps_ = std::make_shared<KdTree>(std::move(gp_positions));
        }

        std::vector<std::vector<std::size_t>> no_gps_indices(num_threads);
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
#pragma endregion

#pragma region test_gps
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
#pragma endregion

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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::GetMesh(
        const Position &boundary_size,
        const Rotation &boundary_rotation,
        const Position &boundary_center,
        const Dtype resolution,
        const Dtype iso_value,
        std::vector<Position> &surface_points,
        std::vector<Face> &faces,
        std::vector<Gradient> &face_normals) const {

        using GridShape = Eigen::Vector<long, Dim>;
        using VoxelCoord = Eigen::Vector<long, Dim>;
        using EdgeCoord = Eigen::Vector<long, Dim + 1>;
        using MC = std::conditional_t<Dim == 2, geometry::MarchingSquares, geometry::MarchingCubes>;
        using namespace common;

        // 1. create grid
        ERL_INFO("Creating grid for mesh extraction");
        GridShape grid_shape;
        Position grid_resolution;
        for (int i = 0; i < Dim; ++i) {
            grid_shape[i] = static_cast<int>(std::ceil(boundary_size[i] / resolution));
            grid_resolution[i] = boundary_size[i] / static_cast<Dtype>(grid_shape[i]);
        }
        const GridShape grid_strides = ComputeCStrides<long, Dim>(grid_shape, 1);
        const Position bound_min = boundary_size.array() * -0.5f;
        auto old_test_query = m_setting_->test_query;  // backup
        m_setting_->test_query.compute_gradient = false;
        m_setting_->test_query.compute_gradient_variance = false;
        m_setting_->test_query.compute_covariance = false;

        // 2. find voxels that are near the surface, i.e. in any cluster
        ERL_INFO("Finding voxels near the surface");
        const KeySet clusters = m_surface_mapping_->GetAllClusters();
        Positions cluster_centers(Dim, clusters.size());
        {
            long idx = 0;
            for (const auto &key: clusters) {
                cluster_centers.col(idx++) = m_surface_mapping_->GetClusterCenter(key);
            }
        }
        const Dtype scaling = 1.0f / m_surface_mapping_->GetScaling();
        cluster_centers *= scaling;
        Dtype radius = m_surface_mapping_->GetClusterSize() * scaling + resolution;
        radius *= std::sqrt(static_cast<Dtype>(Dim));
        KdTree kdtree_clusters(cluster_centers);
        const long n_voxels = grid_shape.prod();
        Eigen::VectorXb flags_near_surface(n_voxels);
#pragma omp parallel for schedule(static) default(none) \
    shared(n_voxels,                                    \
               grid_strides,                            \
               flags_near_surface,                      \
               clusters,                                \
               bound_min,                               \
               grid_resolution,                         \
               boundary_rotation,                       \
               boundary_center,                         \
               kdtree_clusters,                         \
               radius)
        for (long voxel_idx = 0; voxel_idx < n_voxels; ++voxel_idx) {
            VoxelCoord voxel_coord = IndexToCoordsWithStrides(grid_strides, voxel_idx, true);
            Position voxel_center;
            for (int i = 0; i < Dim; ++i) {
                voxel_center[i] = GridToMeter(voxel_coord[i], bound_min[i], grid_resolution[i]);
            }
            voxel_center = boundary_rotation * voxel_center + boundary_center;
            // Key cluster_key = m_surface_mapping_->GetClusterKey(voxel_center);
            // flags_near_surface[voxel_idx] = clusters.contains(cluster_key);
            std::vector<nanoflann::ResultItem<long, Dtype>> indices_dists;
            kdtree_clusters.RadiusSearch(voxel_center, radius, indices_dists, false);
            flags_near_surface[voxel_idx] = !indices_dists.empty();
        }

        struct Voxel {
            int idx = -1;
            VoxelCoord coord = VoxelCoord::Zero();
            int surf_config = 0;
            std::vector<EdgeCoord> unique_edges;
            std::vector<Face> faces;

            Voxel(const int idx, VoxelCoord coord_)
                : idx(idx),
                  coord(std::move(coord_)) {}
        };

        std::vector<Voxel> near_surface_voxels;
        near_surface_voxels.reserve(clusters.size() * (1 << (Dim - 1)));
        for (int i = 0; i < n_voxels; ++i) {
            if (flags_near_surface[i]) {
                near_surface_voxels.emplace_back(
                    i,
                    IndexToCoordsWithStrides<long, Dim>(grid_strides, i, true));
            }
        }

        // 3. Find unique vertices among the near-surface voxels
        ERL_INFO("Finding unique vertices of {} voxels", near_surface_voxels.size());
        constexpr int n_vertices = 1 << Dim;
        const std::size_t num_threads = std::thread::hardware_concurrency();
        const std::size_t batch_size = near_surface_voxels.size() / num_threads;
        std::vector<std::vector<std::pair<VoxelCoord, Position>>> vertices_batches(num_threads);
        std::vector<absl::flat_hash_set<VoxelCoord>> vertex_sets(num_threads);
#pragma omp parallel for default(none) \
    shared(num_threads,                \
               batch_size,             \
               near_surface_voxels,    \
               vertices_batches,       \
               vertex_sets,            \
               bound_min,              \
               grid_resolution)
        for (std::size_t tidx = 0; tidx < num_threads; ++tidx) {
            std::size_t start_idx = tidx * batch_size;
            std::size_t end_idx =
                (tidx == num_threads - 1) ? near_surface_voxels.size() : start_idx + batch_size;
            std::vector<std::pair<VoxelCoord, Position>> &vertices = vertices_batches[tidx];
            absl::flat_hash_set<VoxelCoord> &vertex_set = vertex_sets[tidx];
            vertices.reserve((end_idx - start_idx) * n_vertices);
            vertex_set.reserve((end_idx - start_idx) * n_vertices / 2);
            for (std::size_t idx = start_idx; idx < end_idx; ++idx) {
                const Voxel &voxel = near_surface_voxels[idx];
                VoxelCoord vertex_coord;
                for (int i = 0; i < n_vertices; ++i) {
                    const int *vertex_code = MC::GetVertexCode(i);
                    // compute vertex coordinates
                    for (int dim = 0; dim < Dim; ++dim) {
                        vertex_coord[dim] = voxel.coord[dim] + vertex_code[dim];
                    }
                    // check if the vertex exists
                    auto [it, inserted] = vertex_set.insert(vertex_coord);
                    if (!inserted) { continue; }
                    Position vertex_pos;
                    for (int dim = 0; dim < Dim; ++dim) {
                        vertex_pos[dim] = VertexIndexToMeter<Dtype>(
                            vertex_coord[dim],
                            bound_min[dim],
                            grid_resolution[dim]);
                    }
                    vertices.emplace_back(vertex_coord, vertex_pos);
                }
            }
        }
        // merge vertices from all threads into a single unique set
        std::size_t n_unique_vertices = std::accumulate(
            vertices_batches.begin(),
            vertices_batches.end(),
            0,
            [](std::size_t sum, const std::vector<std::pair<VoxelCoord, Position>> &batch) {
                return sum + batch.size();
            });
        Positions vertices(Dim, n_unique_vertices);
        absl::flat_hash_map<VoxelCoord, long> vertex_map;
        vertex_map.reserve(n_unique_vertices);
        for (const auto &vertices_batch: vertices_batches) {
            auto idx = static_cast<long>(vertex_map.size());
            for (const auto &[vertex_coord, vertex_pos]: vertices_batch) {
                // check if the vertex exists
                auto [it, inserted] = vertex_map.try_emplace(vertex_coord, idx);
                if (!inserted) { continue; }  // vertex already exists
                vertices.col(idx++) = vertex_pos;
            }
        }
        vertices.conservativeResize(Eigen::NoChange, vertex_map.size());
        // transform to world coordinates
        vertices = (boundary_rotation * vertices).colwise() + boundary_center;

        // 4. query SDF at voxels' vertices
        VectorX sdf_values = VectorX::Zero(vertices.cols());
        Gradients gradients(Dim, vertices.cols());
        Variances variances = Variances::Zero(Dim + 1, vertices.cols());
        Covariances covariances;
        ERL_INFO("Querying SDF at {} vertices", vertices.cols());
        const bool success = const_cast<GpSdfMapping *>(this)
                                 ->Test(vertices, sdf_values, gradients, variances, covariances);
        if (!success) {
            ERL_WARN("Failed to query SDF at voxel vertices");
            return;
        }

        // 5. for each voxel, compute surface config
        ERL_INFO("Computing surface configurations for voxels");
#pragma omp parallel for schedule(static) default(none) \
    shared(near_surface_voxels, sdf_values, vertex_map, n_vertices, iso_value)
        for (Voxel &voxel: near_surface_voxels) {
            // collect SDF values at the vertices of the current voxel
            VectorX vertex_values(n_vertices);
            VoxelCoord vertex_coord;
            for (int i = 0; i < n_vertices; ++i) {
                const int *vertex_code = MC::GetVertexCode(i);
                for (int dim = 0; dim < Dim; ++dim) {  // compute vertex coordinates
                    vertex_coord[dim] = voxel.coord[dim] + vertex_code[dim];
                }
                vertex_values[i] = sdf_values[vertex_map.at(vertex_coord)];
            }
            // calculate the surface configuration index based on the vertex SDF values
            voxel.surf_config = MC::CalculateVertexConfigIndex(vertex_values.data(), iso_value);
            const int *unique_edge_indices = MC::GetUniqueEdgeIndices(voxel.surf_config);
            if (unique_edge_indices == nullptr) { continue; }
            int col = 0;
            EdgeCoord edge_coord;
            voxel.unique_edges.reserve(2);
            while (unique_edge_indices[col] != -1) {
                const int *edge_code = MC::GetEdgeCode(unique_edge_indices[col++]);
                for (int dim = 0; dim < Dim; ++dim) {
                    edge_coord[dim] = voxel.coord[dim] + edge_code[dim];
                }
                edge_coord[Dim] = edge_code[Dim];
                voxel.unique_edges.emplace_back(edge_coord);
            }
            const int *vertex_indices = MC::GetVertexIndices(voxel.surf_config);
            while (*vertex_indices != -1) {
                Face face;
                // ref:
                // https://github.com/ExistentialRobotics/erl_geometry/blob/main/src/marching_cubes.cpp#L1168-L1170
                for (int dim = 0; dim < Dim; ++dim) { face[Dim - dim - 1] = *vertex_indices++; }
                voxel.faces.push_back(face);
            }
        }

        // 6. interpolation of surface points on the unique edges
        ERL_INFO("Interpolating surface points on unique edges");
        std::size_t n_unique_edges = std::accumulate(
            near_surface_voxels.begin(),
            near_surface_voxels.end(),
            0,
            [](std::size_t sum, const Voxel &voxel) { return sum + voxel.unique_edges.size(); });
        std::vector<EdgeCoord> unique_edges;
        unique_edges.reserve(n_unique_edges);
        absl::flat_hash_map<EdgeCoord, long> edge_map;
        edge_map.reserve(n_unique_edges);
        for (const Voxel &voxel: near_surface_voxels) {
            for (const EdgeCoord &edge_coord: voxel.unique_edges) {
                auto idx = static_cast<long>(unique_edges.size());
                auto [it, inserted] = edge_map.try_emplace(edge_coord, idx);
                if (!inserted) { continue; }  // edge already exists
                unique_edges.emplace_back(edge_coord);
            }
        }
        n_unique_edges = unique_edges.size();
        surface_points.resize(n_unique_edges);
#pragma omp parallel for schedule(static) default(none) \
    shared(n_unique_edges,                              \
               unique_edges,                            \
               vertex_map,                              \
               sdf_values,                              \
               vertices,                                \
               gradients,                               \
               surface_points,                          \
               iso_value)
        for (long i = 0; i < static_cast<long>(n_unique_edges); ++i) {
            const EdgeCoord &edge_coord = unique_edges[i];
            VoxelCoord v1_coord = edge_coord.template head<Dim>();
            VoxelCoord v2_coord = edge_coord.template head<Dim>();
            ++v2_coord[edge_coord[Dim] - 1];
            long vid1 = vertex_map.at(v1_coord);
            long vid2 = vertex_map.at(v2_coord);
            constexpr Dtype kEpsilon = 1e-6f;
            const Dtype val1 = sdf_values[vid1];
            const Dtype val2 = sdf_values[vid2];
            const Dtype val_diff = val1 - val2;
            Dtype *p = surface_points[i].data();
            const Dtype *p1 = vertices.col(vid1).data();
            const Dtype *p2 = vertices.col(vid2).data();
            if (std::abs(val_diff) >= kEpsilon) {
                Dtype t = (val1 - iso_value) / val_diff;
                for (int dim = 0; dim < Dim; ++dim) { p[dim] = p1[dim] + t * (p2[dim] - p1[dim]); }
            } else {
                for (int dim = 0; dim < Dim; ++dim) { p[dim] = 0.5f * (p1[dim] + p2[dim]); }
            }
        }

        // 7. merge the resulting meshes from all voxels into a single mesh
        ERL_INFO("Merging meshes from {} voxels", near_surface_voxels.size());
        std::vector<std::size_t> start_indices;
        start_indices.reserve(near_surface_voxels.size());
        std::size_t n_faces = 0;
        for (const Voxel &voxel: near_surface_voxels) {
            start_indices.push_back(n_faces);
            n_faces += voxel.faces.size();
        }
        faces.clear();
        faces.resize(n_faces);
        face_normals.resize(n_faces);
#pragma omp parallel for schedule(static) default(none) \
    shared(start_indices, near_surface_voxels, edge_map, surface_points, faces, face_normals)
        for (std::size_t i = 0; i < near_surface_voxels.size(); ++i) {
            const Voxel &voxel = near_surface_voxels[i];
            std::size_t idx = start_indices[i];
            for (const Face &face: voxel.faces) {
                auto &face_out = faces[idx];
                auto &normal_out = face_normals[idx];
                for (int dim = 0; dim < Dim; ++dim) {
                    face_out[dim] = edge_map.at(voxel.unique_edges[face[dim]]);
                }
                if (Dim == 3) {
                    // compute face normal
                    const Position &v0 = surface_points[face_out[0]];
                    const Position &v1 = surface_points[face_out[1]];
                    const Position &v2 = surface_points[face_out[2]];
                    Position v10 = v1 - v0;
                    Position v20 = v2 - v0;
                    normal_out[0] = v10[1] * v20[2] - v10[2] * v20[1];
                    normal_out[1] = v10[2] * v20[0] - v10[0] * v20[2];
                    normal_out[2] = v10[0] * v20[1] - v10[1] * v20[0];
                    normal_out.normalize();
                } else if (Dim == 2) {
                    const Position &v0 = surface_points[face_out[0]];
                    const Position &v1 = surface_points[face_out[1]];
                    face_normals[idx][0] = v1[1] - v0[1];
                    face_normals[idx][1] = v0[0] - v1[0];
                    face_normals[idx].normalize();
                }
                ++idx;
            }
        }

        // 8. cleanup
        m_setting_->test_query = old_test_query;  // restore original settings
        ERL_INFO("Finished mesh extraction");
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::Write(std::ostream &s) const {
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

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::Read(std::istream &s) {
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

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::operator==(const GpSdfMapping &other) const {
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
        // when m_cluster_queue_keys_ is the same, m_cluster_queue_ is the same
        // m_clusters_to_train_, m_candidate_gps_, m_kdtree_candidate_gps_, m_map_boundary_,
        // m_query_to_gps_, m_query_signs_, m_test_buffer and m_query_used_gps_ are temporary data.
        if (m_train_gp_time_us_ != other.m_train_gp_time_us_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::CollectChangedClusters() {
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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::UpdateClusterQueue() {
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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TrainGps() {
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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TrainGpThread(
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

        std::vector<std::pair<Dtype, std::size_t>> surface_data_indices;
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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::SearchCandidateGps(const Eigen::Ref<const Positions> &positions_in) {
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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::SearchGpThread(
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
            std::vector<std::pair<Dtype, KeyGpPair>> &gps = m_query_to_gps_[i];

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

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::SearchGpFallback(const std::vector<std::size_t> &no_gps_indices) {
        if (no_gps_indices.empty()) { return; }

        // CRITICAL SECTION: access m_surface_mapping_ and m_gp_map_
        auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        auto lock = GetLockGuard();

        ERL_WARN_COND(
            !no_gps_indices.empty(),
            "Run fallback search for {} query positions.",
            no_gps_indices.size());

#pragma omp parallel for default(none) shared(no_gps_indices) schedule(dynamic)
        for (const std::size_t &i: no_gps_indices) {
            auto &gps = m_query_to_gps_[i];
            // failed to find GPs in the kd-tree, fall back to search clusters in the area
            // double search area size
            Dtype search_area_hs = 2.0f * m_setting_->test_query.search_area_half_size;
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
                search_area_hs *= 2.0f;       // double search area size
                Aabb new_area = m_map_boundary_.Intersection({test_position, search_area_hs});
                if (new_area.IsValid() && (search_area.min() == new_area.min()) &&
                    (search_area.max() == new_area.max())) {
                    break;  // no need to search again
                }
                search_area = std::move(new_area);  // update area
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TestGpThread(
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
        std::vector<std::pair<long, long>> tested_idx;  // (column index, gps index)
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
            fs.setZero();
            distance_out = m_setting_->test_query.default_invalid_sdf;
            gradient_out.setZero();
            variances.setConstant(1e6);
            variance_out.setConstant(1e6);
            covariances.setZero();
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
            Dtype sign = m_query_signs_[i];  //, sign_sum = 0;
            for (const std::size_t &j: gp_indices) {
                // call selected GPs for inference
                const auto &gp = gps[j].second.second;              // (distance, (key, gp))
                if (!gp->active || !gp->IsTrained()) { continue; }  // skip inactive / untrained GPs
                if (!gp->Test(
                        test_position,
                        fs.col(cnt),
                        variances.col(cnt),
                        covariances.col(cnt),
                        sign,
                        compute_gradient,
                        compute_gradient_variance,
                        compute_covariance,
                        use_gp_covariance)) {
                    continue;
                }
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
                    if (compute_gradient) {
                        gradient_out << fs.col(j).template segment<Dim>(1);
                        gradient_out.normalize();
                    }
                    variance_out << variances.col(j);
                    if (compute_covariance) {
                        m_test_buffer_.covariances->col(i) = covariances.col(j);
                    }
                    used_gps[0] = gps[tested_idx[0].second].second.second;

                    ERL_DEBUG_ASSERT(
                        std::isfinite(gradient_out.norm()),
                        "gradient norm is not finite: {} at index {}. Var: {}.",
                        gradient_out.norm(),
                        i,
                        variance_out[0]);
                } else {
                    // compute a weighted sum
                    ComputeWeightedSum<Dim>(i, tested_idx, fs, variances, covariances);

                    ERL_DEBUG_ASSERT(
                        std::isfinite(gradient_out.norm()),
                        "gradient norm is not finite: {} at index {}. Var: {}.",
                        gradient_out.norm(),
                        i,
                        variance_out[0]);
                }
            } else {
                // the first column is the result
                distance_out = fs(0, 0);
                if (compute_gradient) {
                    gradient_out << fs.col(0).template segment<Dim>(1);
                    gradient_out.normalize();
                }
                variance_out << variances.col(0);
                if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariances.col(0); }
                used_gps[0] = gps[tested_idx[0].second].second.second;

                ERL_DEBUG_ASSERT(
                    std::isfinite(gradient_out.norm()),
                    "gradient norm is not finite: {} at index {}. Var: {}.",
                    gradient_out.norm(),
                    i,
                    variance_out[0]);
            }
        }
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3, void>
    GpSdfMapping<Dtype, Dim>::ComputeWeightedSum(
        uint32_t i,
        const std::vector<std::pair<long, long>> &tested_idx,
        const Eigen::Matrix<Dtype, 7, Eigen::Dynamic> &fs,
        const Variances &variances,
        const Covariances &covariances) {

        const bool compute_gradient = m_setting_->test_query.compute_gradient;
        const bool compute_gradient_variance = m_setting_->test_query.compute_gradient_variance;
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        Dtype max_test_valid_distance_var = m_setting_->test_query.max_test_valid_distance_var;
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

        (*m_test_buffer_.distances)[i] = f[0];  // distance
        if (compute_gradient) {                 // gradient
            auto gradient = (*m_test_buffer_.gradients).col(i);
            gradient << f.template tail<3>();
            gradient.normalize();
        }
        auto var_out = (*m_test_buffer_.variances).col(i);
        var_out[0] = variance_f[0] / w_sum;  // variance
        if (compute_gradient_variance) {
            var_out[1] = variance_f[1] / w_sum;
            var_out[2] = variance_f[2] / w_sum;
            var_out[3] = variance_f[3] / w_sum;
        }
        if (compute_covariance) { (*m_test_buffer_.covariances).col(i) = covariance_f / w_sum; }
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 2, void>
    GpSdfMapping<Dtype, Dim>::ComputeWeightedSum(
        uint32_t i,
        const std::vector<std::pair<long, long>> &tested_idx,
        const Eigen::Matrix<Dtype, 5, Eigen::Dynamic> &fs,
        const Variances &variances,
        const Covariances &covariances) {

        const bool compute_gradient = m_setting_->test_query.compute_gradient;
        const bool compute_gradient_variance = m_setting_->test_query.compute_gradient_variance;
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        Dtype max_test_valid_distance_var = m_setting_->test_query.max_test_valid_distance_var;
        auto &gps = m_query_to_gps_[i];

        // pick the best two results to do the weighted sum
        const long j1 = tested_idx[0].first;
        const long j2 = tested_idx[1].first;
        const Dtype w1 = variances(0, j1) - max_test_valid_distance_var;
        const Dtype w2 = variances(0, j2) - max_test_valid_distance_var;
        const Dtype w12 = w1 + w2;
        // clang-format off
        (*m_test_buffer_.distances)[i] = (fs(0, j1) * w2 + fs(0, j2) * w1) / w12;  // distance
        if (compute_gradient) {                                                    // gradient
            (*m_test_buffer_.gradients).col(i) << (fs(1, j1) * w2 + fs(1, j2) * w1) / w12,
                                                  (fs(2, j1) * w2 + fs(2, j2) * w1) / w12;
        }
        auto var_out = (*m_test_buffer_.variances).col(i);
        var_out[0] = (variances(0, j1) * w2 + variances(0, j2) * w1) / w12;  // variance
        if (compute_gradient_variance) {
            var_out[1] = (variances(1, j1) * w2 + variances(1, j2) * w1) / w12;
            var_out[2] = (variances(2, j1) * w2 + variances(2, j2) * w1) / w12;
        }
        if (compute_covariance) {
            (*m_test_buffer_.covariances).col(i) <<
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

    template class GpSdfMapping<float, 2>;
    template class GpSdfMapping<double, 2>;
    template class GpSdfMapping<float, 3>;
    template class GpSdfMapping<double, 3>;
}  // namespace erl::gp_sdf
