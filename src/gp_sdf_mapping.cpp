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
        const Eigen::Ref<const MatrixDX> &positions_in,
        VectorX &distances_out,
        MatrixDX &gradients_out,
        Variances &variances_out,
        Covariances &covariances_out,
        const bool compute_covariance) {
        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
        const long n = positions_in.cols();
        if (n == 0) { return false; }

        distances_out.resize(n);
        gradients_out.resize(MatrixDX::RowsAtCompileTime, n);
        variances_out.resize(Variances::RowsAtCompileTime, n);
        if (compute_covariance) { covariances_out.resize(Covariances::RowsAtCompileTime, n); }
        this->positions = std::make_unique<Eigen::Ref<const MatrixDX>>(positions_in);
        this->distances = std::make_unique<Eigen::Ref<VectorX>>(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<MatrixDX>>(gradients_out);
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
        : m_setting_(std::move(setting)), m_surface_mapping_(std::move(surface_mapping)) {
        ERL_ASSERTM(m_setting_ != nullptr, "setting is nullptr.");
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");
        ERL_ASSERTM(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1.");

        InitMultiThreading();
    }

    template<typename Dtype, int Dim>
    std::lock_guard<std::mutex>
    GpSdfMapping<Dtype, Dim>::GetLockGuard() const {
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

        double surf_mapping_time = 0;
        bool ok = false;
        {
            const ERL_BLOCK_TIMER_MSG_TIME("Surface mapping update", surf_mapping_time);
            ok = m_surface_mapping_->Update(rotation, translation, scan, are_points, are_local);
        }

        if (ok) {
            const double time_budget_us = 1e6 / m_setting_->update_hz;  // us
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
        UpdateLoadDataQueue();

        const auto dt = timer.Elapsed<double, std::micro>();
        time_budget_us -= dt;
        RunLoadDataQueue(time_budget_us, false);

        ERL_TRACY_FRAME_MARK_END();
        return !m_gps_to_load_data_.empty();  // return true if we loaded any data
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TrainAllGps() {
        // CRITICAL SECTION: access m_load_data_queue_
        const auto lock = GetLockGuard();
        (void) lock;

        RunLoadDataQueue(0, true);  // load all data

        m_gps_to_train_.clear();
        for (auto &[key, gp]: m_gp_map_) {
            if (gp == nullptr || !gp->active || !gp->GpOutdated()) { continue; }
            m_gps_to_train_.emplace_back(0, gp);
        }
        TrainGps();
    }

    template<typename Dtype, int Dim>
    [[nodiscard]] bool
    GpSdfMapping<Dtype, Dim>::Test(
        const Eigen::Ref<const MatrixDX> &positions_in,
        VectorX &distances_out,
        MatrixDX &gradients_out,
        Variances &variances_out,
        Covariances &covariances_out) {

        m_query_used_gps_.clear();

        {
            const auto lock = GetLockGuard();  // CRITICAL SECTION: access m_gp_map_
            (void) lock;
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
        const MatrixDX positions_s = scaling == 1 ? positions_in : positions_in.array() * scaling;
        if (!m_test_buffer_.ConnectBuffers(  // allocate memory for test results
                positions_s,
                distances_out,
                gradients_out,
                variances_out,
                covariances_out,
                m_setting_->test_query.compute_covariance)) {
            ERL_WARN("Failed to connect test buffers.");
            return false;
        }

        const uint32_t num_queries = positions_s.cols();
        const uint32_t num_threads = std::min(m_setting_->num_threads, num_queries);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = num_queries / num_threads;
        const std::size_t leftover = num_queries - batch_size * num_threads;

        {
            // CRITICAL SECTION: access m_surface_mapping_
            const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
            (void) surface_mapping_lock;
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
        SearchCandidateGps(positions_s);

        if (m_candidate_gps_.empty()) {  // no candidate GPs
            ERL_WARN_COND(positions_in.cols() == 1, "No candidate GPs available for testing.");
            return false;
        }
        ERL_INFO("{} candidate GPs for {} query positions.", m_candidate_gps_.size(), num_queries);
#pragma endregion

#pragma region test_search_gps
        std::size_t start_idx = 0;
        std::size_t end_idx = 0;
        m_kdtree_candidate_gps_.reset();
        // build kdtree of candidate GPs to allow fast search.
        // remove inactive GPs and collect GP positions
        MatrixDX gp_positions(Dim, m_candidate_gps_.size());
        for (std::size_t i = 0; i < m_candidate_gps_.size(); ++i) {
            auto &gp = m_candidate_gps_[i];
            gp_positions.col(static_cast<long>(i)) = gp->GetMeanPosition();
        }
        m_kdtree_candidate_gps_ = std::make_shared<KdTree>(std::move(gp_positions));

        std::vector<std::vector<std::size_t>> no_gps_indices(num_threads);
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

        for (auto &gps: m_query_to_gps_) {
            for (auto &gp: gps) {
                if (gp == nullptr) { break; }
                gp->MarkQueried();
            }
        }
#pragma endregion

#pragma region test_train_gps
        CollectGpsToTrain();
        TrainGps();
#pragma endregion

#pragma region test_gps
        bool surf_mapping_sign = false;
        const SignMethod sign_method = m_setting_->sdf_gp->sign_method;
        m_in_free_space_.setConstant(num_queries, false);
        if (const auto &hybrid_sign_methods = m_setting_->sdf_gp->hybrid_sign_methods;
            sign_method == SignMethod::kExternal ||
            (sign_method == SignMethod::kHybrid &&
             (hybrid_sign_methods.first == SignMethod::kExternal ||
              hybrid_sign_methods.second == SignMethod::kExternal))) {
            const ERL_BLOCK_TIMER_MSG("Get sign from surface mapping");
            // collect the sign from the surface mapping, which is not thread-safe
            // CRITICAL SECTION: access m_surface_mapping_
            const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
            (void) surface_mapping_lock;
            surf_mapping_sign = m_surface_mapping_->IsInFreeSpace(positions_in, m_in_free_space_);
            ERL_WARN_COND(!surf_mapping_sign, "Failed to get sign from the surface mapping.");
        }

        if (m_setting_->test_query.use_global_buffer) {
            m_test_buffer_.PrepareGpBuffer(num_queries, m_setting_->test_query.num_neighbor_gps);
        }

        // Compute the inference result for each query position
        m_query_used_gps_.clear();
        m_query_used_gps_.resize(num_queries);
        {
            const ERL_BLOCK_TIMER_MSG("Query GPs");
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
        }
#pragma endregion

        m_test_buffer_.DisconnectBuffers();

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
        const VectorD &boundary_size,
        const Rotation &boundary_rotation,
        const VectorD &boundary_center,
        const Dtype resolution,
        const Dtype iso_value,
        std::vector<VectorD> &surface_points,
        std::vector<Face> &faces,
        std::vector<VectorD> &face_normals) {

        using GridShape = Eigen::Vector<long, Dim>;
        using VoxelCoord = Eigen::Vector<long, Dim>;
        using EdgeCoord = Eigen::Vector<long, Dim + 1>;
        using MC = std::conditional_t<Dim == 2, geometry::MarchingSquares, geometry::MarchingCubes>;
        using namespace common;

        // 1. create grid
        ERL_INFO("Creating grid for mesh extraction");
        GridShape grid_shape;
        VectorD grid_resolution;
        for (int i = 0; i < Dim; ++i) {
            grid_shape[i] = static_cast<int>(std::ceil(boundary_size[i] / resolution));
            grid_resolution[i] = boundary_size[i] / static_cast<Dtype>(grid_shape[i]);
        }
        const GridShape grid_strides = ComputeCStrides<long, Dim>(grid_shape, 1);
        const VectorD bound_min = boundary_size.array() * -0.5f;
        auto old_test_query = m_setting_->test_query;  // backup
        m_setting_->test_query.compute_gradient = false;
        m_setting_->test_query.compute_gradient_variance = false;
        m_setting_->test_query.compute_covariance = false;

        // 2. find voxels that are near the surface, i.e. in any cluster
        ERL_INFO("Finding voxels near the surface");
        const KeySet clusters = m_surface_mapping_->GetAllClusters();
        MatrixDX cluster_centers(Dim, clusters.size());
        {
            long idx = 0;
            for (const auto &key: clusters) {
                cluster_centers.col(idx++) = m_surface_mapping_->GetClusterCenter(key);
            }
        }
        const Dtype scaling = 1.0f / m_surface_mapping_->GetScaling();
        cluster_centers *= scaling;
        Dtype radius = m_surface_mapping_->GetClusterSize() * scaling * std::sqrt(Dim) * 0.5f;
        radius *= std::sqrt(static_cast<Dtype>(Dim));
        const KdTree kdtree_clusters(cluster_centers);
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
            VectorD voxel_center;
            for (int i = 0; i < Dim; ++i) {
                voxel_center[i] = GridToMeter(voxel_coord[i], bound_min[i], grid_resolution[i]);
            }
            voxel_center = boundary_rotation * voxel_center + boundary_center;
            std::vector<typename KdTree::ResultItem> indices_dists;
            kdtree_clusters.RadiusSearch(voxel_center, radius, false, indices_dists);
            flags_near_surface[voxel_idx] = !indices_dists.empty();
        }

        struct Voxel {
            int idx = -1;
            VoxelCoord coord = VoxelCoord::Zero();
            int surf_config = 0;
            std::vector<EdgeCoord> unique_edges;
            std::vector<Face> faces;

            Voxel(const int idx, VoxelCoord coord_)
                : idx(idx), coord(std::move(coord_)) {}
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
        std::vector<std::vector<std::pair<VoxelCoord, VectorD>>> vertices_batches(num_threads);
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
            const std::size_t start_idx = tidx * batch_size;
            const std::size_t end_idx =
                (tidx == num_threads - 1) ? near_surface_voxels.size() : start_idx + batch_size;
            std::vector<std::pair<VoxelCoord, VectorD>> &vertices = vertices_batches[tidx];
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
                    VectorD vertex_pos;
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
        const std::size_t n_unique_vertices = std::accumulate(
            vertices_batches.begin(),
            vertices_batches.end(),
            0,
            [](std::size_t sum, const std::vector<std::pair<VoxelCoord, VectorD>> &batch) {
                return sum + batch.size();
            });
        MatrixDX vertices(Dim, n_unique_vertices);
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
        MatrixDX gradients(Dim, vertices.cols());
        Variances variances = Variances::Zero(Dim + 1, vertices.cols());
        Covariances covariances;
        ERL_INFO("Querying SDF at {} vertices", vertices.cols());
        const bool success = Test(vertices, sdf_values, gradients, variances, covariances);
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
            const VoxelCoord v1_coord = edge_coord.template head<Dim>();
            VoxelCoord v2_coord = edge_coord.template head<Dim>();
            ++v2_coord[edge_coord[Dim] - 1];
            const long vid1 = vertex_map.at(v1_coord);
            const long vid2 = vertex_map.at(v2_coord);
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
                    const VectorD &v0 = surface_points[face_out[0]];
                    const VectorD &v1 = surface_points[face_out[1]];
                    const VectorD &v2 = surface_points[face_out[2]];
                    VectorD v10 = v1 - v0;
                    VectorD v20 = v2 - v0;
                    normal_out[0] = v10[1] * v20[2] - v10[2] * v20[1];
                    normal_out[1] = v10[2] * v20[0] - v10[0] * v20[2];
                    normal_out[2] = v10[0] * v20[1] - v10[1] * v20[0];
                    normal_out.normalize();
                } else if (Dim == 2) {
                    const VectorD &v0 = surface_points[face_out[0]];
                    const VectorD &v1 = surface_points[face_out[1]];
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
    GpSdfMapping<Dtype, Dim>::Write(std::ostream &stream) const {
        using namespace common;
        using namespace common::serialization;

        m_surface_mapping_->FlushSurfaceDataCache();
        auto *mutable_this = const_cast<GpSdfMapping *>(this);  // NOLINT(*-pro-type-const-cast)
        mutable_this->CollectChangedClusters();
        mutable_this->UpdateLoadDataQueue();
        mutable_this->TrainAllGps();
        m_surface_mapping_->ClearChangedClusters();

        static const TokenWriteFunctionPairs<GpSdfMapping> token_function_pairs = {
            {
                "setting",
                [](const GpSdfMapping *self, std::ostream &s) {
                    return self->m_setting_->Write(s) && s.good();
                },
            },
            {
                "surface_mapping",
                [](const GpSdfMapping *self, std::ostream &s) {
                    return self->m_surface_mapping_->Write(s) && s.good();
                },
            },
            {
                "gp_map",
                [](const GpSdfMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_gp_map_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (auto &[key, gp]: self->m_gp_map_) {
                        s.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        bool has_gp = gp != nullptr;
                        s.write(reinterpret_cast<const char *>(&has_gp), sizeof(bool));
                        if (has_gp && !gp->Write(s)) { return false; }
                    }
                    return s.good();
                },
            },
            {
                "queue_keys",
                [](const GpSdfMapping *self, std::ostream &s) {
                    const std::size_t n = self->m_queue_keys_.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, handle]: self->m_queue_keys_) {
                        s.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        s.write(
                            reinterpret_cast<const char *>(&(*handle).priority),
                            sizeof((*handle).priority));
                    }
                    return s.good();
                },
            },
            // m_cluster_queue_ can be reconstructed from m_cluster_queue_keys_.
            {
                "load_surf_data_time_us",
                [](const GpSdfMapping *self, std::ostream &s) {
                    return s.write(
                               reinterpret_cast<const char *>(&self->m_load_surf_data_time_us_),
                               sizeof(double)) &&
                           s.good();
                },
            },
            {
                "train_gp_time_us",
                [](const GpSdfMapping *self, std::ostream &s) {
                    return s.write(
                               reinterpret_cast<const char *>(&self->m_train_gp_time_us_),
                               sizeof(double)) &&
                           s.good();
                },
            },
        };
        return WriteTokens(stream, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    GpSdfMapping<Dtype, Dim>::Read(std::istream &stream) {
        using namespace common;
        using namespace common::serialization;
        static const TokenReadFunctionPairs<GpSdfMapping> token_function_pairs = {
            {
                "setting",
                [](GpSdfMapping *self, std::istream &s) {
                    return self->m_setting_->Read(s) && s.good();
                },
            },
            {
                "surface_mapping",
                [](GpSdfMapping *self, std::istream &s) {
                    return self->m_surface_mapping_->Read(s) && s.good();
                },
            },
            {
                "gp_map",
                [](GpSdfMapping *self, std::istream &s) {
                    std::size_t n = 0;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_gp_map_.clear();
                    self->m_gp_map_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        auto [it, inserted] = self->m_gp_map_.try_emplace(key, nullptr);
                        if (!inserted) {
                            ERL_WARN("Duplicate GP key: {}.", static_cast<std::string>(key));
                            return false;
                        }
                        bool has_gp = false;
                        s.read(reinterpret_cast<char *>(&has_gp), sizeof(bool));
                        if (has_gp) {
                            it->second = std::make_shared<SdfGp>(self->m_setting_->sdf_gp);
                            if (!it->second->Read(s)) {
                                ERL_WARN(
                                    "Failed to read GP of key {}.",
                                    static_cast<std::string>(key));
                                return false;
                            }
                        }
                    }
                    return s.good();
                },
            },
            {
                "queue_keys",
                [](GpSdfMapping *self, std::istream &s) {
                    std::size_t n = 0;
                    s.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_queue_keys_.clear();
                    self->m_queue_keys_.reserve(n);
                    self->m_load_data_queue_.clear();
                    self->m_load_data_queue_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        s.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        Dtype priority;
                        s.read(reinterpret_cast<char *>(&priority), sizeof(priority));
                        auto gp = self->m_gp_map_.at(key);
                        auto [it, inserted] = self->m_queue_keys_.try_emplace(
                            key,
                            self->m_load_data_queue_.push({priority, std::make_pair(key, gp)}));
                        if (!inserted) {
                            ERL_WARN("Duplicate cluster key: {}.", static_cast<std::string>(key));
                            return false;
                        }
                    }
                    return s.good();
                },
            },
            {
                "load_surf_data_time_us",
                [](GpSdfMapping *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->m_load_surf_data_time_us_),
                        sizeof(double));
                    return s.good();
                },
            },
            {
                "train_gp_time_us",
                [](GpSdfMapping *self, std::istream &s) {
                    s.read(reinterpret_cast<char *>(&self->m_train_gp_time_us_), sizeof(double));
                    return s.good();
                },
            },
        };
        const bool success = ReadTokens(stream, this, token_function_pairs);
        if (success) { InitMultiThreading(); }
        return success;
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
        if (m_queue_keys_.size() != other.m_queue_keys_.size()) { return false; }
        for (const auto &[key, handle]: m_queue_keys_) {
            auto it = other.m_queue_keys_.find(key);
            if (it == other.m_queue_keys_.end()) { return false; }
            const auto &[other_key, other_handle] = *it;
            if (key != other_key) { return false; }
            if ((*handle).priority != (*other_handle).priority) { return false; }
        }
        // when m_cluster_queue_keys_ is the same, m_cluster_queue_ is the same.
        // No need to compare the following temporary data:
        // m_clusters_to_train_, m_candidate_gps_, m_kdtree_candidate_gps_, m_map_boundary_,
        // m_query_to_gps_, m_in_free_space_, m_test_buffer and m_query_used_gps_.
        if (m_train_gp_time_us_ != other.m_train_gp_time_us_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::InitMultiThreading() {
        m_surf_data_indices_.resize(m_setting_->num_threads);
        m_surf_data_dist_indices_.resize(m_setting_->num_threads);
        m_gp_load_data_cnt_.resize(m_setting_->num_threads);
        m_key_sets_.resize(m_setting_->num_threads);
        m_key_vectors_.resize(m_setting_->num_threads);
        for (auto &indices: m_surf_data_indices_) { indices.reserve(512); }
        for (auto &indices: m_surf_data_dist_indices_) { indices.reserve(512); }
    }

    template<typename Dtype, int Dim>
    Dtype
    GpSdfMapping<Dtype, Dim>::GetDataCollectionRadius() const {
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        return cluster_size * m_setting_->gp_sdf_area_scale * 0.707f;
    }

    template<typename Dtype, int Dim>
    Dtype
    GpSdfMapping<Dtype, Dim>::GetDataCollectionAabbHalfSize() const {
        const Dtype cluster_size = m_surface_mapping_->GetClusterSize();
        return cluster_size * m_setting_->gp_sdf_area_scale * 0.5f;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::CollectChangedClusters() {
        const ERL_BLOCK_TIMER_MSG("CollectChangedClusters");

        const Dtype radius = GetDataCollectionRadius();

        // CRITICAL SECTION: access m_surface_mapping_
        const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        (void) surface_mapping_lock;
        const KeySet &changed_clusters = m_surface_mapping_->GetChangedClusters();
        if (changed_clusters.empty()) {
            m_clusters_to_load_data_.clear();
            return;
        }

        m_clusters_to_load_data_ = changed_clusters;
        const KeyVector keys(changed_clusters.begin(), changed_clusters.end());

        ERL_INFO("Collecting neighboring clusters for {} changed clusters.", keys.size());

        for (auto &key_set: m_key_sets_) { key_set = changed_clusters; }

#pragma omp parallel for default(none) shared(keys, changed_clusters, radius)
        for (const auto &cluster_key: keys) {
            auto &key_set = m_key_sets_[omp_get_thread_num()];
            auto &key_vec = m_key_vectors_[omp_get_thread_num()];
            const Aabb area(m_surface_mapping_->GetClusterCenter(cluster_key), radius);
            m_surface_mapping_->IterateClustersInAabb(area, [&](const Key &key) {
                // m_clusters_to_load_data_.insert(key);
                if (!key_set.insert(key).second) { return; }  // already included
                key_vec.push_back(key);
            });
        }

        // merge results from all threads
        m_clusters_to_load_data_ = m_key_sets_[0];
        m_key_vectors_[0].clear();
        for (auto &key_vec: m_key_vectors_) {
            m_clusters_to_load_data_.insert(key_vec.begin(), key_vec.end());
            key_vec.clear();
        }
        ERL_INFO("Total {} clusters to load data.", m_clusters_to_load_data_.size());
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::UpdateLoadDataQueue() {
        const ERL_BLOCK_TIMER_MSG("UpdateLoadDataQueue");

        const Dtype area_half_size = GetDataCollectionAabbHalfSize();
        const auto max_c1 = static_cast<Dtype>(m_setting_->queue_priority.max_buf_outdated_count);
        const Dtype alpha = m_setting_->queue_priority.distance_weight;
        const Dtype beta = m_setting_->queue_priority.query_weight_for_loading;

        const auto lock = GetLockGuard();  // CRITICAL SECTION: access m_gp_map_
        (void) lock;

        const VectorD sensor_pos = m_surface_mapping_->GetLastSensorPosition();
        KeyVector &keys = m_key_vectors_[0];
        keys.clear();
        for (const auto &cluster_key: m_clusters_to_load_data_) {
            auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, nullptr);
            auto &gp = it->second;
            Dtype priority;
            if (inserted) {
                // new GP
                const VectorD gp_center = m_surface_mapping_->GetClusterCenter(cluster_key);
                Dtype d = (gp_center - sensor_pos).squaredNorm();
                priority = max_c1 * std::exp(-d * alpha);
                gp = std::make_shared<SdfGp>(m_setting_->sdf_gp);
                gp->Activate();
                gp->buf_outdated_count = static_cast<long>(priority);
                gp->position = gp_center;
                gp->SetMeanPosition(gp_center);
                gp->half_size = area_half_size;
                if (m_setting_->new_gp_load_data_immediately) {
                    keys.push_back(cluster_key);
                    continue;
                }
            } else {
                gp->Activate();
                gp->MarkBufferOutdated();
                priority = gp->GetLoadingPriority(beta);
            }
            // add the cluster to the queue
            if (auto itr = m_queue_keys_.find(cluster_key); itr == m_queue_keys_.end()) {
                // new cluster
                m_queue_keys_.insert(
                    {cluster_key,
                     m_load_data_queue_.push({priority, std::make_pair(cluster_key, gp)})});
            } else {
                auto &heap_key = itr->second;
                (*heap_key).priority = priority;
                m_load_data_queue_.increase(heap_key);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::RunLoadDataQueue(
        const double time_budget_us,
        const bool ignore_budget) {

        if (m_load_data_queue_.empty()) {
            ERL_INFO("Load data queue is empty.");
            return;
        }

        m_clusters_to_collect_data_.clear();
        m_gps_to_load_data_.clear();
        m_clusters_to_collect_data_.reserve(m_queue_keys_.size());
        m_gps_to_load_data_.reserve(m_queue_keys_.size());

        if (m_setting_->new_gp_load_data_immediately) {
            for (const auto &key: m_key_vectors_[0]) {
                const auto &pair = std::make_pair(key, m_gp_map_.at(key));
                m_clusters_to_collect_data_.insert(key);
                m_gps_to_load_data_.emplace_back(pair);
            }
        }

        if (ignore_budget) {
            ERL_DEBUG("Ignoring time budget, loading data for all GPs in the queue.");
            for (const auto &[key, handle]: m_queue_keys_) {
                const auto &pair = (*handle).key_gp_pair;
                if (!pair.second->active) { continue; }  // skip inactive GP
                m_clusters_to_collect_data_.insert(key);
                m_gps_to_load_data_.emplace_back(pair);
            }
            m_queue_keys_.clear();
            m_load_data_queue_.clear();
        } else {
            auto max_num_gps =
                static_cast<long>(std::floor(time_budget_us / m_load_surf_data_time_us_));
            max_num_gps = std::max(max_num_gps, m_setting_->min_num_gps_to_update);

            ERL_DEBUG(
                "time_budget: {:.2f} us, per_gp_time: {:.2f} us, max_num_gps: {}.",
                time_budget_us,
                m_load_surf_data_time_us_,
                max_num_gps);

            while (!m_load_data_queue_.empty() &&
                   m_gps_to_load_data_.size() < static_cast<std::size_t>(max_num_gps)) {
                auto pair = m_load_data_queue_.top().key_gp_pair;
                m_load_data_queue_.pop();
                m_queue_keys_.erase(pair.first);
                if (!pair.second->active) { continue; }  // skip inactive GP
                m_clusters_to_collect_data_.insert(pair.first);
                m_gps_to_load_data_.emplace_back(pair);
            }
        }

        if (!m_gps_to_load_data_.empty()) { LoadSurfaceData(); }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::LoadSurfaceData() {
        const ERL_BLOCK_TIMER_MSG("LoadSurfaceData");

        const std::size_t n = m_gps_to_load_data_.size();
        if (n == 0) { return; }

        ERL_INFO("Load surface data for {} GPs, {} GPs in queue.", n, m_load_data_queue_.size());

        // CRITICAL SECTION: access m_surface_mapping_ in LoadSurfaceDataThread
        const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        (void) surface_mapping_lock;

        const auto t0 = std::chrono::high_resolution_clock::now();

        {
            // we need to include the neighboring clusters' data as well
            KeyVector &keys = m_key_vectors_[0];
            keys.clear();
            keys.reserve(m_clusters_to_collect_data_.size() * 2);
            keys.insert(
                keys.end(),
                m_clusters_to_collect_data_.begin(),
                m_clusters_to_collect_data_.end());
            const Dtype radius = GetDataCollectionRadius();
            for (std::size_t i = 0, n_keys = keys.size(); i < n_keys; ++i) {
                const auto &cluster_key = keys[i];
                const Aabb area(m_surface_mapping_->GetClusterCenter(cluster_key), radius);
                m_surface_mapping_->IterateClustersInAabb(area, [&](const Key &key) {
                    if (!m_clusters_to_collect_data_.insert(key).second) { return; }
                    keys.push_back(key);  // new neighboring cluster
                });
            }

            // collect surface data indices from the clusters
            for (auto &indices: m_surf_data_indices_) { indices.clear(); }
#pragma omp parallel for default(none) shared(keys)
            for (const auto &cluster_key: keys) {
                const int thread_idx = omp_get_thread_num();
                auto &indices = m_surf_data_indices_[thread_idx];
                (void) m_surface_mapping_->CollectSurfaceDataFromCluster(cluster_key, indices);
            }

            keys.clear();
        }

        // build kdtree for surface data points
        std::size_t n_points = 0;
        std::vector<std::size_t> start_indices;
        start_indices.reserve(m_surf_data_indices_.size() + 1);
        for (const auto &indices: m_surf_data_indices_) {
            start_indices.push_back(n_points);
            n_points += indices.size();
        }
        if (n_points == 0) { return; }
        start_indices.emplace_back(n_points);
        MatrixDX surface_points(Dim, n_points);
        m_surf_data_indices_[0].resize(n_points);  // for indexing in LoadSurfaceDataThread
#pragma omp parallel for default(none) shared(surface_points, start_indices)
        for (std::size_t i = 0; i < m_surf_data_indices_.size(); ++i) {
            const auto &buffer = m_surface_mapping_->GetSurfaceDataBuffer();
            const auto &indices = m_surf_data_indices_[i];
            if (i == 0) {
                const long end_idx = static_cast<long>(start_indices[1]);
                for (long idx = 0; idx < end_idx; ++idx) {
                    const std::size_t &point_idx = indices[idx];
                    surface_points.col(idx) = buffer[point_idx].position;
                }
            } else {
                long idx = static_cast<long>(start_indices[i]);
                for (const auto &point_idx: indices) {
                    surface_points.col(idx) = buffer[point_idx].position;
                    m_surf_data_indices_[0][idx++] = point_idx;
                }
            }
        }
        m_kdtree_surf_data_ = std::make_shared<KdTree>(std::move(surface_points));

        const uint32_t n_threads = m_setting_->num_threads;
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        const std::size_t batch_size = n / n_threads;
        const std::size_t left_over = n - batch_size * n_threads;
        std::size_t end = 0;
        for (uint32_t t_idx = 0; t_idx < n_threads; ++t_idx) {
            const std::size_t start = end;
            end = start + batch_size;
            if (t_idx < left_over) { end++; }
            threads.emplace_back(&GpSdfMapping::LoadSurfaceDataThread, this, t_idx, start, end);
        }
        for (uint32_t t_idx = 0; t_idx < n_threads; ++t_idx) { threads[t_idx].join(); }
        threads.clear();

        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        time /= static_cast<double>(n);

        m_load_surf_data_time_us_ = m_load_surf_data_time_us_ * 0.4f + time * 0.6f;

        const std::size_t n_surf_data_loaded =
            std::accumulate(m_gp_load_data_cnt_.begin(), m_gp_load_data_cnt_.end(), 0ul);
        const Dtype ratio = static_cast<Dtype>(n_surf_data_loaded) / static_cast<Dtype>(n);
        ERL_INFO(
            "Loaded surface data for {} / {} GPs ({:.2f}%).",
            n_surf_data_loaded,
            n,
            ratio * 100.0f);
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::LoadSurfaceDataThread(
        const uint32_t thread_idx,
        const std::size_t start_idx,
        const std::size_t end_idx) {

        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());

        const Dtype radius = GetDataCollectionRadius();
        auto sdf_gp_setting = m_setting_->sdf_gp;
        long max_k = std::max(
            sdf_gp_setting->sign_gp->max_num_samples,
            sdf_gp_setting->edf_gp->max_num_samples);
        max_k = static_cast<long>(1.5f * static_cast<float>(max_k));
        std::vector<long> indices;
        std::vector<Dtype> dists;
        auto &data_indices = m_surf_data_dist_indices_[thread_idx];
        auto &load_data_cnt = m_gp_load_data_cnt_[thread_idx];
        load_data_cnt = 0;
        for (std::size_t i = start_idx; i < end_idx; ++i) {
            auto &gp = CHECKED_AT(m_gps_to_load_data_, i).second;
            ERL_DEBUG_ASSERT(gp->active, "GP is not active");

            // collect surface data in the area
            const long k =
                m_kdtree_surf_data_->RadiusKnn(max_k, gp->position, radius, indices, dists);

            if (k == 0) {          // no surface data in the area
                gp->Deactivate();  // deactivate the GP if there is no training data
                continue;
            }
            data_indices.clear();
            data_indices.reserve(k);
            const auto &indices0 = m_surf_data_indices_[0];
            for (long j = 0; j < k; ++j) {
                data_indices.emplace_back(dists[j], indices0[indices[j]]);
            }
            if (!gp->LoadSurfaceData(
                    data_indices,
                    m_surface_mapping_->GetSurfaceDataBuffer(),
                    true,
                    m_setting_->sensor_noise,
                    m_setting_->max_valid_gradient_var,
                    m_setting_->invalid_position_var)) {
                gp->Deactivate();
            }
            ++load_data_cnt;
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::CollectGpsToTrain() {
        m_gps_to_train_.clear();
        absl::flat_hash_set<uint64_t> addr_set;
        addr_set.reserve(64);
        // add must-train GPs
        const bool retrain_outdated = m_setting_->test_query.retrain_outdated;
        const auto highest_priority = std::numeric_limits<Dtype>::max();
        std::size_t n_added = 0;
        for (auto &gp: m_candidate_gps_) {  // assumed active, unique
            if (!gp->IsTrained()) {         // must be trained
                addr_set.insert(reinterpret_cast<uint64_t>(gp.get()));
                m_gps_to_train_.emplace_back(highest_priority, gp);
                std::swap(gp, m_candidate_gps_[n_added++]);
                continue;
            }
            if (!retrain_outdated || !gp->GpOutdated()) {  // the GP is trained.
                // If retrain_outdated is true, we train the GP if it is outdated or not trained.
                // If retrain_outdated is false, we train the GP only if it is not trained.
                std::swap(gp, m_candidate_gps_[n_added++]);
            }
        }
        // add the first GP for each query position
        for (const auto &gps: m_query_to_gps_) {
            if (gps.empty()) { continue; }
            const auto &gp = gps[0];                             // may be in-active
            if (!gp->active || !gp->GpOutdated()) { continue; }  // inactive or not outdated
            if (!addr_set.insert(reinterpret_cast<uint64_t>(gp.get())).second) { continue; }
            m_gps_to_train_.emplace_back(highest_priority, gp);  // prioritize training
        }
        ERL_ASSERT_EQ(addr_set.size(), m_gps_to_train_.size());
        const std::size_t n_must_train = m_gps_to_train_.size();

        const std::size_t max_num_retrain_gps = m_setting_->test_query.max_num_retrain_gps;
        if (max_num_retrain_gps == 0 || m_gps_to_train_.size() < max_num_retrain_gps) {
            // add other to-be-retrained GPs from m_candidate_gps_
            const Dtype gamma = m_setting_->queue_priority.query_weight_for_retrain;
            for (auto it = m_candidate_gps_.begin() + n_added; it != m_candidate_gps_.end(); ++it) {
                const auto &gp = *it;
                if (!gp->active || !gp->GpOutdated()) { continue; }
                if (addr_set.contains(reinterpret_cast<uint64_t>(gp.get()))) { continue; }
                m_gps_to_train_.emplace_back(gp->GetRetrainPriority(gamma), gp);
            }
        }

        if (max_num_retrain_gps > 0 && m_gps_to_train_.size() > max_num_retrain_gps) {
            std::sort(
                m_gps_to_train_.begin() + n_must_train,
                m_gps_to_train_.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });  // descending
            if (n_must_train > max_num_retrain_gps) {
                m_gps_to_train_.resize(n_must_train);
            } else {
                m_gps_to_train_.resize(max_num_retrain_gps);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TrainGps() {
        const ERL_BLOCK_TIMER_MSG("TrainGps");

        const std::size_t n = m_gps_to_train_.size();
        if (n == 0) { return; }

        ERL_INFO("Training {} GPs", n);

        const auto t0 = std::chrono::high_resolution_clock::now();

        const uint32_t n_threads = m_setting_->num_threads;
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        const std::size_t batch_size = n / n_threads;
        const std::size_t left_over = n - batch_size * n_threads;
        std::size_t end_idx = 0;
        for (uint32_t t_idx = 0; t_idx < n_threads; ++t_idx) {
            const std::size_t start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (t_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping::TrainGpThread, this, t_idx, start_idx, end_idx);
        }
        for (uint32_t t_idx = 0; t_idx < n_threads; ++t_idx) { threads[t_idx].join(); }
        threads.clear();

        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        time /= static_cast<double>(n);

        // update timing by (EWMA: Exponentially Weighted Moving Average)
        m_train_gp_time_us_ = m_train_gp_time_us_ * 0.4f + time * 0.6f;
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::TrainGpThread(
        const uint32_t thread_idx,
        const std::size_t start_idx,
        const std::size_t end_idx) {

        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto &gp = CHECKED_AT(m_gps_to_train_, i).second;
            if (!gp->active) { continue; }
            gp->Train();
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::SearchCandidateGps(const Eigen::Ref<const MatrixDX> &positions_in) {
        const ERL_BLOCK_TIMER_MSG("SearchCandidateGps");

        m_candidate_gps_.clear();

        VectorD query_area_min = positions_in.col(0);
        VectorD query_area_max = positions_in.col(0);
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
                const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
                const auto lock = GetLockGuard();
                (void) surface_mapping_lock;
                (void) lock;
                m_surface_mapping_->IterateClustersInAabb(area, [&](const Key &cluster_key) {
                    // search for clusters in the area
                    if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                        const auto &gp = it->second;
                        if (!gp->active || !gp->buf_ever_loaded.load()) { return; }
                        m_candidate_gps_.emplace_back(gp);
                    }
                });
            }
            if (!m_candidate_gps_.empty()) { break; }  // found at least one GP
            search_area_padding *= 2.0f;               // double search area size
            Aabb new_area = m_map_boundary_.Intersection(
                {query_area_min.array() - search_area_padding,
                 query_area_max.array() + search_area_padding});
            if (new_area.IsValid() && area == new_area) { break; }  // the area did not change
            area = std::move(new_area);                             // update area
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

        const long knn = m_setting_->test_query.num_neighbor_gps;
        const Dtype radius = m_setting_->test_query.search_area_half_size;
        Eigen::VectorXl idxs = Eigen::VectorXl::Constant(knn, -1);
        VectorX squared_dists(knn);

        for (std::size_t i = start_idx; i < end_idx; ++i) {
            const VectorD test_pos = m_test_buffer_.positions->col(i);
            std::vector<GpPtr> &gps = m_query_to_gps_[i];

            gps.clear();
            gps.reserve(knn);
            idxs.fill(-1);
            const long n_gps =
                m_kdtree_candidate_gps_->RadiusKnn(knn, test_pos, radius, idxs, squared_dists);
            for (long j = 0; j < n_gps; ++j) {
                const long &index = idxs[j];
                const auto &gp = m_candidate_gps_[index];
                gps.emplace_back(gp);
                if (gps.size() >= static_cast<std::size_t>(knn)) { break; }
            }
            if (gps.empty()) { no_gps_indices.push_back(i); }
        }
    }

    template<typename Dtype, int Dim>
    void
    GpSdfMapping<Dtype, Dim>::SearchGpFallback(const std::vector<std::size_t> &no_gps_indices) {
        if (no_gps_indices.empty()) { return; }

        const ERL_BLOCK_TIMER_MSG("SearchGpFallback");

        // CRITICAL SECTION: access m_surface_mapping_ and m_gp_map_
        const auto surface_mapping_lock = m_surface_mapping_->GetLockGuard();
        const auto lock = GetLockGuard();
        (void) surface_mapping_lock;
        (void) lock;

        ERL_WARN_COND(
            !no_gps_indices.empty(),
            "Run fallback search for {} query positions.",
            no_gps_indices.size());

#pragma omp parallel for default(none) shared(no_gps_indices) schedule(dynamic)
        for (const std::size_t &i: no_gps_indices) {
            // failed to find GPs in the kd-tree, fall back to search clusters in the area
            // double search area size
            Dtype search_area_hs = 2.0f * m_setting_->test_query.search_area_half_size;
            const VectorD test_position = m_test_buffer_.positions->col(i);
            Aabb search_area = m_map_boundary_.Intersection({test_position, search_area_hs});
            const long knn = m_setting_->test_query.num_neighbor_gps;
            std::vector<std::pair<Key, Dtype>> key_and_dists;
            key_and_dists.reserve(knn * 2);
            while (key_and_dists.empty()) {
                // no gp found, maybe the test position is on the query boundary
                if (search_area.IsValid()) {
                    m_surface_mapping_->IterateClustersInAabb(
                        search_area,
                        [&](const Key &cluster_key) {
                            // search for clusters in the area
                            if (auto it = m_gp_map_.find(cluster_key); it != m_gp_map_.end()) {
                                const auto &gp = it->second;
                                if (!gp->active) { return; }  // e.g., due to no training data
                                Dtype dist = (gp->GetMeanPosition() - test_position).norm();
                                key_and_dists.emplace_back(cluster_key, dist);
                            }
                        });
                }
                if (!key_and_dists.empty()) { break; }  // found at least one gp
                search_area_hs *= 2.0f;                 // double search area size
                Aabb new_area = m_map_boundary_.Intersection({test_position, search_area_hs});
                if (new_area.IsValid() && (search_area.min() == new_area.min()) &&
                    (search_area.max() == new_area.max())) {
                    break;  // no need to search again
                }
                search_area = std::move(new_area);  // update area
            }
            auto &gps = m_query_to_gps_[i];
            std::sort(key_and_dists.begin(), key_and_dists.end(), [](const auto &a, const auto &b) {
                return a.second < b.second;
            });
            gps.clear();
            gps.reserve(knn);
            auto n = std::min<std::size_t>(key_and_dists.size(), static_cast<std::size_t>(knn));
            for (std::size_t j = 0; j < n; ++j) {
                gps.emplace_back(m_gp_map_.at(key_and_dists[j].first));
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
        std::vector<IndexedMetric> indexed_metrics;
        indexed_metrics.reserve(num_neighbor_gps);

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

            const std::vector<GpPtr> &gps = m_query_to_gps_[i];  // pre-sorted by distance
            if (gps.empty()) { continue; }

            // test GPs
            const VectorD test_position = m_test_buffer_.positions->col(i);
            indexed_metrics.clear();
            bool need_weighted_sum = false;
            Dtype sign = m_in_free_space_[i] ? 1.0f : -1.0f;
            long cnt = 0;
            for (std::size_t gp_idx = 0; gp_idx < gps.size(); ++gp_idx) {
                // call selected GPs for inference
                const auto gp = gps[gp_idx];
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
                if (use_smallest) {
                    indexed_metrics.emplace_back(cnt, gp_idx, std::abs(fs(0, cnt)));  // distance
                    ++cnt;
                    continue;
                }
                indexed_metrics.emplace_back(cnt, gp_idx, variances(0, cnt));  // dist variance
                ++cnt;
                // the current gp prediction is not good enough,
                // we use more GPs to compute the result.
                if (!need_weighted_sum && gps.size() > 1 &&
                    variances(0, cnt) > max_test_valid_distance_var) {
                    need_weighted_sum = true;
                }
                if (!need_weighted_sum) { break; }
            }
            if (indexed_metrics.empty()) {  //
                continue;
            }

            if (use_smallest && indexed_metrics.size() > 1) {
                std::sort(
                    indexed_metrics.begin(),
                    indexed_metrics.end(),
                    [](const IndexedMetric &a, const IndexedMetric &b) -> bool {
                        return a.metric < b.metric;
                    });
                need_weighted_sum = false;
                const long idx = indexed_metrics[0].idx;
                fs.col(0) = fs.col(idx);
                variances.col(0) = variances.col(idx);
                if (compute_covariance) { covariances.col(0) = covariances.col(idx); }
            }

            if (need_weighted_sum && indexed_metrics.size() > 1) {
                std::sort(
                    indexed_metrics.begin(),
                    indexed_metrics.end(),
                    [](const IndexedMetric &a, const IndexedMetric &b) -> bool {
                        return a.metric < b.metric;
                    });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, indexed_metrics[0].idx) * fs(0, indexed_metrics[1].idx) < 0) {
                    need_weighted_sum = false;
                }
            }

            // store the result
            if (need_weighted_sum) {
                if (const long j = indexed_metrics[0].idx;
                    variances(0, j) <= max_test_valid_distance_var) {
                    // the first result is good enough
                    distance_out = fs(0, j);  // column j is the result
                    if (compute_gradient) {
                        gradient_out << fs.col(j).template segment<Dim>(1);
                        gradient_out.normalize();
                    }
                    variance_out << variances.col(j);
                    if (compute_covariance) {
                        m_test_buffer_.covariances->col(i) = covariances.col(j);
                    }
                    used_gps[0] = gps[indexed_metrics[0].gp_idx];
                } else {
                    // compute a weighted sum
                    ComputeWeightedSum<Dim>(i, indexed_metrics, fs, variances, covariances);
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
                used_gps[0] = gps[indexed_metrics[0].gp_idx];
            }
        }
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3, void>
    GpSdfMapping<Dtype, Dim>::ComputeWeightedSum(
        uint32_t i,
        const std::vector<IndexedMetric> &indexed_metrics,
        const Eigen::Matrix<Dtype, 7, Eigen::Dynamic> &fs,
        const Variances &variances,
        const Covariances &covariances) {

        const bool compute_gradient = m_setting_->test_query.compute_gradient;
        const bool compute_gradient_variance = m_setting_->test_query.compute_gradient_variance;
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        Dtype max_test_valid_distance_var = m_setting_->test_query.max_test_valid_distance_var;
        std::vector<GpPtr> &gps = m_query_to_gps_[i];
        UsedGps &used_gps = m_query_used_gps_[i];
        used_gps.fill(nullptr);

        // pick the best <= 4 results to compute the weighted sum.
        const std::size_t m = std::min(indexed_metrics.size(), 4ul);
        Dtype w_sum = 0;
        Eigen::Vector4<Dtype> f = Eigen::Vector4<Dtype>::Zero();
        Eigen::Vector4<Dtype> variance_f = Eigen::Vector4<Dtype>::Zero();
        Eigen::Vector<Dtype, 6> covariance_f = Eigen::Vector<Dtype, 6>::Zero();
        for (std::size_t k = 0; k < m; ++k) {
            const long jk = indexed_metrics[k].idx;
            Dtype w = 1.0f / std::abs(variances(0, jk) - max_test_valid_distance_var) + 1.e-5f;
            w_sum += w;
            f += fs.col(jk).template head<4>() * w;
            variance_f += variances.col(jk) * w;
            CHECKED_AT(used_gps, k) = CHECKED_AT(gps, jk);
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
        const std::vector<IndexedMetric> &indexed_metrics,
        const Eigen::Matrix<Dtype, 5, Eigen::Dynamic> &fs,
        const Variances &variances,
        const Covariances &covariances) {

        const bool compute_gradient = m_setting_->test_query.compute_gradient;
        const bool compute_gradient_variance = m_setting_->test_query.compute_gradient_variance;
        const bool compute_covariance = m_setting_->test_query.compute_covariance;
        Dtype max_test_valid_distance_var = m_setting_->test_query.max_test_valid_distance_var;
        auto &gps = m_query_to_gps_[i];

        // pick the best two results to do the weighted sum
        const long j1 = indexed_metrics[0].idx;
        const long j2 = indexed_metrics[1].idx;
        Dtype w1 = variances(0, j1) - max_test_valid_distance_var;
        Dtype w2 = variances(0, j2) - max_test_valid_distance_var;
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
        used_gps[0] = gps[indexed_metrics[0].gp_idx];
        used_gps[1] = gps[indexed_metrics[1].gp_idx];
    }

    template class GpSdfMapping<float, 2>;
    template class GpSdfMapping<double, 2>;
    template class GpSdfMapping<float, 3>;
    template class GpSdfMapping<double, 3>;
}  // namespace erl::gp_sdf
