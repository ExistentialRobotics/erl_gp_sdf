#include "erl_gp_sdf/height_map_projector.hpp"

#include <cmath>

namespace erl::gp_sdf {

    namespace {
        /// Ensure a dimension is odd, as required by GridMapInfo.
        template<typename T>
        T
        MakeOdd(T n) {
            return (n % 2 == 0) ? n + 1 : n;
        }
    }  // namespace

    template<typename Dtype, bool Colored>
    HeightMapProjector<Dtype, Colored>::HeightMapProjector(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {}

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::Update(SurfaceMapping &mapping) {
        // Initialize derived constants on first call.
        if (!m_initialized_) {
            const auto &map_setting = mapping.GetSetting();
            const auto tree = mapping.GetTree();
            const Dtype bhm_node_size = tree->GetNodeSize(map_setting->bhm_depth);
            const long surface_grid_size = map_setting->local_bhm->surface_grid_size;
            m_surface_resolution_ = bhm_node_size / static_cast<Dtype>(surface_grid_size);

            const Dtype target = m_setting_->target_resolution;
            if (target >= m_surface_resolution_) {
                m_internal_resolution_ = m_surface_resolution_;
            } else {
                const auto n = static_cast<int>(std::ceil(m_surface_resolution_ / target));
                m_internal_resolution_ = m_surface_resolution_ / static_cast<Dtype>(n);
            }

            m_patch_size_ = static_cast<int>(std::round(bhm_node_size / m_internal_resolution_));

            if (m_setting_->use_bounding_box) {
                // Pre-allocate fixed-size global map.
                const auto &bb = m_setting_->bounding_box;
                const int rows = MakeOdd(
                    static_cast<int>(
                        std::ceil((bb.max()[0] - bb.min()[0]) / m_internal_resolution_)));
                const int cols = MakeOdd(
                    static_cast<int>(
                        std::ceil((bb.max()[1] - bb.min()[1]) / m_internal_resolution_)));
                m_global_height_map_ = Eigen::MatrixX<Dtype>::Constant(rows, cols, kSentinel);
                m_global_origin_x_ = bb.min()[0];
                m_global_origin_y_ = bb.min()[1];
            } else {
                // Initialize origin from sensor position so the map starts centered.
                const auto &sensor_pos = mapping.GetLastSensorPosition();
                m_global_origin_x_ = sensor_pos[0];
                m_global_origin_y_ = sensor_pos[1];
            }

            m_initialized_ = true;
        }

        // Step a: locked — collect changed BHMs and copy triangle data.
        auto triangle_data = CollectChangedBhms(mapping);
        if (triangle_data.empty()) { return; }

        // Step b: unlocked — compute local height maps in parallel.
        auto local_patches = ComputeLocalHeightMaps(triangle_data, mapping);

        // Step c: grow global map if needed.
        if (!m_setting_->use_bounding_box) { GrowGlobalMap(local_patches); }

        // Step d: paste local patches sequentially.
        MergePatches(local_patches);

        // Step e/f: build occupancy grid.
        BuildOccupancyGrid();
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::GetOccupancyGrid(
        std::vector<int8_t> &occupancy_data,
        GridMapInfo &grid_info) const {
        occupancy_data = m_occupancy_data_;
        if (m_output_grid_info_) { grid_info = *m_output_grid_info_; }
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::GetHeightMap(
        Eigen::MatrixX<Dtype> &height_data,
        GridMapInfo &grid_info) const {
        height_data = m_global_height_map_;
        // Build grid info for the internal map: GridMapInfo(origin, resolution, shape).
        const Eigen::Vector2<Dtype> origin(m_global_origin_x_, m_global_origin_y_);
        const Eigen::Vector2<Dtype> res(m_internal_resolution_, m_internal_resolution_);
        const Eigen::Vector2i shape(
            static_cast<int>(m_global_height_map_.rows()),
            static_cast<int>(m_global_height_map_.cols()));
        grid_info = GridMapInfo(origin, res, shape);
    }

    template<typename Dtype, bool Colored>
    std::vector<typename HeightMapProjector<Dtype, Colored>::BhmTriangleData>
    HeightMapProjector<Dtype, Colored>::CollectChangedBhms(SurfaceMapping &mapping) {
        auto lock = mapping.GetLockGuard();

        const Dtype sensor_z = mapping.GetLastSensorPosition()[2];
        UpdateNearGroundBhms(mapping, sensor_z);

        const auto &bhm_dict = mapping.GetLocalBhms();
        const auto &surf_data_buffer = mapping.GetSurfaceDataBuffer();
        const Dtype scaling = mapping.GetScaling();

        std::vector<BhmTriangleData> result;
        result.reserve(m_near_ground_bhms_.size());

        for (auto &[key, last_ts]: m_near_ground_bhms_) {
            auto it = bhm_dict.find(key);
            if (it == bhm_dict.end() || it->second == nullptr) { continue; }
            const auto &local_bhm = *it->second;

            // Check if this BHM has been updated since last processed.
            if (local_bhm.surface_update_timestamp <= last_ts) { continue; }

            // If bounding box is set, skip BHMs outside it.
            if (m_setting_->use_bounding_box) {
                const auto center = mapping.GetClusterCenter(key);
                const Dtype half = mapping.GetClusterSize() * static_cast<Dtype>(0.5);
                const auto &bb = m_setting_->bounding_box;
                if (center[0] + half < bb.min()[0] || center[0] - half > bb.max()[0] ||
                    center[1] + half < bb.min()[1] || center[1] - half > bb.max()[1]) {
                    continue;
                }
            }

            BhmTriangleData data;
            data.key = key;

            // Copy vertex positions from the surface data buffer.
            // Build local edge_idx -> vertex_index map.
            absl::flat_hash_map<typename SurfaceMapping::GridIndex, int> edge_to_vertex;
            edge_to_vertex.reserve(local_bhm.surface_indices.size());
            data.vertices.reserve(local_bhm.surface_indices.size());
            for (const auto &[edge_idx, surf_idx]: local_bhm.surface_indices) {
                VectorD position = surf_data_buffer[surf_idx].position;
                for (int dim = 0; dim < 3; ++dim) { position[dim] /= scaling; }
                edge_to_vertex[edge_idx] = static_cast<int>(data.vertices.size());
                data.vertices.push_back(position);
            }

            // Copy faces, remapping edge indices to local vertex indices.
            data.faces.reserve(local_bhm.num_faces);
            for (const auto &[voxel_idx, voxel]: local_bhm.surf_voxels) {
                if (!voxel.good) { continue; }
                for (const auto &face: voxel.faces) {
                    Face remapped;
                    bool valid = true;
                    for (int d = 0; d < 3; ++d) {
                        auto vit = edge_to_vertex.find(voxel.edges[face[d]]);
                        if (vit == edge_to_vertex.end()) {
                            valid = false;
                            break;
                        }
                        remapped[d] = vit->second;
                    }
                    if (valid) { data.faces.push_back(remapped); }
                }
            }

            // Update timestamp.
            last_ts = local_bhm.surface_update_timestamp;

            result.push_back(std::move(data));
        }

        return result;
    }

    template<typename Dtype, bool Colored>
    std::vector<typename HeightMapProjector<Dtype, Colored>::LocalPatch>
    HeightMapProjector<Dtype, Colored>::ComputeLocalHeightMaps(
        const std::vector<BhmTriangleData> &triangle_data,
        const SurfaceMapping &mapping) {

        const int patch_size = m_patch_size_;
        const Dtype resolution = m_internal_resolution_;
        const Dtype min_nz = m_setting_->min_normal_z;

        std::vector<LocalPatch> patches(triangle_data.size());

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(triangle_data, patches, mapping, patch_size, resolution, min_nz)
        for (std::size_t i = 0; i < triangle_data.size(); ++i) {
            const auto &td = triangle_data[i];
            LocalPatch &patch = patches[i];

            // Compute this BHM's patch offset in global coordinates.
            ComputePatchOffset(td.key, mapping, patch.global_row_offset, patch.global_col_offset);
            patch.rows = patch_size;
            patch.cols = patch_size;

            // Initialize local height map with sentinel.
            patch.height_map = Eigen::MatrixX<Dtype>::Constant(patch_size, patch_size, kSentinel);

            if (td.faces.empty()) { continue; }  // pruned BHM — patch stays sentinel

            // Compute the metric origin of this patch.
            const Dtype origin_x =
                m_global_origin_x_ + static_cast<Dtype>(patch.global_row_offset) * resolution;
            const Dtype origin_y =
                m_global_origin_y_ + static_cast<Dtype>(patch.global_col_offset) * resolution;

            const bool needs_rasterization = (m_internal_resolution_ < m_surface_resolution_);

            for (const auto &face: td.faces) {
                const VectorD &v0 = td.vertices[face[0]];
                const VectorD &v1 = td.vertices[face[1]];
                const VectorD &v2 = td.vertices[face[2]];

                // Compute face normal.
                const VectorD e1 = v1 - v0;
                const VectorD e2 = v2 - v0;
                const VectorD normal = e1.cross(e2);
                const Dtype norm = normal.norm();
                if (norm < static_cast<Dtype>(1e-8)) { continue; }
                const Dtype nz = std::abs(normal[2]) / norm;
                if (nz < min_nz) { continue; }  // not ground-like

                if (needs_rasterization) {
                    RasterizeTriangle(
                        v0,
                        v1,
                        v2,
                        min_nz,
                        origin_x,
                        origin_y,
                        resolution,
                        patch_size,
                        patch_size,
                        patch.height_map);
                } else {
                    // internal_res == surface_res: just compute mean Z and write to one cell.
                    const Dtype mean_z = (v0[2] + v1[2] + v2[2]) / static_cast<Dtype>(3);
                    // Map triangle centroid to grid cell.
                    const Dtype cx = (v0[0] + v1[0] + v2[0]) / static_cast<Dtype>(3);
                    const Dtype cy = (v0[1] + v1[1] + v2[1]) / static_cast<Dtype>(3);
                    const int row = static_cast<int>(std::floor((cx - origin_x) / resolution));
                    const int col = static_cast<int>(std::floor((cy - origin_y) / resolution));
                    if (row >= 0 && row < patch_size && col >= 0 && col < patch_size) {
                        Dtype &cell = patch.height_map(row, col);
                        if (mean_z > cell) { cell = mean_z; }
                    }
                }
            }
        }

        return patches;
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::GrowGlobalMap(std::vector<LocalPatch> &patches) {

        if (patches.empty()) { return; }

        // Find the bounds needed.
        long min_row = 0;
        long max_row = m_global_height_map_.rows() - 1;
        long min_col = 0;
        long max_col = m_global_height_map_.cols() - 1;

        for (const auto &p: patches) {
            const long pr = p.global_row_offset;
            const long pc = p.global_col_offset;
            min_row = std::min(min_row, pr);
            max_row = std::max(max_row, pr + static_cast<long>(p.rows) - 1);
            min_col = std::min(min_col, pc);
            max_col = std::max(max_col, pc + static_cast<long>(p.cols) - 1);
        }

        const long new_rows = MakeOdd(max_row - min_row + 1);
        const long new_cols = MakeOdd(max_col - min_col + 1);

        if (new_rows <= m_global_height_map_.rows() && new_cols <= m_global_height_map_.cols() &&
            min_row >= 0 && min_col >= 0) {
            return;  // No growth needed.
        }

        // Allocate new map.
        Eigen::MatrixX<Dtype> new_map =
            Eigen::MatrixX<Dtype>::Constant(new_rows, new_cols, kSentinel);

        // Copy old data into the new map at the correct offset.
        if (m_global_height_map_.size() > 0) {
            const long row_shift = -min_row;
            const long col_shift = -min_col;
            new_map.block(
                row_shift,
                col_shift,
                m_global_height_map_.rows(),
                m_global_height_map_.cols()) = m_global_height_map_;
        }

        m_global_height_map_ = std::move(new_map);
        m_global_origin_x_ += static_cast<Dtype>(min_row) * m_internal_resolution_;
        m_global_origin_y_ += static_cast<Dtype>(min_col) * m_internal_resolution_;

        // Shift patch offsets to account for the new origin.
        const int row_shift = static_cast<int>(-min_row);
        const int col_shift = static_cast<int>(-min_col);
        for (auto &p: patches) {
            p.global_row_offset += row_shift;
            p.global_col_offset += col_shift;
        }
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::MergePatches(const std::vector<LocalPatch> &patches) {
        const long map_rows = m_global_height_map_.rows();
        const long map_cols = m_global_height_map_.cols();

        for (const auto &p: patches) {
            // Patch offsets are already in global map cell coordinates
            // (adjusted by GrowGlobalMap if the map was expanded).

            // Fast path: patch fits entirely within the global map (no clamping needed).
            if (p.global_row_offset >= 0 && p.global_col_offset >= 0 &&
                p.global_row_offset + p.rows <= map_rows &&
                p.global_col_offset + p.cols <= map_cols) {
                m_global_height_map_
                    .block(p.global_row_offset, p.global_col_offset, p.rows, p.cols) = p.height_map;
                continue;
            }

            // Slow path: clamp to map bounds (bounding box edges).
            const int r0 = std::max(0, p.global_row_offset);
            const int c0 = std::max(0, p.global_col_offset);
            const int r1 = std::min(static_cast<int>(map_rows), p.global_row_offset + p.rows);
            const int c1 = std::min(static_cast<int>(map_cols), p.global_col_offset + p.cols);

            if (r0 >= r1 || c0 >= c1) { continue; }

            m_global_height_map_.block(r0, c0, r1 - r0, c1 - c0) = p.height_map.block(
                r0 - p.global_row_offset,
                c0 - p.global_col_offset,
                r1 - r0,
                c1 - c0);
        }
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::BuildOccupancyGrid() {
        const int rows = m_global_height_map_.rows();
        const int cols = m_global_height_map_.cols();
        if (rows == 0 || cols == 0) { return; }

        // Determine output resolution and dimensions.
        Dtype output_resolution = m_setting_->target_resolution;
        int out_rows = rows;
        int out_cols = cols;
        Eigen::MatrixX<Dtype> output_height;

        if (output_resolution > m_internal_resolution_ * static_cast<Dtype>(1.01)) {
            // Downsample: block-max.
            const int block = std::max(
                1,
                static_cast<int>(std::round(output_resolution / m_internal_resolution_)));
            output_resolution = m_internal_resolution_ * static_cast<Dtype>(block);
            out_rows = MakeOdd((rows + block - 1) / block);
            out_cols = MakeOdd((cols + block - 1) / block);
            output_height = Eigen::MatrixX<Dtype>::Constant(out_rows, out_cols, kSentinel);

            for (int c = 0; c < out_cols; ++c) {
                for (int r = 0; r < out_rows; ++r) {
                    const int sr = r * block;
                    const int sc = c * block;
                    const int er = std::min(sr + block, rows);
                    const int ec = std::min(sc + block, cols);
                    Dtype max_z = kSentinel;
                    for (int cc = sc; cc < ec; ++cc) {
                        for (int rr = sr; rr < er; ++rr) {
                            const Dtype z = m_global_height_map_(rr, cc);
                            if (z > max_z) { max_z = z; }
                        }
                    }
                    output_height(r, c) = max_z;
                }
            }
        } else {
            output_resolution = m_internal_resolution_;
            output_height = m_global_height_map_;
        }

        // Build occupancy from height map.
        const Dtype max_step = m_setting_->max_step_height;
        m_occupancy_data_.resize(out_rows * out_cols);

        for (int c = 0; c < out_cols; ++c) {
            for (int r = 0; r < out_rows; ++r) {
                const Dtype z = output_height(r, c);
                int8_t &cell = m_occupancy_data_[r * out_cols + c];

                if (z <= kSentinel + static_cast<Dtype>(1)) {
                    cell = -1;  // unknown
                    continue;
                }

                // Check step height against 4-neighbors.
                bool obstacle = false;
                const int dr[] = {-1, 1, 0, 0};
                const int dc[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; ++d) {
                    const int nr = r + dr[d];
                    const int nc = c + dc[d];
                    if (nr < 0 || nr >= out_rows || nc < 0 || nc >= out_cols) { continue; }
                    const Dtype nz = output_height(nr, nc);
                    if (nz <= kSentinel + static_cast<Dtype>(1)) { continue; }
                    if (std::abs(z - nz) > max_step) {
                        obstacle = true;
                        break;
                    }
                }

                cell = obstacle ? 100 : 0;
            }
        }

        // Build output grid info: GridMapInfo(origin, resolution, shape).
        const Eigen::Vector2<Dtype> origin(m_global_origin_x_, m_global_origin_y_);
        const Eigen::Vector2<Dtype> res(output_resolution, output_resolution);
        const Eigen::Vector2i shape(out_rows, out_cols);
        m_output_grid_info_ = std::make_unique<GridMapInfo>(origin, res, shape);
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::UpdateNearGroundBhms(
        const SurfaceMapping &mapping,
        const Dtype sensor_z) {

        const bool needs_update =
            !m_near_ground_initialized_ ||
            std::abs(sensor_z - m_last_sensor_z_) > m_setting_->sensor_z_change_threshold;

        if (!needs_update) {
            // Still add any new BHMs that might have been created.
            const auto &bhm_dict = mapping.GetLocalBhms();
            const Dtype z_lo = m_last_sensor_z_ + m_setting_->ground_z_min;
            const Dtype z_hi = m_last_sensor_z_ + m_setting_->ground_z_max;
            const Dtype half = mapping.GetClusterSize() * static_cast<Dtype>(0.5);

            for (const auto &[key, bhm_ptr]: bhm_dict) {
                if (bhm_ptr == nullptr) { continue; }
                if (m_near_ground_bhms_.contains(key)) { continue; }
                const auto center = mapping.GetClusterCenter(key);
                if (center[2] - half <= z_hi && center[2] + half >= z_lo) {
                    m_near_ground_bhms_[key] = 0;
                }
            }
            return;
        }

        // Full re-evaluation.
        m_last_sensor_z_ = sensor_z;
        m_near_ground_initialized_ = true;

        const Dtype z_lo = sensor_z + m_setting_->ground_z_min;
        const Dtype z_hi = sensor_z + m_setting_->ground_z_max;
        const Dtype half = mapping.GetClusterSize() * static_cast<Dtype>(0.5);

        absl::flat_hash_map<Key, std::size_t> new_map;
        const auto &bhm_dict = mapping.GetLocalBhms();
        new_map.reserve(bhm_dict.size());
        for (const auto &[key, bhm_ptr]: bhm_dict) {
            if (bhm_ptr == nullptr) { continue; }
            const auto center = mapping.GetClusterCenter(key);
            if (center[2] - half <= z_hi && center[2] + half >= z_lo) {
                // Preserve existing timestamp if re-adding.
                auto it = m_near_ground_bhms_.find(key);
                const std::size_t ts = (it != m_near_ground_bhms_.end()) ? it->second : 0;
                new_map[key] = ts;
            }
        }
        m_near_ground_bhms_ = std::move(new_map);
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::ComputePatchOffset(
        const Key &key,
        const SurfaceMapping &mapping,
        int &row_offset,
        int &col_offset) const {
        const auto center = mapping.GetClusterCenter(key);
        const Dtype half = mapping.GetClusterSize() * static_cast<Dtype>(0.5);
        // BHM patch min corner in metric coordinates.
        const Dtype patch_min_x = center[0] - half;
        const Dtype patch_min_y = center[1] - half;
        // Convert to global cell coordinates.
        row_offset = static_cast<int>(
            std::round((patch_min_x - m_global_origin_x_) / m_internal_resolution_));
        col_offset = static_cast<int>(
            std::round((patch_min_y - m_global_origin_y_) / m_internal_resolution_));
    }

    template<typename Dtype, bool Colored>
    void
    HeightMapProjector<Dtype, Colored>::RasterizeTriangle(
        const VectorD &v0,
        const VectorD &v1,
        const VectorD &v2,
        const Dtype /*min_normal_z*/,
        const Dtype origin_x,
        const Dtype origin_y,
        const Dtype resolution,
        const int rows,
        const int cols,
        Eigen::MatrixX<Dtype> &height_map) {

        const Dtype mean_z = (v0[2] + v1[2] + v2[2]) / static_cast<Dtype>(3);

        // Bounding box in grid coordinates.
        const Dtype min_x = std::min({v0[0], v1[0], v2[0]});
        const Dtype max_x = std::max({v0[0], v1[0], v2[0]});
        const Dtype min_y = std::min({v0[1], v1[1], v2[1]});
        const Dtype max_y = std::max({v0[1], v1[1], v2[1]});

        const int r0 = std::max(0, static_cast<int>(std::floor((min_x - origin_x) / resolution)));
        const int r1 =
            std::min(rows - 1, static_cast<int>(std::floor((max_x - origin_x) / resolution)));
        const int c0 = std::max(0, static_cast<int>(std::floor((min_y - origin_y) / resolution)));
        const int c1 =
            std::min(cols - 1, static_cast<int>(std::floor((max_y - origin_y) / resolution)));

        const int bbox_span = std::max(r1 - r0, c1 - c0) + 1;

        if (bbox_span <= 4) {
            // Small triangle: brute-force bbox scan with point-in-triangle test.
            const Dtype e0x = v1[0] - v0[0], e0y = v1[1] - v0[1];
            const Dtype e1x = v2[0] - v1[0], e1y = v2[1] - v1[1];
            const Dtype e2x = v0[0] - v2[0], e2y = v0[1] - v2[1];

            for (int c = c0; c <= c1; ++c) {
                const Dtype py =
                    origin_y + (static_cast<Dtype>(c) + static_cast<Dtype>(0.5)) * resolution;
                for (int r = r0; r <= r1; ++r) {
                    const Dtype px =
                        origin_x + (static_cast<Dtype>(r) + static_cast<Dtype>(0.5)) * resolution;

                    const Dtype d0 = e0x * (py - v0[1]) - e0y * (px - v0[0]);
                    const Dtype d1 = e1x * (py - v1[1]) - e1y * (px - v1[0]);
                    const Dtype d2 = e2x * (py - v2[1]) - e2y * (px - v2[0]);

                    if ((d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0)) {
                        Dtype &cell = height_map(r, c);
                        if (mean_z > cell) { cell = mean_z; }
                    }
                }
            }
        } else {
            // Large triangle: scanline rasterization.
            // Sort vertices by Y (column) coordinate for column-major friendly scanline.
            const VectorD *verts[3] = {&v0, &v1, &v2};
            if ((*verts[0])[1] > (*verts[1])[1]) { std::swap(verts[0], verts[1]); }
            if ((*verts[1])[1] > (*verts[2])[1]) { std::swap(verts[1], verts[2]); }
            if ((*verts[0])[1] > (*verts[1])[1]) { std::swap(verts[0], verts[1]); }
            const VectorD &va = *verts[0];  // left (min Y)
            const VectorD &vb = *verts[1];  // middle
            const VectorD &vc = *verts[2];  // right (max Y)

            const Dtype inv_res = static_cast<Dtype>(1) / resolution;
            const Dtype half = static_cast<Dtype>(0.5);

            // Edge slopes: dX/dY for each edge.
            const Dtype dy_ac = vc[1] - va[1];
            const Dtype dy_ab = vb[1] - va[1];
            const Dtype dy_bc = vc[1] - vb[1];

            const Dtype slope_ac = (std::abs(dy_ac) > static_cast<Dtype>(1e-8))
                                       ? (vc[0] - va[0]) / dy_ac
                                       : static_cast<Dtype>(0);
            const Dtype slope_ab = (std::abs(dy_ab) > static_cast<Dtype>(1e-8))
                                       ? (vb[0] - va[0]) / dy_ab
                                       : static_cast<Dtype>(0);
            const Dtype slope_bc = (std::abs(dy_bc) > static_cast<Dtype>(1e-8))
                                       ? (vc[0] - vb[0]) / dy_bc
                                       : static_cast<Dtype>(0);

            for (int c = c0; c <= c1; ++c) {
                const Dtype py = origin_y + (static_cast<Dtype>(c) + half) * resolution;
                Dtype x_edge1, x_edge2;

                // Long edge (va -> vc) is always active.
                x_edge1 = va[0] + slope_ac * (py - va[1]);

                // Short edge depends on which half of the triangle.
                if (py < vb[1]) {
                    x_edge2 = va[0] + slope_ab * (py - va[1]);  // va -> vb
                } else {
                    x_edge2 = vb[0] + slope_bc * (py - vb[1]);  // vb -> vc
                }

                // Ensure x_top <= x_bottom.
                if (x_edge1 > x_edge2) { std::swap(x_edge1, x_edge2); }

                const int rl =
                    std::max(r0, static_cast<int>(std::floor((x_edge1 - origin_x) * inv_res)));
                const int rr =
                    std::min(r1, static_cast<int>(std::floor((x_edge2 - origin_x) * inv_res)));

                for (int r = rl; r <= rr; ++r) {
                    Dtype &cell = height_map(r, c);
                    if (mean_z > cell) { cell = mean_z; }
                }
            }
        }
    }

    // Explicit template instantiations.
    template struct HeightMapProjectorSetting<float>;
    template struct HeightMapProjectorSetting<double>;
    template class HeightMapProjector<float>;
    template class HeightMapProjector<double>;
    template class HeightMapProjector<float, true>;
    template class HeightMapProjector<double, true>;

}  // namespace erl::gp_sdf
