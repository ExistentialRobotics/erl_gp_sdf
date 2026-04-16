#pragma once

#include "bayesian_hilbert_surface_mapping.hpp"
#include "height_map_projector_setting.hpp"

#include "erl_common/grid_map_info.hpp"

#include <omp.h>

#include <memory>

namespace erl::gp_sdf {

    /**
     * Projects 3D BHM surface mesh triangles onto a 2D height map and derives a Nav2-compatible
     * occupancy grid. Only processes ground-like triangles (upward-facing normals). Uses
     * incremental updates via per-BHM surface timestamps.
     *
     * Pipeline (called via Update()):
     *   1. [LOCKED]   Collect changed near-ground BHMs, copy triangle data, update timestamps.
     *   2. [UNLOCKED] Compute local height maps per changed BHM (parallel, non-overlapping).
     *   3. [UNLOCKED] Grow global height map if needed.
     *   4. [UNLOCKED] Paste local height maps into global map (sequential).
     *   5. [UNLOCKED] Downsample if needed, build Nav2 occupancy.
     */
    template<typename Dtype, bool Colored = false>
    class HeightMapProjector {
    public:
        using SurfaceMapping = BayesianHilbertSurfaceMapping<Dtype, 3, Colored>;
        using Key = geometry::OctreeKey;
        using Face = Eigen::Vector3i;
        using VectorD = Eigen::Vector3<Dtype>;
        using Setting = HeightMapProjectorSetting<Dtype>;
        using GridMapInfo = common::GridMapInfo2D<Dtype>;

        /// Copied triangle data from a single BHM during the locked section.
        struct BhmTriangleData {
            Key key;
            std::vector<VectorD> vertices;  // vertex positions (unscaled)
            std::vector<Face> faces;        // face vertex indices (local to this BHM)
        };

        /// Local height map patch for a single BHM.
        struct LocalPatch {
            int global_row_offset = 0;
            int global_col_offset = 0;
            int rows = 0;
            int cols = 0;
            Eigen::MatrixX<Dtype> height_map;
        };

        static constexpr Dtype kSentinel = -std::numeric_limits<Dtype>::max();

    private:
        // settings
        std::shared_ptr<Setting> m_setting_;

        // derived constants (set during first Update)
        Dtype m_surface_resolution_ = 0;
        Dtype m_internal_resolution_ = 0;
        int m_patch_size_ = 0;  // number of internal cells per BHM patch (per axis)
        bool m_initialized_ = false;

        // persistent global height map
        Eigen::MatrixX<Dtype> m_global_height_map_;
        Dtype m_global_origin_x_ = 0;  // metric X of the global map origin (min corner)
        Dtype m_global_origin_y_ = 0;  // metric Y of the global map origin (min corner)

        // near-ground BHM tracking: key -> last processed surface_update_timestamp.
        // A BHM being in this map means it is considered near-ground.
        absl::flat_hash_map<Key, std::size_t> m_near_ground_bhms_;
        Dtype m_last_sensor_z_ = 0;
        bool m_near_ground_initialized_ = false;

        // latest output
        std::vector<int8_t> m_occupancy_data_;
        std::unique_ptr<GridMapInfo> m_output_grid_info_;

    public:
        explicit HeightMapProjector(std::shared_ptr<Setting> setting);

        /// Main entry point. Locks the surface mapping briefly to collect changes,
        /// then processes everything unlocked.
        void
        Update(SurfaceMapping &mapping);

        /// Get the latest Nav2 occupancy grid.
        /// @param occupancy_data Output: -1 (unknown), 0 (free), 100 (occupied).
        /// @param grid_info Output: grid metadata (origin, resolution, dimensions).
        void
        GetOccupancyGrid(std::vector<int8_t> &occupancy_data, GridMapInfo &grid_info) const;

        /// Get the internal height map for visualization/debugging.
        /// @param height_data Output: height values (sentinel = unknown).
        /// @param grid_info Output: grid metadata.
        void
        GetHeightMap(Eigen::MatrixX<Dtype> &height_data, GridMapInfo &grid_info) const;

        [[nodiscard]] Dtype
        GetSurfaceResolution() const {
            return m_surface_resolution_;
        }

        [[nodiscard]] Dtype
        GetInternalResolution() const {
            return m_internal_resolution_;
        }

    private:
        /// [LOCKED] Collect changed BHMs, copy their triangle data, update timestamps.
        std::vector<BhmTriangleData>
        CollectChangedBhms(SurfaceMapping &mapping);

        /// [UNLOCKED] Compute local height maps in parallel.
        std::vector<LocalPatch>
        ComputeLocalHeightMaps(
            const std::vector<BhmTriangleData> &triangle_data,
            const SurfaceMapping &mapping);

        /// [UNLOCKED] Grow the global height map if any patch falls outside bounds.
        /// Adjusts patch offsets to account for the new origin.
        void
        GrowGlobalMap(std::vector<LocalPatch> &patches);

        /// [UNLOCKED] Paste local patches into the global map sequentially.
        void
        MergePatches(const std::vector<LocalPatch> &patches);

        /// [UNLOCKED] Downsample internal map if needed and build the occupancy grid.
        void
        BuildOccupancyGrid();

        /// Update the near-ground BHM list based on current sensor Z.
        void
        UpdateNearGroundBhms(const SurfaceMapping &mapping, Dtype sensor_z);

        /// Compute BHM patch position in the global height map.
        void
        ComputePatchOffset(
            const Key &key,
            const SurfaceMapping &mapping,
            int &row_offset,
            int &col_offset) const;

        /// Rasterize a single triangle onto a local height map patch.
        static void
        RasterizeTriangle(
            const VectorD &v0,
            const VectorD &v1,
            const VectorD &v2,
            Dtype min_normal_z,
            Dtype origin_x,
            Dtype origin_y,
            Dtype resolution,
            int rows,
            int cols,
            Eigen::MatrixX<Dtype> &height_map);
    };

    using HeightMapProjectorF = HeightMapProjector<float>;
    using HeightMapProjectorD = HeightMapProjector<double>;
    using ColoredHeightMapProjectorF = HeightMapProjector<float, true>;
    using ColoredHeightMapProjectorD = HeightMapProjector<double, true>;

    extern template class HeightMapProjector<float>;
    extern template class HeightMapProjector<double>;
    extern template class HeightMapProjector<float, true>;
    extern template class HeightMapProjector<double, true>;

}  // namespace erl::gp_sdf
