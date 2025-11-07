#pragma once

#include "surface_data_manager.hpp"

#include "erl_common/ring_buffer.hpp"
#include "erl_common/yaml.hpp"
#include "erl_covariance/covariance.hpp"
#include "erl_geometry/bayesian_hilbert_map.hpp"

namespace erl::gp_sdf {

    template<typename Dtype>
    class LocalBayesianHilbertMapSetting
        : public common::Yamlable<LocalBayesianHilbertMapSetting<Dtype>> {
    public:
        using Covariance = covariance::Covariance<Dtype>;
        using KernelSetting = typename Covariance::Setting;
        using BhmSetting = geometry::BayesianHilbertMapSetting;

        std::shared_ptr<BhmSetting> bhm = std::make_shared<BhmSetting>();
        std::string kernel_type = type_name<Covariance>();
        std::string kernel_setting_type = type_name<KernelSetting>();
        std::shared_ptr<KernelSetting> kernel = std::make_shared<KernelSetting>();
        long min_dataset_size = 0;             // minimum size of the dataset required to update
        long min_dataset_hit_size = 1;         // minimum number of hit points in the dataset
        long max_dataset_size = -1;            // maximum size of the dataset to store
        long hit_point_buffer_size = -1;       // <=0 means no limit
        long ray_buffer_size = -1;             // <=0 means no limit
        long surface_grid_size = 5;            // size of the surface grid
        Dtype surface_log_odds = 0.0f;         // log-odds value for the surface points
        long surface_log_odds_init_count = 1;  // initial number of log-odds sample count
        Dtype surface_log_odds_min = -20.0f;   // minimum log-odds value for the surface points
        Dtype surface_log_odds_max = 20.0f;    // maximum log-odds value for the surface points
        bool auto_surface_log_odds = true;     // automatically learn the surface log-odds
        bool include_neighbor_voxels = true;   // include neighbor voxels when updating the surface
        // if true, pass faster=true to the Bayesian Hilbert map predict methods, which assumes
        // that the weight covariance is very small.
        bool faster_prediction = false;

        ERL_REFLECT_SCHEMA(
            LocalBayesianHilbertMapSetting,
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, bhm),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, kernel_type),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, kernel_setting_type),
            ERL_REFLECT_MEMBER_POLY(LocalBayesianHilbertMapSetting, kernel, kernel_setting_type),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, min_dataset_size),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, max_dataset_size),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, hit_point_buffer_size),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, ray_buffer_size),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, surface_grid_size),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, surface_log_odds),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, surface_log_odds_min),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, surface_log_odds_max),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, auto_surface_log_odds),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, include_neighbor_voxels),
            ERL_REFLECT_MEMBER(LocalBayesianHilbertMapSetting, faster_prediction));
    };

    template<typename Dtype, int Dim>
    class LocalBayesianHilbertMap {
    public:
        using Setting = LocalBayesianHilbertMapSetting<Dtype>;
        using Covariance = covariance::Covariance<Dtype>;

        using Aabb = geometry::Aabb<Dtype, Dim>;
        using VectorD = Eigen::Vector<Dtype, Dim>;
        using MatrixDX = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Face = Eigen::Vector<int, Dim>;
        using SurfData = SurfaceData<Dtype, Dim>;

        using GridIndex = Eigen::Vector<long, Dim + 1>;
        using SurfaceIndexMap = absl::flat_hash_map<GridIndex, std::size_t>;
        using SurfaceDataMap = absl::flat_hash_map<GridIndex, SurfData>;
        using BayesianHilbertMap = geometry::BayesianHilbertMap<Dtype, Dim>;
        using RayInfo = typename geometry::OccupancyMap<Dtype, Dim>::RayInfo;

        struct Voxel {
            bool good = false;
            int surf_config = 0;
            std::vector<GridIndex> edges{};
            std::vector<Face> faces{};

            [[nodiscard]] bool
            operator==(const Voxel &other) const;

            [[nodiscard]] bool
            operator!=(const Voxel &other) const;
        };

        using SurfaceVoxelMap = absl::flat_hash_map<GridIndex, Voxel>;

        std::shared_ptr<Setting> setting = nullptr;  // settings for the local map
        Aabb tracked_surface_boundary;               // boundary of the surface to track
        VectorD tracked_surface_resolution;          // resolution of the tracked surface
        BayesianHilbertMap bhm;                      // local Bayesian Hilbert map
        SurfaceIndexMap surface_indices;             // grid/edge index -> buffer index
        SurfaceVoxelMap surf_voxels;                 // surface voxels
        long dataset_hit_size = 0;                   // number of hit points in the dataset
        long dataset_size = 0;                       // number of dataset points
        MatrixDX dataset_points;                     // [Dim, N] dataset points
        VectorX dataset_labels;                      // [N, 1] dataset labels
        std::vector<long> hit_indices;               // indices of the hit points in the dataset
        std::vector<VectorD> hit_point_buffer;       // hit point buffer of M points
        std::vector<RayInfo> ray_info_buffer;        // ray info buffer
        common::RingBuffer<VectorD> hit_point_ring_buffer{1};  // hit point ring buffer
        common::RingBuffer<RayInfo> ray_info_ring_buffer{1};   // ray info ring buffer
        long unused_ray_count = 0;       // number of unused rays in the ray buffer
        bool active = false;             // whether the local BHM is active
        SurfaceDataMap surf_data_cache;  // temporary cache
        Dtype surface_log_odds = 0.0f;   // log-odds value for surface points
        uint64_t log_odds_count = 1;     // number of log-odds samples

        LocalBayesianHilbertMap(
            std::shared_ptr<Setting> setting_,
            MatrixDX hinged_points,
            Aabb map_boundary,
            uint64_t seed,
            Aabb track_surface_boundary_);

        [[nodiscard]] bool
        HasRaysUnused() const {
            return unused_ray_count > 0;
        }

        void
        GenerateDataset(
            const Eigen::Ref<const VectorD> &sensor_position,
            const Eigen::Ref<const MatrixDX> &points,
            const std::vector<long> &point_indices);

        bool
        UpdateSurface(const Eigen::Ref<const MatrixDX> &points, bool update_surface_voxels);

        bool
        Update(
            const Eigen::Ref<const VectorD> &sensor_origin,
            const Eigen::Ref<const MatrixDX> &points,
            const std::vector<long> &point_indices,
            bool update_surface_voxels);

        [[nodiscard]] bool
        GetGridCoords(
            const Eigen::Ref<const VectorD> &point,
            bool check_bounds,
            GridIndex &grid_coords) const;

        void
        Predict(
            const Eigen::Ref<const MatrixDX> &points,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            VectorX &prob_occupied,
            Eigen::VectorXb &in_free_space,
            MatrixDX &gradient) const;

        void
        PredictAt(
            const VectorD &point,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            Dtype &prob_occupied,
            bool &in_free_space,
            VectorD &gradient) const;

        void
        PredictGradient(
            const Eigen::Ref<const MatrixDX> &points,
            bool with_sigmoid,
            bool parallel,
            MatrixDX &gradient) const;

        [[nodiscard]] bool
        Write(std::ostream &stream) const;

        [[nodiscard]] bool
        Read(std::istream &stream);

        [[nodiscard]] bool
        operator==(const LocalBayesianHilbertMap &other) const;

        [[nodiscard]] bool
        operator!=(const LocalBayesianHilbertMap &other) const;
    };

    extern template class LocalBayesianHilbertMapSetting<float>;
    extern template class LocalBayesianHilbertMapSetting<double>;
    extern template class LocalBayesianHilbertMap<float, 2>;
    extern template class LocalBayesianHilbertMap<float, 3>;
    extern template class LocalBayesianHilbertMap<double, 2>;
    extern template class LocalBayesianHilbertMap<double, 3>;

    using LocalBayesianHilbertMapSettingF = LocalBayesianHilbertMapSetting<float>;
    using LocalBayesianHilbertMapSettingD = LocalBayesianHilbertMapSetting<double>;
}  // namespace erl::gp_sdf
