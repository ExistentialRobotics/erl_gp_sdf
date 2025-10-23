#pragma once

#include "surface_data_manager.hpp"

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
        long min_dataset_size = 0;            // minimum size of the dataset required to update
        long max_dataset_size = -1;           // maximum size of the dataset to store
        long hit_buffer_size = -1;            // -1 means no limit, 0 means no hit buffer
        long surface_grid_size = 5;           // size of the surface grid
        Dtype surface_log_odds = 0.0f;        // log-odds value for the surface points
        Dtype surface_log_odds_min = -20.0f;  // minimum log-odds value for the surface points
        Dtype surface_log_odds_max = 20.0f;   // maximum log-odds value for the surface points
        Dtype surface_log_odds_lr = 0.1f;     // learning rate for updating the surface log-odds
        // if true, pass faster=true to the Bayesian Hilbert map predict methods, which assumes
        // that the weight covariance is very small.
        bool faster_prediction = false;

        struct YamlConvertImpl {
            static YAML::Node
            encode(const LocalBayesianHilbertMapSetting &setting);

            static bool
            decode(const YAML::Node &node, LocalBayesianHilbertMapSetting &setting);
        };
    };

    template<typename Dtype, int Dim>
    class LocalBayesianHilbertMap {
    public:
        using Setting = LocalBayesianHilbertMapSetting<Dtype>;
        using Covariance = covariance::Covariance<Dtype>;

        using Aabb = geometry::Aabb<Dtype, Dim>;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using Gradient = Position;
        using Gradients = Positions;
        using VectorX = Eigen::VectorX<Dtype>;
        using Face = Eigen::Vector<int, Dim>;
        using SurfData = SurfaceData<Dtype, Dim>;

        using GridIndex = Eigen::Vector<long, Dim + 1>;
        using SurfaceIndexMap = absl::flat_hash_map<GridIndex, std::size_t>;
        using SurfaceDataMap = absl::flat_hash_map<GridIndex, SurfData>;
        using BayesianHilbertMap = geometry::BayesianHilbertMap<Dtype, Dim>;

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
        Aabb tracked_surface_boundary{};             // boundary of the surface to track
        Position tracked_surface_resolution{};       // resolution of the tracked surface
        BayesianHilbertMap bhm;                      // local Bayesian Hilbert map
        SurfaceIndexMap surface_indices;             // grid/edge index -> buffer index
        SurfaceVoxelMap surf_voxels;                 // surface voxels
        long dataset_size = 0;                       // number of dataset points
        Positions dataset_points{};                  // [Dim, N] dataset points
        VectorX dataset_labels{};                    // [N, 1] dataset labels
        std::vector<long> hit_indices{};             // indices of the hit points in the dataset
        std::vector<Position> hit_buffer{};          // hit point buffer of M points
        long hit_buffer_head = 0;                    // head of the hit point buffer
        bool active = false;                         // whether the local BHM is active
        SurfaceDataMap surf_data_cache;              // temporary cache
        Dtype surface_log_odds = 0.0f;               // log-odds value for surface points
        uint64_t log_odds_count = 1;                 // number of log-odds samples

        LocalBayesianHilbertMap(
            std::shared_ptr<Setting> setting_,
            Positions hinged_points,
            Aabb map_boundary,
            uint64_t seed,
            Aabb track_surface_boundary_);

        void
        Reset();

        bool
        GenerateDataset(
            const Eigen::Ref<const Position> &sensor_origin,
            const Eigen::Ref<const Positions> &points,
            const std::vector<long> &point_indices);

        bool
        Update(
            const Eigen::Ref<const Position> &sensor_origin,
            const Eigen::Ref<const Positions> &points,
            const std::vector<long> &point_indices,
            bool update_surface_log_odds);

        void
        UpdateHitBuffer(const Eigen::Ref<const Positions> &points);

        [[nodiscard]] bool
        GetGridCoords(
            const Eigen::Ref<const Position> &point,
            bool check_bounds,
            GridIndex &grid_coords) const;

        void
        Predict(
            const Eigen::Ref<const Positions> &points,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            bool parallel,
            VectorX &prob_occupied,
            Eigen::VectorXb &in_free_space,
            Gradients &gradient) const;

        void
        PredictAt(
            const Position &point,
            bool logodd,
            bool compute_free_space,
            bool compute_gradient,
            bool gradient_with_sigmoid,
            Dtype &prob_occupied,
            bool &in_free_space,
            Gradient &gradient) const;

        void
        PredictGradient(
            const Eigen::Ref<const Positions> &points,
            bool with_sigmoid,
            bool parallel,
            Gradients &gradient) const;

        [[nodiscard]] bool
        Write(std::ostream &s) const;

        [[nodiscard]] bool
        Read(std::istream &s);

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

template<>
struct YAML::convert<erl::gp_sdf::LocalBayesianHilbertMapSettingF>
    : erl::gp_sdf::LocalBayesianHilbertMapSettingF::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::LocalBayesianHilbertMapSettingD>
    : erl::gp_sdf::LocalBayesianHilbertMapSettingD::YamlConvertImpl {};
