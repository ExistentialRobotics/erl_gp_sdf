#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"
#include "surface_mapping_octree.hpp"
#include "surface_mapping_quadtree.hpp"

#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    class GpOccSurfaceMapping : public AbstractSurfaceMapping {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");
        inline static const std::string kClassName = type_name<GpOccSurfaceMapping>();
        inline static const std::string kFileHeader = fmt::format("# {}", kClassName);
        inline static const char *kFileFooter = "end_of_GpOccSurfaceMapping";

    public:
        // type definitions required by GpSdfMapping
        using Key = std::conditional_t<Dim == 2, geometry::QuadtreeKey, geometry::OctreeKey>;
        using KeySet = std::conditional_t<Dim == 2, geometry::QuadtreeKeySet, geometry::OctreeKeySet>;
        using KeyVector = std::conditional_t<Dim == 2, geometry::QuadtreeKeyVector, geometry::OctreeKeyVector>;
        using Tree = std::conditional_t<Dim == 2, SurfaceMappingQuadtree<Dtype>, SurfaceMappingOctree<Dtype>>;
        using TreeNode = std::conditional_t<Dim == 2, SurfaceMappingQuadtreeNode, SurfaceMappingOctreeNode>;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = typename SurfDataManager::Data;

        // other types
        using SensorGp = std::conditional_t<Dim == 2, gaussian_process::LidarGaussianProcess2D<Dtype>, gaussian_process::RangeSensorGaussianProcess3D<Dtype>>;
        using SensorGpSetting = typename SensorGp::Setting;
        using TreeSetting = typename Tree::Setting;
        using Aabb = geometry::Aabb<Dtype, Dim>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Vector<Dtype, Dim>;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Gradient = Position;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;

        struct Setting : common::Yamlable<Setting> {
            struct ComputeVariance {
                Dtype zero_gradient_position_var = 1.;  // position variance to set when the estimated gradient is almost zero.
                Dtype zero_gradient_gradient_var = 1.;  // gradient variance to set when the estimated gradient is almost zero.
                Dtype position_var_alpha = 0.01;        // scaling number of position variance.
                Dtype min_distance_var = 1.;            // allowed minimum distance variance.
                Dtype max_distance_var = 100.;          // allowed maximum distance variance.
                Dtype min_gradient_var = 0.01;          // allowed minimum gradient variance.
                Dtype max_gradient_var = 1.;            // allowed maximum gradient variance.
            };

            struct UpdateMapPoints {
                int max_adjust_tries = 10;
                Dtype min_observable_occ = -0.1;     // points of OCC smaller than this value is considered unobservable, i.e. inside the object.
                Dtype min_position_var = 0.001;      // minimum position variance.
                Dtype min_gradient_var = 0.001;      // minimum gradient variance.
                Dtype max_surface_abs_occ = 0.02;    // maximum absolute value of surface points' OCC, which should be zero ideally.
                Dtype max_valid_gradient_var = 0.5;  // maximum valid gradient variance, above this threshold, it won't be used for the Bayes Update.
                Dtype max_bayes_position_var = 1.;   // if the position variance by Bayes Update is above this threshold, it will be discarded.
                Dtype max_bayes_gradient_var = 0.6;  // if the gradient variance by Bayes Update is above this threshold, it will be discarded.
            };

            ComputeVariance compute_variance;
            UpdateMapPoints update_map_points;
            std::shared_ptr<SensorGpSetting> sensor_gp = std::make_shared<SensorGpSetting>();
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();
            Dtype perturb_delta = 0.01;              // perturbation delta for gradient estimation.
            Dtype zero_gradient_threshold = 1.e-15;  // gradient below this threshold is considered zero.
            bool update_occupancy = true;            // whether to update the occupancy of the occupancy tree.
            int cluster_level = 2;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SensorGp> m_sensor_gp_ = nullptr;
        std::shared_ptr<Tree> m_tree_ = nullptr;
        SurfDataManager m_surf_data_manager_;
        Eigen::Matrix<Dtype, Dim, 2 * Dim> m_pos_perturb_ = {};
        KeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const SensorGp>
        GetSensorGp() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetTree() const;

        [[nodiscard]] const SurfDataManager &
        GetSurfDataManager() const;

        bool
        Update(const Eigen::Ref<const Rotation> &rotation, const Eigen::Ref<const Translation> &translation, const Eigen::Ref<const MatrixX> &ranges);

        [[nodiscard]] bool
        operator==(const AbstractSurfaceMapping &other) const override;

        using AbstractSurfaceMapping::Read;
        using AbstractSurfaceMapping::Write;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

    private:
        static std::pair<Dtype, Dtype>
        Cartesian2Polar(Dtype x, Dtype y);

        void
        UpdateMapPoints();

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2, bool>
        ComputeOcc(const Position &pos_local, Dtype &distance_local, Eigen::Ref<Scalar> distance_pred, Eigen::Ref<Scalar> distance_pred_var, Dtype &occ) const;

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 3, bool>
        ComputeOcc(const Position &pos_local, Dtype &distance_local, Eigen::Ref<Scalar> distance_pred, Eigen::Ref<Scalar> distance_pred_var, Dtype &occ) const;

        template<int D = Dim>
        std::enable_if_t<D == 2>
        UpdateGradient(Dtype var_new, Dtype var_sum, const Gradient &grad_old, Gradient &grad_new);

        template<int D = Dim>
        std::enable_if_t<D == 3>
        UpdateGradient(Dtype var_new, Dtype var_sum, const Gradient &grad_old, Gradient &grad_new);

        void
        UpdateOccupancy();

        void
        AddNewMeasurement();

        void
        RecordChangedKey(const Key &key);

        bool
        ComputeGradient1(const Position &pos_local, Gradient &gradient, Dtype &occ_mean, Dtype &distance_var);

        bool
        ComputeGradient2(const Eigen::Ref<const Position> &pos_local, Gradient &gradient, Dtype &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Position> &pos_local,
            const Gradient &grad_local,
            const Dtype &distance,
            const Dtype &distance_var,
            const Dtype &occ_mean_abs,
            const Dtype &occ_abs,
            bool new_point,
            Dtype &var_position,
            Dtype &var_gradient) const;
    };

    using GpOccSurfaceMapping3Dd = GpOccSurfaceMapping<double, 3>;
    using GpOccSurfaceMapping3Df = GpOccSurfaceMapping<float, 3>;
    using GpOccSurfaceMapping2Dd = GpOccSurfaceMapping<double, 2>;
    using GpOccSurfaceMapping2Df = GpOccSurfaceMapping<float, 2>;

}  // namespace erl::sdf_mapping

#include "gp_occ_surface_mapping.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping3Dd::Setting> : erl::sdf_mapping::GpOccSurfaceMapping3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping3Df::Setting> : erl::sdf_mapping::GpOccSurfaceMapping3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting> : erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting> : erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting::YamlConvertImpl {};
