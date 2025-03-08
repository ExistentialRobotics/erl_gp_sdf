#pragma once

#include "abstract_surface_mapping_2d.hpp"
#include "gp_occ_surface_mapping_base_setting.hpp"
#include "surface_data_manager.hpp"
#include "surface_mapping_quadtree.hpp"

#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"

#include <memory>

namespace erl::sdf_mapping {

    template<typename Dtype>
    class GpOccSurfaceMapping2D : public AbstractSurfaceMapping2D<Dtype> {
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<GpOccSurfaceMapping2D>());

    public:
        using Super = AbstractSurfaceMapping2D<Dtype>;
        using Key = geometry::QuadtreeKey;
        using SensorGp = gaussian_process::LidarGaussianProcess2D<Dtype>;
        using Tree = SurfaceMappingQuadtree<Dtype>;
        using TreeNode = SurfaceMappingQuadtreeNode;
        using SurfaceDataManager2D = SurfaceDataManager<Dtype, 2>;
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using Matrix2 = Eigen::Matrix2<Dtype>;
        using Matrix2X = Eigen::Matrix2X<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Vector2 = Eigen::Vector2<Dtype>;
        using Vector3 = Eigen::Vector3<Dtype>;

        struct Setting : common::Yamlable<Setting, GpOccSurfaceMappingBaseSetting> {
            std::shared_ptr<typename SensorGp::Setting> sensor_gp = std::make_shared<typename SensorGp::Setting>();
            std::shared_ptr<typename Tree::Setting> quadtree = std::make_shared<typename Tree::Setting>();

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SensorGp> m_sensor_gp_ = nullptr;  // the GP of regression between angle and mapped distance
        std::shared_ptr<Tree> m_tree_ = nullptr;
        SurfaceDataManager2D m_surface_data_manager_;
        Eigen::Matrix<Dtype, 2, 4> m_xy_perturb_ = {};
        geometry::QuadtreeKeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping2D(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)),
              m_sensor_gp_(std::make_shared<SensorGp>(m_setting_->sensor_gp)) {
            const auto d = static_cast<Dtype>(m_setting_->perturb_delta);
            // clang-format off
            m_xy_perturb_ << d, -d, 0., 0.,
                             0., 0., d, -d;
            // clang-format on
        }

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const SensorGp>
        GetSensorGp() const;

        geometry::QuadtreeKeySet
        GetChangedClusters() override;

        [[nodiscard]] unsigned int
        GetClusterLevel() const override;

        std::shared_ptr<Tree>
        GetQuadtree() override;

        [[nodiscard]] const SurfaceDataManager2D &
        GetSurfaceDataManager() const override;

        [[nodiscard]] Dtype
        GetSensorNoise() const override;

        // METHODS REQUIRED BY GpSdfMapping
        [[nodiscard]] bool
        Ready() const;

        bool
        Update(const Eigen::Ref<const Matrix2> &rotation, const Eigen::Ref<const Vector2> &translation, const Eigen::Ref<const MatrixX> &ranges) override;

        [[nodiscard]] bool
        operator==(const Super &other) const override;

        [[nodiscard]] bool
        Write(const std::string &filename) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(const std::string &filename) override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

    protected:
        void
        UpdateMapPoints();

        void
        UpdateOccupancy();

        void
        AddNewMeasurement();

        void
        RecordChangedKey(const geometry::QuadtreeKey &key);

        bool
        ComputeGradient1(const Vector2 &xy_local, Vector2 &gradient, Dtype &occ_mean, Dtype &distance_var);

        bool
        ComputeGradient2(const Eigen::Ref<const Vector2> &xy_local, Vector2 &gradient, Dtype &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Vector2> &xy_local,
            const Vector2 &grad_local,
            const Dtype &distance,
            const Dtype &distance_var,
            const Dtype &occ_mean_abs,
            const Dtype &occ_abs,
            bool new_point,
            Dtype &var_position,
            Dtype &var_gradient) const;
    };

    using GpOccSurfaceMapping2Dd = GpOccSurfaceMapping2D<double>;
    using GpOccSurfaceMapping2Df = GpOccSurfaceMapping2D<float>;
}  // namespace erl::sdf_mapping

#include "gp_occ_surface_mapping_2d.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting> : erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting> : erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting::YamlConvertImpl {};
