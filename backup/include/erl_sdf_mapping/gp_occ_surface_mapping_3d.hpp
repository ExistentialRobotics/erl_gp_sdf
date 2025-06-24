#pragma once

#include "abstract_surface_mapping_3d.hpp"
#include "gp_occ_surface_mapping_base_setting.hpp"
#include "surface_data_manager.hpp"
#include "surface_mapping_octree.hpp"

#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"

#include <memory>

namespace erl::gp_sdf {

    template<typename Dtype>
    class GpOccSurfaceMapping3D : public AbstractSurfaceMapping3D<Dtype> {
    public:
        using Super = AbstractSurfaceMapping3D<Dtype>;
        using Key = geometry::OctreeKey;
        using SensorGp = gaussian_process::RangeSensorGaussianProcess3D<Dtype>;
        using Tree = SurfaceMappingOctree<Dtype>;
        using TreeNode = SurfaceMappingOctreeNode;
        using SurfaceDataManager3D = SurfaceDataManager<Dtype, 3>;
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using Matrix3 = Eigen::Matrix3<Dtype>;
        using Matrix3X = Eigen::Matrix3X<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Vector2 = Eigen::Vector2<Dtype>;
        using Vector3 = Eigen::Vector3<Dtype>;

        struct Setting : common::Yamlable<Setting, GpOccSurfaceMappingBaseSetting> {
            std::shared_ptr<typename SensorGp::Setting> sensor_gp = std::make_shared<typename SensorGp::Setting>();
            std::shared_ptr<typename Tree::Setting> octree = std::make_shared<typename Tree::Setting>();

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();
        inline static const std::string kFileHeader = fmt::format("# {}", type_name<GpOccSurfaceMapping3D>());

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SensorGp> m_sensor_gp_ = nullptr;  // the GP of regression between angle and mapped distance
        std::shared_ptr<Tree> m_tree_ = nullptr;
        SurfaceDataManager3D m_surface_data_manager_;
        Eigen::Matrix<Dtype, 3, 6> m_xyz_perturb_ = {};
        geometry::OctreeKeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping3D(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)),
              m_sensor_gp_(std::make_shared<SensorGp>(m_setting_->sensor_gp)) {
            const auto d = static_cast<Dtype>(m_setting_->perturb_delta);
            // clang-format off
            m_xyz_perturb_ << d, -d, 0,  0, 0,  0,
                              0,  0, d, -d, 0,  0,
                              0,  0, 0,  0, d, -d;
            // clang-format on
        }

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const SensorGp>
        GetSensorGp() const;

        geometry::OctreeKeySet
        GetChangedClusters() override;

        [[nodiscard]] unsigned int
        GetClusterLevel() const override;

        std::shared_ptr<Tree>
        GetOctree();

        [[nodiscard]] const SurfaceDataManager3D &
        GetSurfaceDataManager() const override;

        [[nodiscard]] Dtype
        GetSensorNoise() const override;

        // METHODS REQUIRED BY GpSdfMapping
        [[nodiscard]] bool
        Ready() const;

        bool
        Update(const Eigen::Ref<const Matrix3> &rotation, const Eigen::Ref<const Vector3> &translation, const Eigen::Ref<const MatrixX> &ranges) override;

        [[nodiscard]] bool
        operator==(const Super &other) const override;

        // [[nodiscard]] bool
        // Write(const std::string &filename) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        // [[nodiscard]] bool
        // Read(const std::string &filename) override;

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
        RecordChangedKey(const geometry::OctreeKey &key);

        bool
        ComputeGradient1(const Vector3 &xyz_local, Vector3 &gradient, Dtype &occ_mean, Dtype &distance_var);

        bool
        ComputeGradient2(const Vector3 &xyz_local, Vector3 &gradient, Dtype &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Vector3> &xyz_local,
            const Vector3 &grad_local,
            const Dtype &distance,
            const Dtype &distance_var,
            const Dtype &occ_mean_abs,
            const Dtype &occ_abs,
            bool new_point,
            Dtype &var_position,
            Dtype &var_gradient) const;
    };

    using GpOccSurfaceMapping3Dd = GpOccSurfaceMapping3D<double>;
    using GpOccSurfaceMapping3Df = GpOccSurfaceMapping3D<float>;
}  // namespace erl::gp_sdf

#include "gp_occ_surface_mapping_3d.tpp"

template<>
struct YAML::convert<erl::gp_sdf::GpOccSurfaceMapping3Dd::Setting> : erl::gp_sdf::GpOccSurfaceMapping3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::GpOccSurfaceMapping3Df::Setting> : erl::gp_sdf::GpOccSurfaceMapping3Df::Setting::YamlConvertImpl {};
