#pragma once

#include "gp_occ_surface_mapping_base_setting.hpp"

#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_geometry/abstract_surface_mapping_2d.hpp"
#include "erl_geometry/surface_mapping_quadtree.hpp"

#include <memory>

namespace erl::sdf_mapping {

    class GpOccSurfaceMapping2D : public geometry::AbstractSurfaceMapping2D {
    public:
        using SensorGp = gaussian_process::LidarGaussianProcess2D;

        struct Setting : common::Yamlable<Setting, GpOccSurfaceMappingBaseSetting> {
            std::shared_ptr<SensorGp::Setting> sensor_gp = std::make_shared<SensorGp::Setting>();
            std::shared_ptr<geometry::SurfaceMappingQuadtree::Setting> quadtree = std::make_shared<geometry::SurfaceMappingQuadtree::Setting>();
        };

        inline static const volatile bool kSettingRegistered = common::YamlableBase::Register<Setting>();

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<gaussian_process::LidarGaussianProcess2D> m_sensor_gp_ = nullptr;  // the GP of regression between angle and mapped distance
        std::shared_ptr<geometry::SurfaceMappingQuadtree> m_quadtree_ = nullptr;
        Eigen::Matrix24d m_xy_perturb_ = {};
        geometry::QuadtreeKeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping2D(const std::shared_ptr<Setting> &setting)
            : m_setting_(setting),
              m_sensor_gp_(std::make_shared<gaussian_process::LidarGaussianProcess2D>(m_setting_->sensor_gp)) {}

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const {
            return m_setting_;
        }

        geometry::QuadtreeKeySet
        GetChangedClusters() override {
            return m_changed_keys_;
        }

        [[nodiscard]] unsigned int
        GetClusterLevel() const override {
            return m_setting_->cluster_level;
        }

        std::shared_ptr<geometry::SurfaceMappingQuadtree>
        GetQuadtree() override {
            return m_quadtree_;
        }

        [[nodiscard]] double
        GetSensorNoise() const override {
            return m_setting_->sensor_gp->sensor_range_var;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::Matrix2d> &rotation,
            const Eigen::Ref<const Eigen::Vector2d> &translation,
            const Eigen::Ref<const Eigen::MatrixXd> &ranges) override;

        [[nodiscard]] bool
        operator==(const AbstractSurfaceMapping2D &other) const override;

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
        RecordChangedKey(const geometry::QuadtreeKey &key) {
            ERL_DEBUG_ASSERT(m_quadtree_ != nullptr, "Quadtree is not initialized.");
            ERL_DEBUG_ASSERT(m_setting_ != nullptr, "Setting is not initialized.");
            m_changed_keys_.insert(m_quadtree_->AdjustKeyToDepth(key, m_quadtree_->GetTreeDepth() - m_setting_->cluster_level));
        }

        bool
        ComputeGradient1(const Eigen::Vector2d &xy_local, Eigen::Vector2d &gradient, double &occ_mean, double &distance_var);

        bool
        ComputeGradient2(const Eigen::Ref<const Eigen::Vector2d> &xy_local, Eigen::Vector2d &gradient, double &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Eigen::Vector2d> &xy_local,
            const Eigen::Vector2d &grad_local,
            const double &distance,
            const double &distance_var,
            const double &occ_mean_abs,
            const double &occ_abs,
            bool new_point,
            double &var_position,
            double &var_gradient) const;
    };

    ERL_REGISTER_SURFACE_MAPPING(GpOccSurfaceMapping2D);
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting> {
    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMapping2D::Setting &setting) {
        Node node = convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::encode(setting);
        node["sensor_gp"] = setting.sensor_gp;
        node["quadtree"] = setting.quadtree;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMapping2D::Setting &setting) {
        if (!convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::decode(node, setting)) { return false; }
        setting.sensor_gp = node["sensor_gp"].as<decltype(setting.sensor_gp)>();
        setting.quadtree = node["quadtree"].as<decltype(setting.quadtree)>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
