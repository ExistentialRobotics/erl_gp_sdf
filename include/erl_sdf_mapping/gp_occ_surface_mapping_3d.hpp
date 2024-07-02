#pragma once
#include "gp_occ_surface_mapping_base_setting.hpp"

#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_geometry/abstract_surface_mapping_3d.hpp"
#include "erl_geometry/surface_mapping_octree.hpp"

#include <memory>

namespace erl::sdf_mapping {

    class GpOccSurfaceMapping3D : public geometry::AbstractSurfaceMapping3D {
    public:
        using SensorGp = gaussian_process::RangeSensorGaussianProcess3D;

        struct Setting : common::OverrideYamlable<GpOccSurfaceMappingBaseSetting, Setting> {
            std::shared_ptr<SensorGp::Setting> sensor_gp = std::make_shared<SensorGp::Setting>();
            std::shared_ptr<geometry::SurfaceMappingOctree::Setting> octree = std::make_shared<geometry::SurfaceMappingOctree::Setting>();
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<gaussian_process::RangeSensorGaussianProcess3D> m_sensor_gp_ = nullptr;  // the GP of regression between angle and mapped distance
        std::shared_ptr<geometry::SurfaceMappingOctree> m_octree_ = nullptr;
        Eigen::Matrix<double, 3, 6> m_xyz_perturb_ = {};
        geometry::OctreeKeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping3D(std::shared_ptr<Setting> setting)
            : m_setting_(std::move(setting)),
              m_sensor_gp_(std::make_shared<gaussian_process::RangeSensorGaussianProcess3D>(m_setting_->sensor_gp)) {}

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const {
            return m_setting_;
        }

        geometry::OctreeKeySet
        GetChangedClusters() override {
            return m_changed_keys_;
        }

        [[nodiscard]] unsigned int
        GetClusterLevel() const override {
            return m_setting_->cluster_level;
        }

        std::shared_ptr<geometry::SurfaceMappingOctree>
        GetOctree() override {
            return m_octree_;
        }

        std::shared_ptr<gaussian_process::RangeSensorGaussianProcess3D>
        GetSensorGp() {
            return m_sensor_gp_;
        }

        [[nodiscard]] double
        GetSensorNoise() const override {
            return m_setting_->sensor_gp->sensor_range_var;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::Matrix3d> &rotation,
            const Eigen::Ref<const Eigen::Vector3d> &translation,
            const Eigen::Ref<const Eigen::MatrixXd> &ranges) override;

        void
        UpdateMapPoints();

        void
        UpdateOccupancy();

        void
        AddNewMeasurement();

    protected:
        void
        RecordChangedKey(const geometry::OctreeKey &key) {
            ERL_DEBUG_ASSERT(m_octree_ != nullptr, "octree is nullptr.");
            m_changed_keys_.insert(m_octree_->AdjustKeyToDepth(key, m_octree_->GetTreeDepth() - m_setting_->cluster_level));
        }

        bool
        ComputeGradient1(const Eigen::Vector3d &xyz_local, Eigen::Vector3d &gradient, double &occ_mean, double &distance_var);

        bool
        ComputeGradient2(const Eigen::Vector3d &xyz_local, Eigen::Vector3d &gradient, double &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Eigen::Vector3d> &xyz_local,
            const Eigen::Vector3d &grad_local,
            const double &distance,
            const double &distance_var,
            const double &occ_mean_abs,
            const double &occ_abs,
            bool new_point,
            double &var_position,
            double &var_gradient) const;
    };
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping3D::Setting> {
    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMapping3D::Setting &rhs) {
        Node node = convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::encode(rhs);
        node["sensor_gp"] = rhs.sensor_gp;
        node["octree"] = rhs.octree;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMapping3D::Setting &rhs) {
        if (!convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::decode(node, rhs)) { return false; }
        rhs.sensor_gp = node["sensor_gp"].as<std::shared_ptr<erl::gaussian_process::RangeSensorGaussianProcess3D::Setting>>();
        rhs.octree = node["octree"].as<std::shared_ptr<erl::geometry::SurfaceMappingOctree::Setting>>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
