#pragma once

#include "abstract_surface_mapping_2d.hpp"
#include "gp_occ_surface_mapping_base_setting.hpp"
#include "surface_mapping_quadtree.hpp"

#include "erl_common/yaml.hpp"
#include "erl_gaussian_process/lidar_gp_1d.hpp"

#include <memory>

namespace erl::sdf_mapping {

    class GpOccSurfaceMapping2D : public AbstractSurfaceMapping2D {
    public:
        struct Setting : common::OverrideYamlable<GpOccSurfaceMappingBaseSetting, Setting> {
            std::shared_ptr<gaussian_process::LidarGaussianProcess1D::Setting> gp_theta = std::make_shared<gaussian_process::LidarGaussianProcess1D::Setting>();
            std::shared_ptr<SurfaceMappingQuadtree::Setting> quadtree = std::make_shared<SurfaceMappingQuadtree::Setting>();  // parameters used by quadtree.
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<gaussian_process::LidarGaussianProcess1D> m_gp_theta_ = nullptr;  // the GP of regression between angle and mapped distance
        std::shared_ptr<SurfaceMappingQuadtree> m_quadtree_ = nullptr;
        Eigen::Matrix24d m_xy_perturb_ = {};
        geometry::QuadtreeKeySet m_changed_keys_ = {};

    public:
        explicit GpOccSurfaceMapping2D(const std::shared_ptr<Setting> &setting)
            : m_setting_(setting),
              m_gp_theta_(std::make_shared<gaussian_process::LidarGaussianProcess1D>(m_setting_->gp_theta)) {}

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

        std::shared_ptr<SurfaceMappingQuadtree>
        GetQuadtree() override {
            return m_quadtree_;
        }

        [[nodiscard]] double
        GetSensorNoise() const override {
            return m_setting_->gp_theta->sensor_range_var;
        }

        bool
        Update(
            const Eigen::Ref<const Eigen::VectorXd> &angles,
            const Eigen::Ref<const Eigen::VectorXd> &distances,
            const Eigen::Ref<const Eigen::Matrix23d> &pose) override;

        void
        UpdateMapPoints();

        void
        UpdateOccupancy(
            const Eigen::Ref<const Eigen::VectorXd> &angles,
            const Eigen::Ref<const Eigen::VectorXd> &distances,
            const Eigen::Ref<const Eigen::Matrix23d> &pose);

        void
        AddNewMeasurement();

    protected:
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
}  // namespace erl::sdf_mapping

// ReSharper disable CppInconsistentNaming
template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2D::Setting> {
    static Node
    encode(const erl::sdf_mapping::GpOccSurfaceMapping2D::Setting &setting) {
        Node node = convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::encode(setting);
        node["gp_theta"] = setting.gp_theta;
        node["quadtree"] = setting.quadtree;
        return node;
    }

    static bool
    decode(const Node &node, erl::sdf_mapping::GpOccSurfaceMapping2D::Setting &setting) {
        if (!convert<erl::sdf_mapping::GpOccSurfaceMappingBaseSetting>::decode(node, setting)) { return false; }
        setting.gp_theta = node["gp_theta"].as<decltype(setting.gp_theta)>();
        setting.quadtree = node["quadtree"].as<decltype(setting.quadtree)>();
        return true;
    }
};

// ReSharper restore CppInconsistentNaming
