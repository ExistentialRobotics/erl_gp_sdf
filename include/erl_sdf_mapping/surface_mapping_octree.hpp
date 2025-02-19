#pragma once

#include "surface_mapping_octree_node.hpp"

#include "erl_geometry/occupancy_octree_base.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    class SurfaceMappingOctree : public geometry::OccupancyOctreeBase<Dtype, SurfaceMappingOctreeNode, geometry::OccupancyOctreeBaseSetting> {

    public:
        using Setting = geometry::OccupancyOctreeBaseSetting;
        using Super = geometry::OccupancyOctreeBase<Dtype, SurfaceMappingOctreeNode, Setting>;
        using Drawer = geometry::OccupancyOctreeDrawer<SurfaceMappingOctree>;

        explicit SurfaceMappingOctree(const std::shared_ptr<Setting> &setting)
            : Super(setting) {}

        SurfaceMappingOctree()
            : SurfaceMappingOctree(std::make_shared<Setting>()) {}

        explicit SurfaceMappingOctree(const std::string &filename)
            : SurfaceMappingOctree() {  // resolution will be set by LoadData
            ERL_ASSERTM(this->LoadData(filename), "Failed to read SurfaceMappingOctree from file: {}", filename);
        }

        SurfaceMappingOctree(const SurfaceMappingOctree &) = default;
        SurfaceMappingOctree &
        operator=(const SurfaceMappingOctree &) = default;
        SurfaceMappingOctree(SurfaceMappingOctree &&) = default;
        SurfaceMappingOctree &
        operator=(SurfaceMappingOctree &&) = default;

    protected:
        [[nodiscard]] std::shared_ptr<geometry::AbstractOctree<Dtype>>
        Create(const std::shared_ptr<geometry::NdTreeSetting> &setting) const override {
            auto tree_setting = std::dynamic_pointer_cast<Setting>(setting);
            if (tree_setting == nullptr) {
                ERL_DEBUG_ASSERT(setting == nullptr, "setting is not the type for SurfaceMappingOctree.");
                tree_setting = std::make_shared<Setting>();
            }
            return std::make_shared<SurfaceMappingOctree>(tree_setting);
        }
    };

    using SurfaceMappingOctree_d = SurfaceMappingOctree<double>;
    using SurfaceMappingOctree_f = SurfaceMappingOctree<float>;

    ERL_REGISTER_OCTREE(SurfaceMappingOctree_d);
    ERL_REGISTER_OCTREE(SurfaceMappingOctree_f);

}  // namespace erl::sdf_mapping

template<>
struct YAML::convert<erl::sdf_mapping::SurfaceMappingOctree_d::Drawer::Setting> : erl::sdf_mapping::SurfaceMappingOctree_d::Drawer::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::SurfaceMappingOctree_f::Drawer::Setting> : erl::sdf_mapping::SurfaceMappingOctree_f::Drawer::Setting::YamlConvertImpl {};
