#pragma once

#include "surface_mapping_octree_node.hpp"

#include "erl_geometry/occupancy_octree_base.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"

namespace erl::sdf_mapping {

    class SurfaceMappingOctree : public geometry::OccupancyOctreeBase<SurfaceMappingOctreeNode, geometry::OccupancyOctreeBaseSetting> {

    public:
        using Setting = geometry::OccupancyOctreeBaseSetting;
        using Drawer = geometry::OccupancyOctreeDrawer<SurfaceMappingOctree>;

        explicit SurfaceMappingOctree(const std::shared_ptr<Setting> &setting)
            : OccupancyOctreeBase(setting) {}

        SurfaceMappingOctree()
            : SurfaceMappingOctree(std::make_shared<Setting>()) {}

        explicit SurfaceMappingOctree(const std::string &filename)
            : SurfaceMappingOctree() {  // resolution will be set by LoadData
            ERL_ASSERTM(this->LoadData(filename), "Failed to read SurfaceMappingOctree from file: {}", filename);
        }

        SurfaceMappingOctree(const SurfaceMappingOctree &) = delete;  // no copy constructor

    protected:
        [[nodiscard]] std::shared_ptr<AbstractOctree>
        Create(const std::shared_ptr<geometry::NdTreeSetting> &setting) const override {
            auto tree_setting = std::dynamic_pointer_cast<Setting>(setting);
            if (tree_setting == nullptr) { tree_setting = std::make_shared<Setting>(); }
            return std::make_shared<SurfaceMappingOctree>(tree_setting);
        }
    };

    ERL_REGISTER_OCTREE(SurfaceMappingOctree);

}  // namespace erl::sdf_mapping

template<>
struct YAML::convert<erl::sdf_mapping::SurfaceMappingOctree::Drawer::Setting>
    : public ConvertOccupancyOctreeDrawerSetting<erl::sdf_mapping::SurfaceMappingOctree::Drawer::Setting> {};  // namespace YAML
