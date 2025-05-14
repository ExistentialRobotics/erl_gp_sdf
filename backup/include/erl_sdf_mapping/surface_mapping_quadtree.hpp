#pragma once

#include "surface_mapping_quadtree_node.hpp"

#include "erl_geometry/occupancy_quadtree_base.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype>
    class SurfaceMappingQuadtree : public geometry::OccupancyQuadtreeBase<Dtype, SurfaceMappingQuadtreeNode, geometry::OccupancyQuadtreeBaseSetting> {

    public:
        using Setting = geometry::OccupancyQuadtreeBaseSetting;
        using Super = geometry::OccupancyQuadtreeBase<Dtype, SurfaceMappingQuadtreeNode, Setting>;

        explicit SurfaceMappingQuadtree(const std::shared_ptr<Setting> &setting)
            : Super(setting) {}

        SurfaceMappingQuadtree()
            : SurfaceMappingQuadtree(std::make_shared<Setting>()) {}

        explicit SurfaceMappingQuadtree(const std::string &filename)
            : SurfaceMappingQuadtree() {  // resolution will be set by LoadData
            ERL_ASSERTM(this->LoadData(filename), "Failed to read SurfaceMappingQuadtree from file: {}", filename);
        }

        SurfaceMappingQuadtree(const SurfaceMappingQuadtree &) = default;
        SurfaceMappingQuadtree &
        operator=(const SurfaceMappingQuadtree &) = default;
        SurfaceMappingQuadtree(SurfaceMappingQuadtree &&) = default;
        SurfaceMappingQuadtree &
        operator=(SurfaceMappingQuadtree &&) = default;

    protected:
        [[nodiscard]] std::shared_ptr<geometry::AbstractQuadtree<Dtype>>
        Create(const std::shared_ptr<geometry::NdTreeSetting> &setting) const override {
            auto tree_setting = std::dynamic_pointer_cast<Setting>(setting);
            if (tree_setting == nullptr) {
                ERL_DEBUG_ASSERT(setting == nullptr, "setting is not the type for SurfaceMappingQuadtree.");
                tree_setting = std::make_shared<Setting>();
            }
            return std::make_shared<SurfaceMappingQuadtree>(tree_setting);
        }
    };

    using SurfaceMappingQuadtreeD = SurfaceMappingQuadtree<double>;
    using SurfaceMappingQuadtreeF = SurfaceMappingQuadtree<float>;
}  // namespace erl::sdf_mapping
