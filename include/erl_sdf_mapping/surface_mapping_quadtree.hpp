#pragma once

#include "erl_geometry/occupancy_quadtree_base.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "surface_mapping_quadtree_node.hpp"

namespace erl::sdf_mapping {

    class SurfaceMappingQuadtree : public geometry::OccupancyQuadtreeBase<SurfaceMappingQuadtreeNode, geometry::OccupancyQuadtreeBaseSetting> {

    public:
        using Setting = geometry::OccupancyQuadtreeBaseSetting;
        typedef geometry::OccupancyQuadtreeDrawer<SurfaceMappingQuadtree> Drawer;

        explicit SurfaceMappingQuadtree(const std::shared_ptr<Setting> &setting)
            : OccupancyQuadtreeBase<SurfaceMappingQuadtreeNode, geometry::OccupancyQuadtreeBaseSetting>(setting) {
            s_init_.EnsureLinking();
        }

        SurfaceMappingQuadtree()
            : SurfaceMappingQuadtree(std::make_shared<Setting>()) {}

        explicit SurfaceMappingQuadtree(const std::string &filename)
            : SurfaceMappingQuadtree() {  // resolution will be set by LoadData
            ERL_ASSERTM(this->LoadData(filename), "Failed to read SurfaceMappingQuadtree from file: %s", filename.c_str());
        }

        SurfaceMappingQuadtree(const SurfaceMappingQuadtree &) = delete;  // no copy constructor

        [[nodiscard]] inline std::string
        GetTreeType() const override {
            return "SurfaceMappingQuadtree";
        }

    protected:
        [[nodiscard]] inline std::shared_ptr<AbstractQuadtree>
        Create() const override {
            return std::make_shared<SurfaceMappingQuadtree>();
        }

        /**
         * Static member object which ensures that this OcTree's prototype
         * ends up in the classIDMapping only once. You need this as a
         * static member in any derived octree class in order to read .ot
         * files through the AbstractOcTree factory. You should also call
         * ensureLinking() once from the constructor.
         */
        class StaticMemberInitializer {
        public:
            StaticMemberInitializer() {
                auto tree = std::make_shared<SurfaceMappingQuadtree>();
                tree->ClearKeyRays();
                AbstractQuadtree::RegisterTreeType(tree);
            }

            /**
             * Dummy function to ensure that MSVC does not drop the
             * StaticMemberInitializer, causing this tree failing to register.
             * Needs to be called from the constructor of this octree.
             */
            void
            EnsureLinking() {}
        };

        inline static StaticMemberInitializer s_init_ = {};
    };

}  // namespace erl::sdf_mapping

namespace YAML {

    template<>
    struct convert<erl::sdf_mapping::SurfaceMappingQuadtree::Drawer::Setting>
        : public ConvertOccupancyQuadtreeDrawerSetting<erl::sdf_mapping::SurfaceMappingQuadtree::Drawer::Setting> {};
}  // namespace YAML
