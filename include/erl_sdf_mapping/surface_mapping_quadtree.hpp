#pragma once

#include "erl_geometry/occupancy_quadtree_base.hpp"
#include "surface_mapping_quadtree_node.hpp"

namespace erl::sdf_mapping {

    class SurfaceMappingQuadtree : public geometry::OccupancyQuadtreeBase<SurfaceMappingQuadtreeNode> {

    public:
        explicit SurfaceMappingQuadtree(double resolution)
            : OccupancyQuadtreeBase<SurfaceMappingQuadtreeNode>(resolution) {
            s_init_.EnsureLinking();
        }

        explicit SurfaceMappingQuadtree(const std::string &filename)
            : OccupancyQuadtreeBase<SurfaceMappingQuadtreeNode>(0.1) {  // resolution will be set by readBinary
            this->ReadBinary(filename);
        }

        [[nodiscard]] std::shared_ptr<AbstractQuadtree>
        Create() const override {
            return std::make_shared<SurfaceMappingQuadtree>(0.1);
        }

        [[nodiscard]] std::string
        GetTreeType() const override {
            return ERL_AS_STRING(SurfaceMappingQuadtree);
        }

        bool
        IsNodeCollapsible(const std::shared_ptr<SurfaceMappingQuadtreeNode> &node) const override {
            // all children must exist
            if (node->GetNumChildren() != 4) { return false; }

            auto first_child = this->GetNodeChild(node, 0);
            if (first_child->GetSurfaceData() != nullptr || first_child->HasAnyChild()) { return false; }

            for (unsigned int i = 1; i < 4; ++i) {
                auto child = this->GetNodeChild(node, i);
                // child should be a leaf node
                if (child->GetSurfaceData() != nullptr || child->HasAnyChild() || *child != *first_child) { return false; }
            }

            return true;
        }

    protected:
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
                auto tree = std::make_shared<SurfaceMappingQuadtree>(0.1);
                tree->ClearKeyRays();
                AbstractQuadtree::RegisterTreeType(tree);
            }

            /**
             * Dummy function to ensure that MSVC does not drop the
             * StaticMemberInitializer, causing this tree failing to register.
             * Needs to be called from the constructor of this octree.
             */
            void
            EnsureLinking(){};
        };

        inline static StaticMemberInitializer s_init_ = {};
    };

}  // namespace erl::sdf_mapping
