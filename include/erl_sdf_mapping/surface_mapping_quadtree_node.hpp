#pragma once

#include "erl_geometry/occupancy_quadtree_node.hpp"

namespace erl::sdf_mapping {

    struct SurfaceMappingQuadtreeNode : geometry::OccupancyQuadtreeNode {
        std::size_t surface_data_index = -1;

        explicit SurfaceMappingQuadtreeNode(const uint32_t depth = 0, const int child_index = -1, const float log_odds = 0)
            : OccupancyQuadtreeNode(depth, child_index, log_odds) {}

        SurfaceMappingQuadtreeNode(const SurfaceMappingQuadtreeNode &) = default;
        SurfaceMappingQuadtreeNode &
        operator=(const SurfaceMappingQuadtreeNode &other) = default;
        SurfaceMappingQuadtreeNode(SurfaceMappingQuadtreeNode &&other) noexcept = default;
        SurfaceMappingQuadtreeNode &
        operator=(SurfaceMappingQuadtreeNode &&other) noexcept = default;

        [[nodiscard]] AbstractQuadtreeNode *
        Create(const uint32_t depth, const int child_index) const override {
            AbstractQuadtreeNode *node = new SurfaceMappingQuadtreeNode(depth, child_index, /*log_odds*/ 0);
            ERL_TRACY_RECORD_ALLOC(node, sizeof(SurfaceMappingQuadtreeNode));
            return node;
        }

        [[nodiscard]] AbstractQuadtreeNode *
        Clone() const override {
            AbstractQuadtreeNode *node = new SurfaceMappingQuadtreeNode(*this);
            ERL_TRACY_RECORD_ALLOC(node, sizeof(SurfaceMappingQuadtreeNode));
            return node;
        }

        bool
        operator==(const AbstractQuadtreeNode &other) const override {
            if (!OccupancyQuadtreeNode::operator==(other)) { return false; }
            const auto *other_ptr = dynamic_cast<const SurfaceMappingQuadtreeNode *>(&other);
            if (other_ptr == nullptr) { return false; }
            return surface_data_index == other_ptr->surface_data_index;
        }

        [[nodiscard]] bool
        AllowMerge(const AbstractQuadtreeNode *other) const override {
            if (surface_data_index != static_cast<std::size_t>(-1)) { return false; }
            ERL_DEBUG_ASSERT(dynamic_cast<const SurfaceMappingQuadtreeNode *>(other) != nullptr, "other node is not SurfaceMappingOctreeNode.");
            if (const auto *other_node = reinterpret_cast<const SurfaceMappingQuadtreeNode *>(other);
                other_node->surface_data_index != static_cast<std::size_t>(-1)) {
                return false;
            }
            return OccupancyQuadtreeNode::AllowMerge(other);
        }

        [[nodiscard]] bool
        HasSurfaceData() const {
            return surface_data_index != static_cast<std::size_t>(-1);
        }

        void
        ResetSurfaceDataIndex() {
            surface_data_index = static_cast<std::size_t>(-1);
        }

        std::istream &
        ReadData(std::istream &s) override {
            OccupancyQuadtreeNode::ReadData(s);
            s.read(reinterpret_cast<char *>(&surface_data_index), sizeof(std::size_t));
            return s;
        }

        std::ostream &
        WriteData(std::ostream &s) const override {
            OccupancyQuadtreeNode::WriteData(s);
            s.write(reinterpret_cast<const char *>(&surface_data_index), sizeof(std::size_t));
            return s;
        }
    };

    ERL_REGISTER_QUADTREE_NODE(SurfaceMappingQuadtreeNode);
}  // namespace erl::sdf_mapping
