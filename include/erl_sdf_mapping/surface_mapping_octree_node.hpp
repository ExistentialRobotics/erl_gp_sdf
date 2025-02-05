#pragma once

#include "erl_geometry/occupancy_octree_node.hpp"

namespace erl::sdf_mapping {

    struct SurfaceMappingOctreeNode : geometry::OccupancyOctreeNode {
        std::size_t surface_data_index = -1;

        explicit SurfaceMappingOctreeNode(const uint32_t depth = 0, const int child_index = -1, const float log_odds = 0)
            : OccupancyOctreeNode(depth, child_index, log_odds) {}

        SurfaceMappingOctreeNode(const SurfaceMappingOctreeNode &) = default;
        SurfaceMappingOctreeNode &
        operator=(const SurfaceMappingOctreeNode &other) = default;
        SurfaceMappingOctreeNode(SurfaceMappingOctreeNode &&other) noexcept = default;
        SurfaceMappingOctreeNode &
        operator=(SurfaceMappingOctreeNode &&other) noexcept = default;

        [[nodiscard]] AbstractOctreeNode *
        Create(const uint32_t depth, const int child_index) const override {
            auto node = new SurfaceMappingOctreeNode(depth, child_index, /*log_odds*/ 0);
            ERL_TRACY_RECORD_ALLOC(node, sizeof(SurfaceMappingOctreeNode));
            return node;
        }

        [[nodiscard]] AbstractOctreeNode *
        Clone() const override {
            auto node = new SurfaceMappingOctreeNode(*this);
            ERL_TRACY_RECORD_ALLOC(node, sizeof(SurfaceMappingOctreeNode));
            return node;
        }

        bool
        operator==(const AbstractOctreeNode &other) const override {
            if (!OccupancyOctreeNode::operator==(other)) { return false; }
            const auto *other_ptr = dynamic_cast<const SurfaceMappingOctreeNode *>(&other);
            if (other_ptr == nullptr) { return false; }
            return surface_data_index == other_ptr->surface_data_index;
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
            OccupancyOctreeNode::ReadData(s);
            s.read(reinterpret_cast<char *>(&surface_data_index), sizeof(std::size_t));
            return s;
        }

        std::ostream &
        WriteData(std::ostream &s) const override {
            OccupancyOctreeNode::WriteData(s);
            s.write(reinterpret_cast<const char *>(&surface_data_index), sizeof(std::size_t));
            return s;
        }
    };

    ERL_REGISTER_OCTREE_NODE(SurfaceMappingOctreeNode);
}  // namespace erl::sdf_mapping
