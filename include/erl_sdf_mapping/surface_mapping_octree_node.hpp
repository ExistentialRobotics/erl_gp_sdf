#pragma once

#include "erl_geometry/occupancy_octree_node.hpp"

namespace erl::sdf_mapping {

    struct SurfaceMappingOctreeNode : geometry::OccupancyOctreeNode {
        std::size_t surface_data_index = -1;

        explicit SurfaceMappingOctreeNode(const uint32_t depth = 0, const int child_index = -1, const float log_odds = 0)
            : OccupancyOctreeNode(depth, child_index, log_odds) {}

        SurfaceMappingOctreeNode(const SurfaceMappingOctreeNode &other)
            : OccupancyOctreeNode(other) {
            ERL_WARN("Copy constructor called.");
            surface_data_index = other.surface_data_index;
        }

        SurfaceMappingOctreeNode &
        operator=(const SurfaceMappingOctreeNode &other) {
            ERL_WARN("Copy assignment operator called.");
            if (this != &other) {
                OccupancyOctreeNode::operator=(other);
                surface_data_index = other.surface_data_index;
            }
            return *this;
        }

        SurfaceMappingOctreeNode(SurfaceMappingOctreeNode &&other) noexcept {
            ERL_WARN("Move constructor called.");
            surface_data_index = other.surface_data_index;
            other.surface_data_index = static_cast<std::size_t>(-1);
        }

        SurfaceMappingOctreeNode &
        operator=(SurfaceMappingOctreeNode &&other) noexcept {
            ERL_WARN("Move assignment operator called.");
            if (this != &other) {
                OccupancyOctreeNode::operator=(other);
                surface_data_index = other.surface_data_index;
                other.surface_data_index = static_cast<std::size_t>(-1);
            }
            return *this;
        }

        [[nodiscard]] AbstractOctreeNode *
        Create(const uint32_t depth, const int child_index) const override {
            AbstractOctreeNode *node = new SurfaceMappingOctreeNode(depth, child_index, /*log_odds*/ 0);
            ERL_TRACY_RECORD_ALLOC(node, sizeof(SurfaceMappingOctreeNode));
            return node;
        }

        [[nodiscard]] AbstractOctreeNode *
        Clone() const override {
            AbstractOctreeNode *node = new SurfaceMappingOctreeNode(*this);
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

        // uncomment this to block merging when surface_data_index is not -1
        // then the occupancy tree cannot help to reject noise points, the surface mapping algorithm has to do it well
        // [[nodiscard]] bool
        // AllowMerge(const AbstractOctreeNode *other) const override {
        //     if (surface_data_index != static_cast<std::size_t>(-1)) { return false; }
        //     ERL_DEBUG_ASSERT(dynamic_cast<const SurfaceMappingOctreeNode *>(other) != nullptr, "other node is not SurfaceMappingOctreeNode.");
        //     if (const auto *other_node = reinterpret_cast<const SurfaceMappingOctreeNode *>(other);
        //         other_node->surface_data_index != static_cast<std::size_t>(-1)) {
        //         return false;
        //     }
        //     return OccupancyOctreeNode::AllowMerge(other);
        // }

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

}  // namespace erl::sdf_mapping
