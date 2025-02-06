#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree.hpp"
#include "erl_sdf_mapping/surface_mapping_quadtree_node.hpp"

void
BindSurfaceMappingQuadtreeNode(const py::module& m) {
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;
    auto node = BindOccupancyQuadtreeNode<SurfaceMappingQuadtreeNode, OccupancyQuadtreeNode>(m, "SurfaceMappingQuadtreeNode");
    node.def_readwrite("surface_data_index", &SurfaceMappingQuadtreeNode::surface_data_index);
}
