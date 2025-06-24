#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree.hpp"
#include "erl_gp_sdf/surface_mapping_quadtree_node.hpp"

void
BindSurfaceMappingQuadtreeNode(const py::module& m) {
    using namespace erl::geometry;
    using namespace erl::gp_sdf;
    BindOccupancyQuadtreeNode<SurfaceMappingQuadtreeNode, OccupancyQuadtreeNode>(m, "SurfaceMappingQuadtreeNode")
        .def_readwrite("surface_data_index", &SurfaceMappingQuadtreeNode::surface_data_index);
}
