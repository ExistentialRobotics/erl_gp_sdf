#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_octree.hpp"
#include "erl_sdf_mapping/surface_mapping_octree_node.hpp"

void
BindSurfaceMappingOctreeNode(const py::module& m) {
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;
    BindOccupancyOctreeNode<SurfaceMappingOctreeNode, OccupancyOctreeNode>(m, "SurfaceMappingOctreeNode")
        .def_readwrite("surface_data_index", &SurfaceMappingOctreeNode::surface_data_index);
}
