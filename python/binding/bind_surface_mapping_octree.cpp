#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_octree.hpp"
#include "erl_sdf_mapping/surface_mapping_octree.hpp"

void
BindSurfaceMappingOctree(const py::module& m) {
    using namespace erl::sdf_mapping;
    BindOccupancyOctree<SurfaceMappingOctreeD, SurfaceMappingOctreeNode>(m, "SurfaceMappingOctreeD");
    BindOccupancyOctree<SurfaceMappingOctreeF, SurfaceMappingOctreeNode>(m, "SurfaceMappingOctreeF");
}
