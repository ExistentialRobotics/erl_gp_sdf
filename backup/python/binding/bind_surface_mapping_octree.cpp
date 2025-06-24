#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_octree.hpp"
#include "erl_gp_sdf/surface_mapping_octree.hpp"

void
BindSurfaceMappingOctree(const py::module& m) {
    using namespace erl::gp_sdf;
    BindOccupancyOctree<SurfaceMappingOctreeD, SurfaceMappingOctreeNode>(m, "SurfaceMappingOctreeD");
    BindOccupancyOctree<SurfaceMappingOctreeF, SurfaceMappingOctreeNode>(m, "SurfaceMappingOctreeF");
}
