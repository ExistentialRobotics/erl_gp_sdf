#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree.hpp"
#include "erl_gp_sdf/surface_mapping_quadtree.hpp"

void
BindSurfaceMappingQuadtree(const py::module& m) {
    using namespace erl::gp_sdf;
    BindOccupancyQuadtree<SurfaceMappingQuadtreeD, SurfaceMappingQuadtreeNode>(m, "SurfaceMappingQuadtreeD");
    BindOccupancyQuadtree<SurfaceMappingQuadtreeF, SurfaceMappingQuadtreeNode>(m, "SurfaceMappingQuadtreeF");
}
