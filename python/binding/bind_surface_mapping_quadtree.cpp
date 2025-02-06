#include "erl_common/pybind11.hpp"
#include "erl_geometry/pybind11_occupancy_quadtree.hpp"
#include "erl_sdf_mapping/surface_mapping_quadtree.hpp"

void
BindSurfaceMappingQuadtree(const py::module& m) {
    using namespace erl::sdf_mapping;
    BindOccupancyQuadtree<SurfaceMappingQuadtree, SurfaceMappingQuadtreeNode>(m, "SurfaceMappingQuadtree");
}
