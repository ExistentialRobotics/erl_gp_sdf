#include "erl_sdf_mapping/init.hpp"

#include "erl_gaussian_process/init.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
// #include "erl_sdf_mapping/gp_occ_surface_mapping_2d.hpp"
// #include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"
// #include "erl_sdf_mapping/gp_sdf_mapping.hpp"
// #include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

namespace erl::sdf_mapping {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = false;

    bool
    Init() {
        if (initialized) { return true; }

        if (!gaussian_process::Init()) { return false; }

        // REGISTER(GpSdfMapping3Dd::Setting);
        // REGISTER(GpSdfMapping3Df::Setting);
        REGISTER(LogEdfGaussianProcessD::Setting);
        REGISTER(LogEdfGaussianProcessF::Setting);
        REGISTER(GpOccSurfaceMapping3Dd::Setting);
        REGISTER(GpOccSurfaceMapping3Df::Setting);
        REGISTER(GpOccSurfaceMapping3Dd);
        REGISTER(GpOccSurfaceMapping3Df);
        // REGISTER(GpOccSurfaceMapping2Dd::Setting);
        // REGISTER(GpOccSurfaceMapping2Df::Setting);
        // REGISTER(GpOccSurfaceMapping2Dd);
        // REGISTER(GpOccSurfaceMapping2Df);

        REGISTER(SurfaceMappingQuadtreeNode);
        REGISTER(SurfaceMappingOctreeNode);
        REGISTER(SurfaceMappingOctreeD);
        REGISTER(SurfaceMappingOctreeF);

        ERL_INFO("erl_sdf_mapping initialized");
        initialized = true;

        return true;
    }
}  // namespace erl::sdf_mapping
