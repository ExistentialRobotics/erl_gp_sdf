#include "erl_sdf_mapping/init.hpp"

#include "erl_gaussian_process/init.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_setting.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

namespace erl::sdf_mapping {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = Init();

    bool
    Init() {
        static bool initialized_ = false;

        if (initialized_) { return true; }

        if (!gaussian_process::Init()) { return false; }

        // REGISTER(SurfaceMappingQuadtreeNode);
        // REGISTER(SurfaceMappingOctreeNode);
        // REGISTER(SurfaceMappingQuadtreeD);
        // REGISTER(SurfaceMappingQuadtreeF);
        // REGISTER(SurfaceMappingOctreeD);
        // REGISTER(SurfaceMappingOctreeF);

        REGISTER(LogEdfGaussianProcessD::Setting);
        REGISTER(LogEdfGaussianProcessF::Setting);

        REGISTER(GpOccSurfaceMapping3Dd::Setting);
        REGISTER(GpOccSurfaceMapping3Df::Setting);
        REGISTER(GpOccSurfaceMapping3Dd);
        REGISTER(GpOccSurfaceMapping3Df);
        REGISTER(GpOccSurfaceMapping2Dd::Setting);
        REGISTER(GpOccSurfaceMapping2Df::Setting);
        REGISTER(GpOccSurfaceMapping2Dd);
        REGISTER(GpOccSurfaceMapping2Df);

        REGISTER(GpSdfMappingSetting3Dd);
        REGISTER(GpSdfMappingSetting3Df);
        REGISTER(GpSdfMappingSetting2Dd);
        REGISTER(GpSdfMappingSetting2Df);

        using GpOccSurfaceSdfMapping3Dd = GpSdfMapping<double, 3, GpOccSurfaceMapping3Dd>;
        using GpOccSurfaceSdfMapping3Df = GpSdfMapping<float, 3, GpOccSurfaceMapping3Df>;
        using GpOccSurfaceSdfMapping2Dd = GpSdfMapping<double, 2, GpOccSurfaceMapping2Dd>;
        using GpOccSurfaceSdfMapping2Df = GpSdfMapping<float, 2, GpOccSurfaceMapping2Df>;
        REGISTER(GpOccSurfaceSdfMapping3Dd);
        REGISTER(GpOccSurfaceSdfMapping3Df);
        REGISTER(GpOccSurfaceSdfMapping2Dd);
        REGISTER(GpOccSurfaceSdfMapping2Df);

        ERL_INFO("erl_sdf_mapping initialized");
        initialized_ = true;

        return true;
    }
}  // namespace erl::sdf_mapping
