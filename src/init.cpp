#include "erl_sdf_mapping/init.hpp"

#include "erl_gaussian_process/init.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

namespace erl::sdf_mapping {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = false;

    bool
    Init() {
        if (initialized) { return true; }

        if (!gaussian_process::Init()) { return false; }

        REGISTER(GpSdfMapping3Dd::Setting);
        REGISTER(GpSdfMapping3Df::Setting);
        REGISTER(LogEdfGaussianProcess_d::Setting);
        REGISTER(LogEdfGaussianProcess_f::Setting);
        REGISTER(GpOccSurfaceMapping3Dd::Setting);
        REGISTER(GpOccSurfaceMapping3Df::Setting);

        ERL_INFO("erl_sdf_mapping initialized");
        initialized = true;

        return true;
    }
}  // namespace erl::sdf_mapping
