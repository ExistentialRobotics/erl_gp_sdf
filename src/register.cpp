#include "erl_sdf_mapping/register.hpp"

#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"
#include "erl_sdf_mapping/log_edf_gp.hpp"

namespace erl::sdf_mapping {

#define REGISTER(x) (void) x::Register<x>()

    const bool kRegistered = []() -> bool {
        REGISTER(GpSdfMapping3Dd::Setting);
        REGISTER(GpSdfMapping3Df::Setting);
        REGISTER(LogEdfGaussianProcess_d::Setting);
        REGISTER(LogEdfGaussianProcess_f::Setting);
        REGISTER(GpOccSurfaceMapping3Dd::Setting);
        REGISTER(GpOccSurfaceMapping3Df::Setting);
        return true;
    }();
}  // namespace erl::sdf_mapping
