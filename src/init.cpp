#include "erl_gp_sdf/init.hpp"

#include "erl_gaussian_process/init.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping_setting.hpp"
#include "erl_gp_sdf/log_edf_gp.hpp"

namespace erl::gp_sdf {

#define REGISTER(x) (void) x::Register<x>()

    bool initialized = Init();

    bool
    Init() {
        static bool initialized_ = false;

        if (initialized_) { return true; }

        if (!gaussian_process::Init()) { return false; }

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

        // using GpOccSurfaceSdfMapping3Dd = GpSdfMapping<double, 3, GpOccSurfaceMapping3Dd>;
        // using GpOccSurfaceSdfMapping3Df = GpSdfMapping<float, 3, GpOccSurfaceMapping3Df>;
        // using GpOccSurfaceSdfMapping2Dd = GpSdfMapping<double, 2, GpOccSurfaceMapping2Dd>;
        // using GpOccSurfaceSdfMapping2Df = GpSdfMapping<float, 2, GpOccSurfaceMapping2Df>;
        // REGISTER(GpOccSurfaceSdfMapping3Dd);
        // REGISTER(GpOccSurfaceSdfMapping3Df);
        // REGISTER(GpOccSurfaceSdfMapping2Dd);
        // REGISTER(GpOccSurfaceSdfMapping2Df);

        REGISTER(LocalBayesianHilbertMapSettingF);
        REGISTER(LocalBayesianHilbertMapSettingD);
        REGISTER(BayesianHilbertSurfaceMapping2Df::Setting);
        REGISTER(BayesianHilbertSurfaceMapping2Dd::Setting);
        REGISTER(BayesianHilbertSurfaceMapping2Df);
        REGISTER(BayesianHilbertSurfaceMapping2Dd);
        REGISTER(BayesianHilbertSurfaceMapping3Df::Setting);
        REGISTER(BayesianHilbertSurfaceMapping3Dd::Setting);
        REGISTER(BayesianHilbertSurfaceMapping3Df);
        REGISTER(BayesianHilbertSurfaceMapping3Dd);

        // using BayesianHilbertSdfMapping2Df =
        //     GpSdfMapping<float, 2, BayesianHilbertSurfaceMapping2Df>;
        // using BayesianHilbertSdfMapping2Dd =
        //     GpSdfMapping<double, 2, BayesianHilbertSurfaceMapping2Dd>;
        // using BayesianHilbertSdfMapping3Df =
        //     GpSdfMapping<float, 3, BayesianHilbertSurfaceMapping3Df>;
        // using BayesianHilbertSdfMapping3Dd =
        //     GpSdfMapping<double, 3, BayesianHilbertSurfaceMapping3Dd>;
        // REGISTER(BayesianHilbertSdfMapping2Df);
        // REGISTER(BayesianHilbertSdfMapping2Dd);
        // REGISTER(BayesianHilbertSdfMapping3Df);
        // REGISTER(BayesianHilbertSdfMapping3Dd);

        ERL_INFO("erl_gp_sdf initialized");
        initialized_ = true;

        return true;
    }
}  // namespace erl::gp_sdf
