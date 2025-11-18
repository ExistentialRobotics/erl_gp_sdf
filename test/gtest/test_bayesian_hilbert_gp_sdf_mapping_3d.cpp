#include "test_sdf_mapping_3d.hpp"

#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype>
using BayesianHilbertSurfaceMapping3D = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 3>;

TEST(GpSdfMapping, BayesianHilbert3Dd) {
    TestSdfMapping3D<double, BayesianHilbertSurfaceMapping3D<double>> test(g_argc, g_argv);
    test.mapping_uses_points = true;
    test.Run();
}

TEST(GpSdfMapping, BayesianHilbert3Df) {
    ERL_INFO("logging level: {}", static_cast<int>(erl::common::Logging::GetLevel()));
    TestSdfMapping3D<float, BayesianHilbertSurfaceMapping3D<float>> test(g_argc, g_argv);
    test.mapping_uses_points = true;
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    erl::common::SetGlobalRandomSeed(0);
    return RUN_ALL_TESTS();
}
