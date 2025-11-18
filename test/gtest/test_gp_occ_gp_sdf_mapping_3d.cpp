#include "test_sdf_mapping_3d.hpp"

#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"

int g_argc = 0;
char **g_argv = nullptr;

TEST(GpSdfMapping, GpOcc3Dd) {
    TestSdfMapping3D<double, erl::gp_sdf::GpOccSurfaceMapping3Dd> test(g_argc, g_argv);
    test.mapping_uses_points = false;
    test.Run();
}

TEST(GpSdfMapping, GpOcc3Df) {
    TestSdfMapping3D<float, erl::gp_sdf::GpOccSurfaceMapping3Df> test(g_argc, g_argv);
    test.mapping_uses_points = false;
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
