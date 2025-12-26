#include "test_surf_mapping_3d.hpp"

#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"

// expected performance (Intel i9-14900K):
// - Cow and Lady, Depth: 30 to 60 fps (float) / 25 to 50 fps (double)
// - Newer College, LiDAR: 20 to 30 fps (float)
// - Replica Hotel, LiDAR, 360: 50 to 100 fps
// - Replica Hotel, Depth Camera: 40 to 60 fps (float) / 30 to 50 fps (double)
// - Gazebo Room: 300 to 500 fps (2D)
// - House Expo: 1000 to 1400 fps (2D)
// - UCSD FAH: 400 to 800 fps (2D)

template<typename Dtype>
using GpOccSurfaceMapping3D = erl::gp_sdf::GpOccSurfaceMapping<Dtype, 3>;

template<typename Dtype>
struct TestGpOccSurfaceMapping3D : public TestSurfMapping3D<Dtype, GpOccSurfaceMapping3D<Dtype>> {

    using Super = TestSurfMapping3D<Dtype, GpOccSurfaceMapping3D<Dtype>>;

    using typename Super::Matrix3X;
    using typename Super::OptionType;
    using typename Super::Vector3;

    TestGpOccSurfaceMapping3D(int argc, char *argv[])
        : Super(argc, argv, std::make_shared<OptionType>()) {}

protected:
    void
    UpdateWholeMapPrediction() override {}

    void
    UpdateFollowingMapPrediction() override {}

    void
    UpdatePredictionAtPosition() override {}

    void
    TestGrid(const Matrix3X & /*grid_points*/) override {}

    std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>>
    GetBuiltMesh() override {
        return {};
    }

    std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>>
    ExtractMesh() override {
        return {};
    }
};

int g_argc = 0;
char **g_argv = nullptr;

TEST(SurfMapping, GpOcc3Dd) {
    TestGpOccSurfaceMapping3D<double> test(g_argc, g_argv);
    test.Run();
}

TEST(SurfMapping, GpOcc3Df) {
    TestGpOccSurfaceMapping3D<float> test(g_argc, g_argv);
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
