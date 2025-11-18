#include "test_sdf_mapping_2d.hpp"

#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"

template<typename Dtype>
struct TestSdfMappingWithGpOccSurfaceMapping2D
    : public TestSdfMapping2D<Dtype, erl::gp_sdf::GpOccSurfaceMapping<Dtype, 2>> {

    TestSdfMappingWithGpOccSurfaceMapping2D(int argc, char **argv)
        : TestSdfMapping2D<Dtype, erl::gp_sdf::GpOccSurfaceMapping<Dtype, 2>>(argc, argv) {}

    void
    InitSceneImg() override {
        this->img_scene.setTo(cv::Scalar(128, 128, 128, 255));
        if (this->surf_map_setting->update_occupancy) {
            this->quadtree_drawer->DrawLeaves(this->img_scene);
        } else {
            this->quadtree_drawer->DrawTree(this->img_scene);
        }
        this->DrawSurfaceData(this->img_scene);
    }
};

int g_argc = 0;
char **g_argv = nullptr;

TEST(GpSdfMapping, GpOcc2Dd) {
    TestSdfMappingWithGpOccSurfaceMapping2D<double> test(g_argc, g_argv);
    test.Run();
}

TEST(GpSdfMapping, GpOcc2Df) {
    TestSdfMappingWithGpOccSurfaceMapping2D<float> test(g_argc, g_argv);
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
