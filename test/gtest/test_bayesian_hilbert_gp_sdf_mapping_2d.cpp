#include "test_sdf_mapping_2d.hpp"

#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

template<typename Dtype>
struct TestSdfMappingWithBayesianHilbertMap2D
    : public TestSdfMapping2D<Dtype, erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 2>> {

    TestSdfMappingWithBayesianHilbertMap2D(int argc, char **argv)
        : TestSdfMapping2D<Dtype, erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 2>>(
              argc,
              argv) {
        this->color_normal_vec = {255, 255, 255, 255};
    }

protected:
    void
    InitSceneImg() override {
        this->img_scene.setTo(cv::Scalar(128, 128, 128, 255));
        this->quadtree_drawer->DrawLeaves(this->img_scene);
        Eigen::VectorX<Dtype> prob_occupied;
        Eigen::Matrix2X<Dtype> gradients;
        this->surf_map->Predict(  //
            this->grid_points,
            false /*logodd*/,
            false /*compute gradient*/,
            false /*gradient with sigmoid*/,
            true /*parallel*/,
            prob_occupied,
            gradients);
        cv::Mat prob_occupied_img(
            this->grid_map_info->Shape(1),  // height(y)
            this->grid_map_info->Shape(0),  // width(x)
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            prob_occupied.data());
        cv::flip(prob_occupied_img, prob_occupied_img, 0);
        cv::normalize(prob_occupied_img, prob_occupied_img, 0, 255, cv::NORM_MINMAX);
        prob_occupied_img.convertTo(prob_occupied_img, CV_8UC1);
        cv::applyColorMap(prob_occupied_img, prob_occupied_img, cv::COLORMAP_JET);
        cv::cvtColor(prob_occupied_img, prob_occupied_img, cv::COLOR_BGR2BGRA);
        cv::addWeighted(prob_occupied_img, 0.7, this->img_scene, 0.3, 0.0, this->img_scene);
        this->DrawSurfaceData(this->img_scene);
    }
};

int g_argc = 0;
char **g_argv = nullptr;

TEST(GpSdfMapping, BayesianHilbert2Dd) {
    TestSdfMappingWithBayesianHilbertMap2D<double> test(g_argc, g_argv);
    test.mapping_uses_points = true;
    test.Run();
}

TEST(GpSdfMapping, BayesianHilbert2Df) {
    TestSdfMappingWithBayesianHilbertMap2D<float> test(g_argc, g_argv);
    test.mapping_uses_points = true;
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
