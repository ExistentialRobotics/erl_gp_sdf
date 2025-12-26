#include "test_surf_mapping_2d.hpp"

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
using GpOccSurfaceMapping = erl::gp_sdf::GpOccSurfaceMapping<Dtype, 2>;

template<typename Dtype>
struct TestGpOccSurfaceMapping2D : public TestSurfMapping2D<Dtype, GpOccSurfaceMapping<Dtype>> {
    using Super = TestSurfMapping2D<Dtype, GpOccSurfaceMapping<Dtype>>;

    using typename Super::Matrix2X;
    using typename Super::OptionType;
    using typename Super::Vector2;

    using Super::cur_traj;
    using Super::grid_map_info;
    using Super::img_canvas;
    using Super::img_dir;
    using Super::options;
    using Super::quadtree_drawer;
    using Super::surf_map;
    using Super::surf_map_setting;
    using Super::update_map_fps;
    using Super::update_pred_fps;
    using Super::update_vis_fps;
    using Super::window_name;

    long img_cnt = 0;

    TestGpOccSurfaceMapping2D(int argc, char *argv[])
        : Super(argc, argv, std::make_shared<OptionType>()) {
        Super::mapping_uses_points = false;
    }

protected:
    void
    UpdatePrediction() override {}

    void
    UpdateVisualization() override {
        if (surf_map_setting->update_occupancy) {
            quadtree_drawer->DrawLeaves(img_canvas);
        } else {
            quadtree_drawer->DrawTree(img_canvas);
        }

        const cv::Scalar red(0, 0, 255, 255);
        const cv::Scalar black(0, 0, 0, 255);

        for (auto it = surf_map->BeginSurfaceData(), end = surf_map->EndSurfaceData();  //
             it != end;
             ++it) {
            Eigen::Vector2i position_px =
                quadtree_drawer->template GetPixelCoordsForPositions<Dtype>(it->position, true);
            cv::Point position_px_cv(position_px[0], position_px[1]);
            // draw surface point
            cv::circle(img_canvas, position_px_cv, 2, red, -1);
            Eigen::Vector2i normal_px = quadtree_drawer->template GetPixelCoordsForVectors<Dtype>(
                it->normal * options->surf_normal_scale);
            cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
            // draw surface normal
            cv::arrowedLine(img_canvas, position_px_cv, arrow_end_px, red, 1, cv::LINE_8, 0, 0.1);
        }

        // draw trajectory
        Eigen::Map<const Matrix2X> traj(cur_traj[0].data(), 2, cur_traj.size());
        constexpr bool pixel_based = true;
        using namespace erl::common;
        DrawTrajectoryInplace<Dtype>(img_canvas, traj, grid_map_info, black, 2, pixel_based);

        // draw fps
        cv::putText(
            img_canvas,
            fmt::format("update: {:.2f}", update_map_fps),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);
        cv::putText(
            img_canvas,
            fmt::format("pred: {:.2f}", update_pred_fps),
            cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);
        cv::putText(
            img_canvas,
            fmt::format("GUI: {:.2f}", update_vis_fps),
            cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);

        cv::imshow(window_name, img_canvas);
        cv::imwrite(img_dir / fmt::format("{:04d}.png", img_cnt++), img_canvas);
    }

    void
    TestGrid(const Matrix2X & /*grid_points*/) override {}

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    GetBuiltMesh() override {
        return {};
    }

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    ExtractMesh() override {
        return {};
    }
};

int g_argc = 0;
char **g_argv = nullptr;

TEST(SurfMapping, GpOcc2Dd) {
    TestGpOccSurfaceMapping2D<double> test(g_argc, g_argv);
    test.Run();
}

TEST(SurfMapping, GpOcc2Df) {
    TestGpOccSurfaceMapping2D<float> test(g_argc, g_argv);
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
