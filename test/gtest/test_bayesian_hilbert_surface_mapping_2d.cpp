#include "test_surf_mapping_2d.hpp"

#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

template<typename Dtype>
struct TestBayesianHilbertSurfaceMapping2D
    : public TestSurfMapping2D<Dtype, erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 2>> {

    using SurfMap = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 2>;
    using Super = TestSurfMapping2D<Dtype, SurfMap>;

    using typename Super::Matrix2X;
    using typename Super::OptionType;
    using typename Super::Vector2;
    using typename Super::VectorX;

    using Super::cur_traj;
    using Super::grid_map_info;
    using Super::img_canvas;
    using Super::img_dir;
    using Super::options;
    using Super::quadtree;
    using Super::quadtree_drawer;
    using Super::surf_map;
    using Super::surf_map_setting;
    using Super::surface_points_cv;
    using Super::update_map_fps;
    using Super::update_pred_fps;
    using Super::update_vis_fps;
    using Super::window_name;

    // prediction results

    Matrix2X grid_points;
    VectorX logodd_values;
    VectorX prob_occupied;
    Eigen::VectorXb in_free_space;
    Matrix2X gradients;

    cv::Mat img_tree;
    cv::Mat img_iter_cnt;
    cv::Mat img_logodd;
    cv::Mat img_prob_occ;
    cv::Mat img_grad_norms;
    cv::Mat img_final;

    cv::Mat mat_iter_cnt;
    cv::Mat mat_logodd;
    cv::Mat mat_prob_occ;
    cv::Mat mat_grad_norms;

    std::filesystem::path img_canvas_dir;
    std::filesystem::path img_tree_dir;
    std::filesystem::path img_iter_cnt_dir;
    std::filesystem::path img_logodd_dir;
    std::filesystem::path img_prob_occ_dir;
    std::filesystem::path img_grad_norms_dir;
    long img_cnt = 0;

    static constexpr int x_space = 10;
    static constexpr int y_space = 10;

    TestBayesianHilbertSurfaceMapping2D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionType> options = std::make_shared<OptionType>())
        : TestSurfMapping2D<Dtype, SurfMap>(argc, argv, std::move(options)) {
        Super::mapping_uses_points = true;
    }

protected:
    void
    PrepareVisualization() override {
        Super::PrepareVisualization();

        if (!options->visualize) { return; }

        grid_points = grid_map_info->GenerateMeterCoordinates(false /*c_stride*/);

        quadtree_drawer->DrawLeaves(img_tree);                 // BGRA
        cv::cvtColor(img_tree, img_tree, cv::COLOR_BGRA2BGR);  // BGR

        const int rows = img_tree.rows;
        const int cols = img_tree.cols;

        img_iter_cnt = cv::Mat(rows, cols, img_tree.type());
        img_logodd = cv::Mat(rows, cols, img_tree.type());
        img_prob_occ = cv::Mat(rows, cols, img_tree.type());
        img_grad_norms = cv::Mat(rows, cols, img_tree.type());

        img_canvas =
            cv::Mat(img_tree.rows * 2 + x_space, img_tree.cols * 3 + y_space * 2, img_tree.type());

        cv::imshow(window_name, img_canvas);
        cv::waitKey(1);
    }

    void
    PrepareOutputFolders() override {
        Super::PrepareOutputFolders();

        img_canvas_dir = img_dir / "canvas";
        img_tree_dir = img_dir / "tree";
        img_iter_cnt_dir = img_dir / "iter_cnt";
        img_logodd_dir = img_dir / "logodd";
        img_prob_occ_dir = img_dir / "prob_occ";
        img_grad_norms_dir = img_dir / "grad_norms";

        std::filesystem::create_directories(img_canvas_dir);
        std::filesystem::create_directories(img_tree_dir);
        std::filesystem::create_directories(img_iter_cnt_dir);
        std::filesystem::create_directories(img_logodd_dir);
        std::filesystem::create_directories(img_prob_occ_dir);
        std::filesystem::create_directories(img_grad_norms_dir);
    }

    void
    UpdatePrediction() override {
        surf_map->Predict(  //
                grid_points,
                true /*logodd*/,
                false /*compute_free_space*/,
                true /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                true /*parallel*/,
                logodd_values,
                in_free_space,
                gradients);
        prob_occupied.resize(logodd_values.size());
        for (long j = 0; j < logodd_values.size(); ++j) {
            prob_occupied[j] = erl::geometry::logodd::Probability(logodd_values[j]);
        }
    }

    void
    UpdateVisualization() override {
        img_tree.setTo(cv::Scalar(128, 128, 128, 255));
        const cv::Scalar black(0, 0, 0, 255);
        const cv::Scalar white(255, 255, 255, 255);
        const cv::Scalar red(0, 0, 255, 255);
        using namespace erl::common;

        const int rows = img_tree.rows;
        const int cols = img_tree.cols;

        // draw tree
        quadtree_drawer->DrawLeaves(img_tree);
        cv::cvtColor(img_tree, img_tree, cv::COLOR_BGRA2BGR);  // BGR
        const Eigen::Map<const Matrix2X> traj(cur_traj[0].data(), 2, cur_traj.size());
        DrawTrajectoryInplace<Dtype>(img_tree, traj, grid_map_info, black, 2, true);
        // draw sensor observation
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(img_tree, px, red, cv::MARKER_CROSS, 10, 2);
        }
        // draw fps
        cv::putText(
            img_tree,
            fmt::format("update: {:.2f}", update_map_fps),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);
        cv::putText(
            img_tree,
            fmt::format("pred: {:.2f}", update_pred_fps),
            cv::Point(10, 70),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);
        cv::putText(
            img_tree,
            fmt::format("GUI: {:.2f}", update_vis_fps),
            cv::Point(10, 110),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);

        // draw the iteration count
        Eigen::VectorXi iter_cnt = Eigen::VectorXi::Zero(prob_occupied.size());
        auto tree = quadtree.get();
        const uint32_t bhm_depth = surf_map_setting->bhm_depth;
        const auto &local_bhms = surf_map->GetLocalBhms();
        for (long j = 0; j < grid_points.cols(); ++j) {
            erl::geometry::QuadtreeKey key;
            if (!tree->CoordToKeyChecked(grid_points.col(j), bhm_depth, key)) { continue; }
            if (!local_bhms.contains(key)) { continue; }
            iter_cnt[j] = local_bhms.at(key)->bhm.GetIterationCount();
        }
        mat_iter_cnt = cv::Mat(rows, cols, CV_32SC1, iter_cnt.data());
        cv::normalize(mat_iter_cnt, mat_iter_cnt, 0, 255, cv::NORM_MINMAX);
        mat_iter_cnt.convertTo(img_iter_cnt, CV_8UC1);
        cv::applyColorMap(img_iter_cnt, img_iter_cnt, cv::COLORMAP_JET);
        cv::flip(img_iter_cnt, img_iter_cnt, 0);
        DrawTrajectoryInplace<Dtype>(img_iter_cnt, traj, grid_map_info, white, 2, true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(img_iter_cnt, px, white, cv::MARKER_CROSS, 10, 2);
        }
        for (const auto &[key, bhm]: local_bhms) {
            const auto &boundary = bhm->bhm.GetMapBoundary();
            Eigen::Vector2i px1 = grid_map_info->MeterToPixelForPoints(boundary.min());
            Eigen::Vector2i px2 = grid_map_info->MeterToPixelForPoints(boundary.max());
            const cv::Point p1(px1[0], px1[1]);
            const cv::Point p2(px2[0], px2[1]);
            cv::rectangle(img_iter_cnt, p1, p2, black, 2);
        }
        cv::putText(
            img_iter_cnt,
            "Iter Count",
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            white,
            2);

        // draw the log odds
        const int type = sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1;
        mat_logodd = cv::Mat(rows, cols, type, logodd_values.data());
        cv::normalize(mat_logodd, mat_logodd, 0, 255, cv::NORM_MINMAX);
        mat_logodd.convertTo(img_logodd, CV_8UC1);
        cv::applyColorMap(img_logodd, img_logodd, cv::COLORMAP_JET);
        cv::flip(img_logodd, img_logodd, 0);
        DrawTrajectoryInplace<Dtype>(img_logodd, traj, grid_map_info, white, 2, true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(img_logodd, px, white, cv::MARKER_CROSS, 10, 2);
        }
        cv::putText(
            img_logodd,
            "Log Odds",
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            black,
            2);

        // draw the occupancy probability
        mat_prob_occ = cv::Mat(rows, cols, type, prob_occupied.data());
        cv::normalize(mat_prob_occ, mat_prob_occ, 0, 255, cv::NORM_MINMAX);
        mat_prob_occ.convertTo(img_prob_occ, CV_8UC1);
        cv::applyColorMap(img_prob_occ, img_prob_occ, cv::COLORMAP_JET);
        cv::flip(img_prob_occ, img_prob_occ, 0);
        DrawTrajectoryInplace<Dtype>(img_prob_occ, traj, grid_map_info, white, 2, true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(img_prob_occ, px, white, cv::MARKER_CROSS, 10, 2);
        }
        cv::putText(
            img_prob_occ,
            "Prob Occupied",
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            white,
            2);

        // draw the gradient
        VectorX gradient_norms = gradients.colwise().norm();
        mat_grad_norms = cv::Mat(rows, cols, type, gradient_norms.data());
        cv::normalize(mat_grad_norms, mat_grad_norms, 0, 255, cv::NORM_MINMAX);
        mat_grad_norms.convertTo(img_grad_norms, CV_8UC1);
        cv::applyColorMap(img_grad_norms, img_grad_norms, cv::COLORMAP_JET);
        cv::flip(img_grad_norms, img_grad_norms, 0);
        DrawTrajectoryInplace<Dtype>(img_grad_norms, traj, grid_map_info, white, 2, true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(img_grad_norms, px, white, cv::MARKER_CROSS, 10, 2);
        }
        //// draw the surface normals
        const auto &surf_data_buffer = surf_map->GetSurfaceDataBuffer();
        for (auto &[key, local_bhm]: local_bhms) {
            for (auto &[grid_idx, surf_idx]: local_bhm->surface_indices) {
                const auto &surf = surf_data_buffer[surf_idx];
                Eigen::Vector2i px1 = grid_map_info->MeterToPixelForPoints(surf.position);
                Eigen::Vector2i px2 =
                    grid_map_info->MeterToPixelForPoints(surf.position + surf.normal);
                const cv::Point p1(px1[0], px1[1]);
                const cv::Point p2(px2[0], px2[1]);
                cv::arrowedLine(img_grad_norms, p1, p2, white, 2, cv::LINE_AA);
            }
        }
        cv::putText(
            img_grad_norms,
            "Grad Norms",
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            white,
            2);

        // 2 x 3 grid
        img_canvas.setTo(cv::Scalar(128, 128, 128, 255));
        img_tree.copyTo(img_canvas(cv::Rect(0, 0, cols, rows)));
        img_iter_cnt.copyTo(img_canvas(cv::Rect(cols + x_space, 0, cols, rows)));
        img_logodd.copyTo(img_canvas(cv::Rect(2 * (cols + x_space), 0, cols, rows)));
        img_prob_occ.copyTo(img_canvas(cv::Rect(0, rows + y_space, cols, rows)));
        img_grad_norms.copyTo(img_canvas(cv::Rect(cols + x_space, rows + y_space, cols, rows)));
        cv::imshow(window_name, img_canvas);
        cv::waitKey(1);

        if (options->save_images) {
            const std::string filename = fmt::format("{:04d}.png", img_cnt++);
            cv::imwrite(img_canvas_dir / filename, img_canvas);
            cv::imwrite(img_tree_dir / filename, img_tree);
            cv::imwrite(img_iter_cnt_dir / filename, img_iter_cnt);
            cv::imwrite(img_logodd_dir / filename, img_logodd);
            cv::imwrite(img_prob_occ_dir / filename, img_prob_occ);
            cv::imwrite(img_grad_norms_dir / filename, img_grad_norms);
        }
    }

    void
    ShowFinalResults() override {
        Super::ShowFinalResults();

        if (prob_occupied.size() == 0) { UpdatePrediction(); }

        const int rows = img_tree.rows;
        const int cols = img_tree.cols;
        const int type = sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1;
        img_final = cv::Mat(rows, cols, type, prob_occupied.data());
        cv::normalize(img_final, img_final, 0, 255, cv::NORM_MINMAX);
        img_final.convertTo(img_final, CV_8UC1);
        cv::applyColorMap(img_final, img_final, cv::COLORMAP_JET);
        cv::flip(img_final, img_final, 0);

        const std::string filepath = img_dir / "final_result.png";
        cv::imwrite(filepath, img_final);
        if (options->visualize) {
            cv::imshow("final_result", img_final);
            cv::waitKey(1);
        }
    }

    void
    TestGrid(const Matrix2X &grid_points) override {
        const ERL_BLOCK_TIMER_MSG("TestGrid");

        VectorX pred_logodds;
        Eigen::VectorXb pred_in_free_space;
        Matrix2X pred_gradients;

        surf_map->Predict(
            grid_points,
            true /*logodd*/,
            true /*compute_free_space*/,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            true /*parallel*/,
            pred_logodds,
            pred_in_free_space,
            pred_gradients);

        std::filesystem::path file = options->output_dir / "test_grid_points.bin";
        ERL_INFO("Saving test grid points to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, grid_points));

        file = options->output_dir / "test_grid_logodds.bin";
        ERL_INFO("Saving test grid logodds to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_logodds));

        file = options->output_dir / "test_grid_in_free_space.bin";
        ERL_INFO("Saving test grid in_free_space to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<bool>(file, pred_in_free_space));

        file = options->output_dir / "test_grid_gradients.bin";
        ERL_INFO("Saving test grid gradients to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_gradients));
    }

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    GetBuiltMesh() override {
        std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>> mesh_data;
        if (surf_map_setting->update_map.method != 2) { return mesh_data; }
        surf_map->GetMesh(false, mesh_data.first, mesh_data.second);

        const cv::Mat img_mesh = this->VisualizeMesh(mesh_data.first, mesh_data.second, img_final);
        const std::string filepath = img_dir / "built_mesh.png";
        cv::imwrite(filepath, img_mesh);

        if (options->visualize) {
            cv::imshow("built_mesh", img_mesh);
            cv::waitKey(1);
        }

        return mesh_data;
    }

    std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>>
    ExtractMesh() override {
        std::pair<std::vector<Vector2>, std::vector<Eigen::Vector2i>> mesh_data;
        if (surf_map_setting->update_map.method != 2) { return mesh_data; }
        surf_map->GetMesh(options->extract_mesh_res, mesh_data.first, mesh_data.second);

        const cv::Mat img_mesh = this->VisualizeMesh(mesh_data.first, mesh_data.second, img_final);
        const std::string filepath = img_dir / "extracted_mesh.png";
        cv::imwrite(filepath, img_mesh);

        if (options->visualize) {
            cv::imshow("extracted_mesh", img_mesh);
            cv::waitKey(1);
        }

        return mesh_data;
    }
};

// Update FPS:
// Gazebo Room: 350 fps (float)

static int g_argc = 0;
static char **g_argv = nullptr;

TEST(SurfMapping, BayesianHilbert2Dd) {
    TestBayesianHilbertSurfaceMapping2D<double> test(g_argc, g_argv);
    test.Run();
}

TEST(SurfMapping, BayesianHilbert2Df) {
    TestBayesianHilbertSurfaceMapping2D<float> test(g_argc, g_argv);
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
