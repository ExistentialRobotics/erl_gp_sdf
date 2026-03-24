#include "test_surf_mapping_3d.hpp"

#include "erl_common/plplot_fig.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

#include <open3d/io/PointCloudIO.h>

using PlplotFig = erl::common::PlplotFig;

template<typename Dtype>
using BayesianHilbertSurfaceMapping3D = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 3>;

template<typename Dtype>
struct TestBayesianHilbertSurfaceMapping3D
    : public TestSurfMapping3D<Dtype, BayesianHilbertSurfaceMapping3D<Dtype>> {

    using Super = TestSurfMapping3D<Dtype, BayesianHilbertSurfaceMapping3D<Dtype>>;
    using typename Super::Matrix3X;
    using typename Super::OptionType;
    using typename Super::Vector3;
    using typename Super::VectorX;

    using Open3dVisualizerWrapper = erl::geometry::Open3dVisualizerWrapper;
    using Octree = erl::geometry::OccupancyOctree<Dtype>;
    using OctreeDrawer = erl::geometry::OccupancyOctreeDrawer<Octree>;

    using Super::geometries;
    using Super::gradient_follow;
    using Super::gradient_whole_map;
    using Super::gt_surface_points;
    using Super::in_free_space_follow;
    using Super::in_free_space_whole_map;
    using Super::map_max;
    using Super::map_min;
    using Super::mesh_surf;
    using Super::mesh_surf_faces;
    using Super::mesh_surf_vertices;
    using Super::options;
    using Super::pcd_obs;
    using Super::pcd_surf_points;
    using Super::positions_test_follow;
    using Super::positions_test_follow_org;
    using Super::positions_test_whole_map;
    using Super::prob_occupied_follow;
    using Super::prob_occupied_whole_map;
    using Super::rotation_sensor;
    using Super::surf_map;
    using Super::surf_map_setting;
    using Super::test_dt;
    using Super::test_success;
    using Super::translation_sensor;
    using Super::vis_setting;
    using Super::visualizer;
    using Super::voxel_grid_pred;
    using Super::whole_map_xs;
    using Super::whole_map_ys;

    TestBayesianHilbertSurfaceMapping3D(int argc, char *argv[])
        : Super(argc, argv, std::make_shared<OptionType>()) {
        Super::mapping_uses_points = true;
    }

    void
    Run() override {
        Super::Run();

        this->vis_range_min = this->surf_map_setting->local_bhm->bhm->min_distance;
        this->vis_range_max = this->surf_map_setting->local_bhm->bhm->max_distance;
        const int wait_time_seconds = options->hold ? 0 : 5;

        // analyze the map after the animation ends

        VectorX log_odd_values;
        Eigen::VectorXb in_free_space;
        Matrix3X gradients;
        GetPrediction(
            gt_surface_points,
            true /*logodd*/,
            false /*compute_free_space*/,
            false /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            true /*parallel*/,
            log_odd_values,
            in_free_space,
            gradients);
        Dtype mean = log_odd_values.mean();
        Dtype squared_mean =
            log_odd_values.squaredNorm() / static_cast<Dtype>(log_odd_values.size());
        Dtype std = std::sqrt(squared_mean - mean * mean);
        Dtype min = log_odd_values.minCoeff();
        Dtype max = log_odd_values.maxCoeff();
        ERL_INFO(
            "Statistics of log-odd on ground truth surface points: mean={}, std={}, min={}, max={}",
            mean,
            std,
            min,
            max);

        PlplotFig fig(1200, 800, true);
        Eigen::VectorXd log_odd_values_d = log_odd_values.template cast<double>();
        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(1)
            .DrawHist(log_odd_values_d.data(), log_odd_values_d.size(), min, max, 50, {})
            .SetAxisLabelX("log odd value")
            .SetAxisLabelY("Frequency");
        cv::Mat img = fig.ToCvMat();
        cv::imwrite(options->output_dir / "gt_surface_points_logodd_hist.png", img);
        cv::imshow("log_odd_value histogram (gt pcd)", img);
        cv::waitKey(1000 * wait_time_seconds);

        // visualize the log-odd on the ground truth surface points
        pcd_surf_points->points_.clear();
        pcd_surf_points->colors_.clear();
        pcd_surf_points->points_.reserve(gt_surface_points.cols());
        pcd_surf_points->colors_.reserve(gt_surface_points.cols());
        const open3d::visualization::ColorMapJet color_map;
        min = -20.0f;
        max = 20.0f;
        for (long i = 0; i < gt_surface_points.cols(); ++i) {
            pcd_surf_points->points_.push_back(gt_surface_points.col(i).template cast<double>());
            const double v = std::min(std::max(log_odd_values[i], min), max);
            pcd_surf_points->colors_.push_back(color_map.GetColor((v - min) / (max - min)));
        }
        open3d::io::WritePointCloud(
            options->output_dir / "gt_surface_points_logodd.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }
        visualizer->Show(wait_time_seconds);

        // check the surf_log_odds of local BHMs
        // check the number of unused rays
        pcd_surf_points->points_.clear();
        pcd_surf_points->colors_.clear();
        std::vector<double> log_odds;
        long min_unused_ray_count = std::numeric_limits<long>::max();
        long max_unused_ray_count = std::numeric_limits<long>::min();
        std::vector<long> unused_ray_counts;
        auto &surf_data_buf = surf_map->GetSurfaceDataBuffer();
        const Dtype scaling = 1.0f / surf_map_setting->scaling;
        mean = 0;
        squared_mean = 0;
        for (const auto &[key, local_bhm]: surf_map->GetLocalBhms()) {
            // if (!local_bhm->active) { continue; }
            for (const auto &[grid_index, index]: local_bhm->surface_indices) {
                const Eigen::Vector3d point =
                    surf_data_buf[index].position.template cast<double>() * scaling;
                pcd_surf_points->points_.push_back(point);
                log_odds.push_back(local_bhm->surface_log_odds);
                mean += local_bhm->surface_log_odds;
                squared_mean += local_bhm->surface_log_odds * local_bhm->surface_log_odds;
                unused_ray_counts.push_back(local_bhm->unused_ray_count);
            }
            min_unused_ray_count = std::min(min_unused_ray_count, local_bhm->unused_ray_count);
            max_unused_ray_count = std::max(max_unused_ray_count, local_bhm->unused_ray_count);
        }

        mean /= static_cast<Dtype>(log_odds.size());
        squared_mean /= static_cast<Dtype>(log_odds.size());
        std = std::sqrt(squared_mean - mean * mean);
        // min = *std::min_element(log_odds.begin(), log_odds.end());
        // max = *std::max_element(log_odds.begin(), log_odds.end());
        ERL_INFO(
            "log odd values of local bhm surface points: mean={}, std={}, min={}, max={}",
            mean,
            std,
            *std::min_element(log_odds.begin(), log_odds.end()),
            *std::max_element(log_odds.begin(), log_odds.end()));

        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(1)
            .DrawHist(log_odds.data(), log_odds.size(), min, max, 50, {})
            .SetAxisLabelX("log odd value")
            .SetAxisLabelY("Frequency");
        img = fig.ToCvMat();
        cv::imwrite(options->output_dir / "local_bhm_surface_points_logodd_hist.png", img);
        cv::imshow("log_odd_value histogram (local bhm)", img);
        cv::waitKey(1000 * wait_time_seconds);

        pcd_surf_points->colors_.reserve(log_odds.size());
        for (auto v: log_odds) {
            v = std::min(std::max(v, static_cast<double>(min)), static_cast<double>(max));
            pcd_surf_points->colors_.push_back(color_map.GetColor((v - min) / (max - min)));
        }
        open3d::io::WritePointCloud(
            options->output_dir / "local_bhm_surface_logodd.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }
        visualizer->Show(wait_time_seconds);

        // visualize the number of unused rays of local BHMs
        ERL_INFO(
            "unused ray counts of local bhm surface points: min={}, max={}",
            min_unused_ray_count,
            max_unused_ray_count);
        pcd_surf_points->colors_.clear();
        for (const auto &v: unused_ray_counts) {
            pcd_surf_points->colors_.push_back(color_map.GetColor(
                static_cast<double>(v - min_unused_ray_count) /
                static_cast<double>(max_unused_ray_count - min_unused_ray_count)));
        }
        open3d::io::WritePointCloud(
            options->output_dir / "local_bhm_surface_unused_ray_count.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }
        visualizer->Show(wait_time_seconds);

        // visualize the local BHMs
        auto o3d_line_set = std::make_shared<open3d::geometry::LineSet>();
        const double hs = surf_map->GetClusterSize() * 0.5f;
        auto box_active = open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            open3d::geometry::AxisAlignedBoundingBox(
                Eigen::Vector3d(-hs, -hs, -hs),
                Eigen::Vector3d(hs, hs, hs)));
        auto box_inactive = std::make_shared<open3d::geometry::LineSet>(*box_active);
        auto box_empty = std::make_shared<open3d::geometry::LineSet>(*box_active);
        box_active->PaintUniformColor({0.0, 1.0, 0.0});    // green
        box_inactive->PaintUniformColor({1.0, 0.0, 0.0});  // red
        box_empty->PaintUniformColor({1.0, 0.5, 0.0});     // orange
        for (const auto &[key, local_bhm]: surf_map->GetLocalBhms()) {
            open3d::geometry::LineSet box;
            if (local_bhm->active) {
                if (local_bhm->surface_indices.empty()) {
                    box = *box_empty;
                } else {
                    box = *box_active;
                }
            } else {
                box = *box_inactive;
            }
            box.Translate(local_bhm->bhm.GetMapBoundary().center.template cast<double>());
            *o3d_line_set += box;
        }
        o3d_line_set->Scale(1.0f / surf_map_setting->scaling, Eigen::Vector3d::Zero());
        visualizer->Reset();
        visualizer->GetVisualizer()->GetRenderOption().line_width_ = 10.0f;
        visualizer->AddGeometries({geometries.front(), o3d_line_set});
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }
        visualizer->Show(wait_time_seconds);

        auto drawer_setting = std::make_shared<typename OctreeDrawer::Setting>();
        drawer_setting->scaling = 1.0f / surf_map_setting->scaling;  // inverse scaling
        drawer_setting->area_min = map_min.template cast<double>();
        drawer_setting->area_max = map_max.template cast<double>();
        drawer_setting->draw_occupied = true;
        const OctreeDrawer octree_drawer(drawer_setting, surf_map->GetTree());
        auto gt_mesh = geometries[0];
        geometries = octree_drawer.GetBlankGeometries();
        geometries.push_back(gt_mesh);
        octree_drawer.DrawLeaves(geometries);
        visualizer->Reset();
        visualizer->AddGeometries(geometries);
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }
        visualizer->Show(wait_time_seconds);

        std::filesystem::path file = options->output_dir / "surface_points.ply";
        ERL_INFO("Writing point clouds to {}", file);
        open3d::io::WritePointCloud(file, *pcd_surf_points);

        file = options->output_dir / "observed_points.ply";
        ERL_INFO("Writing point clouds to {}", file);
        open3d::io::WritePointCloud(file, *pcd_obs);

        if (surf_map_setting->update_map.method == 2) {
            std::vector<Vector3> vertices;
            std::vector<Eigen::Vector3i> faces;
            surf_map->GetMesh(false, vertices, faces);
            auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
            Super::ConvertToOpen3dMesh(mesh, vertices, faces);
            file = options->output_dir / "surface_mesh.ply";
            open3d::io::WriteTriangleMesh(file, *mesh);

            visualizer->Reset();
            visualizer->AddGeometries({mesh});
            if (!options->o3d_view_status_file.empty()) {
                visualizer->SetViewStatus(options->o3d_view_status_file);
            }
            visualizer->SetKeyboardCallback(
                [this, &mesh, &vertices, &faces](
                    const Open3dVisualizerWrapper *wrapper,
                    open3d::visualization::Visualizer *vis) -> bool {
                    auto iso_value = static_cast<Dtype>(wrapper->GetSetting()->x);
                    if (surf_map_setting->local_bhm->surface_log_odds == iso_value) {
                        return false;
                    }
                    ERL_INFO("Generating mesh with iso_value={}", iso_value);
                    surf_map_setting->local_bhm->surface_log_odds = iso_value;
                    for (auto &[key, local_bhm]: surf_map->GetLocalBhms()) {
                        local_bhm->surface_log_odds = iso_value;
                    }
                    surf_map->ResetMarchingResults();
                    surf_map->GetMesh(false, vertices, faces);
                    Super::ConvertToOpen3dMesh(mesh, vertices, faces);
                    const std::filesystem::path mesh_file =
                        options->output_dir / "surface_mesh_iso.ply";
                    open3d::io::WriteTriangleMesh(mesh_file, *mesh);
                    vis->UpdateGeometry(mesh);
                    return true;
                });
            visualizer->GetSetting()->x = surf_map_setting->local_bhm->surface_log_odds;
            ERL_INFO(
                "Press left/right arrow keys to change the iso_value and regenerate the mesh.");
            visualizer->Show(wait_time_seconds);
        }
    }

protected:
    void
    GetPrediction(
        const Matrix3X &positions,
        const bool logodd,
        const bool compute_free_space,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &pred_logodds,
        Eigen::VectorXb &pred_in_free_space,
        Matrix3X &pred_gradients) {

        const long batch_size = options->test_batch_size;
        const long n = positions.cols();

        if (n <= batch_size) {
            surf_map->Predict(
                positions,
                logodd,
                compute_free_space,
                compute_gradient,
                gradient_with_sigmoid,
                parallel,
                pred_logodds,
                pred_in_free_space,
                pred_gradients);
            return;
        }

        pred_logodds.resize(n);
        if (compute_free_space) { pred_in_free_space.resize(n); }
        if (compute_gradient) { pred_gradients.resize(3, n); }

        VectorX pred_logodds_batch;
        Eigen::VectorXb pred_in_free_space_batch;
        Matrix3X pred_gradients_batch;

        const long i_max = (n + batch_size - 1) / batch_size;
        for (long i = 0; i < n; i += batch_size) {
            const long j = std::min(i + batch_size, n);  // end index (exclusive)
            const long m = j - i;                        // actual batch size
            ERL_INFO("Batch {}/{}: {} to {}, total {}", i / batch_size, i_max, i, j - 1, n);
            surf_map->Predict(
                positions.middleCols(i, m),
                logodd,
                compute_free_space,
                compute_gradient,
                gradient_with_sigmoid,
                parallel,
                pred_logodds_batch,
                pred_in_free_space_batch,
                pred_gradients_batch);
            // store results
            pred_logodds.segment(i, m) = pred_logodds_batch.head(m);
            if (compute_free_space) {
                pred_in_free_space.segment(i, m) = pred_in_free_space_batch.head(m);
            }
            if (compute_gradient) {
                pred_gradients.middleCols(i, m) = pred_gradients_batch.leftCols(m);
            }
        }
    }

    void
    UpdateWholeMapPrediction() override {
        options->test_whole_map_z = static_cast<Dtype>(vis_setting->z);
        positions_test_whole_map.row(2).setConstant(options->test_whole_map_z);

        {
            const ERL_BLOCK_TIMER_MSG("surf_map.Test");
            GetPrediction(
                positions_test_whole_map,
                false /*logodd*/,
                true /*compute_free_space*/,
                false /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                true /*parallel*/,
                prob_occupied_whole_map,
                in_free_space_whole_map,
                gradient_whole_map);
        }
        cv::Mat prob_occupied_img = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            prob_occupied_whole_map,
            true,
            0,
            1);
        cv::Mat occupancy_img = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            (prob_occupied_whole_map.array() > 0.5f).template cast<Dtype>(),
            true,
            0,
            1);
        cv::Mat free_space_img = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            in_free_space_whole_map.template cast<Dtype>(),
            false,
            0,
            1);
        ConvertToVoxelGrid<Dtype>(prob_occupied_img, positions_test_whole_map, voxel_grid_pred);
        visualizer->GetVisualizer()->UpdateGeometry(voxel_grid_pred);

        Dtype resize_scale = options->image_resize_scale;
        resize_scale = std::min(
            resize_scale,
            static_cast<Dtype>(1920.0f) / static_cast<Dtype>(prob_occupied_img.cols));
        resize_scale = std::min(
            resize_scale,
            static_cast<Dtype>(1920.0f) / static_cast<Dtype>(prob_occupied_img.rows));

        cv::resize(prob_occupied_img, prob_occupied_img, cv::Size(), resize_scale, resize_scale);
        cv::resize(occupancy_img, occupancy_img, cv::Size(), resize_scale, resize_scale);
        cv::resize(free_space_img, free_space_img, cv::Size(), resize_scale, resize_scale);

        cv::imshow("prob_occupied", prob_occupied_img);
        cv::imshow("occupancy", occupancy_img);
        cv::imshow("free_space", free_space_img);
        cv::waitKey(1);
    }

    void
    UpdateFollowingMapPrediction() override {
        positions_test_follow =
            (rotation_sensor * positions_test_follow_org).colwise() + translation_sensor;

        {
            const ERL_BLOCK_TIMER_MSG_TIME("surf_map.Test", test_dt);
            GetPrediction(
                positions_test_follow,
                false /*logodd*/,
                true /*compute_free_space*/,
                false /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                true /*parallel*/,
                prob_occupied_follow,
                in_free_space_follow,
                gradient_follow);
            test_success = true;
        }

        if (std::find(geometries.begin(), geometries.end(), mesh_surf) != geometries.end()) {
            surf_map->GetMesh(true, mesh_surf_vertices, mesh_surf_faces);
            Super::ConvertToOpen3dMesh(mesh_surf, mesh_surf_vertices, mesh_surf_faces);
        }
    }

    void
    UpdatePredictionAtPosition() override {}

    void
    VisualizePrediction() override {
        cv::Mat prob_occupied_img = ConvertVectorToImage<Dtype>(
            options->test_follow_map_xs,
            options->test_follow_map_ys,
            prob_occupied_follow,
            true,
            0,
            1);
        ConvertToVoxelGrid<Dtype>(prob_occupied_img, positions_test_follow, voxel_grid_pred);
        cv::Mat occupancy_img = ConvertVectorToImage<Dtype>(
            options->test_follow_map_xs,
            options->test_follow_map_ys,
            (prob_occupied_follow.array() > 0.5f).template cast<Dtype>(),
            true,
            0,
            1);
        cv::Mat free_space_img = ConvertVectorToImage<Dtype>(
            options->test_follow_map_xs,
            options->test_follow_map_ys,
            in_free_space_follow.template cast<Dtype>(),
            false,
            0,
            1);

        Dtype resize_scale = options->image_resize_scale;
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(prob_occupied_img.cols));
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(prob_occupied_img.rows));

        cv::resize(prob_occupied_img, prob_occupied_img, cv::Size(), resize_scale, resize_scale);
        cv::resize(occupancy_img, occupancy_img, cv::Size(), resize_scale, resize_scale);
        cv::resize(free_space_img, free_space_img, cv::Size(), resize_scale, resize_scale);

        cv::imshow("prob_occupied", prob_occupied_img);
        cv::imshow("occupancy", occupancy_img);
        cv::imshow("free_space", free_space_img);
        cv::waitKey(1);

        Super::VisualizePrediction();
    }

    void
    TestGrid(const Matrix3X &grid_points) override {
        const ERL_BLOCK_TIMER_MSG("TestGrid");

        VectorX pred_logodds;
        Eigen::VectorXb pred_in_free_space;
        Matrix3X pred_gradients;

        GetPrediction(
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

    std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>>
    GetBuiltMesh() override {
        std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>> mesh_data;
        if (surf_map_setting->update_map.method != 2) { return mesh_data; }
        surf_map->GetMesh(false, mesh_data.first, mesh_data.second);
        return mesh_data;
    }

    std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>>
    ExtractMesh() override {
        std::pair<std::vector<Vector3>, std::vector<Eigen::Vector3i>> mesh_data;
        if (surf_map_setting->update_map.method != 2) { return mesh_data; }
        surf_map->GetMesh(options->extract_mesh_res, mesh_data.first, mesh_data.second);
        return mesh_data;
    }
};

// Update FPS:
// Replica Lidar-271: 40-70 fps (float/double)

static int g_argc = 0;
static char **g_argv = nullptr;

TEST(SurfMapping, BayesianHilbert3Dd) {
    TestBayesianHilbertSurfaceMapping3D<double> test(g_argc, g_argv);
    test.Run();
}

TEST(SurfMapping, BayesianHilbert3Df) {
    TestBayesianHilbertSurfaceMapping3D<float> test(g_argc, g_argv);
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
