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
    using Super::test_output_folder;
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

        // analyze the map after the animation ends

        VectorX log_odd_values;
        Eigen::VectorXb in_free_space;
        Matrix3X gradients;
        surf_map->Predict(
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
        cv::imwrite(test_output_folder / "gt_surface_points_logodd_hist.png", img);
        cv::imshow("log_odd_value histogram (gt pcd)", img);
        cv::waitKey(0);

        // visualize the log-odd on the ground truth surface points
        pcd_surf_points->points_.clear();
        pcd_surf_points->colors_.clear();
        pcd_surf_points->points_.reserve(gt_surface_points.cols());
        pcd_surf_points->colors_.reserve(gt_surface_points.cols());
        open3d::visualization::ColorMapJet color_map;
        min = -30.0f;
        max = 30.0f;
        for (long i = 0; i < gt_surface_points.cols(); ++i) {
            pcd_surf_points->points_.push_back(gt_surface_points.col(i).template cast<double>());
            double v = std::min(std::max(log_odd_values[i], min), max);
            pcd_surf_points->colors_.push_back(color_map.GetColor((v - min) / (max - min)));
        }
        open3d::io::WritePointCloud(
            test_output_folder / "gt_surface_points_logodd.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        visualizer->SetViewStatus(options->o3d_view_status_file);
        visualizer->Show();

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
            if (!local_bhm->active) { continue; }
            for (const auto &[grid_index, index]: local_bhm->surface_indices) {
                Eigen::Vector3d point =
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
        min = *std::min_element(log_odds.begin(), log_odds.end());
        max = *std::max_element(log_odds.begin(), log_odds.end());
        ERL_INFO(
            "log odd values of local bhm surface points: mean={}, std={}, min={}, max={}",
            mean,
            std,
            min,
            max);

        fig.Clear()
            .SetMargin(0.15, 0.85, 0.15, 0.85)
            .SetCurrentColor(PlplotFig::Color0::Black)
            .SetPenWidth(1)
            .DrawHist(log_odds.data(), log_odds.size(), min, max, 50, {})
            .SetAxisLabelX("log odd value")
            .SetAxisLabelY("Frequency");
        img = fig.ToCvMat();
        cv::imwrite(test_output_folder / "local_bhm_surface_points_logodd_hist.png", img);
        cv::imshow("log_odd_value histogram (local bhm)", img);
        cv::waitKey(0);

        pcd_surf_points->colors_.reserve(log_odds.size());
        for (const auto &v: log_odds) {
            pcd_surf_points->colors_.push_back(color_map.GetColor((v - min) / (max - min)));
        }
        open3d::io::WritePointCloud(
            test_output_folder / "local_bhm_surface_logodd.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        visualizer->SetViewStatus(options->o3d_view_status_file);
        visualizer->Show();

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
            test_output_folder / "local_bhm_surface_unused_ray_count.ply",
            *pcd_surf_points);
        visualizer->Reset();
        visualizer->AddGeometries({pcd_surf_points});
        visualizer->SetViewStatus(options->o3d_view_status_file);
        visualizer->Show();

        // visualize the local BHMs
        auto o3d_line_set = std::make_shared<open3d::geometry::LineSet>();
        const double hs = surf_map->GetClusterSize() * 0.5f;
        auto box_active = open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            open3d::geometry::AxisAlignedBoundingBox(
                Eigen::Vector3d(-hs, -hs, -hs),
                Eigen::Vector3d(hs, hs, hs)));
        auto box_inactive = std::make_shared<open3d::geometry::LineSet>(*box_active);
        box_active->PaintUniformColor({0.0, 1.0, 0.0});
        box_inactive->PaintUniformColor({1.0, 0.0, 0.0});
        for (const auto &[key, local_bhm]: surf_map->GetLocalBhms()) {
            open3d::geometry::LineSet box = local_bhm->active ? *box_active : *box_inactive;
            box.Translate(local_bhm->bhm.GetMapBoundary().center.template cast<double>());
            *o3d_line_set += box;
        }
        o3d_line_set->Scale(1.0f / surf_map_setting->scaling, Eigen::Vector3d::Zero());
        visualizer->Reset();
        visualizer->GetVisualizer()->GetRenderOption().line_width_ = 10.0f;
        visualizer->AddGeometries({geometries.front(), o3d_line_set});
        visualizer->SetViewStatus(options->o3d_view_status_file);
        visualizer->Show();

        auto drawer_setting = std::make_shared<typename OctreeDrawer::Setting>();
        drawer_setting->scaling = 1.0f / surf_map_setting->scaling;  // inverse scaling
        drawer_setting->area_min = map_min.template cast<double>();
        drawer_setting->area_max = map_max.template cast<double>();
        drawer_setting->occupied_only = true;
        OctreeDrawer octree_drawer(drawer_setting, surf_map->GetTree());
        auto gt_mesh = geometries[0];
        geometries = octree_drawer.GetBlankGeometries();
        geometries.push_back(gt_mesh);
        octree_drawer.DrawLeaves(geometries);
        visualizer->Reset();
        visualizer->AddGeometries(geometries);
        visualizer->SetViewStatus(options->o3d_view_status_file);
        visualizer->Show();

        ERL_INFO("Writing point clouds to {}", test_output_folder);
        open3d::io::WritePointCloud(test_output_folder / "surface_points.ply", *pcd_surf_points);
        open3d::io::WritePointCloud(test_output_folder / "observed_points.ply", *pcd_obs);

        if (surf_map_setting->update_map.method == 2) {
            std::vector<Vector3> vertices;
            std::vector<Eigen::Vector3i> faces;
            surf_map->GetMesh(vertices, faces);
            auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
            mesh->vertices_.reserve(vertices.size());
            mesh->triangles_.reserve(faces.size());
            for (const auto &v: vertices) { mesh->vertices_.emplace_back(v.x(), v.y(), v.z()); }
            for (const auto &f: faces) { mesh->triangles_.emplace_back(f.x(), f.y(), f.z()); }
            mesh->ComputeVertexNormals();
            open3d::io::WriteTriangleMesh(test_output_folder / "surface_mesh.ply", *mesh);

            visualizer->Reset();
            visualizer->AddGeometries({mesh});
            visualizer->SetViewStatus(options->o3d_view_status_file);
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
                    surf_map->GetMesh(vertices, faces);
                    mesh->vertices_.clear();
                    mesh->triangles_.clear();
                    mesh->vertices_.reserve(vertices.size());
                    mesh->triangles_.reserve(faces.size());
                    for (const auto &v: vertices) {
                        mesh->vertices_.emplace_back(v.x(), v.y(), v.z());
                    }
                    for (const auto &f: faces) {
                        mesh->triangles_.emplace_back(f.x(), f.y(), f.z());
                    }
                    mesh->ComputeVertexNormals();
                    open3d::io::WriteTriangleMesh(test_output_folder / "surface_mesh.ply", *mesh);
                    vis->UpdateGeometry(mesh);
                    return true;
                });
            visualizer->GetSetting()->x = surf_map_setting->local_bhm->surface_log_odds;
            ERL_INFO(
                "Press left/right arrow keys to change the iso_value and regenerate the mesh.");
            visualizer->Show();
        }
    }

protected:
    void
    UpdateWholeMapPrediction() override {
        options->test_z = static_cast<Dtype>(vis_setting->z);
        positions_test_whole_map.row(2).setConstant(options->test_z);

        {
            ERL_BLOCK_TIMER_MSG("surf_map.Test");
            const double scaling = surf_map_setting->scaling;
            surf_map->Predict(
                positions_test_whole_map * scaling,
                false /*logodd*/,
                true /*compute_free_space*/,
                false /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                true /*parallel*/,
                prob_occupied_whole_map,
                in_free_space_whole_map,
                gradient_whole_map);
        }
        const cv::Mat prob_occupied_img =
            ConvertVectorToImage(whole_map_xs, whole_map_ys, prob_occupied_whole_map, true);
        const cv::Mat occupancy_img = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            (prob_occupied_whole_map.array() > 0.5f).template cast<Dtype>(),
            true);
        const cv::Mat free_space_img = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            in_free_space_whole_map.template cast<Dtype>(),
            false);
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

        ERL_BLOCK_TIMER_MSG_TIME("surf_map.Test", test_dt);
        const double scaling = surf_map_setting->scaling;
        surf_map->Predict(
            positions_test_follow * scaling,
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

    void
    UpdatePredictionAtPosition() override {}

    void
    VisualizePrediction() override {
        cv::Mat prob_occupied_img =
            ConvertVectorToImage(options->test_xs, options->test_ys, prob_occupied_follow, true);
        ConvertToVoxelGrid<Dtype>(prob_occupied_img, positions_test_follow, voxel_grid_pred);
        cv::Mat occupancy_img = ConvertVectorToImage<Dtype>(
            options->test_xs,
            options->test_ys,
            (prob_occupied_follow.array() > 0.5f).template cast<Dtype>(),
            true);
        cv::Mat free_space_img = ConvertVectorToImage<Dtype>(
            options->test_xs,
            options->test_ys,
            in_free_space_follow.template cast<Dtype>(),
            false);

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
};

// Update FPS:
// Replica Lidar-271: 40-70 fps (float/double)

int g_argc = 0;
char **g_argv = nullptr;

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
