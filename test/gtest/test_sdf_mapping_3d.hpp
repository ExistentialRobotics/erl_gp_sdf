#pragma once

#include "test_mapping_3d.hpp"

#include "erl_gp_sdf/gp_sdf_mapping.hpp"

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>, OptionsForTestMapping3D<Dtype>> {

    using Super = OptionsForTestMapping3D<Dtype>;

    std::string surf_map_config_file;
    std::string sdf_map_config_file;

    ERL_REFLECT_SCHEMA(
        Options,
        ERL_REFLECT_MEMBER(Options, surf_map_config_file),
        ERL_REFLECT_MEMBER(Options, sdf_map_config_file));

    bool
    PostDeserialization() override {
        if (!Super::PostDeserialization()) { return false; }
        ERL_ASSERTM(
            !surf_map_config_file.empty(),
            "Please provide the surface mapping config file via --surf_map_config_file");
        ERL_ASSERTM(
            !sdf_map_config_file.empty(),
            "Please provide the SDF mapping config file via --sdf_map_config_file");
        return true;
    }
};

template<typename Dtype, typename SurfaceMappingType>
struct TestSdfMapping3D : public TestMapping3D<Dtype, erl::gp_sdf::GpSdfMapping<Dtype, 3>> {

    using SurfaceMapping = SurfaceMappingType;
    using SdfMapping = erl::gp_sdf::GpSdfMapping<Dtype, 3>;
    using Super = TestMapping3D<Dtype, SdfMapping>;
    using SurfaceMappingSetting = typename SurfaceMapping::Setting;
    using SdfMappingSetting = typename SdfMapping::Setting;
    using OptionType = Options<Dtype>;

    // bring in base class types
    using typename Super::Matrix3;
    using typename Super::Matrix3X;
    using typename Super::Matrix4;
    using typename Super::Matrix4X;
    using typename Super::Matrix6X;
    using typename Super::MatrixX;
    using typename Super::Vector3;
    using typename Super::VectorX;

    // bring in base class members

    using Super::cluster_edge_index_map;
    using Super::cluster_half_size;
    using Super::cluster_indices;
    using Super::cluster_vertex_index_map;
    using Super::fps_data;
    using Super::frame_idx;
    using Super::frame_points;
    using Super::frame_ranges;
    using Super::geometries;
    using Super::gui_dt;
    using Super::img_dir;
    using Super::inactive_cluster_keys;
    using Super::line_set_cluster_box;
    using Super::line_set_clusters;
    using Super::line_set_clusters_map;
    using Super::line_set_surf_normals;
    using Super::line_set_traj;
    using Super::map_max;
    using Super::map_min;
    using Super::mapping;
    using Super::mapping_uses_points;
    using Super::max_wp_idx;
    using Super::mesh_sensor;
    using Super::mesh_surf;
    using Super::mesh_surf_faces;
    using Super::mesh_surf_vertices;
    using Super::pcd_cluster_samples;
    using Super::pcd_obs;
    using Super::pcd_surf_points;
    using Super::position_test;
    using Super::positions_test_follow;
    using Super::positions_test_follow_org;
    using Super::positions_test_whole_map;
    using Super::range_sensor_frame;
    using Super::ranges_img;
    using Super::ranges_img_texts;
    using Super::rotation_frame;
    using Super::rotation_sensor;
    using Super::scaling;
    using Super::surf_data_buffer;
    using Super::test_output_folder;
    using Super::translation_frame;
    using Super::translation_sensor;
    using Super::unused_surf_data_indices;
    using Super::vis_setting;
    using Super::visualizer;
    using Super::voxel_grid_pred;
    using Super::whole_map_xs;
    using Super::whole_map_ys;
    using Super::wp_idx;

    // bring in base class methods

    using Super::LoadData;

    std::shared_ptr<OptionType> options = std::make_shared<OptionType>();

    std::shared_ptr<SurfaceMappingSetting> surf_map_setting = nullptr;
    std::shared_ptr<SdfMappingSetting> sdf_map_setting = nullptr;
    std::shared_ptr<SurfaceMapping> surf_map = nullptr;
    std::shared_ptr<SdfMapping> sdf_map = nullptr;

    // open3d visualization
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sdf_sphere = nullptr;
    bool surf_map_supports_mesh = true;

    // test data
    VectorX sdf_pred_follow;
    Matrix3X gradients_follow;
    Matrix4X variances_follow;
    Matrix6X covairances_follow;

    VectorX sdf_pred_whole_map;
    Matrix3X gradients_whole_map;
    Matrix4X variances_whole_map;
    Matrix6X covairances_whole_map;

    // logging

    bool surf_map_updated = false;
    bool sdf_map_updated = false;
    bool test_success = false;
    double surf_map_update_dt = 0;
    double sdf_map_update_dt = 0;
    double test_dt = 0;
    double surf_map_update_fps = 0;
    double sdf_map_update_fps = 0;
    double test_fps = 0;
    double gui_fps = 0;
    absl::flat_hash_map<uint64_t, long> gp_index_map;

    TestSdfMapping3D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionType> options = std::make_shared<OptionType>())
        : Super(argc, argv, options), options(options) {}

protected:
    void
    Init() override {
        // load settings
        surf_map_setting = std::make_shared<SurfaceMappingSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options->surf_map_config_file),
            "Failed to load surf_map_config_file: {}",
            options->surf_map_config_file);
        surf_map_setting->AsYamlFile(test_output_folder / "surf_mapping.yaml");

        sdf_map_setting = std::make_shared<SdfMappingSetting>();
        ERL_ASSERTM(
            sdf_map_setting->FromYamlFile(options->sdf_map_config_file),
            "Failed to load sdf_map_config_file: {}",
            options->sdf_map_config_file);
        sdf_map_setting->AsYamlFile(test_output_folder / "sdf_mapping.yaml");

        ERL_INFO("Surface mapping config: {}", options->surf_map_config_file);
        std::cout << surf_map_setting->AsYamlString() << std::endl;

        ERL_INFO("SDF mapping config: {}", options->sdf_map_config_file);
        std::cout << sdf_map_setting->AsYamlString() << std::endl;

        // create mappings
        surf_map = std::make_shared<SurfaceMapping>(surf_map_setting);
        sdf_map = std::make_shared<SdfMapping>(sdf_map_setting, surf_map);
        mapping = sdf_map;

        // cluster size
        scaling = surf_map_setting->scaling;
        cluster_half_size = surf_map->GetClusterSize() * 0.5f / surf_map_setting->scaling;

        // base init
        Super::Init();

        // other
        sdf_pred_follow.resize(positions_test_follow_org.cols());
        sdf_pred_follow.setZero();
        gradients_follow.resize(3, positions_test_follow.cols());

        sdf_pred_whole_map.resize(positions_test_whole_map.cols());
        sdf_pred_whole_map.setZero();
        gradients_whole_map.resize(3, positions_test_whole_map.cols());

        fps_data.setConstant(4, Super::GetNumOfFrames(), 0.0);

        try {
            surf_map->GetMesh(true, mesh_surf_vertices, mesh_surf_faces);
        } catch (std::exception &e) {
            ERL_WARN("Surface mapping does not support mesh extraction: {}", e.what());
            surf_map_supports_mesh = false;
        }
    }

    void
    PrepareVisualizer() override {
        Super::PrepareVisualizer();

        // for visualizing SDF at a position
        mesh_sdf_sphere = std::make_shared<open3d::geometry::TriangleMesh>();
        if (!options->test_whole_map_at_end) {
            if (std::find(geometries.begin(), geometries.end(), mesh_sdf_sphere) ==
                geometries.end()) {
                geometries.push_back(mesh_sdf_sphere);
                visualizer->AddGeometries({mesh_sdf_sphere});
            }
        }
    }

    bool
    GetPrediction(
        const Matrix3X &positions,
        VectorX &pred_sdf,
        Matrix3X &pred_grads,
        Matrix4X &pred_vars,
        Matrix6X &pred_covars) {

        const long batch_size = options->test_batch_size;
        const long n = positions.cols();

        if (n <= batch_size) {
            return sdf_map->Test(positions, pred_sdf, pred_grads, pred_vars, pred_covars);
        }

        pred_sdf.resize(n);

        VectorX pred_sdf_batch;
        Matrix3X pred_grads_batch;
        Matrix4X pred_vars_batch;
        Matrix6X pred_covars_batch;

        const long i_max = (n + batch_size - 1) / batch_size;
        for (long i = 0; i < n; i += batch_size) {
            const long j = std::min(i + batch_size, n);
            const long m = j - i;
            ERL_INFO("Batch {}/{}: {} to {}, total {}", i / batch_size, i_max, i, j - 1, n);
            if (!sdf_map->Test(
                    positions.middleCols(i, m),
                    pred_sdf_batch,
                    pred_grads_batch,
                    pred_vars_batch,
                    pred_covars_batch)) {
                return false;
            }
            pred_sdf.segment(i, m) = pred_sdf_batch.head(m);
            if (pred_grads_batch.cols() > 0) {
                if (pred_grads.cols() < n) { pred_grads.resize(3, n); }
                pred_grads.middleCols(i, m) = pred_grads_batch.leftCols(m);
            }
            if (pred_vars_batch.cols() > 0) {
                if (pred_vars.cols() < n) { pred_vars.resize(4, n); }
                pred_vars.middleCols(i, m) = pred_vars_batch.leftCols(m);
            }
            if (pred_covars_batch.cols() > 0) {
                if (pred_covars.cols() < n) { pred_covars.resize(6, n); }
                pred_covars.middleCols(i, m) = pred_covars_batch.leftCols(m);
            }
        }
        return true;
    }

    void
    UpdateWholeMapPrediction() override {
        options->test_whole_map_z = static_cast<Dtype>(vis_setting->z);
        positions_test_whole_map.row(2).setConstant(options->test_whole_map_z);

        {
            const ERL_BLOCK_TIMER_MSG("sdf_map.Test");
            ERL_ASSERT(GetPrediction(
                positions_test_whole_map,
                sdf_pred_whole_map,
                gradients_whole_map,
                variances_whole_map,
                covairances_whole_map));
        }

        ERL_INFO(
            "sdf min: {}, max: {}",
            sdf_pred_whole_map.minCoeff(),
            sdf_pred_whole_map.maxCoeff());

        cv::Mat img_sdf =
            ConvertVectorToImage(whole_map_xs, whole_map_ys, sdf_pred_whole_map, true);

        VectorX signs = (sdf_pred_whole_map.array() >= 0.0f).template cast<Dtype>();
        cv::Mat img_sdf_sign = ConvertVectorToImage(whole_map_xs, whole_map_ys, signs, false);

        ConvertToVoxelGrid(img_sdf, positions_test_whole_map, voxel_grid_pred);
        visualizer->GetVisualizer()->UpdateGeometry(voxel_grid_pred);

        Eigen::VectorXb in_free_space;
        ERL_ASSERT(surf_map->IsInFreeSpace(positions_test_whole_map, in_free_space));
        cv::Mat img_surf_mapping_sign = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            in_free_space.cast<Dtype>(),
            false);

        Dtype resize_scale = options->image_resize_scale;
        resize_scale =
            std::min(resize_scale, static_cast<Dtype>(1920.0f) / static_cast<Dtype>(img_sdf.cols));
        resize_scale =
            std::min(resize_scale, static_cast<Dtype>(1920.0f) / static_cast<Dtype>(img_sdf.rows));

        cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
        cv::resize(
            img_surf_mapping_sign,
            img_surf_mapping_sign,
            cv::Size(),
            resize_scale,
            resize_scale);

        cv::imshow("sdf_whole_map", img_sdf);
        cv::imshow("sdf_sign_whole_map", img_sdf_sign);
        cv::imshow("surf_mapping_sign_whole_map", img_surf_mapping_sign);

        cv::imwrite(test_output_folder / "sdf_whole_map.png", img_sdf);
        cv::imwrite(test_output_folder / "sdf_sign_whole_map.png", img_sdf_sign);
        cv::imwrite(test_output_folder / "surf_mapping_sign_whole_map.png", img_surf_mapping_sign);

        cv::waitKey(1);
    }

    void
    UpdateFollowingMapPrediction() override {
        positions_test_follow =
            (rotation_sensor * positions_test_follow_org).colwise() + translation_sensor;

        {
            const ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Test", test_dt);
            test_success = GetPrediction(
                positions_test_follow,
                sdf_pred_follow,
                gradients_follow,
                variances_follow,
                covairances_follow);
        }

        const auto &used_gps = sdf_map->GetUsedGps();
        gp_index_map.clear();
        gp_index_map.reserve(used_gps.size());
        cluster_indices.setConstant(-1);
        if (test_success && !used_gps.empty()) {
            for (long i = 0; i < cluster_indices.size(); ++i) {
                auto &gp = CHECKED_AT(used_gps, i)[0];
                if (gp == nullptr) { continue; }
                auto key = reinterpret_cast<uint64_t>(gp.get());
                auto [it, inserted] = gp_index_map.try_emplace(key, gp_index_map.size());
                cluster_indices[i] = it->second;
            }
            ERL_INFO(
                "{} unique GPs used for {} test points",
                gp_index_map.size(),
                cluster_indices.size());
        }

        if (test_success && surf_map_supports_mesh &&
            std::find(geometries.begin(), geometries.end(), mesh_surf) != geometries.end()) {
            surf_map->GetMesh(true, mesh_surf_vertices, mesh_surf_faces);
            Super::ConvertToOpen3dMesh(mesh_surf, mesh_surf_vertices, mesh_surf_faces);

            // Too slow, not suitable for online mesh extraction
            // Vector3 boundary_size;
            // boundary_size[0] = map_max[0] - map_min[0];
            // boundary_size[1] = map_max[1] - map_min[1];
            // boundary_size[2] = map_max[2] - map_min[2];
            // std::vector<Vector3> face_normals;
            // sdf_map->GetMesh(
            //     boundary_size,
            //     Matrix3::Identity(),
            //     Vector3::Zero(),
            //     options->test_res_grid,
            //     0.0,
            //     mesh_surf_vertices,
            //     mesh_surf_faces,
            //     face_normals);
            // Super::ConvertToOpen3dMesh(mesh_surf, mesh_surf_vertices, mesh_surf_faces);
        }
    }

    void
    UpdatePredictionAtPosition() override {
        position_test[0] = vis_setting->x;
        position_test[1] = vis_setting->y;
        position_test[2] = vis_setting->z;

        erl::common::Logging::SetLevel(erl::common::LoggingLevel::kDebug);
        if (!sdf_map->Test(
                position_test,
                sdf_pred_follow,
                gradients_follow,
                variances_follow,
                covairances_follow)) {
            ERL_WARN("Failed to predict SDF at position {}", position_test.transpose());
            return;
        }
        ERL_INFO(
            "SDF at [{}] = {}, gradient = [{}]",
            position_test.transpose(),
            sdf_pred_follow[0],
            gradients_follow.col(0).transpose());

        const Dtype radius = std::abs(sdf_pred_follow[0]);
        *mesh_sdf_sphere = *open3d::geometry::TriangleMesh::CreateSphere(radius);
        mesh_sdf_sphere->PaintUniformColor({0.0, 1.0, 0.0});
        mesh_sdf_sphere->Translate(position_test.template cast<double>());

        auto vis = visualizer->GetVisualizer();
        vis->UpdateGeometry(mesh_sdf_sphere);

        auto &gp = sdf_map->GetUsedGps()[0][0];
        if (gp != nullptr) {
            pcd_cluster_samples->Clear();
            auto &buf = gp->edf_gp->GetTrainBuffer();
            for (long i = 0; i < buf.num_samples; ++i) {
                pcd_cluster_samples->points_.emplace_back(buf.x.col(i).template cast<double>());
            }
            pcd_cluster_samples->PaintUniformColor({1.0, 0.5, 0.0});  // orange
            vis->UpdateGeometry(pcd_cluster_samples);
        }
    }

    void
    VisualizeSensorData() override {
        if (surf_map_updated) { surf_map_update_fps = 1000.0 / surf_map_update_dt; }
        if (sdf_map_updated) { sdf_map_update_fps = 1000.0 / sdf_map_update_dt; }
        if (test_success) { test_fps = 1000.0 / test_dt; }
        if (gui_dt > 0) { gui_fps = 1000.0 / gui_dt; }

        fps_data.col(frame_idx - 1) << surf_map_update_fps, sdf_map_update_fps, test_fps, gui_fps;

        ranges_img_texts.clear();
        ranges_img_texts.push_back(fmt::format("frame {}", wp_idx));
        ranges_img_texts.push_back(fmt::format("surf_map.update: {:.2f} fps", surf_map_update_fps));
        ranges_img_texts.push_back(fmt::format("sdf_map.update: {:.2f} fps", sdf_map_update_fps));
        ranges_img_texts.push_back(fmt::format("sdf_map.test: {:.2f} fps", test_fps));
        ranges_img_texts.push_back(fmt::format("gui.update: {:.2f} fps", gui_fps));

        Super::VisualizeSensorData();
    }

    void
    VisualizePrediction() override {
        const long xs = options->test_follow_map_xs;
        const long ys = options->test_follow_map_ys;
        cv::Mat img_sdf = ConvertVectorToImage(xs, ys, sdf_pred_follow, true);
        ConvertToVoxelGrid(img_sdf, positions_test_follow, voxel_grid_pred);
        cv::Mat img_sdf_sign = ConvertVectorToImage<Dtype>(
            xs,
            ys,
            (sdf_pred_follow.array() > 0.0).template cast<Dtype>(),
            true);
        cv::Mat img_cluster_indices =
            ConvertVectorToImage<Dtype>(xs, ys, cluster_indices.template cast<Dtype>(), true);

        Dtype resize_scale = options->image_resize_scale;
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(img_sdf.cols));
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(img_sdf.rows));

        cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
        cv::resize(
            img_cluster_indices,
            img_cluster_indices,
            cv::Size(),
            resize_scale,
            resize_scale);

        cv::imshow("sdf", img_sdf);
        cv::imshow("sdf_sign", img_sdf_sign);
        cv::imshow("cluster_indices", img_cluster_indices);
        cv::waitKey(1);

        Super::VisualizePrediction();
    }

    void
    VisualizeSurfaceMapping() override {
        surf_data_buffer = &surf_map->GetSurfaceDataBuffer();
        unused_surf_data_indices = &surf_map->GetUnusedSurfaceDataIndices();
        Super::VisualizeSurfaceMapping();
    }

    void
    UpdateClusterBoxes() override {
        inactive_cluster_keys.clear();
        auto &gps = sdf_map->GetGpMap();
        for (const auto &key: surf_map->GetChangedClusters()) {
            auto it_gp = gps.find(key);
            if (it_gp == gps.end()) { continue; }
            const auto &gp = it_gp->second;
            auto addr = reinterpret_cast<uint64_t>(gp.get());
            if (!gp->active) { inactive_cluster_keys.push_back(addr); }
            Super::UpdateClusterBox(addr, gp->position.template cast<double>());
        }
    }

    std::string
    GetBinFileName() override {
        std::string bin_file = fmt::format("sdf_mapping_3d_{}.bin", type_name<Dtype>());
        bin_file = test_output_folder / bin_file;
        return bin_file;
    }

    void
    TestIo() override {
        auto surface_mapping_read =
            std::make_shared<SurfaceMapping>(std::make_shared<typename SurfaceMapping::Setting>());
        SdfMapping sdf_mapping_read(
            std::make_shared<typename SdfMapping::Setting>(),
            surface_mapping_read);
        Super::TestIo(sdf_mapping_read);
    }

    bool
    UpdateSurfaceMap() {
        if (!mapping_uses_points) {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            // are_points: false, are_local: true
            return surf_map->Update(rotation_frame, translation_frame, frame_ranges, false, true);
        }

        // transform points from sensor frame to world frame
#pragma omp parallel for default(none) schedule(static)
        for (long i = 0; i < frame_points.cols(); ++i) {
            frame_points.col(i) = rotation_frame * frame_points.col(i) + translation_frame;
        }

        {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            // are_points: true, are_local: false
            return surf_map->Update(rotation_frame, translation_frame, frame_points, true, false);
        }
    }

    bool
    UpdateMap() override {
        ERL_BLOCK_TIMER_MSG_TIME("sdf_map.Update", sdf_map_update_dt);

        surf_map_updated = UpdateSurfaceMap();
        ERL_WARN_COND(!surf_map_updated, "Sdf mapping update failed");
        if (!surf_map_updated) { return false; }

        const double time_budget_us = 1e6 / sdf_map_setting->update_hz;  // us
        sdf_map_updated = sdf_map->UpdateGpSdf(time_budget_us - surf_map_update_dt * 1000);
        return sdf_map_updated;
    }

    void
    TestGrid(const Matrix3X &grid_positions) override {
        VectorX pred_sdf;
        Matrix3X pred_grads;
        Matrix4X pred_vars;
        Matrix6X pred_covars;
        {
            const ERL_BLOCK_TIMER_MSG("sdf_map.Test grid");
            ERL_ASSERT(GetPrediction(grid_positions, pred_sdf, pred_grads, pred_vars, pred_covars));
        }

        std::filesystem::path file = test_output_folder / "test_grid_positions.bin";
        ERL_INFO("Saving test grid positions to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, grid_positions));

        file = test_output_folder / "test_grid_sdf.bin";
        ERL_INFO("Saving test grid sdf to {}", file.string());
        ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_sdf));

        if (pred_grads.cols() > 0) {
            file = test_output_folder / "test_grid_gradients.bin";
            ERL_INFO("Saving test grid gradients to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_grads));
        }

        if (pred_vars.cols() > 0) {
            file = test_output_folder / "test_grid_variances.bin";
            ERL_INFO("Saving test grid variances to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_vars));
        }

        if (pred_covars.cols() > 0) {
            file = test_output_folder / "test_grid_covariances.bin";
            ERL_INFO("Saving test grid covariances to {}", file.string());
            ERL_ASSERT(erl::common::SaveEigenMatrixToBinaryFile<Dtype>(file, pred_covars));
        }
    }
};
