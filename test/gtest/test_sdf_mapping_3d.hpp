#pragma once

#include "test_mapping_3d.hpp"

#include "erl_common/macros.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>, OptionsForTestMapping3D<Dtype>> {

    using Super = OptionsForTestMapping3D<Dtype>;

    std::string surface_mapping_config_file;
    std::string sdf_mapping_config_file;

    ERL_REFLECT_SCHEMA(
        Options,
        ERL_REFLECT_MEMBER(Options, surface_mapping_config_file),
        ERL_REFLECT_MEMBER(Options, sdf_mapping_config_file));

    bool
    PostDeserialization() override {
        if (!Super::PostDeserialization()) { return false; }
        ERL_ASSERTM(
            !surface_mapping_config_file.empty(),
            "Please provide the surface mapping config file via --surface_mapping_config_file");
        ERL_ASSERTM(
            !sdf_mapping_config_file.empty(),
            "Please provide the SDF mapping config file via --sdf_mapping_config_file");
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
    using Super::mapping;
    using Super::mapping_uses_points;
    using Super::max_wp_idx;
    using Super::mesh_sensor;
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
            surf_map_setting->FromYamlFile(options->surface_mapping_config_file),
            "Failed to load surface_mapping_config_file: {}",
            options->surface_mapping_config_file);
        surf_map_setting->AsYamlFile(test_output_folder / "surf_mapping.yaml");

        sdf_map_setting = std::make_shared<SdfMappingSetting>();
        ERL_ASSERTM(
            sdf_map_setting->FromYamlFile(options->sdf_mapping_config_file),
            "Failed to load sdf_mapping_config_file: {}",
            options->sdf_mapping_config_file);
        sdf_map_setting->AsYamlFile(test_output_folder / "sdf_mapping.yaml");

        ERL_INFO("Surface mapping config: {}", options->surface_mapping_config_file);
        std::cout << surf_map_setting->AsYamlString() << std::endl;

        ERL_INFO("SDF mapping config: {}", options->sdf_mapping_config_file);
        std::cout << sdf_map_setting->AsYamlString() << std::endl;

        // create mappings
        surf_map = std::make_shared<SurfaceMapping>(surf_map_setting);
        sdf_map = std::make_shared<SdfMapping>(sdf_map_setting, surf_map);
        mapping = sdf_map;

        // cluster size
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

        fps_data.resize(4, (max_wp_idx + options->seq_stride - 1) / options->seq_stride);
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

    void
    UpdateWholeMapPrediction() override {
        options->test_z = static_cast<Dtype>(vis_setting->z);
        positions_test_whole_map.row(2).setConstant(options->test_z);

        {
            ERL_BLOCK_TIMER_MSG("sdf_map.Test");
            ASSERT_TRUE(sdf_map->Test(
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
        ASSERT_TRUE(surf_map->IsInFreeSpace(positions_test_whole_map, in_free_space));
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
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Test", test_dt);
            test_success = sdf_map->Test(
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
                auto &gp = VEC_ACCESS(used_gps, i)[0];
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

        fps_data.col(wp_idx / options->seq_stride) << surf_map_update_fps, sdf_map_update_fps,
            test_fps, gui_fps;
        wp_idx += options->seq_stride;

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
        cv::Mat img_sdf =
            ConvertVectorToImage(options->test_xs, options->test_ys, sdf_pred_follow, true);
        ConvertToVoxelGrid(img_sdf, positions_test_follow, voxel_grid_pred);
        cv::Mat img_sdf_sign = ConvertVectorToImage<Dtype>(
            options->test_xs,
            options->test_ys,
            (sdf_pred_follow.array() > 0.0).template cast<Dtype>(),
            true);
        cv::Mat img_cluster_indices = ConvertVectorToImage<Dtype>(
            options->test_xs,
            options->test_ys,
            cluster_indices.template cast<Dtype>(),
            true);

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
        LoadData();

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
};
