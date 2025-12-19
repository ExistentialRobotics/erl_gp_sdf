#pragma once

#include "test_mapping_3d.hpp"

#include <open3d/visualization/utility/DrawGeometry.h>

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>, OptionsForTestMapping3D<Dtype>> {

    using Super = OptionsForTestMapping3D<Dtype>;

    std::string surf_map_config_file;

    ERL_REFLECT_SCHEMA(Options, ERL_REFLECT_MEMBER(Options, surf_map_config_file));

    bool
    PostDeserialization() override {
        if (!Super::PostDeserialization()) { return false; }
        ERL_ASSERTM(
            !surf_map_config_file.empty(),
            "Please provide the surface mapping config file via --surf_map_config_file");
        return true;
    }
};

template<typename Dtype, typename SurfaceMappingType>
struct TestSurfMapping3D : public TestMapping3D<Dtype, SurfaceMappingType> {

    using SurfaceMapping = SurfaceMappingType;
    using Super = TestMapping3D<Dtype, SurfaceMapping>;
    using SurfaceMappingSetting = typename SurfaceMapping::Setting;
    using OptionType = Options<Dtype>;

    using typename Super::Matrix3X;
    using typename Super::VectorX;

    using Super::cluster_half_size;
    using Super::fps_data;
    using Super::frame_points;
    using Super::frame_ranges;
    using Super::gui_dt;
    using Super::mapping;
    using Super::mapping_uses_points;
    using Super::max_wp_idx;
    using Super::positions_test_follow;
    using Super::positions_test_follow_org;
    using Super::positions_test_whole_map;
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

    using Super::LoadData;

    std::shared_ptr<OptionType> options = nullptr;

    std::shared_ptr<SurfaceMappingSetting> surf_map_setting = nullptr;
    std::shared_ptr<SurfaceMapping> surf_map = nullptr;

    // test data

    VectorX prob_occupied_follow;
    Eigen::VectorXb in_free_space_follow;
    Matrix3X gradient_follow;
    VectorX prob_occupied_whole_map;
    Eigen::VectorXb in_free_space_whole_map;
    Matrix3X gradient_whole_map;

    // logging

    bool test_success = false;
    double surf_map_update_dt = 0;
    double surf_map_update_fps = 0.0;
    double test_dt = 0;
    double test_fps = 0.0;
    double gui_fps = 0.0;

    TestSurfMapping3D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionType> options = std::make_shared<OptionType>())
        : TestMapping3D<Dtype, SurfaceMapping>(argc, argv, options), options(options) {}

protected:
    void
    Init() override {
        // load settings
        surf_map_setting = std::make_shared<SurfaceMappingSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options->surf_map_config_file),
            "Failed to load surf_map_config_file: {}",
            options->surf_map_config_file);
        surf_map_setting->AsYamlFile(test_output_folder / "config.yaml");

        ERL_INFO("Surface mapping config: {}", options->surf_map_config_file);
        std::cout << surf_map_setting->AsYamlString() << std::endl;

        // create mapping
        surf_map = std::make_shared<SurfaceMapping>(surf_map_setting);
        mapping = surf_map;

        // cluster size
        scaling = surf_map_setting->scaling;
        cluster_half_size = surf_map->GetClusterSize() * 0.5f / surf_map_setting->scaling;

        // base init
        Super::Init();

        // other
        prob_occupied_follow.resize(positions_test_follow_org.cols());
        in_free_space_follow.resize(positions_test_follow_org.cols());
        gradient_follow.resize(3, positions_test_follow_org.cols());

        prob_occupied_whole_map.resize(positions_test_whole_map.cols());
        in_free_space_whole_map.resize(positions_test_whole_map.cols());
        gradient_whole_map.resize(3, positions_test_whole_map.cols());

        fps_data.resize(3, (max_wp_idx + options->seq_stride - 1) / options->seq_stride);
    }

    void
    VisualizeSensorData() override {
        surf_map_update_fps = 1000.0 / surf_map_update_dt;
        if (test_success) { test_fps = 1000.0 / test_dt; }
        if (gui_dt > 0) { gui_fps = 1000.0 / gui_dt; }

        fps_data.col(wp_idx / options->seq_stride - 1) << surf_map_update_fps, test_fps, gui_fps;

        ranges_img_texts.clear();
        ranges_img_texts.push_back(fmt::format("frame {}", wp_idx));
        ranges_img_texts.push_back(fmt::format("surf_map.update: {:.2f} fps", surf_map_update_fps));
        if (test_fps > 0) {
            ranges_img_texts.push_back(fmt::format("surf_map.test: {:.2f} fps", test_fps));
        }
        ranges_img_texts.push_back(fmt::format("gui.update: {:.2f} fps", gui_fps));

        Super::VisualizeSensorData();
    }

    void
    VisualizeSurfaceMapping() override {
        surf_data_buffer = &surf_map->GetSurfaceDataBuffer();
        unused_surf_data_indices = &surf_map->GetUnusedSurfaceDataIndices();
        Super::VisualizeSurfaceMapping();
    }

    std::string
    GetBinFileName() override {
        std::string bin_file = fmt::format("surf_mapping_3d_{}.bin", type_name<Dtype>());
        bin_file = test_output_folder / bin_file;
        return bin_file;
    }

    void
    TestIo() override {
        SurfaceMapping surface_mapping_read(std::make_shared<SurfaceMappingSetting>());
        Super::TestIo(surface_mapping_read);
    }

    bool
    UpdateMap() override {
        if (!mapping_uses_points) {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            return surf_map->Update(rotation_frame, translation_frame, frame_ranges, false, true);
        }

        // transform points from sensor frame to world frame
#pragma omp parallel for default(none) schedule(static)
        for (long i = 0; i < frame_points.cols(); ++i) {
            frame_points.col(i) = rotation_frame * frame_points.col(i) + translation_frame;
        }

        {
            ERL_BLOCK_TIMER_MSG_TIME("surf_map.Update", surf_map_update_dt);
            return surf_map->Update(rotation_frame, translation_frame, frame_points, true, false);
        }
    }
};
