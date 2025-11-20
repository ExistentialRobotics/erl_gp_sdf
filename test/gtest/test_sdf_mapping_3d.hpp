#pragma once

#include "utils.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/macros.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"
#include "erl_geometry/newer_college.hpp"
#include "erl_geometry/open3d_helper.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

enum class DataSetType {
    CowAndLady,
    Mesh,
    NewerCollege,
};

template<typename Dtype>
struct Options : public erl::common::Yamlable<Options<Dtype>> {
    inline static const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
    inline static const std::filesystem::path kDataDir = kProjectRootDir / "data";
    inline static const std::filesystem::path kConfigDir = kProjectRootDir / "config";

    std::string dataset_name = "cow_and_lady";
    std::string cow_and_lady_dir;
    std::string newer_college_dir;
    std::string mesh_file = kDataDir / "replica-hotel-0.ply";       // mesh file
    std::string traj_file = kDataDir / "replica-hotel-0-traj.txt";  // trajectory file
    std::string surface_mapping_config_file;
    std::string sdf_mapping_config_file;
    std::string o3d_view_status_file;
    std::string sdf_mapping_bin_file;
    std::string sensor_frame_type = type_name<erl::geometry::LidarFrame3D<Dtype>>();
    std::string sensor_frame_config_file = kConfigDir / "sensors" / "lidar_frame_3d_360.yaml";
    long start_wp_idx = 0;
    long end_wp_idx = -1;  // -1 means all waypoints
    long seq_stride = 1;
    Eigen::VectorXl scan_stride;          // linear scan stride or per axis scan stride
    bool random_scan_downsample = false;  // use random downsample the scan points
    long vis_stride = 1;                  // visualization stride
    std::size_t pcd_stride = 10;          // stride of pcd for visualization
    std::vector<std::string> show_geometries =
        {"all", "-pcd_obs", "-pcd_surf_points", "-line_set_surf_normals", "-line_set_gps"};
    Dtype test_res = 0.02;               // test resolution
    Dtype test_x_min = 0.0f;             // x min for testing of the whole map
    Dtype test_x_max = 0.0f;             // x max for testing of the whole map
    Dtype test_y_min = 0.0f;             // y min for testing of the whole map
    Dtype test_y_max = 0.0f;             // y max for testing of the whole map
    Dtype test_z = 0.0;                  // test z for the whole map/ single query point
    long test_xs = 150;                  // num of x for testing of the follow map
    long test_ys = 100;                  // num of y for testing of the follow map
    bool test_whole_map_at_end = false;  // test the whole map at the end
    Dtype image_resize_scale = 10;       // image resize scale
    bool save_images = false;
    bool test_io = false;
    bool hold = false;
    bool load_sdf_mapping_bin = false;

    ERL_REFLECT_SCHEMA(
        Options,
        ERL_REFLECT_MEMBER(Options, dataset_name),
        ERL_REFLECT_MEMBER(Options, cow_and_lady_dir),
        ERL_REFLECT_MEMBER(Options, newer_college_dir),
        ERL_REFLECT_MEMBER(Options, mesh_file),
        ERL_REFLECT_MEMBER(Options, traj_file),
        ERL_REFLECT_MEMBER(Options, surface_mapping_config_file),
        ERL_REFLECT_MEMBER(Options, sdf_mapping_config_file),
        ERL_REFLECT_MEMBER(Options, o3d_view_status_file),
        ERL_REFLECT_MEMBER(Options, sdf_mapping_bin_file),
        ERL_REFLECT_MEMBER(Options, sensor_frame_type),
        ERL_REFLECT_MEMBER(Options, sensor_frame_config_file),
        ERL_REFLECT_MEMBER(Options, start_wp_idx),
        ERL_REFLECT_MEMBER(Options, end_wp_idx),
        ERL_REFLECT_MEMBER(Options, seq_stride),
        ERL_REFLECT_MEMBER(Options, scan_stride),
        ERL_REFLECT_MEMBER(Options, random_scan_downsample),
        ERL_REFLECT_MEMBER(Options, vis_stride),
        ERL_REFLECT_MEMBER(Options, show_geometries),
        ERL_REFLECT_MEMBER(Options, test_res),
        ERL_REFLECT_MEMBER(Options, test_x_min),
        ERL_REFLECT_MEMBER(Options, test_x_max),
        ERL_REFLECT_MEMBER(Options, test_y_min),
        ERL_REFLECT_MEMBER(Options, test_y_max),
        ERL_REFLECT_MEMBER(Options, test_z),
        ERL_REFLECT_MEMBER(Options, test_xs),
        ERL_REFLECT_MEMBER(Options, test_ys),
        ERL_REFLECT_MEMBER(Options, test_whole_map_at_end),
        ERL_REFLECT_MEMBER(Options, image_resize_scale),
        ERL_REFLECT_MEMBER(Options, save_images),
        ERL_REFLECT_MEMBER(Options, test_io),
        ERL_REFLECT_MEMBER(Options, hold),
        ERL_REFLECT_MEMBER(Options, load_sdf_mapping_bin));
};

template<typename Dtype, typename SurfaceMappingType>
struct TestSdfMapping3D {
    using SurfaceMapping = SurfaceMappingType;
    using SdfMapping = erl::gp_sdf::GpSdfMapping<Dtype, 3>;
    using SurfaceMappingSetting = typename SurfaceMapping::Setting;
    using SdfMappingSetting = typename SdfMapping::Setting;

    using DepthFrame = erl::geometry::DepthFrame3D<Dtype>;
    using LidarFrame = erl::geometry::LidarFrame3D<Dtype>;
    using RangeSensorFrame = erl::geometry::RangeSensorFrame3D<Dtype>;
    using DepthCamera = erl::geometry::DepthCamera3D<Dtype>;
    using Lidar = erl::geometry::Lidar3D<Dtype>;
    using RangeSensor = erl::geometry::RangeSensor3D<Dtype>;
    using CowAndLady = erl::geometry::CowAndLady;
    using NewerCollege = erl::geometry::NewerCollege;
    using Open3dVisualizerWrapper = erl::geometry::Open3dVisualizerWrapper;
    using Trajectory = erl::geometry::Trajectory<Dtype>;

    using VectorX = Eigen::VectorX<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using Matrix4X = Eigen::Matrix4X<Dtype>;
    using Matrix6X = Eigen::Matrix<Dtype, 6, Eigen::Dynamic>;

    Options<Dtype> options;

    std::shared_ptr<SurfaceMappingSetting> surf_map_setting = nullptr;
    std::shared_ptr<SdfMappingSetting> sdf_map_setting = nullptr;
    std::shared_ptr<SurfaceMapping> surf_map = nullptr;
    std::shared_ptr<SdfMapping> sdf_map = nullptr;

    // datasets

    DataSetType dataset_type = DataSetType::Mesh;
    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::shared_ptr<NewerCollege> newer_college = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    long max_wp_idx = 0;
    long wp_idx = 0;
    bool mapping_uses_points = false;  // should be set externally
    bool raw_data_is_points = false;
    bool raw_data_is_row_major = false;
    MatrixX frame_ranges;
    Matrix3X frame_points;

    // sensor

    std::shared_ptr<RangeSensorFrame> range_sensor_frame = nullptr;
    std::shared_ptr<RangeSensor> range_sensor = nullptr;
    bool is_lidar = false;
    Matrix3 rotation_sensor;
    Matrix3 rotation_frame;
    Vector3 translation_sensor;
    Vector3 translation_frame;

    // open3d data structures

    std::shared_ptr<Open3dVisualizerWrapper::Setting> vis_setting = nullptr;
    std::shared_ptr<Open3dVisualizerWrapper> visualizer = nullptr;
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;
    std::shared_ptr<open3d::geometry::Geometry3D> gt_scene = nullptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sensor = nullptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sensor_xyz = nullptr;
    std::shared_ptr<open3d::geometry::LineSet> line_set_traj = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_obs = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_surf_points = nullptr;
    std::shared_ptr<open3d::geometry::LineSet> line_set_surf_normals = nullptr;
    std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_sdf = nullptr;
    std::shared_ptr<open3d::geometry::LineSet> line_set_gps = nullptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sdf_sphere = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_gp_points = nullptr;
    open3d::geometry::LineSet line_set_gp_box;
    using LineSetInfo = std::pair<std::array<std::size_t, 8>, std::array<std::size_t, 12>>;
    absl::flat_hash_map<uint64_t, LineSetInfo> line_set_gps_map;
    absl::flat_hash_map<Eigen::Vector3l, std::size_t> gp_vertex_index_map;
    absl::flat_hash_map<Eigen::Vector2i, std::size_t> gp_edge_index_map;

    // opencv data structures

    cv::Mat ranges_img;

    // test data

    Vector3 map_min, map_max;
    Matrix3 grid_rotation = Matrix3::Identity();
    Vector3 grid_translation = Vector3::Zero();
    Vector3 position_test;
    Matrix3X positions_test_follow_org;
    Matrix3X positions_test_follow;
    VectorX sdf_pred_follow;
    Matrix3X gradients_follow;
    Matrix4X variances_follow;
    Matrix6X covairances_follow;
    long whole_map_xs = 0;
    long whole_map_ys = 0;
    Matrix3X positions_test_whole_map;
    VectorX sdf_pred_whole_map;
    Matrix3X gradients_whole_map;
    Matrix4X variances_whole_map;
    Matrix6X covairances_whole_map;

    // animation control

    bool animation_ended = false;
    Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();

    // logging

    bool surf_map_updated = false;
    bool sdf_map_updated = false;
    bool test_success = false;
    Eigen::Matrix4Xd fps_data;
    double surf_map_update_dt = 0;
    double sdf_map_update_dt = 0;
    double test_dt = 0;
    double gui_dt = 0;
    double surf_map_update_fps = 0;
    double sdf_map_update_fps = 0;
    double test_fps = 0;
    double gui_fps = 0;
    Eigen::VectorXl gp_indices;
    absl::flat_hash_map<uint64_t, long> gp_index_map;

    // output folders

    std::filesystem::path test_output_folder;
    std::filesystem::path img_dir;

    TestSdfMapping3D(const int argc, char *argv[]) {
        ParseOptions(argc, argv);
        LoadSetting();
        PrepareDataset();
        PrepareOutputFolders();
        PrepareVisualizer();
    }

    void
    Run() {
        visualizer->SetAnimationCallback(
            [this](auto *wrapper, auto *vis) { return this->AnimationCallback(wrapper, vis); });
        visualizer->SetViewStatus(options.o3d_view_status_file);

        if (options.load_sdf_mapping_bin) {
            ReadSdfMappingBin(*sdf_map);
            animation_ended = true;
            visualizer->Show();
        } else {
            if (options.test_io) { TestIo(); }
            visualizer->Show();
            if (options.test_io) { TestIo(); }

            erl::common::SaveEigenMatrixToTextFile<double>(
                test_output_folder / "fps.csv",
                fps_data,
                erl::common::EigenTextFormat::kCsvFmt);
        }
    }

protected:
    // initialization

    void
    ParseOptions(int argc, char **argv) {
        options.FromCommandLine(argc, argv);

        if (options.scan_stride.size() > 0) {
            ERL_ASSERT_LE(options.scan_stride.size(), 2);
            for (long i = 0; i < options.scan_stride.size(); ++i) {
                ERL_ASSERT_POS_GT(options.scan_stride[i], 0);
            }
        }

        if (options.dataset_name == "cow_and_lady") {
            dataset_type = DataSetType::CowAndLady;
            ERL_ASSERTM(
                !options.cow_and_lady_dir.empty(),
                "Please provide the Cow and Lady dataset directory via --cow_and_lady_dir");
        } else if (options.dataset_name == "mesh") {
            dataset_type = DataSetType::Mesh;
            ERL_ASSERTM(!options.mesh_file.empty(), "Please provide the mesh file via --mesh_file");
            ERL_ASSERTM(
                !options.traj_file.empty(),
                "Please provide the trajectory file via --traj_file");
        } else if (options.dataset_name == "newer_college") {
            dataset_type = DataSetType::NewerCollege;
            ERL_ASSERTM(
                !options.newer_college_dir.empty(),
                "Please provide the Newer College dataset directory via --newer_college_dir");
        } else {
            ERL_FATAL("Unknown dataset name {} for 3D", options.dataset_name);
        }
    }

    void
    LoadSetting() {
        surf_map_setting = std::make_shared<SurfaceMappingSetting>();
        ERL_ASSERTM(
            surf_map_setting->FromYamlFile(options.surface_mapping_config_file),
            "Failed to load surface_mapping_config_file: {}",
            options.surface_mapping_config_file);

        sdf_map_setting = std::make_shared<SdfMappingSetting>();
        ERL_ASSERTM(
            sdf_map_setting->FromYamlFile(options.sdf_mapping_config_file),
            "Failed to load sdf_mapping_config_file: {}",
            options.sdf_mapping_config_file);

        ERL_INFO("Surface mapping config: {}", options.surface_mapping_config_file);
        std::cout << surf_map_setting->AsYamlString() << std::endl;

        ERL_INFO("SDF mapping config: {}", options.sdf_mapping_config_file);
        std::cout << sdf_map_setting->AsYamlString() << std::endl;

        surf_map = std::make_shared<SurfaceMapping>(surf_map_setting);
        sdf_map = std::make_shared<SdfMapping>(sdf_map_setting, surf_map);

        vis_setting = std::make_shared<Open3dVisualizerWrapper::Setting>();
    }

    void
    PrepareCowAndLady() {
        // dataset
        cow_and_lady = std::make_shared<CowAndLady>(options.cow_and_lady_dir);
        max_wp_idx = cow_and_lady->Size();
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LT(options.end_wp_idx, max_wp_idx);
        raw_data_is_points = false;
        raw_data_is_row_major = false;
        // sensor
        const auto depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = CowAndLady::kCameraCy;
        range_sensor_frame = std::make_shared<DepthFrame>(depth_frame_setting);
        // open3d
        gt_scene = cow_and_lady->GetGroundTruthPointCloud();
        // test data
        map_min = cow_and_lady->GetMapMin().cast<Dtype>();
        map_max = cow_and_lady->GetMapMax().cast<Dtype>();
        if (options.test_x_min == options.test_x_max || options.test_y_min == options.test_y_max) {
            options.test_x_min = map_min[0];
            options.test_x_max = map_max[0];
            options.test_y_min = map_min[1];
            options.test_y_max = map_max[1];
        }
    }

    void
    PrepareMeshDataset() {
        // dataset
        const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options.mesh_file);
        poses = Trajectory::LoadSe3(options.traj_file, false);
        max_wp_idx = static_cast<long>(poses.size());
        raw_data_is_points = false;
        raw_data_is_row_major = false;
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_GT(options.end_wp_idx, max_wp_idx);
        // sensor
        if (options.sensor_frame_type == type_name<LidarFrame>()) {
            const auto lidar_frame_setting = std::make_shared<typename LidarFrame::Setting>();
            ASSERT_TRUE(lidar_frame_setting->FromYamlFile(options.sensor_frame_config_file));
            const auto lidar_setting = std::make_shared<typename Lidar::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<Lidar>(lidar_setting);
            is_lidar = true;
            range_sensor_frame = std::make_shared<LidarFrame>(lidar_frame_setting);
        } else if (options.sensor_frame_type == type_name<DepthFrame>()) {
            const auto depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
            ASSERT_TRUE(depth_frame_setting->FromYamlFile(options.sensor_frame_config_file));
            auto depth_camera_setting = std::make_shared<typename DepthCamera::Setting>(
                depth_frame_setting->camera_intrinsic);
            range_sensor = std::make_shared<DepthCamera>(depth_camera_setting);
            is_lidar = false;
            range_sensor_frame = std::make_shared<DepthFrame>(depth_frame_setting);
        } else {
            ERL_FATAL("Unknown sensor_frame_type: {}", options.sensor_frame_type);
        }
        range_sensor->AddMesh(options.mesh_file);
        // open3d
        gt_scene = mesh;
        // test data
        map_min = mesh->GetMinBound().template cast<Dtype>();
        map_max = mesh->GetMaxBound().template cast<Dtype>();
        if (options.test_x_min == options.test_x_max || options.test_y_min == options.test_y_max) {
            options.test_x_min = map_min[0];
            options.test_x_max = map_max[0];
            options.test_y_min = map_min[1];
            options.test_y_max = map_max[1];
        }
    }

    void
    PrepareNewerCollege() {
        // dataset
        newer_college = std::make_shared<NewerCollege>(options.newer_college_dir);
        max_wp_idx = erl::geometry::NewerCollege::Size();
        is_lidar = true;
        raw_data_is_points = true;
        raw_data_is_row_major = true;
        ERL_ASSERT_LT(options.start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_GT(options.end_wp_idx, max_wp_idx);
        // sensor
        const auto lidar_frame_setting = std::make_shared<typename LidarFrame::Setting>();
        lidar_frame_setting->azimuth_min = -M_PI;
        lidar_frame_setting->azimuth_max = M_PI;
        lidar_frame_setting->num_azimuth_lines = NewerCollege::kNumAzimuthLines;
        lidar_frame_setting->elevation_min = -NewerCollege::kVerticalFov / 2;
        lidar_frame_setting->elevation_max = NewerCollege::kVerticalFov / 2;
        lidar_frame_setting->num_elevation_lines = NewerCollege::kNumElevationLines;
        is_lidar = true;
        range_sensor_frame = std::make_shared<LidarFrame>(lidar_frame_setting);
        // open3d
        auto pcd = newer_college->GetGroundTruthPointCloud();
        pcd = pcd->RandomDownSample(0.05);
        gt_scene = pcd;
        // test data
        Eigen::Matrix3d box_rotation;
        Eigen::Vector3d box_translation;
        Eigen::Vector3d box_size;
        erl::geometry::GetOrientedBoundingBoxWithAxisUp(
            newer_college->GetGroundTruthMesh()->GetMinimalOrientedBoundingBox(),
            2,
            box_translation,
            box_rotation,
            box_size);
        grid_rotation = box_rotation.cast<Dtype>();
        grid_translation = box_translation.cast<Dtype>();

        ERL_INFO(
            "rotation: \n{}, \n"
            "translation: {}, size: {}",
            box_rotation,
            box_translation.transpose(),
            box_size.transpose());

        map_max = box_size.cast<Dtype>() / 2;
        map_min = -map_max;

        if (options.test_x_min == options.test_x_max || options.test_y_min == options.test_y_max) {
            options.test_x_min = -box_size[0] / 2;
            options.test_x_max = box_size[0] / 2;
            options.test_y_min = -box_size[1] / 2;
            options.test_y_max = box_size[1] / 2;
        }
    }

    void
    PrepareDataset() {
        // 1. load dataset
        // 2. create range sensor frame for converting data to points
        // 3. set map_min, map_max, max_wp_idx
        // 4. initialize positions_test_org, positions_test, sdf_pred_follow and fps_data

        switch (dataset_type) {
            case DataSetType::CowAndLady:
                PrepareCowAndLady();
                break;
            case DataSetType::Mesh:
                PrepareMeshDataset();
                break;
            case DataSetType::NewerCollege:
                PrepareNewerCollege();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }

        max_wp_idx = (options.end_wp_idx == -1) ? max_wp_idx : options.end_wp_idx;
        options.test_z = (map_min[2] + map_max[2]) / 2;

        vis_setting->z = options.test_z;
        position_test[0] = vis_setting->x;
        position_test[1] = vis_setting->y;
        position_test[2] = options.test_z;

        positions_test_follow_org.resize(3, options.test_xs * options.test_ys);
        const Vector3 offset(
            static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_xs),
            static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_ys),
            0.0);
        // x: down, y: right
        long idx = 0;
        for (long j = 0; j < options.test_ys; ++j) {
            const Dtype y = static_cast<Dtype>(j) * options.test_res + offset[1];
            for (long i = 0; i < options.test_xs; ++i) {
                const Dtype x = static_cast<Dtype>(i) * options.test_res + offset[0];
                positions_test_follow_org.col(idx++) << x, y, offset[2];
            }
        }
        positions_test_follow.resize(3, positions_test_follow_org.cols());
        sdf_pred_follow.resize(positions_test_follow_org.cols());
        sdf_pred_follow.setZero();
        gradients_follow.resize(3, positions_test_follow.cols());

        if (options.test_x_min == options.test_x_max || options.test_y_min == options.test_y_max) {
            ERL_INFO("Map boundary is not fully defined, using surface mapping boundary.");
            options.test_x_min = map_min[0];
            options.test_x_max = map_max[0];
            options.test_y_min = map_min[1];
            options.test_y_max = map_max[1];
        }
        erl::common::GridMapInfo2D<Dtype> grid_map_info(
            Eigen::Vector2<Dtype>(options.test_x_min, options.test_y_min),
            Eigen::Vector2<Dtype>(options.test_x_max, options.test_y_max),
            Eigen::Vector2<Dtype>(options.test_res, options.test_res),
            Eigen::Vector2i(0, 0));
        whole_map_xs = grid_map_info.Shape(0);
        whole_map_ys = grid_map_info.Shape(1);
        positions_test_whole_map.resize(3, grid_map_info.Size());
        positions_test_whole_map.topRows(2) =
            grid_map_info.GenerateMeterCoordinates(false).template cast<Dtype>();
        positions_test_whole_map =
            (grid_rotation * positions_test_whole_map).colwise() + grid_translation;
        sdf_pred_whole_map.resize(positions_test_whole_map.cols());
        sdf_pred_whole_map.setZero();
        gradients_whole_map.resize(3, positions_test_whole_map.cols());

        gp_indices.resize(positions_test_follow_org.cols());
        fps_data.resize(4, (max_wp_idx + options.seq_stride - 1) / options.seq_stride);
    }

    void
    PrepareOutputFolders() {
        GTEST_PREPARE_OUTPUT_DIR();
        test_output_folder = test_output_dir;
        img_dir = test_output_folder / "images";
        std::filesystem::create_directory(img_dir);
        vis_setting->window_name = test_info->name();
    }

    void
    PrepareVisualizer() {
        vis_setting->mesh_show_back_face = false;
        vis_setting->translate_step = dataset_type == DataSetType::NewerCollege ? 0.1 : 0.01;
        vis_setting->x = (options.test_x_min + options.test_x_max) / 2;
        vis_setting->y = (options.test_y_min + options.test_y_max) / 2;
        vis_setting->z = options.test_z;
        visualizer = std::make_shared<Open3dVisualizerWrapper>(vis_setting);

        if (dataset_type == DataSetType::NewerCollege) {
            mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.5);
            mesh_sensor_xyz = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1);
        } else {
            mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
            mesh_sensor_xyz = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1);
        }
        mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});

        line_set_traj = std::make_shared<open3d::geometry::LineSet>();
        pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
        pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
        line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();

        voxel_grid_sdf = std::make_shared<open3d::geometry::VoxelGrid>();
        voxel_grid_sdf->origin_.setZero();
        voxel_grid_sdf->voxel_size_ = options.test_res * 1.42f;

        line_set_gps = std::make_shared<open3d::geometry::LineSet>();
        auto o3d_line_set = std::make_shared<open3d::geometry::LineSet>();
        const double hs = surf_map->GetClusterSize() * 0.5f / surf_map_setting->scaling;
        line_set_gp_box = *open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            open3d::geometry::AxisAlignedBoundingBox(
                Eigen::Vector3d(-hs, -hs, -hs),
                Eigen::Vector3d(hs, hs, hs)));

        mesh_sdf_sphere = std::make_shared<open3d::geometry::TriangleMesh>();
        pcd_gp_points = std::make_shared<open3d::geometry::PointCloud>();

        absl::flat_hash_map<std::string, std::shared_ptr<open3d::geometry::Geometry3D>> geo_map;
        geo_map["gt_scene"] = gt_scene;
        geo_map["mesh_sensor"] = mesh_sensor;
        geo_map["mesh_sensor_xyz"] = mesh_sensor_xyz;
        geo_map["line_set_traj"] = line_set_traj;
        geo_map["pcd_obs"] = pcd_obs;
        geo_map["pcd_surf_points"] = pcd_surf_points;
        geo_map["line_set_surf_normals"] = line_set_surf_normals;
        geo_map["voxel_grid_sdf"] = voxel_grid_sdf;
        geo_map["line_set_gps"] = line_set_gps;
        std::vector<std::string> geo_names;
        geo_names.reserve(geo_map.size());
        for (const auto &[name, geo]: geo_map) { geo_names.push_back(name); }

        for (auto geo_name: options.show_geometries) {
            bool add = true;
            if (geo_name[0] == '-') {
                add = false;
                geo_name = geo_name.substr(1);
            } else if (geo_name[0] == '+') {
                geo_name = geo_name.substr(1);
            }
            if (geo_name == "all") {
                visualizer->ClearGeometries();
                geometries.clear();
                geometries.push_back(gt_scene);
                geometries.push_back(mesh_sensor);
                geometries.push_back(mesh_sensor_xyz);
                geometries.push_back(line_set_traj);
                geometries.push_back(pcd_obs);
                geometries.push_back(pcd_surf_points);
                geometries.push_back(line_set_surf_normals);
                geometries.push_back(voxel_grid_sdf);
                geometries.push_back(line_set_gps);
                continue;
            }
            const auto it = geo_map.find(geo_name);
            if (it == geo_map.end()) {
                ERL_WARN(
                    "Unknown geometry name to show: {}. Available geometries: {}",
                    geo_name,
                    geo_names);
                continue;
            }
            auto it2 = std::find(geometries.begin(), geometries.end(), it->second);
            if (add) {
                if (it2 != geometries.end()) { continue; }
                geometries.push_back(it->second);
            } else {
                if (it2 == geometries.end()) { continue; }
                geometries.erase(it2);
            }
        }

        if (options.test_whole_map_at_end) {
            if (std::find(geometries.begin(), geometries.end(), voxel_grid_sdf) ==
                geometries.end()) {
                geometries.push_back(voxel_grid_sdf);
            }
        } else {
            if (std::find(geometries.begin(), geometries.end(), mesh_sdf_sphere) ==
                geometries.end()) {
                geometries.push_back(mesh_sdf_sphere);
            }
            if (std::find(geometries.begin(), geometries.end(), pcd_gp_points) ==
                geometries.end()) {
                geometries.push_back(pcd_gp_points);
            }
        }

        visualizer->AddGeometries(geometries);
    }

    std::string
    GetBinFileName() {
        std::string bin_file = fmt::format("sdf_mapping_3d_{}.bin", type_name<Dtype>());
        bin_file = test_output_folder / bin_file;
        return bin_file;
    }

    void
    WriteSdfMappingBin() {
        ERL_BLOCK_TIMER_MSG("WriteSdfMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<SdfMapping>::Write(bin_file, sdf_map),
            "Failed to write to file: {}",
            bin_file);
    }

    void
    ReadSdfMappingBin(SdfMapping &sdf_mapping_read) {
        ERL_BLOCK_TIMER_MSG("ReadSdfMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<SdfMapping>::Read(bin_file, &sdf_mapping_read),
            "Failed to read from file: {}",
            bin_file);
    }

    void
    TestIo() {
        ERL_BLOCK_TIMER_MSG("IO");
        WriteSdfMappingBin();

        auto surface_mapping_read =
            std::make_shared<SurfaceMapping>(std::make_shared<typename SurfaceMapping::Setting>());
        SdfMapping sdf_mapping_read(
            std::make_shared<typename SdfMapping::Setting>(),
            surface_mapping_read);
        ReadSdfMappingBin(sdf_mapping_read);

        ERL_ASSERTM(*sdf_map == sdf_mapping_read, "sdf_map != sdf_mapping_read");
    }

    void
    RangesToPoints() {
        ERL_ASSERT_PTR(range_sensor_frame);
        if (raw_data_is_points) { return; }
        range_sensor_frame->UpdateRanges(rotation_frame, translation_frame, frame_ranges);
        frame_points = Eigen::Map<const Matrix3X>(
            range_sensor_frame->GetHitPointsFrame().data()->data(),
            3,
            range_sensor_frame->GetNumHitRays());
    }

    void
    ApplyScanStride() {
        if (options.scan_stride.size() == 0) { return; }
        using namespace erl::common;
        if (options.random_scan_downsample) {
            std::uniform_int_distribution<long> distribution(0);
            if (mapping_uses_points) {
                // for points, randomly downsample the points
                long n_points = frame_points.cols() / options.scan_stride.prod();
                ERL_ASSERT_GT(n_points, 0);
                for (long i = 0; i < n_points; ++i) {
                    long idx = distribution(g_random_engine);
                    idx = idx % (frame_points.cols() - i) + i;
                    Vector3 tmp = frame_points.col(i);
                    frame_points.col(i) = frame_points.col(idx);
                    frame_points.col(idx) = tmp;
                }
                frame_points.conservativeResize(3, n_points);
                return;
            }
            // for ranges, randomly set ranges to 0
            long n_ranges = frame_ranges.size() / options.scan_stride.prod();
            ERL_ASSERT_GT(n_ranges, 0);
            const long n_zeros = frame_ranges.size() - n_ranges;
            const auto &frame_shape = range_sensor_frame->GetFrameShape();
            const long s = frame_shape.first * frame_shape.second;
            absl::flat_hash_set<long> zero_indices;
            zero_indices.reserve(n_zeros);
            while (static_cast<long>(zero_indices.size()) < n_zeros) {
                const long row = distribution(g_random_engine) % frame_ranges.rows();
                const long col = distribution(g_random_engine) % frame_ranges.cols();
                const long idx = row * s + col;
                if (!zero_indices.insert(idx).second) { continue; }
                frame_ranges.data()[idx] = 0;
            }
        } else {
            const long row_stride = options.scan_stride[0];
            const long col_stride =
                options.scan_stride.size() == 2 ? options.scan_stride[1] : row_stride;
            const auto [rows, cols] = range_sensor_frame->GetFrameShape();

            if (mapping_uses_points) {
                // for points, we assume the points are stored in the order of a 2D range array
                if (options.scan_stride.size() == 1) {
                    // downsample the points by the linear stride
                    const long stride = options.scan_stride[0];
                    long n_points = (frame_points.cols() + stride - 1) / stride;
                    Matrix3X new_points(3, n_points);
                    for (long i = 0, ii = 0; i < n_points; ++i, ii += stride) {
                        new_points.col(i) = frame_points.col(ii);
                    }
                    return;
                }
                // downsample the points by the per axis stride
                const long new_rows = (rows + row_stride - 1) / row_stride;
                const long new_cols = (cols + col_stride - 1) / col_stride;
                Matrix3X new_points(3, new_rows * new_cols);
                const long s_row = (raw_data_is_row_major ? cols : 1) * row_stride;
                const long s_col = (raw_data_is_row_major ? 1 : rows) * col_stride;
                for (long j = 0, jj = 0, js = 0; j < new_cols; ++j, jj += new_rows, js += s_col) {
                    for (long i = 0, is = 0; i < new_rows; ++i, is += s_row) {
                        new_points.col(jj + i) = frame_points.col(is + js);
                    }
                }
                frame_points = new_points;
                return;
            }

            // for ranges, we downsample the range image by setting pixels to 0
            for (long j = 0; j < cols; j += col_stride) {
                for (long i = 0; i < rows; i += row_stride) { frame_ranges(i, j) = 0; }
            }
        }
    }

    void
    LoadDataFromCowAndLady() {
        const auto frame = (*cow_and_lady)[wp_idx];
        rotation_frame = frame.rotation.template cast<Dtype>();
        translation_frame = frame.translation.template cast<Dtype>();
        std::tie(rotation_sensor, translation_sensor) =
            erl::geometry::CameraBase3D<Dtype>::ComputeCameraPose(
                rotation_frame,
                translation_frame);
        frame_ranges = frame.depth.template cast<Dtype>();
        ranges_img = frame.depth_jet;
    }

    void
    LoadDataFromMeshDataset() {
        std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
        std::tie(rotation_frame, translation_frame) =
            range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);

        frame_ranges = range_sensor->Scan(rotation_sensor, translation_sensor);
        ranges_img = ConvertMatrixToImage(frame_ranges, true);
        if (is_lidar) {                           // azimuth: down, elevation: right
            ranges_img = ranges_img.t();          // elevation: down, azimuth: right
            cv::flip(ranges_img, ranges_img, 0);  // elevation: up, azimuth: right
        }
    }

    void
    LoadDataFromNewerCollege() {
        const auto frame = (*newer_college)[wp_idx];
        rotation_frame = frame.rotation.template cast<Dtype>();
        translation_frame = frame.translation.template cast<Dtype>();
        rotation_sensor = rotation_frame;
        translation_sensor = translation_frame;

        frame_points = frame.points.template cast<Dtype>();
        frame_ranges = frame.GetRangeMatrix().template cast<Dtype>();
        ranges_img = ConvertMatrixToImage(frame_ranges, true);
        ranges_img = ranges_img.t();
        cv::flip(ranges_img, ranges_img, 0);
    }

    void
    LoadData() {
        ERL_BLOCK_TIMER_MSG("data loading");
        switch (dataset_type) {
            case DataSetType::CowAndLady:
                LoadDataFromCowAndLady();
                break;
            case DataSetType::Mesh:
                LoadDataFromMeshDataset();
                break;
            case DataSetType::NewerCollege:
                LoadDataFromNewerCollege();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }
        RangesToPoints();
        ApplyScanStride();
    }

    void
    VisualizeWholeMap() {
        options.test_z = static_cast<Dtype>(vis_setting->z);
        positions_test_whole_map.row(2).setConstant(options.test_z);
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

        ConvertToVoxelGrid(img_sdf, positions_test_whole_map, voxel_grid_sdf);

        Eigen::VectorXb in_free_space;
        ASSERT_TRUE(surf_map->IsInFreeSpace(positions_test_whole_map, in_free_space));
        cv::Mat img_surf_mapping_sign = ConvertVectorToImage<Dtype>(
            whole_map_xs,
            whole_map_ys,
            in_free_space.cast<Dtype>(),
            false);

        Dtype resize_scale = options.image_resize_scale;
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
    VisualizeSdfSphere() {
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
        visualizer->GetVisualizer()->UpdateGeometry(mesh_sdf_sphere);

        auto &gp = sdf_map->GetUsedGps()[0][0];
        if (gp != nullptr) {
            pcd_gp_points->Clear();
            auto &buf = gp->edf_gp->GetTrainBuffer();
            for (long i = 0; i < buf.num_samples; ++i) {
                pcd_gp_points->points_.emplace_back(buf.x.col(i).template cast<double>());
            }
            pcd_gp_points->PaintUniformColor({1.0, 0.5, 0.0});  // orange
            visualizer->GetVisualizer()->UpdateGeometry(pcd_gp_points);
        }
    }

    bool
    UpdateSurfaceMap() {
        LoadData();

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

    bool
    UpdateSdfMap() {
        ERL_BLOCK_TIMER_MSG_TIME("sdf_map.Update", sdf_map_update_dt);

        surf_map_updated = UpdateSurfaceMap();
        ERL_WARN_COND(!surf_map_updated, "Sdf mapping update failed");
        if (!surf_map_updated) { return false; }

        const double time_budget_us = 1e6 / sdf_map_setting->update_hz;  // us
        sdf_map_updated = sdf_map->UpdateGpSdf(time_budget_us - surf_map_update_dt * 1000);
        return sdf_map_updated;
    }

    void
    PredSdfFollow() {
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
        gp_indices.setConstant(-1);
        if (test_success && !used_gps.empty()) {
            for (long i = 0; i < gp_indices.size(); ++i) {
                auto &gp = VEC_ACCESS(used_gps, i)[0];
                if (gp == nullptr) { continue; }
                auto key = reinterpret_cast<uint64_t>(gp.get());
                auto [it, inserted] = gp_index_map.try_emplace(key, gp_index_map.size());
                gp_indices[i] = it->second;
            }
            ERL_INFO(
                "{} unique GPs used for {} test points",
                gp_index_map.size(),
                gp_indices.size());
        }
    }

    void
    UpdateVisualization() {
        ERL_BLOCK_TIMER_MSG("UpdateVisualization");

        if (surf_map_updated) { surf_map_update_fps = 1000.0 / surf_map_update_dt; }
        if (sdf_map_updated) { sdf_map_update_fps = 1000.0 / sdf_map_update_dt; }
        if (test_success) { test_fps = 1000.0 / test_dt; }
        if (gui_dt > 0) { gui_fps = 1000.0 / gui_dt; }

        fps_data.col(wp_idx / options.seq_stride) << surf_map_update_fps, sdf_map_update_fps,
            test_fps, gui_fps;
        wp_idx += options.seq_stride;

        constexpr int kFontFace = cv::FONT_HERSHEY_PLAIN;
        constexpr double kFontScale = 1.5;
        const cv::Scalar kTextColor = {255, 255, 255, 255};
        constexpr int kFontThickness = 2;

        // visualize sensor data
        Dtype resize_scale = options.image_resize_scale;
        resize_scale = std::min(resize_scale, 600.0f / static_cast<Dtype>(ranges_img.cols));
        resize_scale = std::min(resize_scale, 600.0f / static_cast<Dtype>(ranges_img.rows));
        cv::resize(ranges_img, ranges_img, cv::Size(), resize_scale, resize_scale);
        cv::putText(
            ranges_img,
            fmt::format("frame {}", wp_idx),
            cv::Point(10, 30),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            ranges_img,
            fmt::format("surf_map.update: {:.2f} fps", surf_map_update_fps),
            cv::Point(10, 60),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            ranges_img,
            fmt::format("sdf_map.update: {:.2f} fps", sdf_map_update_fps),
            cv::Point(10, 90),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            ranges_img,
            fmt::format("sdf_map.test: {:.2f} fps", test_fps),
            cv::Point(10, 120),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            ranges_img,
            fmt::format("gui.update: {:.2f} fps", gui_fps),
            cv::Point(10, 150),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);

        // visualize prediction
        cv::Mat img_sdf =
            ConvertVectorToImage(options.test_xs, options.test_ys, sdf_pred_follow, true);
        ConvertToVoxelGrid(img_sdf, positions_test_follow, voxel_grid_sdf);
        cv::Mat img_sdf_sign = ConvertVectorToImage<Dtype>(
            options.test_xs,
            options.test_ys,
            (sdf_pred_follow.array() > 0.0).template cast<Dtype>(),
            true);
        cv::Mat img_gp_indices = ConvertVectorToImage<Dtype>(
            options.test_xs,
            options.test_ys,
            gp_indices.cast<Dtype>(),
            true);

        resize_scale = options.image_resize_scale;
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(img_sdf.cols));
        resize_scale = std::min(resize_scale, 1920.0f / static_cast<Dtype>(img_sdf.rows));

        cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_gp_indices, img_gp_indices, cv::Size(), resize_scale, resize_scale);

        cv::imshow("ranges", ranges_img);
        cv::imshow("sdf", img_sdf);
        cv::imshow("sdf_sign", img_sdf_sign);
        cv::imshow("gp_indices", img_gp_indices);
        cv::waitKey(1);

        // update open3d objects
        auto vis = visualizer->GetVisualizer();
        /// update the sensor mesh
        const Eigen::Matrix4d last_pose_inv = last_pose.inverse();
        Eigen::Matrix4d cur_pose = Eigen::Matrix4d::Identity();
        cur_pose.topLeftCorner<3, 3>() = rotation_sensor.template cast<double>();
        cur_pose.topRightCorner<3, 1>() = translation_sensor.template cast<double>();
        const Eigen::Matrix4d delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose);
        mesh_sensor_xyz->Transform(delta_pose);
        if (std::find(geometries.begin(), geometries.end(), mesh_sensor) != geometries.end()) {
            vis->UpdateGeometry(mesh_sensor);
        }
        if (std::find(geometries.begin(), geometries.end(), mesh_sensor_xyz) != geometries.end()) {
            vis->UpdateGeometry(mesh_sensor_xyz);
        }
        /// update the trajectory line set
        if (std::find(geometries.begin(), geometries.end(), line_set_traj) != geometries.end()) {
            line_set_traj->points_.emplace_back(translation_sensor.template cast<double>());
            const size_t n_points = line_set_traj->points_.size();
            if (n_points >= 2) {
                line_set_traj->lines_.emplace_back(n_points - 2, n_points - 1);
                line_set_traj->colors_.emplace_back(0.0, 0.0, 0.0);  // black
            }
            vis->UpdateGeometry(line_set_traj);
        }
        /// update the observation point cloud
        if (std::find(geometries.begin(), geometries.end(), pcd_obs) != geometries.end()) {
            pcd_obs->points_.clear();
            pcd_obs->colors_.clear();
            const auto &hit_points = range_sensor_frame->GetHitPointsWorld();
            const auto n_points = (hit_points.size() + options.pcd_stride - 1) / options.pcd_stride;
            pcd_obs->points_.reserve(n_points);
            for (std::size_t i = 0; i < hit_points.size(); i += options.pcd_stride) {
                pcd_obs->points_.emplace_back(hit_points[i].template cast<double>());
            }
            pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});  // green
            vis->UpdateGeometry(pcd_obs);
        }

        /// update the surface point cloud and normals
        auto it1 = std::find(geometries.begin(), geometries.end(), pcd_surf_points);
        auto it2 = std::find(geometries.begin(), geometries.end(), line_set_surf_normals);
        if (const auto tree = surf_map->GetTree();
            (it1 != geometries.end() || it2 != geometries.end()) && tree != nullptr) {
            pcd_surf_points->points_.clear();
            line_set_surf_normals->points_.clear();
            line_set_surf_normals->lines_.clear();
            const auto &surf_data_buffer = surf_map->GetSurfaceDataBuffer();
            const auto &unused_indices = surf_map->GetUnusedSurfaceDataIndices();
            std::unordered_set<std::size_t> unused_set(
                unused_indices.begin(),
                unused_indices.end());
            std::size_t n_points = surf_data_buffer.size() - unused_set.size();
            n_points = (n_points + options.pcd_stride - 1) / options.pcd_stride;
            pcd_surf_points->points_.reserve(n_points);
            line_set_surf_normals->points_.reserve(n_points * 2);
            line_set_surf_normals->lines_.reserve(n_points);
            for (std::size_t i = 0; i < surf_data_buffer.size(); i += options.pcd_stride) {
                if (unused_set.count(i)) { continue; }
                const auto &surface_data = surf_data_buffer[i];

                const Vector3 &position = surface_data.position;
                const Vector3 &normal = surface_data.normal;
                ERL_ASSERTM(
                    std::abs(normal.norm() - 1.0) < 1.e-4,
                    "normal.norm() = {:.6f}",
                    normal.norm());
                pcd_surf_points->points_.emplace_back(position.template cast<double>());
                line_set_surf_normals->points_.emplace_back(position.template cast<double>());
                line_set_surf_normals->points_.emplace_back(
                    (position + 0.1 * normal).template cast<double>());
                line_set_surf_normals->lines_.emplace_back(
                    line_set_surf_normals->points_.size() - 2,
                    line_set_surf_normals->points_.size() - 1);
            }
            pcd_surf_points->PaintUniformColor({0.0, 0.0, 1.0});        // blue
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});  // red
            if (it1 != geometries.end()) { vis->UpdateGeometry(pcd_surf_points); }
            if (it2 != geometries.end()) { vis->UpdateGeometry(line_set_surf_normals); }
        }
        /// update the voxel_grid_sdf
        if (std::find(geometries.begin(), geometries.end(), voxel_grid_sdf) != geometries.end()) {
            vis->UpdateGeometry(voxel_grid_sdf);
        }
        /// update the gps line set
        if (std::find(geometries.begin(), geometries.end(), line_set_gps) != geometries.end()) {
            std::vector<uint64_t> inactive_gp_addr;
            auto &gps = sdf_map->GetGpMap();
            for (const auto &key: surf_map->GetChangedClusters()) {
                auto it_gp = gps.find(key);
                if (it_gp == gps.end()) { continue; }
                const auto &gp = it_gp->second;
                auto addr = reinterpret_cast<uint64_t>(gp.get());
                if (!gp->active) { inactive_gp_addr.push_back(addr); }
                auto [it, inserted] = line_set_gps_map.try_emplace(addr, LineSetInfo());
                if (!inserted) { continue; }
                auto box = line_set_gp_box;
                box.Translate(gp->position.template cast<double>());
                for (std::size_t i = 0; i < box.points_.size(); ++i) {
                    Eigen::Vector3l p = (box.points_[i] / 0.005).template cast<long>();
                    auto [it_p, inserted_p] =
                        gp_vertex_index_map.try_emplace(p, gp_vertex_index_map.size());
                    if (inserted_p) { line_set_gps->points_.emplace_back(box.points_[i]); }
                    it->second.first[i] = it_p->second;
                }
                for (std::size_t i = 0; i < box.lines_.size(); ++i) {
                    Eigen::Vector2i l = box.lines_[i];
                    l[0] = static_cast<int>(it->second.first[l[0]]);
                    l[1] = static_cast<int>(it->second.first[l[1]]);
                    auto [it_l, inserted_l] =
                        gp_edge_index_map.try_emplace(l, gp_edge_index_map.size());
                    if (inserted_l) { line_set_gps->lines_.emplace_back(l); }
                    it->second.second[i] = it_l->second;
                }
            }
            line_set_gps->colors_.clear();
            line_set_gps->PaintUniformColor({0.0, 1.0, 1.0});  // cyan
            for (const auto &addr: inactive_gp_addr) {
                auto it = line_set_gps_map.find(addr);
                ERL_ASSERT(it != line_set_gps_map.end());
                for (const auto &edge_idx: it->second.second) {
                    line_set_gps->colors_[edge_idx] = Eigen::Vector3d(0.5, 0.5, 0.5);  // gray
                }
            }
            vis->UpdateGeometry(line_set_gps);
        }
    }

    bool
    AnimationCallback(Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) {

        if (options.save_images) {
            vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx), false);
        }

        if (animation_ended) {
            // options.hold is true, so the window is not closed yet
            if (options.test_whole_map_at_end) {
                if (options.test_z != static_cast<Dtype>(vis_setting->z)) { VisualizeWholeMap(); }
            } else {
                if (position_test[0] != static_cast<Dtype>(vis_setting->x) ||
                    position_test[1] != static_cast<Dtype>(vis_setting->y) ||
                    position_test[2] != static_cast<Dtype>(vis_setting->z)) {
                    VisualizeSdfSphere();
                }
            }
            cv::waitKey(1);
            return false;
        }

        if (wp_idx >= max_wp_idx) {  // end of animation
            animation_ended = true;
            if (options.test_whole_map_at_end) {
                VisualizeWholeMap();
            } else {
                VisualizeSdfSphere();
            }
            if (options.save_images) {
                vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx + 1), true);
            }
            if (!options.hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }

        ERL_INFO("wp_idx: {}", wp_idx);
        {
            ERL_BLOCK_TIMER_MSG_TIME("gui_update", gui_dt);
            if (UpdateSdfMap()) { PredSdfFollow(); }
            if (wp_idx % options.vis_stride == 0) { UpdateVisualization(); }
        }
        ERL_INFO("gui_update (fps): {:.2f}", 1000.0 / gui_dt);
        return true;
    }
};
