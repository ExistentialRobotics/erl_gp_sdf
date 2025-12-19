#pragma once

#include "utils.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"
#include "erl_geometry/newer_college.hpp"
#include "erl_geometry/open3d_helper.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/replica_rgbd.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_gp_sdf/surface_data_manager.hpp"

#include <absl/container/flat_hash_set.h>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

enum class DataSetType {
    CowAndLady = 1,
    Mesh = 2,
    NewerCollege = 3,
    ReplicaRgbd = 4,
};
ERL_REFLECT_ENUM_SCHEMA(
    DataSetType,
    4,
    ERL_REFLECT_ENUM_MEMBER("cow_and_lady", DataSetType::CowAndLady),
    ERL_REFLECT_ENUM_MEMBER("mesh", DataSetType::Mesh),
    ERL_REFLECT_ENUM_MEMBER("newer_college", DataSetType::NewerCollege),
    ERL_REFLECT_ENUM_MEMBER("replica_rgbd", DataSetType::ReplicaRgbd));
ERL_PARSE_ENUM(DataSetType, 4);

template<typename Dtype>
struct OptionsForTestMapping3D : public erl::common::Yamlable<OptionsForTestMapping3D<Dtype>> {
    inline static const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
    inline static const std::filesystem::path kDataDir = kProjectRootDir / "data";
    inline static const std::filesystem::path kConfigDir = kProjectRootDir / "config";

    uint64_t random_seed = 0;

    DataSetType dataset_type = DataSetType::CowAndLady;
    std::string cow_and_lady_dir;
    std::string newer_college_dir;
    std::string replica_rgbd_dir;
    std::string replica_rgbd_scene_name;
    std::string mesh_file = kDataDir / "replica-hotel-0.ply";       // mesh file
    std::string traj_file = kDataDir / "replica-hotel-0-traj.txt";  // trajectory file
    std::string o3d_view_status_file;
    bool add_sensor_noise = false;
    Dtype sensor_noise_std = 0.01f;

    std::string sensor_frame_type = type_name<erl::geometry::LidarFrame3D<Dtype>>();
    std::string sensor_frame_config_file = kConfigDir / "sensors" / "lidar_frame_3d_360.yaml";
    long start_wp_idx = 0;
    long end_wp_idx = -1;  // -1 means all waypoints
    long seq_stride = 1;
    bool exhausting = false;              // update until cached data is exhausted
    Eigen::VectorXl scan_stride;          // linear scan stride or per axis scan stride
    bool random_scan_downsample = false;  // use random downsample the scan points
    long vis_stride = 1;                  // visualization stride
    std::size_t pcd_stride = 10;          // stride of pcd for visualization
    std::vector<std::string> show_geometries = {
        "all",
        "-pcd_obs",
        "-pcd_surf_points",
        "-line_set_surf_normals",
        "-line_set_clusters",
    };
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
    Dtype surf_normal_scale = 0.25;      // surface normal visualization scale
    bool save_images = false;
    bool test_io = false;
    bool hold = false;

    std::string mapping_bin_file;
    bool load_mapping_bin = false;

    ERL_REFLECT_SCHEMA(
        OptionsForTestMapping3D,
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, random_seed),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, dataset_type),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, cow_and_lady_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, newer_college_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, replica_rgbd_dir),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, replica_rgbd_scene_name),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, mesh_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, traj_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, o3d_view_status_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, add_sensor_noise),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, sensor_noise_std),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, sensor_frame_type),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, sensor_frame_config_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, start_wp_idx),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, end_wp_idx),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, seq_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, exhausting),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, scan_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, random_scan_downsample),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, vis_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, pcd_stride),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, show_geometries),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_res),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_x_min),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_x_max),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_y_min),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_y_max),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_z),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_xs),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_ys),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_whole_map_at_end),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, image_resize_scale),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, surf_normal_scale),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, save_images),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, test_io),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, hold),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, mapping_bin_file),
        ERL_REFLECT_MEMBER(OptionsForTestMapping3D, load_mapping_bin));

    bool
    PostDeserialization() override {
        if (scan_stride.size() > 0) {
            ERL_ASSERT_LE(scan_stride.size(), 2);
            for (long i = 0; i < scan_stride.size(); ++i) { ERL_ASSERT_POS_GT(scan_stride[i], 0); }
        }

        if (show_geometries.size() == 1) {
            show_geometries = erl::common::SplitString(show_geometries[0], ',');
        }

        switch (dataset_type) {
            case DataSetType::CowAndLady:
                ERL_ASSERTM(
                    !cow_and_lady_dir.empty(),
                    "Please provide the Cow and Lady dataset directory via --cow_and_lady_dir");
                break;
            case DataSetType::Mesh:
                ERL_ASSERTM(!mesh_file.empty(), "Please provide the mesh file via --mesh_file");
                ERL_ASSERTM(
                    !traj_file.empty(),
                    "Please provide the trajectory file via --traj_file");
                break;
            case DataSetType::NewerCollege:
                ERL_ASSERTM(
                    !newer_college_dir.empty(),
                    "Please provide the Newer College dataset directory via --newer_college_dir");
                break;
            case DataSetType::ReplicaRgbd:
                ERL_ASSERTM(
                    !replica_rgbd_dir.empty(),
                    "Please provide the Replica RGBD dataset directory via --replica_rgbd_dir");
                ERL_ASSERTM(
                    !replica_rgbd_scene_name.empty(),
                    "Please provide the Replica RGBD scene name via --replica_rgbd_scene_name");
                break;
            default:
                ERL_FATAL("Unknown dataset type.");
        }

        if (load_mapping_bin) {
            ERL_ASSERTM(
                !mapping_bin_file.empty(),
                "Please provide the SDF mapping bin file via --mapping_bin_file");
        }
        return true;
    }
};

template<typename Dtype, typename MappingType>
struct TestMapping3D {
    using DepthFrame = erl::geometry::DepthFrame3D<Dtype>;
    using LidarFrame = erl::geometry::LidarFrame3D<Dtype>;
    using RangeSensorFrame = erl::geometry::RangeSensorFrame3D<Dtype>;
    using DepthCamera = erl::geometry::DepthCamera3D<Dtype>;
    using Lidar = erl::geometry::Lidar3D<Dtype>;
    using RangeSensor = erl::geometry::RangeSensor3D<Dtype>;
    using CowAndLady = erl::geometry::CowAndLady;
    using NewerCollege = erl::geometry::NewerCollege;
    using ReplicaRgbd = erl::geometry::ReplicaRgbd;
    using Open3dVisualizerWrapper = erl::geometry::Open3dVisualizerWrapper;
    using Trajectory = erl::geometry::Trajectory<Dtype>;
    using SurfDataManager = erl::gp_sdf::SurfaceDataManager<Dtype, 3>;
    using SurfDataBuffer = typename SurfDataManager::DataBuffer;

    using VectorX = Eigen::VectorX<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using Matrix4X = Eigen::Matrix4X<Dtype>;
    using Matrix6X = Eigen::Matrix<Dtype, 6, Eigen::Dynamic>;

    std::shared_ptr<MappingType> mapping = nullptr;

    // datasets

    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::shared_ptr<NewerCollege> newer_college = nullptr;
    std::shared_ptr<ReplicaRgbd> replica_rgbd = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Matrix3X gt_surface_points;
    long max_wp_idx = 0;
    long wp_idx = 0;
    bool mapping_uses_points = false;  // should be set externally
    bool raw_data_is_points = false;
    bool raw_data_is_row_major = false;
    MatrixX frame_ranges;
    Matrix3X frame_points;

    // sensor

    std::shared_ptr<typename DepthFrame::Setting> depth_frame_setting = nullptr;
    std::shared_ptr<typename LidarFrame::Setting> lidar_frame_setting = nullptr;
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
    std::shared_ptr<open3d::geometry::LineSet> line_set_traj = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_obs = nullptr;
    Eigen::Matrix4d last_sensor_pose = Eigen::Matrix4d::Identity();

    // for surface mapping

    std::shared_ptr<open3d::geometry::PointCloud> pcd_surf_points = nullptr;
    std::shared_ptr<open3d::geometry::LineSet> line_set_surf_normals = nullptr;
    const SurfDataBuffer *surf_data_buffer = nullptr;
    const std::vector<std::size_t> *unused_surf_data_indices = nullptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_surf = nullptr;
    std::vector<Vector3> mesh_surf_vertices;
    std::vector<Eigen::Vector3i> mesh_surf_faces;

    // for field prediction

    std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_pred = nullptr;
    std::shared_ptr<open3d::geometry::LineSet> line_set_clusters = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_cluster_samples = nullptr;
    open3d::geometry::LineSet line_set_cluster_box;
    using LineSetInfo = std::pair<std::array<std::size_t, 8>, std::array<std::size_t, 12>>;
    absl::flat_hash_map<uint64_t, LineSetInfo> line_set_clusters_map;
    absl::flat_hash_map<Eigen::Vector3l, std::size_t> cluster_vertex_index_map;
    absl::flat_hash_map<Eigen::Vector2i, std::size_t> cluster_edge_index_map;
    std::vector<uint64_t> inactive_cluster_keys;

    // opencv data structures

    cv::Mat ranges_img;
    std::vector<std::string> ranges_img_texts;

    // test data

    Vector3 map_min, map_max;
    Matrix3 grid_rotation = Matrix3::Identity();
    Vector3 grid_translation = Vector3::Zero();
    Vector3 position_test;
    Matrix3X positions_test_follow_org;
    Matrix3X positions_test_follow;
    long whole_map_xs = 0;
    long whole_map_ys = 0;
    Matrix3X positions_test_whole_map;

    // output folders

    std::filesystem::path test_output_folder;
    std::filesystem::path img_dir;

    // logging
    bool animation_ended = false;
    Eigen::VectorXl cluster_indices;
    Eigen::MatrixXd fps_data;
    double gui_dt = 0;

private:
    std::shared_ptr<OptionsForTestMapping3D<Dtype>> options = nullptr;

protected:
    Dtype scaling = 1.0;
    Dtype cluster_half_size = 0.01;

public:
    TestMapping3D(
        const int argc,
        char *argv[],
        std::shared_ptr<OptionsForTestMapping3D<Dtype>> options_in)
        : options(std::move(options_in)) {
        ERL_ASSERT(options->FromCommandLine(argc, argv));
        erl::common::SetGlobalRandomSeed(options->random_seed);
    }

    TestMapping3D(const TestMapping3D &) = delete;
    TestMapping3D &
    operator=(const TestMapping3D &) = delete;
    TestMapping3D(TestMapping3D &&) = delete;
    TestMapping3D &
    operator=(TestMapping3D &&) = delete;

    virtual ~TestMapping3D() = default;

    virtual void
    Run() {
        Init();

        visualizer->SetAnimationCallback(
            [this](auto *wrapper, auto *vis) { return this->AnimationCallback(wrapper, vis); });
        if (!options->o3d_view_status_file.empty()) {
            visualizer->SetViewStatus(options->o3d_view_status_file);
        }

        if (options->load_mapping_bin) {
            ReadMappingBin(*mapping);
            animation_ended = true;
            visualizer->Show();
        } else {
            if (options->test_io) { TestIo(); }
            visualizer->Show();
            if (options->test_io) { TestIo(); }

            erl::common::SaveEigenMatrixToTextFile<double>(
                test_output_folder / "fps.csv",
                fps_data,
                erl::common::EigenTextFormat::kCsvFmt);

            if (!pcd_surf_points->IsEmpty()) {
                open3d::io::WritePointCloud(
                    test_output_folder / "surf_points.ply",
                    *pcd_surf_points);
            }
        }
    }

protected:
    virtual void
    Init() {
        vis_setting = std::make_shared<Open3dVisualizerWrapper::Setting>();

        PrepareDataset();
        PrepareOutputFolders();
        PrepareVisualizer();
    }

#pragma region dataset_prep

    void
    PrepareCowAndLady() {
        // dataset
        cow_and_lady = std::make_shared<CowAndLady>(options->cow_and_lady_dir);
        max_wp_idx = cow_and_lady->Size();
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LE(options->end_wp_idx, max_wp_idx);
        raw_data_is_points = false;
        raw_data_is_row_major = false;
        // sensor
        depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = CowAndLady::kCameraCy;
        range_sensor_frame = std::make_shared<DepthFrame>(depth_frame_setting);
        // open3d
        auto pcd = cow_and_lady->GetGroundTruthPointCloud();
        gt_scene = pcd;
        gt_surface_points.resize(3, pcd->points_.size());
        for (size_t i = 0; i < pcd->points_.size(); ++i) {
            gt_surface_points.col(i) = pcd->points_[i].cast<Dtype>();
        }
        // test data
        map_min = cow_and_lady->GetMapMin().cast<Dtype>();
        map_max = cow_and_lady->GetMapMax().cast<Dtype>();
        if (options->test_x_min == options->test_x_max ||
            options->test_y_min == options->test_y_max) {
            options->test_x_min = map_min[0];
            options->test_x_max = map_max[0];
            options->test_y_min = map_min[1];
            options->test_y_max = map_max[1];
        }
    }

    void
    PrepareMeshDataset() {
        // dataset
        const auto mesh = open3d::io::CreateMeshFromFile(options->mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options->mesh_file);
        poses = Trajectory::LoadSe3(options->traj_file, false);
        max_wp_idx = static_cast<long>(poses.size());
        raw_data_is_points = false;
        raw_data_is_row_major = false;
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LE(options->end_wp_idx, max_wp_idx);
        // sensor
        if (options->sensor_frame_type == type_name<LidarFrame>()) {
            lidar_frame_setting = std::make_shared<typename LidarFrame::Setting>();
            ASSERT_TRUE(lidar_frame_setting->FromYamlFile(options->sensor_frame_config_file));
            ERL_INFO("Lidar frame setting: \n{}", lidar_frame_setting->AsYamlString());
            const auto lidar_setting = std::make_shared<typename Lidar::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            ERL_INFO("Lidar setting: \n{}", lidar_setting->AsYamlString());
            range_sensor = std::make_shared<Lidar>(lidar_setting);
            is_lidar = true;
            range_sensor_frame = std::make_shared<LidarFrame>(lidar_frame_setting);
        } else if (options->sensor_frame_type == type_name<DepthFrame>()) {
            depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
            ASSERT_TRUE(depth_frame_setting->FromYamlFile(options->sensor_frame_config_file));
            ERL_INFO("Depth frame setting: \n{}", depth_frame_setting->AsYamlString());
            auto depth_camera_setting = std::make_shared<typename DepthCamera::Setting>(
                depth_frame_setting->camera_intrinsic);
            ERL_INFO("Depth camera setting: \n{}", depth_camera_setting->AsYamlString());
            range_sensor = std::make_shared<DepthCamera>(depth_camera_setting);
            is_lidar = false;
            range_sensor_frame = std::make_shared<DepthFrame>(depth_frame_setting);
        } else {
            ERL_FATAL("Unknown sensor_frame_type: {}", options->sensor_frame_type);
        }
        range_sensor->AddMesh(options->mesh_file);
        // open3d
        gt_scene = mesh;
        gt_surface_points.resize(3, mesh->vertices_.size());
        for (size_t i = 0; i < mesh->vertices_.size(); ++i) {
            gt_surface_points.col(i) = mesh->vertices_[i].template cast<Dtype>();
        }
        // test data
        map_min = mesh->GetMinBound().template cast<Dtype>();
        map_max = mesh->GetMaxBound().template cast<Dtype>();
        if (options->test_x_min == options->test_x_max ||
            options->test_y_min == options->test_y_max) {
            options->test_x_min = map_min[0];
            options->test_x_max = map_max[0];
            options->test_y_min = map_min[1];
            options->test_y_max = map_max[1];
        }
    }

    void
    PrepareNewerCollege() {
        // dataset
        newer_college = std::make_shared<NewerCollege>(options->newer_college_dir);
        max_wp_idx = erl::geometry::NewerCollege::Size();
        is_lidar = true;
        raw_data_is_points = true;
        raw_data_is_row_major = true;
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LE(options->end_wp_idx, max_wp_idx);
        // sensor
        lidar_frame_setting = std::make_shared<typename LidarFrame::Setting>();
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
        gt_surface_points.resize(3, pcd->points_.size());
        for (size_t i = 0; i < pcd->points_.size(); ++i) {
            gt_surface_points.col(i) = pcd->points_[i].cast<Dtype>();
        }
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

        if (options->test_x_min == options->test_x_max ||
            options->test_y_min == options->test_y_max) {
            options->test_x_min = -box_size[0] / 2;
            options->test_x_max = box_size[0] / 2;
            options->test_y_min = -box_size[1] / 2;
            options->test_y_max = box_size[1] / 2;
        }
    }

    void
    PrepareReplicaRgbd() {
        // dataset
        replica_rgbd = std::make_shared<ReplicaRgbd>(
            options->replica_rgbd_dir,
            options->replica_rgbd_scene_name);
        max_wp_idx = replica_rgbd->Size();
        ERL_ASSERT_LT(options->start_wp_idx, max_wp_idx);
        ERL_ASSERT_POS_LE(options->end_wp_idx, max_wp_idx);
        raw_data_is_points = false;
        raw_data_is_row_major = false;
        // sensor
        depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = ReplicaRgbd::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = ReplicaRgbd::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = ReplicaRgbd::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = ReplicaRgbd::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = ReplicaRgbd::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = ReplicaRgbd::kCameraCy;
        range_sensor_frame = std::make_shared<DepthFrame>(depth_frame_setting);
        // open3d
        auto mesh = open3d::io::CreateMeshFromFile(replica_rgbd->GetMeshPath());
        gt_scene = mesh;
        gt_surface_points.resize(3, mesh->vertices_.size());
        for (size_t i = 0; i < mesh->vertices_.size(); ++i) {
            gt_surface_points.col(i) = mesh->vertices_[i].cast<Dtype>();
        }
        // test data
        map_min = replica_rgbd->GetMapMin().cast<Dtype>();
        map_max = replica_rgbd->GetMapMax().cast<Dtype>();
        if (options->test_x_min == options->test_x_max ||
            options->test_y_min == options->test_y_max) {
            options->test_x_min = map_min[0];
            options->test_x_max = map_max[0];
            options->test_y_min = map_min[1];
            options->test_y_max = map_max[1];
        }
    }

    virtual void
    PrepareDataset() {
        // 1. load dataset
        // 2. create range sensor frame for converting data to points
        // 3. set map_min, map_max, max_wp_idx
        // 4. initialize positions_test_org, positions_test and fps_data

        switch (options->dataset_type) {
            case DataSetType::CowAndLady:
                PrepareCowAndLady();
                break;
            case DataSetType::Mesh:
                PrepareMeshDataset();
                break;
            case DataSetType::NewerCollege:
                PrepareNewerCollege();
                break;
            case DataSetType::ReplicaRgbd:
                PrepareReplicaRgbd();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }

        wp_idx = std::max(options->start_wp_idx, 0l);
        if (options->end_wp_idx > 0) { max_wp_idx = options->end_wp_idx; }
        options->test_z = (map_min[2] + map_max[2]) / 2;

        vis_setting->z = options->test_z;
        position_test[0] = vis_setting->x;
        position_test[1] = vis_setting->y;
        position_test[2] = options->test_z;

        positions_test_follow_org.resize(3, options->test_xs * options->test_ys);
        const Vector3 offset(
            static_cast<Dtype>(-0.5f) * options->test_res * static_cast<Dtype>(options->test_xs),
            static_cast<Dtype>(-0.5f) * options->test_res * static_cast<Dtype>(options->test_ys),
            0.0);
        // x: down, y: right
        long idx = 0;
        for (long j = 0; j < options->test_ys; ++j) {
            const Dtype y = static_cast<Dtype>(j) * options->test_res + offset[1];
            for (long i = 0; i < options->test_xs; ++i) {
                const Dtype x = static_cast<Dtype>(i) * options->test_res + offset[0];
                positions_test_follow_org.col(idx++) << x, y, offset[2];
            }
        }
        positions_test_follow.resize(3, positions_test_follow_org.cols());

        if (options->test_x_min == options->test_x_max ||
            options->test_y_min == options->test_y_max) {
            ERL_INFO("Map boundary is not fully defined, using surface mapping boundary.");
            options->test_x_min = map_min[0];
            options->test_x_max = map_max[0];
            options->test_y_min = map_min[1];
            options->test_y_max = map_max[1];
        }
        const erl::common::GridMapInfo2D<Dtype> grid_map_info(
            Eigen::Vector2<Dtype>(options->test_x_min, options->test_y_min),
            Eigen::Vector2<Dtype>(options->test_x_max, options->test_y_max),
            Eigen::Vector2<Dtype>(options->test_res, options->test_res),
            Eigen::Vector2i(0, 0));
        whole_map_xs = grid_map_info.Shape(0);
        whole_map_ys = grid_map_info.Shape(1);
        positions_test_whole_map.resize(3, grid_map_info.Size());
        positions_test_whole_map.topRows(2) =
            grid_map_info.GenerateMeterCoordinates(false).template cast<Dtype>();
        positions_test_whole_map =
            (grid_rotation * positions_test_whole_map).colwise() + grid_translation;

        cluster_indices.resize(positions_test_follow_org.cols());
    }

#pragma endregion

    void
    PrepareOutputFolders() {
        GTEST_PREPARE_OUTPUT_DIR();
        test_output_folder = test_output_dir;
        img_dir = test_output_folder / "images";
        std::filesystem::create_directory(img_dir);
        vis_setting->window_name = test_info->name();
    }

    virtual void
    PrepareVisualizer() {
        vis_setting->mesh_show_back_face = false;
        vis_setting->translate_step =
            options->dataset_type == DataSetType::NewerCollege ? 0.1 : 0.01;
        vis_setting->x = (options->test_x_min + options->test_x_max) / 2;
        vis_setting->y = (options->test_y_min + options->test_y_max) / 2;
        vis_setting->z = options->test_z;
        visualizer = std::make_shared<Open3dVisualizerWrapper>(vis_setting);

        switch (options->dataset_type) {
            case DataSetType::CowAndLady:
                mesh_sensor = erl::geometry::CreateCameraMesh(
                    CowAndLady::kImageWidth,
                    CowAndLady::kImageHeight,
                    CowAndLady::kCameraFx);
                break;
            case DataSetType::Mesh:
                if (options->sensor_frame_type == type_name<DepthFrame>()) {
                    mesh_sensor = erl::geometry::CreateCameraMesh(
                        depth_frame_setting->camera_intrinsic.image_width,
                        depth_frame_setting->camera_intrinsic.image_height,
                        depth_frame_setting->camera_intrinsic.camera_fx);
                } else {  // lidar
                    mesh_sensor = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1);
                }
                break;
            case DataSetType::NewerCollege:  // lidar
                mesh_sensor = open3d::geometry::TriangleMesh::CreateCoordinateFrame(1);
                break;
            case DataSetType::ReplicaRgbd:
                mesh_sensor = erl::geometry::CreateCameraMesh(
                    ReplicaRgbd::kImageWidth,
                    ReplicaRgbd::kImageHeight,
                    ReplicaRgbd::kCameraFx);
                break;
            default:
                break;
        }

        line_set_traj = std::make_shared<open3d::geometry::LineSet>();
        pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
        pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
        line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
        mesh_surf = std::make_shared<open3d::geometry::TriangleMesh>();

        voxel_grid_pred = std::make_shared<open3d::geometry::VoxelGrid>();
        voxel_grid_pred->origin_.setZero();
        voxel_grid_pred->voxel_size_ = options->test_res * 1.42f;

        line_set_clusters = std::make_shared<open3d::geometry::LineSet>();
        const double hs = cluster_half_size;
        line_set_cluster_box = *open3d::geometry::LineSet::CreateFromAxisAlignedBoundingBox(
            open3d::geometry::AxisAlignedBoundingBox(
                Eigen::Vector3d(-hs, -hs, -hs),
                Eigen::Vector3d(hs, hs, hs)));

        pcd_cluster_samples = std::make_shared<open3d::geometry::PointCloud>();

        absl::flat_hash_map<std::string, std::shared_ptr<open3d::geometry::Geometry3D>> geo_map;
        geo_map["gt_scene"] = gt_scene;
        geo_map["mesh_sensor"] = mesh_sensor;
        geo_map["line_set_traj"] = line_set_traj;
        geo_map["pcd_obs"] = pcd_obs;
        geo_map["pcd_surf_points"] = pcd_surf_points;
        geo_map["line_set_surf_normals"] = line_set_surf_normals;
        geo_map["mesh_surf"] = mesh_surf;
        geo_map["voxel_grid_pred"] = voxel_grid_pred;
        geo_map["line_set_clusters"] = line_set_clusters;
        std::vector<std::string> geo_names;
        geo_names.reserve(geo_map.size());
        for (const auto &[name, geo]: geo_map) { geo_names.push_back(name); }

        for (auto geo_name: options->show_geometries) {
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
                geometries.push_back(line_set_traj);
                geometries.push_back(pcd_obs);
                geometries.push_back(pcd_surf_points);
                geometries.push_back(line_set_surf_normals);
                geometries.push_back(mesh_surf);
                geometries.push_back(voxel_grid_pred);
                geometries.push_back(line_set_clusters);
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
        if (options->test_whole_map_at_end) {
            if (std::find(geometries.begin(), geometries.end(), voxel_grid_pred) ==
                geometries.end()) {
                geometries.push_back(voxel_grid_pred);
            }
        } else {
            if (std::find(geometries.begin(), geometries.end(), pcd_cluster_samples) ==
                geometries.end()) {
                geometries.push_back(pcd_cluster_samples);
            }
        }

        visualizer->AddGeometries(geometries);
    }

#pragma region data_loading

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
        if (options->scan_stride.size() == 0) { return; }
        using namespace erl::common;
        if (options->random_scan_downsample) {
            std::uniform_int_distribution<long> distribution(0);
            if (mapping_uses_points) {
                // for points, randomly downsample the points
                long n_points = frame_points.cols() / options->scan_stride.prod();
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
            long n_ranges = frame_ranges.size() / options->scan_stride.prod();
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
            const long row_stride = options->scan_stride[0];
            const long col_stride =
                options->scan_stride.size() == 2 ? options->scan_stride[1] : row_stride;
            const auto [rows, cols] = range_sensor_frame->GetFrameShape();

            if (mapping_uses_points) {
                // for points, we assume the points are stored in the order of a 2D range array
                if (options->scan_stride.size() == 1) {
                    // downsample the points by the linear stride
                    const long stride = options->scan_stride[0];
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
        // CowAndLady is real dataset, so we do not add noise here.
    }

    void
    LoadDataFromMeshDataset() {
        std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
        std::tie(rotation_frame, translation_frame) =
            range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);

        frame_ranges = range_sensor->Scan(
            rotation_sensor,
            translation_sensor,
            options->add_sensor_noise,
            options->sensor_noise_std);
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
        // Newer College is real dataset, so we do not add noise here.
    }

    void
    LoadDataFromReplicaRgbd() {
        const auto frame = (*replica_rgbd)[wp_idx];
        rotation_frame = frame.rotation.template cast<Dtype>();
        translation_frame = frame.translation.template cast<Dtype>();
        std::tie(rotation_sensor, translation_sensor) =
            erl::geometry::CameraBase3D<Dtype>::ComputeCameraPose(
                rotation_frame,
                translation_frame);
        frame_ranges = frame.depth.template cast<Dtype>();
        ranges_img = frame.depth_jet;
        AddSensorNoise();
    }

    void
    AddSensorNoise() {
        if (!options->add_sensor_noise) { return; }
        const long n_cols = frame_ranges.cols();
        std::vector<uint64_t> random_seeds(n_cols);
        for (long i = 0; i < n_cols; ++i) { random_seeds[i] = erl::common::g_random_engine(); }

#pragma omp parallel for default(none) shared(random_seeds, n_cols)
        for (long v = 0; v < n_cols; ++v) {
            std::mt19937_64 generator(random_seeds[v]);
            std::normal_distribution<Dtype> distribution(0, options->sensor_noise_std);
            for (long u = 0; u < frame_ranges.rows(); ++u) {
                Dtype &r = frame_ranges(u, v);
                if (r <= 0) { continue; }
                r += distribution(generator);
                if (r < 0) { r = 0; }
            }
        }
    }

    void
    LoadData() {
        const ERL_BLOCK_TIMER_MSG("data loading");
        switch (options->dataset_type) {
            case DataSetType::CowAndLady:
                LoadDataFromCowAndLady();
                break;
            case DataSetType::Mesh:
                LoadDataFromMeshDataset();
                break;
            case DataSetType::NewerCollege:
                LoadDataFromNewerCollege();
                break;
            case DataSetType::ReplicaRgbd:
                LoadDataFromReplicaRgbd();
                break;
            default:
                ERL_FATAL("Unsupported dataset type.");
        }
        RangesToPoints();
        ApplyScanStride();
        wp_idx += options->seq_stride;
    }

#pragma endregion

#pragma region visualization

    void
    VisualizeSensorMesh() {
        const Eigen::Matrix4d last_pose_inv = last_sensor_pose.inverse();
        Eigen::Matrix4d cur_pose = Eigen::Matrix4d::Identity();
        cur_pose.topLeftCorner<3, 3>() = rotation_frame.template cast<double>();
        cur_pose.topRightCorner<3, 1>() = translation_frame.template cast<double>();
        const Eigen::Matrix4d delta_pose = cur_pose * last_pose_inv;
        last_sensor_pose = cur_pose;
        mesh_sensor->Transform(delta_pose);
        if (std::find(geometries.begin(), geometries.end(), mesh_sensor) != geometries.end()) {
            visualizer->GetVisualizer()->UpdateGeometry(mesh_sensor);
        }
    }

    virtual void
    VisualizeSensorData() {
        constexpr int kFontFace = cv::FONT_HERSHEY_PLAIN;
        const cv::Scalar kTextColor = {255, 255, 255, 255};

        Dtype resize_scale = options->image_resize_scale;
        resize_scale = std::min(resize_scale, 1000.0f / static_cast<Dtype>(ranges_img.cols));
        resize_scale = std::min(resize_scale, 1000.0f / static_cast<Dtype>(ranges_img.rows));
        cv::Mat ranges_img_resize;
        cv::resize(ranges_img, ranges_img_resize, cv::Size(), resize_scale, resize_scale);
        int y = 30;
        for (const auto &text: ranges_img_texts) {
            constexpr int kFontThickness = 2;
            constexpr double kFontScale = 1.5;
            cv::putText(
                ranges_img_resize,
                text,
                cv::Point(10, y),
                kFontFace,
                kFontScale,
                kTextColor,
                kFontThickness);
            y += 30;
        }
        cv::imshow("ranges", ranges_img_resize);
        cv::waitKey(1);

        if (std::find(geometries.begin(), geometries.end(), pcd_obs) != geometries.end()) {
            pcd_obs->points_.clear();
            pcd_obs->colors_.clear();
            const auto &hit_points = range_sensor_frame->GetHitPointsWorld();
            const auto n_points =
                (hit_points.size() + options->pcd_stride - 1) / options->pcd_stride;
            pcd_obs->points_.reserve(n_points);
            for (std::size_t i = 0; i < hit_points.size(); i += options->pcd_stride) {
                pcd_obs->points_.emplace_back(hit_points[i].template cast<double>());
            }
            pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});  // green
            visualizer->GetVisualizer()->UpdateGeometry(pcd_obs);
        }
    }

    virtual void
    VisualizeTrajectory() {
        if (std::find(geometries.begin(), geometries.end(), line_set_traj) != geometries.end()) {
            line_set_traj->points_.emplace_back(translation_sensor.template cast<double>());
            const size_t n_points = line_set_traj->points_.size();
            if (n_points >= 2) {
                line_set_traj->lines_.emplace_back(n_points - 2, n_points - 1);
                line_set_traj->colors_.emplace_back(0.0, 1.0, 0.0);  // green
            }
            visualizer->GetVisualizer()->UpdateGeometry(line_set_traj);
        }
    }

    virtual void
    VisualizeSurfaceMapping() {
        if (surf_data_buffer == nullptr || unused_surf_data_indices == nullptr) { return; }

        auto it1 = std::find(geometries.begin(), geometries.end(), pcd_surf_points);
        auto it2 = std::find(geometries.begin(), geometries.end(), line_set_surf_normals);

        if (it1 != geometries.end() || it2 != geometries.end()) {
            pcd_surf_points->points_.clear();
            line_set_surf_normals->points_.clear();
            line_set_surf_normals->lines_.clear();
            const auto &unused_indices = *unused_surf_data_indices;
            const std::unordered_set<std::size_t> unused_set(
                unused_indices.begin(),
                unused_indices.end());
            std::size_t n_points = surf_data_buffer->size() - unused_set.size();
            n_points = (n_points + options->pcd_stride - 1) / options->pcd_stride;
            pcd_surf_points->points_.reserve(n_points);
            pcd_surf_points->normals_.reserve(n_points);
            line_set_surf_normals->points_.reserve(n_points * 2);
            line_set_surf_normals->lines_.reserve(n_points);
            const Dtype scale = options->surf_normal_scale;
            for (std::size_t i = 0; i < surf_data_buffer->size(); i += options->pcd_stride) {
                if (unused_set.count(i) != 0u) { continue; }  // skip unused surface data
                const auto &surface_data = (*surf_data_buffer)[i];
                if (surface_data.var_position >= 1.e6f) { continue; }  // invalid surface data

                const Vector3 position = surface_data.position / scaling;
                const Vector3 &normal = surface_data.normal;
                ERL_ASSERTM(
                    std::abs(normal.norm() - 1.0) < 1.e-4,
                    "normal.norm() = {:.6f}",
                    normal.norm());
                pcd_surf_points->points_.emplace_back(position.template cast<double>());
                pcd_surf_points->normals_.emplace_back(normal.template cast<double>());
                line_set_surf_normals->points_.emplace_back(position.template cast<double>());
                line_set_surf_normals->points_.emplace_back(
                    (position + scale * normal).template cast<double>());
                line_set_surf_normals->lines_.emplace_back(
                    line_set_surf_normals->points_.size() - 2,
                    line_set_surf_normals->points_.size() - 1);
            }
            pcd_surf_points->PaintUniformColor({0.0, 0.0, 1.0});        // blue
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});  // red

            const auto vis = visualizer->GetVisualizer();
            if (it1 != geometries.end()) { vis->UpdateGeometry(pcd_surf_points); }
            if (it2 != geometries.end()) { vis->UpdateGeometry(line_set_surf_normals); }
        }

        if (std::find(geometries.begin(), geometries.end(), mesh_surf) != geometries.end()) {
            visualizer->GetVisualizer()->UpdateGeometry(mesh_surf);
        }
    }

    virtual void
    VisualizePrediction() {
        if (std::find(geometries.begin(), geometries.end(), voxel_grid_pred) != geometries.end()) {
            visualizer->GetVisualizer()->UpdateGeometry(voxel_grid_pred);
        }
    }

    void
    UpdateClusterBox(const uint64_t cluster_key, const Eigen::Vector3d &cluster_position) {
        auto [it, inserted] = line_set_clusters_map.try_emplace(cluster_key, LineSetInfo());
        if (!inserted) { return; }

        auto box = line_set_cluster_box;
        box.Translate(cluster_position);
        for (std::size_t i = 0; i < box.points_.size(); ++i) {
            const Eigen::Vector3l p = (box.points_[i] / 0.005f).template cast<long>();
            auto [it_p, inserted_p] =
                cluster_vertex_index_map.try_emplace(p, cluster_vertex_index_map.size());
            if (inserted_p) { line_set_clusters->points_.emplace_back(box.points_[i]); }
            CHECKED_AT(it->second.first, i) = it_p->second;
        }
        for (std::size_t i = 0; i < box.lines_.size(); ++i) {
            Eigen::Vector2i l = box.lines_[i];
            l[0] = static_cast<int>(CHECKED_AT(it->second.first, l[0]));
            l[1] = static_cast<int>(CHECKED_AT(it->second.first, l[1]));
            auto [it_l, inserted_l] =
                cluster_edge_index_map.try_emplace(l, cluster_edge_index_map.size());
            if (inserted_l) { line_set_clusters->lines_.emplace_back(l); }
            CHECKED_AT(it->second.second, i) = it_l->second;
        }
    }

    virtual void
    UpdateClusterBoxes() {}

    void
    VisualizeClusters() {
        if (std::find(geometries.begin(), geometries.end(), line_set_clusters) !=
            geometries.end()) {
            UpdateClusterBoxes();
            line_set_clusters->colors_.clear();
            line_set_clusters->PaintUniformColor({0.0, 1.0, 1.0});  // cyan
            for (const auto &key: inactive_cluster_keys) {
                auto it = line_set_clusters_map.find(key);
                ERL_ASSERT(it != line_set_clusters_map.end());
                for (const auto &edge_idx: it->second.second) {
                    line_set_clusters->colors_[edge_idx] = Eigen::Vector3d(0.5, 0.5, 0.5);  // gray
                }
            }
            visualizer->GetVisualizer()->UpdateGeometry(line_set_clusters);
        }
    }

    void
    UpdateVisualization() {
        const ERL_BLOCK_TIMER_MSG("UpdateVisualization");

        VisualizeSensorMesh();
        VisualizeSensorData();
        VisualizeTrajectory();

        VisualizeSurfaceMapping();
        VisualizePrediction();
        VisualizeClusters();
    }

    /**
     * Visualize the whole map at options->test_z height. Results are stored in voxel_grid_pred.
     * VisualizePrediction() will be called to trigger the update in the visualizer.
     */
    virtual void
    UpdateWholeMapPrediction() = 0;

    /**
     * Update the visualization of the map following the sensor pose. Results are stored in
     * voxel_grid_pred. VisualizePrediction() will be called to trigger the update in the
     * visualizer.
     */
    virtual void
    UpdateFollowingMapPrediction() = 0;

    virtual void
    UpdatePredictionAtPosition() = 0;

    virtual bool
    UpdateMap() = 0;

    virtual bool
    AnimationCallback(Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) {
        if (options->save_images) {
            vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx), false);
        }

        if (animation_ended) {
            // options->hold is true, so the window is not closed yet
            if (options->test_whole_map_at_end) {
                if (options->test_z != static_cast<Dtype>(vis_setting->z)) {
                    UpdateWholeMapPrediction();
                }
            } else {
                if (position_test[0] != static_cast<Dtype>(vis_setting->x) ||
                    position_test[1] != static_cast<Dtype>(vis_setting->y) ||
                    position_test[2] != static_cast<Dtype>(vis_setting->z)) {
                    UpdatePredictionAtPosition();
                }
            }
            cv::waitKey(1);
            return false;
        }

        if (wp_idx >= max_wp_idx) {  // end of animation
            animation_ended = true;
            if (options->exhausting) {
                // clear frame data
                frame_points = Matrix3X();
                frame_ranges = MatrixX();

                while (UpdateMap()) {
                    UpdateFollowingMapPrediction();
                    UpdateVisualization();
                }
            }
            if (options->test_whole_map_at_end) {
                UpdateWholeMapPrediction();
            } else {
                UpdatePredictionAtPosition();
            }
            if (options->save_images) {
                vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx + 1), true);
            }
            if (!options->hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }

        ERL_INFO("wp_idx: {}", wp_idx);
        {
            const ERL_BLOCK_TIMER_MSG_TIME("gui_update", gui_dt);
            LoadData();
            if (UpdateMap()) { UpdateFollowingMapPrediction(); }
            if (wp_idx % options->vis_stride == 0) { UpdateVisualization(); }
        }
        ERL_INFO("gui_update (fps): {:.2f}", 1000.0 / gui_dt);
        return true;
    }

#pragma endregion

#pragma region test_io

    virtual std::string
    GetBinFileName() = 0;

    virtual void
    TestIo() = 0;

    void
    WriteMappingBin() {
        const ERL_BLOCK_TIMER_MSG("WriteMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Write(bin_file, mapping.get()),
            "Failed to write to file: {}",
            bin_file);
    }

    void
    ReadMappingBin(MappingType &mapping_read) {
        const ERL_BLOCK_TIMER_MSG("ReadMappingBin");
        std::string bin_file = GetBinFileName();
        using namespace erl::common::serialization;
        ERL_ASSERTM(
            Serialization<MappingType>::Read(bin_file, &mapping_read),
            "Failed to read from file: {}",
            bin_file);
    }

    void
    TestIo(MappingType &mapping_read) {
        const ERL_BLOCK_TIMER_MSG("TestIo");
        WriteMappingBin();
        ReadMappingBin(mapping_read);
        ERL_ASSERT(*mapping == mapping_read);
    }

#pragma endregion
};
