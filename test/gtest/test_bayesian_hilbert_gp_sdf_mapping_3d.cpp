#include "utils.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"
#include "erl_geometry/newer_college.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
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

int g_argc = 0;
char **g_argv = nullptr;
const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
const std::filesystem::path kDataDir = kProjectRootDir / "data";
const std::filesystem::path kConfigDir = kProjectRootDir / "config";

template<typename Dtype>
struct TestImpl3D {

    using SurfaceMapping = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 3>;
    using SdfMapping = erl::gp_sdf::GpSdfMapping<Dtype, 3>;
    using DepthFrame3D = erl::geometry::DepthFrame3D<Dtype>;
    using LidarFrame3D = erl::geometry::LidarFrame3D<Dtype>;
    using RangeSensor3D = erl::geometry::RangeSensor3D<Dtype>;
    using RangeSensorFrame3D = erl::geometry::RangeSensorFrame3D<Dtype>;
    using DepthCamera3D = erl::geometry::DepthCamera3D<Dtype>;
    using Lidar3D = erl::geometry::Lidar3D<Dtype>;
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

    struct Options {
        std::string dataset_name = "cow_and_lady";
        std::string cow_and_lady_dir;
        std::string newer_college_dir;
        std::string mesh_file = kDataDir / "replica-hotel-0.ply";       // mesh file
        std::string traj_file = kDataDir / "replica-hotel-0-traj.txt";  // trajectory file
        std::string surface_mapping_config_file =
            kConfigDir / "template" /
            fmt::format("bayesian_hilbert_mapping_3d_{}.yaml", type_name<Dtype>());
        std::string sdf_mapping_config_file =
            kConfigDir / "template" / fmt::format("sdf_mapping_3d_{}.yaml", type_name<Dtype>());
        std::string o3d_view_status_file = kConfigDir / "template" / "open3d_view_status.json";
        std::string sdf_mapping_bin_file;
        std::string sensor_frame_type = type_name<LidarFrame3D>();
        std::string sensor_frame_config_file = kConfigDir / "sensors" / "lidar_frame_3d_360.yaml";
        long start_wp_idx = 0;
        long end_wp_idx = -1;  // -1 means all waypoints
        long seq_stride = 1;
        long scan_stride = 1;
        long vis_stride = 1;
        Dtype test_res = 0.02;
        Dtype test_x_min = 0.0f;
        Dtype test_x_max = 0.0f;
        Dtype test_y_min = 0.0f;
        Dtype test_y_max = 0.0f;
        long test_xs = 150;
        long test_ys = 100;
        bool test_whole_map_at_end = false;  // test the whole map at the end
        Dtype test_z = 0.0;                  // test z for the whole map
        Dtype image_resize_scale = 10;       // image resize scale
        bool save_images = false;
        bool test_io = false;
        bool hold = false;
    };

    Options options;
    std::shared_ptr<typename SurfaceMapping::Setting> surface_mapping_setting =
        std::make_shared<typename SurfaceMapping::Setting>();
    std::shared_ptr<typename SdfMapping::Setting> sdf_mapping_setting =
        std::make_shared<typename SdfMapping::Setting>();

    DataSetType dataset_type = DataSetType::Mesh;
    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::shared_ptr<NewerCollege> newer_college = nullptr;
    std::shared_ptr<RangeSensorFrame3D> range_sensor_frame = nullptr;
    std::shared_ptr<RangeSensor3D> range_sensor = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    bool is_lidar = false;
    Vector3 map_min, map_max;
    long max_wp_idx = 0;
    long wp_idx = 0;
    bool animation_ended = false;
    Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();
    cv::Mat ranges_img;
    Matrix3X points;
    Matrix3 rotation_sensor, rotation;
    Vector3 translation_sensor, translation;

    std::shared_ptr<SurfaceMapping> surface_mapping = nullptr;
    std::shared_ptr<SdfMapping> sdf_mapping = nullptr;

    std::shared_ptr<Open3dVisualizerWrapper::Setting> vis_setting =
        std::make_shared<Open3dVisualizerWrapper::Setting>();
    std::shared_ptr<Open3dVisualizerWrapper> visualizer = nullptr;
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sensor = nullptr;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_sensor_xyz = nullptr;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_obs =
        std::make_shared<open3d::geometry::PointCloud>();
    std::shared_ptr<open3d::geometry::PointCloud> pcd_surf_points =
        std::make_shared<open3d::geometry::PointCloud>();
    std::shared_ptr<open3d::geometry::LineSet> line_set_surf_normals =
        std::make_shared<open3d::geometry::LineSet>();
    std::shared_ptr<open3d::geometry::VoxelGrid> voxel_grid_sdf =
        std::make_shared<open3d::geometry::VoxelGrid>();

    Eigen::MatrixX<Vector3> positions_test_org;
    Matrix3X positions_test;
    VectorX sdf_pred;
    std::filesystem::path test_output_dir;
    std::filesystem::path img_dir;

    // logging
    Eigen::Matrix2Xd fps_data;
    double update_dt = 0;
    double update_fps = 0;
    double test_dt = 0;
    double test_fps = 0;

    TestImpl3D() {
        GTEST_PREPARE_OUTPUT_DIR();
        this->test_output_dir = test_output_dir;
        this->img_dir = test_output_dir / "images";
        std::filesystem::create_directory(img_dir);
        vis_setting->window_name = test_info->name();

        ParseOptions();
        LoadSetting();
        PrepareScene();
        PrepareVisualizer();
    }

    void
    ParseOptions() {
        bool options_parsed = false;

        try {
            namespace po = boost::program_options;
            po::options_description desc;
            // clang-format off
            desc.add_options()
            ("help", "produce help message")
            (
                "dataset-name",
                po::value<std::string>(&options.dataset_name)->default_value(options.dataset_name)->value_name("name"),
                "name of the dataset to use, options: cow_and_lady, mesh, newer_college"
            )(
                "cow-and-lady-dir",
                po::value<std::string>(&options.cow_and_lady_dir)->default_value(options.cow_and_lady_dir)->value_name("dir"),
                "directory containing the Cow and Lady dataset"
            )(
                "newer-college-dir",
                po::value<std::string>(&options.newer_college_dir)->default_value(options.newer_college_dir)->value_name("dir"),
                "directory containing the Newer College dataset"
            )
            ("mesh-file", po::value<std::string>(&options.mesh_file)->default_value(options.mesh_file)->value_name("file"), "mesh file")
            ("traj-file", po::value<std::string>(&options.traj_file)->default_value(options.traj_file)->value_name("file"), "trajectory file")
            (
                "surface-mapping-config-file",
                po::value<std::string>(&options.surface_mapping_config_file)->default_value(options.surface_mapping_config_file)->value_name("file"),
                "surface mapping config file"
            )
            (
                "sdf-mapping-config-file",
                po::value<std::string>(&options.sdf_mapping_config_file)->default_value(options.sdf_mapping_config_file)->value_name("file"),
                "SDF mapping config file"
            )
            (
                "o3d-view-status-file",
                po::value<std::string>(&options.o3d_view_status_file)->default_value(options.o3d_view_status_file)->value_name("file"),
                "Open3D view status file, used to set the view of the visualization window"
            )
            (
                "sdf-mapping-bin-file",
                po::value<std::string>(&options.sdf_mapping_bin_file)->default_value(options.sdf_mapping_bin_file)->value_name("file"),
                "SDF mapping bin file"
            )
            (
                "sensor-frame-type",
                po::value<std::string>(&options.sensor_frame_type)->default_value(options.sensor_frame_type)->value_name("type"),
                fmt::format("sensor frame type used when the mesh file is used: {} or {}",
                            type_name<LidarFrame3D>(),
                            type_name<DepthFrame3D>()).c_str()
            )
            (
                "sensor-frame-config-file",
                po::value<std::string>(&options.sensor_frame_config_file)->default_value(options.sensor_frame_config_file)->value_name("file"),
                "sensor frame config file, used when the mesh file is used"
            )
            ("start-wp-idx", po::value<long>(&options.start_wp_idx)->default_value(options.start_wp_idx)->value_name("idx"), "start waypoint index")
            ("end-wp-idx", po::value<long>(&options.end_wp_idx)->default_value(options.end_wp_idx)->value_name("idx"), "end waypoint index (-1 means all waypoints)")
            ("seq-stride", po::value<long>(&options.seq_stride)->default_value(options.seq_stride)->value_name("stride"), "sequence stride")
            ("scan-stride", po::value<long>(&options.scan_stride)->default_value(options.scan_stride)->value_name("stride"), "scan stride")
            ("vis-stride", po::value<long>(&options.vis_stride)->default_value(options.vis_stride)->value_name("stride"), "visualization stride")
            ("test-res", po::value<Dtype>(&options.test_res)->default_value(options.test_res)->value_name("res"), "test resolution")
            ("test-x-min", po::value<Dtype>(&options.test_x_min)->default_value(options.test_x_min)->value_name("x_min"), "test x min")
            ("test-x-max", po::value<Dtype>(&options.test_x_max)->default_value(options.test_x_max)->value_name("x_max"), "test x max")
            ("test-y-min", po::value<Dtype>(&options.test_y_min)->default_value(options.test_y_min)->value_name("y_min"), "test y min")
            ("test-y-max", po::value<Dtype>(&options.test_y_max)->default_value(options.test_y_max)->value_name("y_max"), "test y max")
            ("test-xs", po::value<long>(&options.test_xs)->default_value(options.test_xs)->value_name("xs"), "test xs")
            ("test-ys", po::value<long>(&options.test_ys)->default_value(options.test_ys)->value_name("ys"), "test ys")
            ("test-whole-map-at-end", po::bool_switch(&options.test_whole_map_at_end), "test the whole map at the end")
            ("test-z", po::value<Dtype>(&options.test_z)->default_value(options.test_z)->value_name("z"), "test z for the whole map")
            ("image-resize-scale", po::value<Dtype>(&options.image_resize_scale)->default_value(options.image_resize_scale)->value_name("scale"), "image resize scale")
            ("save-images", po::bool_switch(&options.save_images), "save images")
            ("test-io", po::bool_switch(&options.test_io), "test IO")
            ("hold", po::bool_switch(&options.hold), "hold the window");
            // clang-format on

            po::variables_map vm;
            po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
            if (vm.count("help")) {
                std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl
                          << desc << std::endl;
                return;
            }
            po::notify(vm);
            options_parsed = true;
        } catch (std::exception &e) { std::cerr << e.what() << std::endl; }
        ASSERT_TRUE(options_parsed);
        ASSERT_TRUE(options.scan_stride > 0);

        if (options.dataset_name == "cow_and_lady") {
            dataset_type = DataSetType::CowAndLady;
            ERL_ASSERTM(
                !options.cow_and_lady_dir.empty(),
                "Please provide the Cow and Lady dataset directory via --cow-and-lady-dir");
        } else if (options.dataset_name == "mesh") {
            dataset_type = DataSetType::Mesh;
            ERL_ASSERTM(!options.mesh_file.empty(), "Please provide the mesh file via --mesh-file");
            ERL_ASSERTM(
                !options.traj_file.empty(),
                "Please provide the trajectory file via --traj-file");
        } else if (options.dataset_name == "newer_college") {
            dataset_type = DataSetType::NewerCollege;
            ERL_ASSERTM(
                !options.newer_college_dir.empty(),
                "Please provide the Newer College dataset directory via --newer-college-dir");
        } else {
            ERL_FATAL("Unknown dataset name {} for 3D", options.dataset_name);
        }
    }

    void
    LoadSetting() {
        ERL_ASSERTM(
            surface_mapping_setting->FromYamlFile(options.surface_mapping_config_file),
            "Failed to load surface_mapping_config_file: {}",
            options.surface_mapping_config_file);
        ERL_ASSERTM(
            sdf_mapping_setting->FromYamlFile(options.sdf_mapping_config_file),
            "Failed to load sdf_mapping_config_file: {}",
            options.sdf_mapping_config_file);
        ERL_INFO("Surface mapping config: {}", options.surface_mapping_config_file);
        std::cout << surface_mapping_setting->AsYamlString() << std::endl;
        ERL_INFO("SDF mapping config: {}", options.sdf_mapping_config_file);
        std::cout << sdf_mapping_setting->AsYamlString() << std::endl;
    }

    void
    PrepareScene() {
        // 1. load dataset
        // 2. create range sensor frame for converting data to points
        // 3. set map_min, map_max, max_wp_idx
        // 4. initialize positions_test_org, positions_test, sdf_pred and fps_data

        switch (dataset_type) {
            case DataSetType::CowAndLady: {
                cow_and_lady = std::make_shared<CowAndLady>(options.cow_and_lady_dir);
                max_wp_idx = cow_and_lady->Size();
                geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
                map_min = cow_and_lady->GetMapMin().cast<Dtype>();
                map_max = cow_and_lady->GetMapMax().cast<Dtype>();
                const auto depth_frame_setting = std::make_shared<typename DepthFrame3D::Setting>();
                depth_frame_setting->camera_intrinsic.image_height = CowAndLady::kImageHeight;
                depth_frame_setting->camera_intrinsic.image_width = CowAndLady::kImageWidth;
                depth_frame_setting->camera_intrinsic.camera_fx = CowAndLady::kCameraFx;
                depth_frame_setting->camera_intrinsic.camera_fy = CowAndLady::kCameraFy;
                depth_frame_setting->camera_intrinsic.camera_cx = CowAndLady::kCameraCx;
                depth_frame_setting->camera_intrinsic.camera_cy = CowAndLady::kCameraCy;
                if (options.scan_stride > 1) {
                    depth_frame_setting->Resize(1.0f / static_cast<Dtype>(options.scan_stride));
                }
                range_sensor_frame = std::make_shared<DepthFrame3D>(depth_frame_setting);
                ERL_ASSERTM(
                    options.start_wp_idx < cow_and_lady->Size(),
                    "Start waypoint index {} is out of range [0, {}]",
                    options.start_wp_idx,
                    cow_and_lady->Size());
                ERL_ASSERTM(
                    options.end_wp_idx < cow_and_lady->Size() || options.end_wp_idx == -1,
                    "End waypoint index {} is out of range [-1, {}), -1 means all waypoints",
                    options.end_wp_idx,
                    cow_and_lady->Size());
                break;
            }
            case DataSetType::Mesh: {
                const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
                ERL_ASSERTM(
                    !mesh->vertices_.empty(),
                    "Failed to load mesh file: {}",
                    options.mesh_file);
                map_min = mesh->GetMinBound().template cast<Dtype>();
                map_max = mesh->GetMaxBound().template cast<Dtype>();
                geometries.push_back(mesh);

                if (options.sensor_frame_type == type_name<LidarFrame3D>()) {
                    const auto lidar_frame_setting =
                        std::make_shared<typename LidarFrame3D::Setting>();
                    ASSERT_TRUE(
                        lidar_frame_setting->FromYamlFile(options.sensor_frame_config_file));
                    const auto lidar_setting = std::make_shared<typename Lidar3D::Setting>();
                    lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
                    lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
                    lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
                    lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
                    lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
                    lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
                    range_sensor = std::make_shared<Lidar3D>(lidar_setting);
                    is_lidar = true;
                    if (options.scan_stride > 1) {
                        lidar_frame_setting->Resize(1.0f / static_cast<Dtype>(options.scan_stride));
                    }
                    range_sensor_frame = std::make_shared<LidarFrame3D>(lidar_frame_setting);
                } else if (options.sensor_frame_type == type_name<DepthFrame3D>()) {
                    const auto depth_frame_setting =
                        std::make_shared<typename DepthFrame3D::Setting>();
                    ASSERT_TRUE(
                        depth_frame_setting->FromYamlFile(options.sensor_frame_config_file));
                    auto depth_camera_setting = std::make_shared<typename DepthCamera3D::Setting>(
                        depth_frame_setting->camera_intrinsic);
                    range_sensor = std::make_shared<DepthCamera3D>(depth_camera_setting);
                    if (options.scan_stride > 1) {
                        depth_frame_setting->Resize(1.0f / static_cast<Dtype>(options.scan_stride));
                    }
                    range_sensor_frame = std::make_shared<DepthFrame3D>(depth_frame_setting);
                } else {
                    ERL_FATAL("Unknown sensor_frame_type: {}", options.sensor_frame_type);
                }
                range_sensor->AddMesh(options.mesh_file);
                poses = Trajectory::LoadSe3(options.traj_file, false);
                max_wp_idx = static_cast<long>(poses.size());
                ERL_ASSERTM(
                    options.start_wp_idx < max_wp_idx,
                    "Start waypoint index {} is out of range [0, {}]",
                    options.start_wp_idx,
                    max_wp_idx);
                ERL_ASSERTM(
                    options.end_wp_idx < max_wp_idx || options.end_wp_idx == -1,
                    "End waypoint index {} is out of range [-1, {}), -1 means all waypoints",
                    options.end_wp_idx,
                    max_wp_idx);
                break;
            }
            case DataSetType::NewerCollege: {
                ERL_ASSERTM(
                    !options.newer_college_dir.empty(),
                    "Please provide the Newer College dataset directory via --newer-college-dir");
                newer_college = std::make_shared<NewerCollege>(options.newer_college_dir);
                max_wp_idx = newer_college->Size();
                auto pcd = newer_college->GetGroundTruthPointCloud();
                pcd = pcd->RandomDownSample(0.05);
                geometries.push_back(pcd);
                map_min = newer_college->GetMapMin().cast<Dtype>();
                map_max = newer_college->GetMapMax().cast<Dtype>();
                is_lidar = true;
                ERL_ASSERTM(
                    options.start_wp_idx < newer_college->Size(),
                    "Start waypoint index {} is out of range [0, {}]",
                    options.start_wp_idx,
                    newer_college->Size());
                ERL_ASSERTM(
                    options.end_wp_idx < newer_college->Size() || options.end_wp_idx == -1,
                    "End waypoint index {} is out of range [-1, {}), -1 means all waypoints",
                    options.end_wp_idx,
                    newer_college->Size());
                break;
            }
            default:
                ERL_FATAL("Unsupported dataset type.");
        }

        max_wp_idx = (options.end_wp_idx == -1) ? max_wp_idx : options.end_wp_idx;

        positions_test_org.resize(options.test_xs, options.test_ys);
        const Vector3 offset(
            static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_xs),
            static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_ys),
            0.0);
        // x: down, y: right
        for (long j = 0; j < positions_test_org.cols(); ++j) {
            const Dtype y = static_cast<Dtype>(j) * options.test_res + offset[1];
            for (long i = 0; i < positions_test_org.rows(); ++i) {
                const Dtype x = static_cast<Dtype>(i) * options.test_res + offset[0];
                positions_test_org(i, j) << x, y, offset[2];
            }
        }
        sdf_pred.resize(positions_test_org.size());
        sdf_pred.setZero();
        positions_test.resize(3, positions_test_org.size());

        fps_data.resize(2, (max_wp_idx + options.seq_stride - 1) / options.seq_stride);
    }

    void
    PrepareVisualizer() {
        vis_setting->mesh_show_back_face = false;
        vis_setting->translate_step = dataset_type == DataSetType::NewerCollege ? 0.1 : 0.01;
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
        voxel_grid_sdf->origin_.setZero();
        voxel_grid_sdf->voxel_size_ = options.test_res;

        geometries.push_back(mesh_sensor);
        geometries.push_back(mesh_sensor_xyz);
        // geometries.push_back(pcd_obs);
        // geometries.push_back(pcd_surf_points);
        // geometries.push_back(line_set_surf_normals);
        geometries.push_back(voxel_grid_sdf);

        visualizer->AddGeometries(geometries);
    }

    void
    VisualizeWholeMap() {
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
            Eigen::Vector2i(10, 10));
        Matrix3X test_positions(3, grid_map_info.Size());
        test_positions.topRows(2) =
            grid_map_info.GenerateMeterCoordinates(false).template cast<Dtype>();
        test_positions.row(2).setConstant(options.test_z);
        VectorX distances;
        {
            Matrix3X gradients(3, test_positions.cols());
            Matrix4X variances;
            Matrix6X covairances;
            ERL_BLOCK_TIMER_MSG("sdf_mapping.Test");
            EXPECT_TRUE(
                sdf_mapping->Test(test_positions, distances, gradients, variances, covairances));
        }

        const int xs = grid_map_info.Shape(0);
        const int ys = grid_map_info.Shape(1);
        cv::Mat img_sdf = ConvertVectorToImage(xs, ys, distances, true);
        VectorX signs = (distances.array() >= 0.0f).template cast<Dtype>();
        cv::Mat img_sdf_sign = ConvertVectorToImage(xs, ys, signs, false);

        ConvertToVoxelGrid(img_sdf, test_positions, voxel_grid_sdf);

        VectorX in_free_space;
        EXPECT_TRUE(surface_mapping->IsInFreeSpace(test_positions, in_free_space));
        cv::Mat img_surf_mapping_sign = ConvertVectorToImage(xs, ys, in_free_space, false);

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
        cv::imwrite(test_output_dir / "sdf_whole_map.png", img_sdf);
        cv::imwrite(test_output_dir / "sdf_sign_whole_map.png", img_sdf_sign);
        cv::imwrite(test_output_dir / "surf_mapping_sign_whole_map.png", img_surf_mapping_sign);
        cv::waitKey(1);
    }

    void
    TestIo() {
        ERL_BLOCK_TIMER_MSG("IO");
        std::string bin_file = fmt::format("sdf_mapping_3d_{}.bin", type_name<Dtype>());
        bin_file = test_output_dir / bin_file;
        ERL_ASSERTM(
            erl::common::Serialization<SdfMapping>::Write(bin_file, sdf_mapping),
            "Failed to write to file: {}",
            bin_file);
        auto surface_mapping_read =
            std::make_shared<SurfaceMapping>(std::make_shared<typename SurfaceMapping::Setting>());
        SdfMapping sdf_mapping_read(
            std::make_shared<typename SdfMapping::Setting>(),
            surface_mapping_read);
        ERL_ASSERTM(
            erl::common::Serialization<SdfMapping>::Read(bin_file, &sdf_mapping_read),
            "Failed to read from file: {}",
            bin_file);
        ERL_ASSERTM(*sdf_mapping == sdf_mapping_read, "sdf_mapping != sdf_mapping_read");
    }

    Matrix3X
    RangesToPoints(MatrixX ranges) {
        ERL_ASSERTM(range_sensor_frame != nullptr, "range_sensor_frame is null");
        if (options.scan_stride > 1) {
            auto [rows, cols] = range_sensor_frame->GetFrameShape();
            MatrixX new_ranges(rows, cols);
            for (long j = 0; j < cols; ++j) {
                const Dtype *ranges_j = ranges.col(j * options.scan_stride).data();
                Dtype *new_ranges_j = new_ranges.col(j).data();
                for (long i = 0; i < rows; ++i) {
                    new_ranges_j[i] = ranges_j[i * options.scan_stride];
                }
            }
            ranges = new_ranges;
        }
        range_sensor_frame->UpdateRanges(rotation, translation, ranges);
        return Eigen::Map<const Matrix3X>(
            range_sensor_frame->GetHitPointsFrame().data()->data(),
            3,
            range_sensor_frame->GetNumHitRays());
    }

    void
    LoadData() {
        ERL_BLOCK_TIMER_MSG("data loading");
        switch (dataset_type) {
            case DataSetType::CowAndLady: {
                const auto frame = (*cow_and_lady)[wp_idx];
                rotation = frame.rotation.cast<Dtype>();
                translation = frame.translation.cast<Dtype>();
                std::tie(rotation_sensor, translation_sensor) =
                    erl::geometry::CameraBase3D<Dtype>::ComputeCameraPose(rotation, translation);

                points = RangesToPoints(frame.depth.cast<Dtype>());
                ranges_img = frame.depth_jet;

                break;
            }
            case DataSetType::Mesh: {
                std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
                std::tie(rotation, translation) =
                    range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);

                MatrixX ranges = range_sensor->Scan(rotation_sensor, translation_sensor);
                points = RangesToPoints(ranges);
                ranges_img = ConvertMatrixToImage(ranges, true);
                if (is_lidar) {                           // azimuth: down, elevation: right
                    ranges_img = ranges_img.t();          // elevation: down, azimuth: right
                    cv::flip(ranges_img, ranges_img, 0);  // elevation: up, azimuth: right
                }

                break;
            }
            case DataSetType::NewerCollege: {
                const auto frame = (*newer_college)[wp_idx];
                rotation = frame.rotation.cast<Dtype>();
                translation = frame.translation.cast<Dtype>();
                rotation_sensor = rotation;
                translation_sensor = translation;

                points = frame.points.cast<Dtype>();
                if (options.scan_stride > 1) {
                    long n = (points.cols() + options.scan_stride - 1) / options.scan_stride;
                    Matrix3X new_points(3, n);
                    for (long i = 0; i < n; ++i) {
                        new_points.col(i) = points.col(i * options.scan_stride);
                    }
                    points = new_points;
                }
                MatrixX ranges = frame.GetRangeMatrix().cast<Dtype>();
                ranges_img = ConvertMatrixToImage(ranges, true);
                if (is_lidar) {
                    ranges_img = ranges_img.t();
                    cv::flip(ranges_img, ranges_img, 0);
                }

                break;
            }
            default:
                ERL_FATAL("Unsupported dataset type.");
        }
    }

    void
    UpdateMap() {
        LoadData();

        // Transform points to world coordinates
#pragma omp parallel for default(none) shared(points, rotation, translation) schedule(static)
        for (long i = 0; i < points.cols(); ++i) {
            points.col(i) = rotation * points.col(i) + translation;
        }

        {
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Update", update_dt);
            // the following code is the same as
            // bool ok = sdf_mapping.Update(rotation, translation, points, true, false);
            // but for unknown reason it is 2x faster than calling sdf_mapping.Update directly.
            // TODO: investigate this.
            double surf_mapping_time;
            bool ok;
            {
                ERL_BLOCK_TIMER_MSG_TIME("Surface mapping update", surf_mapping_time);
                ok = sdf_mapping->GetSurfaceMapping()
                         ->Update(rotation_sensor, translation_sensor, points, true, false);
            }

            {
                double time_budget_us = 1e6 / sdf_mapping_setting->update_hz;  // us
                ERL_BLOCK_TIMER_MSG("Update SDF GPs");
                if (ok) { sdf_mapping->UpdateGpSdf(time_budget_us - surf_mapping_time * 1000); }
            }

            ERL_WARN_COND(!ok, "Surface mapping update failed, skipping SDF mapping update.");
        }

        update_fps = 1000.0 / update_dt;
        ERL_TRACY_PLOT("sdf_mapping_update (ms)", update_dt);
        ERL_TRACY_PLOT("sdf_mapping_update (fps)", update_fps);

        // test
        for (long j = 0; j < positions_test_org.cols(); ++j) {
            for (long i = 0; i < positions_test_org.rows(); ++i) {
                const Vector3 &position = positions_test_org(i, j);
                positions_test.col(i + j * positions_test_org.rows()) =
                    rotation_sensor * position + translation_sensor;
            }
        }

        {
            Matrix3X gradients(3, positions_test.cols());
            Matrix4X variances;
            Matrix6X covairances;
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Test", test_dt);
            EXPECT_TRUE(
                sdf_mapping->Test(positions_test, sdf_pred, gradients, variances, covairances));
        }
        test_fps = 1000.0 / test_dt;
        ERL_TRACY_PLOT("sdf_map_test (ms)", test_dt);
        ERL_TRACY_PLOT("sdf_map_test (fps)", test_fps);

        fps_data.col(wp_idx / options.seq_stride) << update_fps, test_fps;
    }

    void
    UpdateVisualization() {
        constexpr int kFontFace = cv::FONT_HERSHEY_PLAIN;
        constexpr double kFontScale = 1.5;
        const cv::Scalar kTextColor = {255, 255, 255, 255};
        constexpr int kFontThickness = 2;
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
            fmt::format("update {:.2f} fps", update_fps),
            cv::Point(10, 60),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::Mat img_sdf = ConvertVectorToImage(options.test_xs, options.test_ys, sdf_pred, true);
        cv::Mat img_sdf_sign = ConvertVectorToImage<Dtype>(
            options.test_xs,
            options.test_ys,
            (sdf_pred.array() > 0.0).template cast<Dtype>(),
            true);
        ConvertToVoxelGrid(img_sdf, positions_test, voxel_grid_sdf);
        Dtype resize_scale = options.image_resize_scale;
        resize_scale =
            std::min(resize_scale, static_cast<Dtype>(1920.0) / static_cast<Dtype>(img_sdf.cols));
        resize_scale =
            std::min(resize_scale, static_cast<Dtype>(1920.0) / static_cast<Dtype>(img_sdf.rows));
        cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
        cv::putText(
            img_sdf,
            fmt::format("frame {}", wp_idx),
            cv::Point(10, 30),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            img_sdf,
            fmt::format("update {:.2f} fps", update_fps),
            cv::Point(10, 60),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::putText(
            img_sdf,
            fmt::format("test {:.2f} fps", test_fps),
            cv::Point(10, 90),
            kFontFace,
            kFontScale,
            kTextColor,
            kFontThickness);
        cv::imshow("ranges", ranges_img);
        cv::imshow("sdf", img_sdf);
        cv::imshow("sdf_sign", img_sdf_sign);
        cv::waitKey(1);
        /// update the sensor mesh
        Eigen::Matrix4d last_pose_inv = last_pose.inverse();
        Eigen::Matrix4d cur_pose = Eigen::Matrix4d::Identity();
        cur_pose.topLeftCorner<3, 3>() = rotation_sensor.template cast<double>();
        cur_pose.topRightCorner<3, 1>() = translation_sensor.template cast<double>();
        Eigen::Matrix4d delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose);
        mesh_sensor_xyz->Transform(delta_pose);
        /// update the observation point cloud
        if (std::find(geometries.begin(), geometries.end(), pcd_obs) != geometries.end()) {
            pcd_obs->points_.clear();
            pcd_obs->colors_.clear();
            const auto &hit_points = range_sensor_frame->GetHitPointsWorld();
            pcd_obs->points_.reserve(hit_points.size());
            for (const auto &hit_point: hit_points) {
                pcd_obs->points_.emplace_back(hit_point.template cast<double>());
            }
            pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        }
        /// update the surface point cloud and normals
        auto it1 = std::find(geometries.begin(), geometries.end(), pcd_surf_points);
        auto it2 = std::find(geometries.begin(), geometries.end(), line_set_surf_normals);
        if (const auto tree = surface_mapping->GetTree();
            (it1 != geometries.end() || it2 != geometries.end()) && tree != nullptr) {
            pcd_surf_points->points_.clear();
            line_set_surf_normals->points_.clear();
            line_set_surf_normals->lines_.clear();
            auto end = surface_mapping->EndSurfaceData();
            for (auto it = surface_mapping->BeginSurfaceData(); it != end; ++it) {
                erl::gp_sdf::SurfaceData<Dtype, 3> &surface_data = *it;
                const Vector3 &position = surface_data.position;
                const Vector3 &normal = surface_data.normal;
                ERL_ASSERTM(
                    std::abs(normal.norm() - 1.0) < 1.e-6,
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
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        }
    }

    bool
    Callback(Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) {
        ERL_TRACY_FRAME_MARK_START();

        if (options.save_images) {
            vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx), false);
        }

        if (animation_ended) {  // options.hold is true, so the window is not closed yet
            if (options.test_z != static_cast<Dtype>(vis_setting->z)) {
                options.test_z = static_cast<Dtype>(vis_setting->z);
                VisualizeWholeMap();
            }
            cv::waitKey(1);
            return false;
        }

        if (wp_idx >= max_wp_idx) {  // end of animation
            animation_ended = true;
            if (options.test_whole_map_at_end) { VisualizeWholeMap(); }
            if (options.save_images) {
                vis->CaptureScreenImage(img_dir / fmt::format("{:04d}.png", wp_idx + 1), true);
            }
            if (!options.hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }

        double dt;
        ERL_INFO("wp_idx: {}", wp_idx);
        {
            ERL_BLOCK_TIMER_MSG_TIME("gui_update", dt);
            UpdateMap();
            wp_idx += options.seq_stride;
            UpdateVisualization();
        }
        ERL_INFO("gui_update (fps): {:.2f}", 1000.0 / dt);

        ERL_TRACY_PLOT("gui_update (ms)", dt);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / dt);

        return true;
    }

    void
    Run() {
        surface_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);
        sdf_mapping = std::make_shared<SdfMapping>(sdf_mapping_setting, surface_mapping);

        // test IO with empty mapping
        if (options.test_io) { TestIo(); }

        // start the mapping
        auto callback = [this](
                            Open3dVisualizerWrapper *wrapper,
                            open3d::visualization::Visualizer *vis) -> bool {
            return this->Callback(wrapper, vis);
        };
        visualizer->SetAnimationCallback(callback);
        visualizer->SetViewStatus(options.o3d_view_status_file);
        visualizer->Show();

        // test IO after mapping
        if (options.test_io) { TestIo(); }

        erl::common::SaveEigenMatrixToTextFile<double>(
            test_output_dir / "fps.csv",
            fps_data,
            erl::common::EigenTextFormat::kCsvFmt);
    }
};

TEST(GpSdfMapping, BayesianHilbert3Dd) {
    TestImpl3D<double> test;
    test.Run();
}

TEST(GpSdfMapping, BayesianHilbert3Df) {
    TestImpl3D<float> test;
    test.Run();
}

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
