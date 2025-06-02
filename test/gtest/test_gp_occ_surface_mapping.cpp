#include "erl_common/csv.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/range_sensor_3d.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"

#include <boost/program_options.hpp>

// expected performance (Intel i9-14900K):
// - Cow and Lady, Depth: 30 to 60 fps (float) / 25 to 50 fps (double)
// - Replica Hotel, LiDAR, 360: 50 to 100 fps
// - Replica Hotel, Depth Camera: 40 to 60 fps (float) / 30 to 50 fps (double)
// - Gazebo Room: 300 to 500 fps
// - House Expo: 1000 to 1400 fps
// - UCSD FAH: 400 to 800 fps

int g_argc = 0;
char **g_argv = nullptr;
const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;

template<typename Dtype>
void
TestImpl3D() {
    GTEST_PREPARE_OUTPUT_DIR();

    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    using SurfaceMapping = GpOccSurfaceMapping<Dtype, 3>;
    using Octree = OccupancyOctree<Dtype>;
    using OctreeDrawer = OccupancyOctreeDrawer<Octree>;
    using RangeSensor = RangeSensor3D<Dtype>;
    using Lidar = Lidar3D<Dtype>;
    using LidarFrame = LidarFrame3D<Dtype>;
    using DepthCamera = DepthCamera3D<Dtype>;
    using DepthFrame = DepthFrame3D<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;

#pragma region options_3d

    struct Options {
        bool use_cow_and_lady = false;
        std::string cow_and_lady_dir;
        std::string mesh_file = kProjectRootDir / "data" / "replica-hotel-0.ply";
        std::string traj_file = kProjectRootDir / "data" / "replica-hotel-0-traj.txt";
        std::string surface_mapping_config_file =
            kProjectRootDir / "config" / "template" /
            fmt::format("gp_occ_mapping_3d_lidar_{}.yaml", type_name<Dtype>());
        long stride = 1;
        Dtype surf_normal_scale = 0.5;
        Dtype test_res = 0.02;
        Dtype test_z = 0.0;
        long test_xs = 150;
        long test_ys = 100;
        bool test_io = false;
        bool hold = false;
    };

#pragma endregion

    Options options;
    bool options_parsed = false;
    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            (
                "use-cow-and-lady",
                po::bool_switch(&options.use_cow_and_lady),
                "use Cow and Lady dataset, otherwise use the mesh file and trajectory file"
            )(
                "cow-and-lady-dir",
                po::value<std::string>(&options.cow_and_lady_dir)->default_value(options.cow_and_lady_dir)->value_name("dir"),
                "directory containing the Cow and Lady dataset"
            )
            ("mesh-file", po::value<std::string>(&options.mesh_file)->default_value(options.mesh_file)->value_name("file"), "mesh file")
            ("traj-file", po::value<std::string>(&options.traj_file)->default_value(options.traj_file)->value_name("file"), "trajectory file")
            (
                "surface-mapping-config-file",
                po::value<std::string>(&options.surface_mapping_config_file)->default_value(options.surface_mapping_config_file)->value_name("file"),
                "SDF mapping config file"
            )
            ("stride", po::value<long>(&options.stride)->default_value(options.stride)->value_name("stride"), "stride")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale)->value_name("scale"), "surface normal scale")
            ("test-res", po::value<Dtype>(&options.test_res)->default_value(options.test_res)->value_name("res"), "test resolution")
            ("test-z", po::value<Dtype>(&options.test_z)->default_value(options.test_z)->value_name("z"), "test z")
            ("test-xs", po::value<long>(&options.test_xs)->default_value(options.test_xs)->value_name("xs"), "test xs")
            ("test-ys", po::value<long>(&options.test_ys)->default_value(options.test_ys)->value_name("ys"), "test ys")
            ("test-io", po::bool_switch(&options.test_io), "test IO")
            ("hold", po::bool_switch(&options.hold), "hold the window");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
            return;
        }
        po::notify(vm);
        options_parsed = true;
    } catch (std::exception &e) { std::cerr << e.what() << std::endl; }
    ASSERT_TRUE(options_parsed);

    // load setting
    const auto gp_surf_setting = std::make_shared<typename SurfaceMapping::Setting>();
    ASSERT_TRUE(gp_surf_setting->FromYamlFile(options.surface_mapping_config_file));

    // prepare the scene
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<RangeSensor> range_sensor = nullptr;
    bool is_lidar = false;
    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Vector3 area_min, area_max;
    if (options.use_cow_and_lady) {
        cow_and_lady = std::make_shared<CowAndLady>(options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        area_min = cow_and_lady->GetMapMin().cast<Dtype>();
        area_max = cow_and_lady->GetMapMax().cast<Dtype>();
        const auto depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = CowAndLady::kCameraCy;
    } else {
        const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options.mesh_file);
        area_min = mesh->GetMinBound().template cast<Dtype>();
        area_max = mesh->GetMaxBound().template cast<Dtype>();
        if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<LidarFrame>()) {
            auto lidar_frame_setting = std::dynamic_pointer_cast<typename LidarFrame::Setting>(
                gp_surf_setting->sensor_gp->sensor_frame);
            const auto lidar_setting = std::make_shared<typename Lidar::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<Lidar>(lidar_setting);
            range_sensor->AddMesh(options.mesh_file);
            is_lidar = true;
        } else if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<DepthFrame>()) {
            auto depth_frame_setting = std::dynamic_pointer_cast<typename DepthFrame::Setting>(
                gp_surf_setting->sensor_gp->sensor_frame);
            const auto depth_camera_setting = std::make_shared<typename DepthCamera::Setting>();
            *depth_camera_setting = depth_frame_setting->camera_intrinsic;
            range_sensor = std::make_shared<DepthCamera>(depth_camera_setting);
            range_sensor->AddMesh(options.mesh_file);
        } else {
            ERL_FATAL(
                "Unknown sensor_frame_type: {}",
                gp_surf_setting->sensor_gp->sensor_frame_type);
        }
        geometries.push_back(mesh);
        poses = Trajectory<Dtype>::LoadSe3(options.traj_file, false);
    }
    (void) is_lidar;

    // prepare the mapping
    SurfaceMapping gp(gp_surf_setting);

    // prepare the visualizer
    auto visualizer_setting = std::make_shared<Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    auto mesh_sensor_xyz = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1);
    const auto pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
    const auto pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
    const auto line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
    geometries.push_back(mesh_sensor);
    geometries.push_back(mesh_sensor_xyz);
    geometries.push_back(pcd_obs);
    geometries.push_back(pcd_surf_points);
    geometries.push_back(line_set_surf_normals);
    visualizer.AddGeometries(geometries);

    // animation callback
    long wp_idx = 0;
    double gp_update_dt = 0.0;
    double cnt = 0.0;
    long max_wp_idx =
        options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size());
    (void) wp_idx, (void) gp_update_dt, (void) cnt, (void) max_wp_idx;
    Matrix4 last_pose = Matrix4::Identity();
    auto callback = [&](Open3dVisualizerWrapper *wrapper,
                        open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();

        if (wp_idx >= max_wp_idx) {
            if (options.test_io) {
                ERL_BLOCK_TIMER_MSG("IO");
                const std::filesystem::path filename =
                    test_output_dir /
                    fmt::format("gp_occ_surface_mapping_3d_{}.bin", type_name<Dtype>());
                ERL_ASSERTM(
                    Serialization<SurfaceMapping>::Write(filename.string(), &gp),
                    "Failed to write to file: {}",
                    filename);
                // ERL_ASSERTM(gp.Write(filename), "Failed to write to file: {}", filename);
                SurfaceMapping gp_load(std::make_shared<typename SurfaceMapping::Setting>());
                ERL_ASSERTM(
                    erl::common::Serialization<SurfaceMapping>::Read(filename.string(), &gp_load),
                    "Failed to read from file: {}",
                    filename);
                // ERL_ASSERTM(gp_load.Read(filename), "Failed to read from file: {}", filename);
                ERL_ASSERTM(gp == gp_load, "gp != gp_load");
            }
            wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
            vis->Close();                            // close the window
            return false;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        Matrix3 rotation;
        Vector3 translation;
        ERL_INFO("wp_idx: {}", wp_idx);

        cv::Mat ranges_img;
        MatrixX ranges;
        double dt;
        {
            ERL_BLOCK_TIMER_MSG_TIME("data loading", dt);
            if (options.use_cow_and_lady) {
                const auto frame = (*cow_and_lady)[wp_idx];
                rotation = frame.rotation.cast<Dtype>();
                translation = frame.translation.cast<Dtype>();
                ranges = frame.depth.cast<Dtype>();
                ranges_img = frame.depth_jet;
            } else {
                std::tie(rotation, translation) = poses[wp_idx];
                ranges = range_sensor->Scan(rotation, translation);
                std::tie(rotation, translation) =
                    range_sensor->GetOpticalPose(rotation, translation);
            }
        }
        ERL_TRACY_PLOT("data loading (ms)", dt);
        wp_idx += options.stride;

        {
            ERL_BLOCK_TIMER_MSG_TIME("gp.Update", dt);
            ERL_WARN_COND(!gp.Update(rotation, translation, ranges), "gp.Update failed.");
        }
        gp_update_dt = (gp_update_dt * cnt + dt) / (cnt + 1.0);
        cnt += 1.0;
        double gp_update_fps = 1000.0 / gp_update_dt;
        ERL_INFO("gp_update_dt: {:.3f} ms, gp_update_fps: {:.3f} fps", gp_update_dt, gp_update_fps);
        ERL_TRACY_PLOT("gp_update (ms)", gp_update_dt);
        ERL_TRACY_PLOT("gp_update (fps)", gp_update_fps);

        // update visualization
        /// update the image
        if (!options.use_cow_and_lady) {
            for (long i = 0; i < ranges.size(); ++i) {
                Dtype &range = ranges.data()[i];
                if (range < 0.0 || range > 1000.0) { range = 0.0; }
            }
            if (is_lidar) {
                cv::eigen2cv(MatrixX(ranges.transpose()), ranges_img);
                cv::flip(ranges_img, ranges_img, 0);
                cv::resize(ranges_img, ranges_img, {0, 0}, 2, 2);
            } else {
                cv::eigen2cv(ranges, ranges_img);
            }
            cv::normalize(ranges_img, ranges_img, 0, 255, cv::NORM_MINMAX);
            ranges_img.convertTo(ranges_img, CV_8UC1);
            cv::applyColorMap(ranges_img, ranges_img, cv::COLORMAP_JET);
        }
        cv::putText(
            ranges_img,
            fmt::format("update {:.2f} fps", gp_update_fps),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        cv::imshow("ranges", ranges_img);
        cv::waitKey(1);
        /// update the sensor mesh
        Matrix4 last_pose_inv = last_pose.inverse();
        Matrix4 cur_pose = Matrix4::Identity();
        cur_pose.template topLeftCorner<3, 3>() = rotation;
        cur_pose.template topRightCorner<3, 1>() = translation;
        Matrix4 delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose.template cast<double>());
        mesh_sensor_xyz->Transform(delta_pose.template cast<double>());
        /// update the observation point cloud
        pcd_obs->points_.clear();
        pcd_obs->colors_.clear();
        pcd_obs->points_.reserve(gp.GetSensorGp()->GetSensorFrame()->GetHitPointsWorld().size());
        for (const auto &point: gp.GetSensorGp()->GetSensorFrame()->GetHitPointsWorld()) {
            pcd_obs->points_.emplace_back(point.template cast<double>());
        }
        pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        /// update the surface point cloud and normals
        pcd_surf_points->points_.clear();
        line_set_surf_normals->points_.clear();
        line_set_surf_normals->lines_.clear();
        for (auto it = gp.BeginSurfaceData(), end = gp.EndSurfaceData(); it != end; ++it) {
            auto &surface_data = *it;
            const Vector3 &position = surface_data.position;
            const Vector3 &normal = surface_data.normal;
            ERL_ASSERTM(
                std::abs(normal.norm() - 1.0) < 1.e-5,
                "normal.norm() = {:.6f}",
                normal.norm());
            pcd_surf_points->points_.emplace_back(position.template cast<double>());
            line_set_surf_normals->points_.emplace_back(position.template cast<double>());
            line_set_surf_normals->points_.emplace_back(
                (position + options.surf_normal_scale * normal).template cast<double>());
            line_set_surf_normals->lines_.emplace_back(
                line_set_surf_normals->points_.size() - 2,
                line_set_surf_normals->points_.size() - 1);
        }
        if (!pcd_surf_points->points_.empty()) {
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        auto duration_total = std::chrono::duration<Dtype, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && options.test_io) {
            ERL_BLOCK_TIMER_MSG("IO");
            std::string bin_file = fmt::format("gp_occ_surf_mapping_3d_{}.bin", type_name<Dtype>());
            bin_file = test_output_dir / bin_file;
            ERL_ASSERTM(
                Serialization<SurfaceMapping>::Write(bin_file, &gp),
                "Failed to write to file: {}",
                bin_file);
            SurfaceMapping gp_load(std::make_shared<typename SurfaceMapping::Setting>());
            ERL_ASSERTM(
                Serialization<SurfaceMapping>::Read(bin_file, &gp_load),
                "Failed to read from file: {}",
                bin_file);
            ERL_ASSERTM(gp == gp_load, "gp != gp_load");
        }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();

    auto drawer_setting = std::make_shared<typename OctreeDrawer::Setting>();
    drawer_setting->area_min = area_min.template cast<double>();
    drawer_setting->area_max = area_max.template cast<double>();
    drawer_setting->occupied_only = true;
    OctreeDrawer octree_drawer(drawer_setting, gp.GetTree());
    auto mesh = geometries[0];
    geometries = octree_drawer.GetBlankGeometries();
    geometries.push_back(mesh);
    octree_drawer.DrawLeaves(geometries);
    visualizer.Reset();
    visualizer.AddGeometries(geometries);
    visualizer.Show();
}

template<typename Dtype>
void
TestImpl2D() {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;

    using SurfaceMapping = GpOccSurfaceMapping<Dtype, 2>;
    using Quadtree = OccupancyQuadtree<Dtype>;
    using QuadtreeDrawer = OccupancyQuadtreeDrawer<Quadtree>;
    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

#pragma region options_2d

    struct Options {
        std::string gazebo_train_file = kProjectRootDir / "data" / "gazebo";
        std::string house_expo_map_file = kProjectRootDir / "data" / "house_expo_room_1451.json";
        std::string house_expo_traj_file = kProjectRootDir / "data" / "house_expo_room_1451.csv";
        std::string ucsd_fah_2d_file = kProjectRootDir / "data" / "ucsd_fah_2d.dat";
        std::string surface_mapping_config_file =
            kProjectRootDir / "config" / "template" /
            fmt::format("gp_occ_mapping_2d_{}.yaml", type_name<Dtype>());
        bool use_gazebo_room_2d = false;
        bool use_house_expo_lidar_2d = false;
        bool use_ucsd_fah_2d = false;
        bool visualize = false;
        bool test_io = false;
        bool hold = false;
        long stride = 1;
        long init_frame = 0;
        Dtype map_resolution = 0.025;
        Dtype surf_normal_scale = 1.0;
    };

#pragma endregion
    Options options;
    bool options_parsed = false;
    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            ("use-gazebo-room-2d", po::bool_switch(&options.use_gazebo_room_2d)->default_value(options.use_gazebo_room_2d), "Use Gazebo data")
            ("use-house-expo-lidar-2d", po::bool_switch(&options.use_house_expo_lidar_2d)->default_value(options.use_house_expo_lidar_2d), "Use HouseExpo data")
            ("use-ucsd-fah-2d", po::bool_switch(&options.use_ucsd_fah_2d)->default_value(options.use_ucsd_fah_2d), "Use UCSD FAH 2D data")
            ("stride", po::value<long>(&options.stride)->default_value(options.stride), "stride for running the sequence")
            ("init-frame", po::value<long>(&options.init_frame)->default_value(options.init_frame), "Initial frame index")
            ("map-resolution", po::value<Dtype>(&options.map_resolution)->default_value(options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale), "Surface normal scale")
            ("visualize", po::bool_switch(&options.visualize)->default_value(options.visualize), "Visualize the mapping")
            ("test-io", po::bool_switch(&options.test_io)->default_value(options.test_io), "Test IO")
            ("hold", po::bool_switch(&options.hold)->default_value(options.hold), "Hold the test until a key is pressed")
            (
                "house-expo-map-file",
                po::value<std::string>(&options.house_expo_map_file)->default_value(options.house_expo_map_file)->value_name("file"),
                "HouseExpo map file"
            )(
                "house-expo-traj-file",
                po::value<std::string>(&options.house_expo_traj_file)->default_value(options.house_expo_traj_file)->value_name("file"),
                "HouseExpo trajectory file"
            )(
                "gazebo-train-file",
                po::value<std::string>(&options.gazebo_train_file)->default_value(options.gazebo_train_file)->value_name("file"),
                "Gazebo train data file"
            )(
                "ucsd-fah-2d-dat-file",
                po::value<std::string>(&options.ucsd_fah_2d_file)->default_value(options.ucsd_fah_2d_file)->value_name("file"),
                "UCSD FAH 2D dat file"
            )(
                "surface-mapping-config-file",
                po::value<std::string>(&options.surface_mapping_config_file)->default_value(options.surface_mapping_config_file)->value_name("file"),
                "Surface mapping config file");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
            return;
        }
        po::notify(vm);
        options_parsed = true;
    } catch (std::exception &e) { std::cerr << e.what() << "\n"; }
    ASSERT_TRUE(options_parsed);

    ASSERT_TRUE(
        options.use_gazebo_room_2d || options.use_house_expo_lidar_2d || options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-room-2d, --use-house-expo-lidar-2d, "
           "--use-ucsd-fah-2d.";
    if (options.use_gazebo_room_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.gazebo_train_file))
            << "Gazebo train data file " << options.gazebo_train_file << " does not exist.";
    }
    if (options.use_house_expo_lidar_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_map_file))
            << "HouseExpo map file " << options.house_expo_map_file << " does not exist.";
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_traj_file))
            << "HouseExpo trajectory file " << options.house_expo_traj_file << " does not exist.";
    }
    if (options.use_ucsd_fah_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.ucsd_fah_2d_file))
            << "ROS bag dat file " << options.ucsd_fah_2d_file << " does not exist.";
    }

    // load the scene
    long max_update_cnt;
    std::vector<VectorX> train_angles;
    std::vector<VectorX> train_ranges;
    std::vector<std::pair<Matrix2, Vector2>> train_poses;
    Vector2 map_min(0, 0);
    Vector2 map_max(0, 0);
    Vector2 map_resolution(options.map_resolution, options.map_resolution);
    Eigen::Vector2i map_padding(10, 10);
    Matrix2X cur_traj;

    if (options.use_gazebo_room_2d) {
        auto train_data_loader = GazeboRoom2D::TrainDataLoader(options.gazebo_train_file);
        max_update_cnt = (train_data_loader.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (long i = options.init_frame; i < train_data_loader.size(); i += options.stride) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles.cast<Dtype>());
            train_ranges.push_back(df.ranges.cast<Dtype>());
            train_poses.emplace_back(df.rotation.cast<Dtype>(), df.translation.cast<Dtype>());
            cur_traj.col(cnt) << df.translation.cast<Dtype>();
            bar->Update();
            ++cnt;
        }
        map_min = GazeboRoom2D::kMapMin.cast<Dtype>();
        map_max = GazeboRoom2D::kMapMax.cast<Dtype>();
        map_padding.setZero();

        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_house_expo_lidar_2d) {
        HouseExpoMap house_expo_map(options.house_expo_map_file, 0.2);
        map_min = house_expo_map.GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .minCoeff()
                      .cast<Dtype>();
        map_max = house_expo_map.GetMeterSpace()
                      ->GetSurface()
                      ->vertices.rowwise()
                      .maxCoeff()
                      .cast<Dtype>();
        auto lidar_setting = std::make_shared<Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
        auto trajectory = LoadAndCastCsvFile<double>(
            options.house_expo_traj_file.c_str(),
            [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt =
            (static_cast<long>(trajectory.size()) - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = options.init_frame; i < trajectory.size(); i += options.stride) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << static_cast<Dtype>(waypoint[0]), static_cast<Dtype>(waypoint[1]);

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            VectorX ranges = lidar.Scan(rotation, translation, scan_in_parallel).cast<Dtype>();
            ranges += GenerateGaussianNoise<Dtype>(ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles().cast<Dtype>());
            train_ranges.push_back(ranges);
            train_poses.emplace_back(rotation.cast<Dtype>(), translation.cast<Dtype>());
            bar->Update();
            ++cnt;
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_ucsd_fah_2d) {
        UcsdFah2D ucsd_fah(options.ucsd_fah_2d_file);
        map_min = UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = UcsdFah2D::kMapMax.cast<Dtype>();
        // prepare buffer
        max_update_cnt =
            (ucsd_fah.Size() - options.init_frame + options.stride - 1) / options.stride;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        // load data into buffer
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        long cnt = 0;
        for (long i = options.init_frame; i < ucsd_fah.Size(); i += options.stride, ++cnt) {
            auto
                [sequence_number,
                 timestamp,
                 header_timestamp,
                 rotation,
                 translation,
                 angles,
                 ranges] = ucsd_fah[i];
            cur_traj.col(cnt) << translation.cast<Dtype>();
            train_angles.emplace_back(angles.cast<Dtype>());
            train_ranges.emplace_back(ranges.cast<Dtype>());
            train_poses.emplace_back(rotation.cast<Dtype>(), translation.cast<Dtype>());
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    }
    max_update_cnt = cur_traj.cols();

    // load setting
    const auto gp_setting = std::make_shared<typename SurfaceMapping::Setting>();
    ASSERT_TRUE(gp_setting->FromYamlFile(options.surface_mapping_config_file));
    gp_setting->sensor_gp->sensor_frame->angle_min = train_angles[0].minCoeff();
    gp_setting->sensor_gp->sensor_frame->angle_max = train_angles[0].maxCoeff();
    gp_setting->sensor_gp->sensor_frame->num_rays = train_angles[0].size();
    ERL_INFO(
        "Sensor frame angle range: [{}, {}], num rays: {}",
        gp_setting->sensor_gp->sensor_frame->angle_min,
        gp_setting->sensor_gp->sensor_frame->angle_max,
        gp_setting->sensor_gp->sensor_frame->num_rays);
    SurfaceMapping gp(gp_setting);

    // prepare the visualizer
    auto drawer_setting = std::make_shared<typename QuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->scaling = gp_setting->scaling;
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    QuadtreeDrawer drawer(drawer_setting);

    std::filesystem::path img_dir = test_output_dir / "images";
    std::filesystem::create_directories(img_dir);
    int img_cnt = 0;

    cv::Scalar trajectory_color(0, 0, 0, 255);
    cv::Mat img;
    bool drawer_connected = false;
    const bool update_occupancy = gp_setting->update_occupancy;
    if (options.visualize) {
        if (update_occupancy) {
            drawer.DrawLeaves(img);
        } else {
            drawer.DrawTree(img);
        }
    }
    if (options.hold) {
        std::cout << "Press any key to start" << std::endl;
        cv::waitKey();  // wait for any key
    }

    // start the mapping
    std::string bin_file = fmt::format("gp_occ_surface_mapping_2d_{}.bin", type_name<Dtype>());
    bin_file = test_output_dir / bin_file;
    double t_ms = 0;
    for (long i = 0; i < max_update_cnt; i++) {
        const auto &[rotation, translation] = train_poses[i];
        const VectorX &ranges = train_ranges[i];
        auto t0 = std::chrono::high_resolution_clock::now();
        EXPECT_TRUE(gp.Update(rotation, translation, ranges));
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update time: {:f} ms.", dt);
        t_ms += dt;

        if (options.visualize) {
            bool pixel_based = true;
            if (!drawer_connected) {
                drawer.SetQuadtree(gp.GetTree());
                drawer_connected = true;
            }
            img.setTo(cv::Scalar(128, 128, 128, 255));
            if (update_occupancy) {
                drawer.DrawLeaves(img);
            } else {
                drawer.DrawTree(img);
            }
            for (auto it = gp.BeginSurfaceData(), end = gp.EndSurfaceData(); it != end; ++it) {
                Eigen::Vector2i position_px = drawer.GetPixelCoordsForPositions(it->position, true);
                cv::Point position_px_cv(position_px[0], position_px[1]);
                cv::circle(
                    img,
                    position_px_cv,
                    2,
                    cv::Scalar(0, 0, 255, 255),
                    -1);  // draw surface point
                Eigen::Vector2i normal_px =
                    drawer.GetPixelCoordsForVectors(it->normal * options.surf_normal_scale);
                cv::Point arrow_end_px(
                    position_px[0] + normal_px[0],
                    position_px[1] + normal_px[1]);
                cv::arrowedLine(
                    img,
                    position_px_cv,
                    arrow_end_px,
                    cv::Scalar(0, 0, 255, 255),
                    1,
                    cv::LINE_8,
                    0,
                    0.1);
            }

            // draw trajectory
            DrawTrajectoryInplace<Dtype>(
                img,
                cur_traj.block(0, 0, 2, i),
                drawer.GetGridMapInfo(),
                trajectory_color,
                2,
                pixel_based);

            // draw fps
            cv::putText(
                img,
                std::to_string(1000.0 / dt),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                1,
                cv::Scalar(0, 255, 0, 255),
                2);

            cv::imshow("GP Occ Surface Mapping", img);
            cv::imwrite(img_dir / fmt::format("{:04d}.png", img_cnt++), img);

            int key = cv::waitKey(10);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
        double avg_dt = t_ms / static_cast<double>(i + 1);
        double fps = 1000.0 / avg_dt;
        ERL_INFO("Average update time: {:f} ms, Average fps: {:f}", avg_dt, fps);
        std::cout << "=====================================" << std::endl;
    }

    if (options.test_io) {
        ASSERT_TRUE(Serialization<SurfaceMapping>::Write(bin_file, &gp));
        SurfaceMapping gp_read(std::make_shared<typename SurfaceMapping::Setting>());
        ASSERT_TRUE(Serialization<SurfaceMapping>::Read(bin_file, &gp_read));
        ASSERT_TRUE(gp == gp_read) << "Loaded GP is not equal to the original GP.";
    }

    if (options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        cv::waitKey();  // wait for any key
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        double wait_time = 10.0;
        while (
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() <
            wait_time) {
            int key = cv::waitKey(10);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
    }
}

TEST(GpOccSurfaceMapping, 3Dd) { TestImpl3D<double>(); }

TEST(GpOccSurfaceMapping, 3Df) { TestImpl3D<float>(); }

TEST(GpOccSurfaceMapping, 2Dd) { TestImpl2D<double>(); }

TEST(GpOccSurfaceMapping, 2Df) { TestImpl2D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
