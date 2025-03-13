#include "erl_common/csv.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/range_sensor_3d.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"

#include <boost/program_options.hpp>

int g_argc = 0;
char **g_argv = nullptr;
const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;

template<typename Dtype>
void
TestImpl3D() {
    GTEST_PREPARE_OUTPUT_DIR();

    using GpOccSurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 3>;
    using SurfaceMappingOctree = erl::sdf_mapping::SurfaceMappingOctree<Dtype>;
    using RangeSensor3D = erl::geometry::RangeSensor3D<Dtype>;
    using Lidar3D = erl::geometry::Lidar3D<Dtype>;
    using LidarFrame3D = erl::geometry::LidarFrame3D<Dtype>;
    using DepthCamera3D = erl::geometry::DepthCamera3D<Dtype>;
    using DepthFrame3D = erl::geometry::DepthFrame3D<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;

    // load setting from command line
    struct Options {
        bool use_cow_and_lady = false;
        std::string cow_and_lady_dir;
        std::string mesh_file = kProjectRootDir / "data" / "replica-hotel-0.ply";
        std::string traj_file = kProjectRootDir / "data" / "replica-hotel-0-traj.txt";
        std::string surface_mapping_config_file = kProjectRootDir / "config" / fmt::format("gp_occ_mapping_3d_lidar_{}.yaml", type_name<Dtype>());
        long stride = 1;
        Dtype test_res = 0.02;
        Dtype test_z = 0.0;
        long test_xs = 150;
        long test_ys = 100;
        bool test_io = false;
        bool hold = false;
    };

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
    const auto gp_surf_setting = std::make_shared<typename GpOccSurfaceMapping::Setting>();
    ASSERT_TRUE(gp_surf_setting->FromYamlFile(options.surface_mapping_config_file));

    // prepare the scene
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<RangeSensor3D> range_sensor = nullptr;
    std::shared_ptr<erl::geometry::CowAndLady> cow_and_lady = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Vector3 area_min, area_max;
    if (options.use_cow_and_lady) {
        cow_and_lady = std::make_shared<erl::geometry::CowAndLady>(options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        area_min = cow_and_lady->GetMapMin().cast<Dtype>();
        area_max = cow_and_lady->GetMapMax().cast<Dtype>();
        const auto depth_frame_setting = std::make_shared<typename DepthFrame3D::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = erl::geometry::CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = erl::geometry::CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = erl::geometry::CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = erl::geometry::CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = erl::geometry::CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = erl::geometry::CowAndLady::kCameraCy;
    } else {
        const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options.mesh_file);
        area_min = mesh->GetMinBound().template cast<Dtype>();
        area_max = mesh->GetMaxBound().template cast<Dtype>();
        if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<LidarFrame3D>()) {
            const auto lidar_frame_setting = std::dynamic_pointer_cast<typename LidarFrame3D::Setting>(gp_surf_setting->sensor_gp->sensor_frame);
            const auto lidar_setting = std::make_shared<typename Lidar3D::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<Lidar3D>(lidar_setting);
            range_sensor->AddMesh(options.mesh_file);
        } else if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<DepthFrame3D>()) {
            const auto depth_frame_setting = std::dynamic_pointer_cast<typename DepthFrame3D::Setting>(gp_surf_setting->sensor_gp->sensor_frame);
            const auto depth_camera_setting = std::make_shared<typename DepthCamera3D::Setting>();
            *depth_camera_setting = depth_frame_setting->camera_intrinsic;
            range_sensor = std::make_shared<DepthCamera3D>(depth_camera_setting);
            range_sensor->AddMesh(options.mesh_file);
        } else {
            ERL_FATAL("Unknown sensor_frame_type: {}", gp_surf_setting->sensor_gp->sensor_frame_type);
        }
        geometries.push_back(mesh);
        poses = erl::geometry::Trajectory<Dtype>::LoadSe3(options.traj_file, false);
    }

    // prepare the mapping
    GpOccSurfaceMapping gp(gp_surf_setting);

    // prepare the visualizer
    const auto visualizer_setting = std::make_shared<erl::geometry::Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    erl::geometry::Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    const auto mesh_sensor_xyz = erl::geometry::CreateAxisMesh(Matrix4::Identity().template cast<double>(), 0.1);
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
    const long max_wp_idx = options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size());
    Matrix4 last_pose = Matrix4::Identity();
    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();

        if (wp_idx >= max_wp_idx) {
            if (options.test_io) {
                const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
                (void) timer;
                const auto filename = test_output_dir / fmt::format("gp_occ_surface_mapping_3d_{}.bin", type_name<Dtype>());
                ERL_ASSERTM(gp.Write(filename), "Failed to write to file: {}", filename);
                GpOccSurfaceMapping gp_load(std::make_shared<typename GpOccSurfaceMapping::Setting>());
                ERL_ASSERTM(gp_load.Read(filename), "Failed to read from file: {}", filename);
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

        cv::Mat depth_jet;
        Eigen::MatrixX<Dtype> ranges;
        double dt;
        {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("data loading", &dt);
            (void) timer;
            if (options.use_cow_and_lady) {
                // ReSharper disable once CppUseStructuredBinding
                const auto frame = (*cow_and_lady)[wp_idx];
                rotation = frame.rotation.cast<Dtype>();
                translation = frame.translation.cast<Dtype>();
                ranges = frame.depth.cast<Dtype>();
                depth_jet = frame.depth_jet;
            } else {
                std::tie(rotation, translation) = poses[wp_idx];
                ranges = range_sensor->Scan(rotation, translation);
                std::tie(rotation, translation) = range_sensor->GetOpticalPose(rotation, translation);
            }
        }
        ERL_TRACY_PLOT("data loading (ms)", dt);
        wp_idx += options.stride;

        {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Update", &dt);
            (void) timer;
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
            Eigen::MatrixXd ranges_img = Eigen::MatrixXd::Zero(ranges.rows(), ranges.cols());
            Dtype min_range = std::numeric_limits<Dtype>::max();
            Dtype max_range = std::numeric_limits<Dtype>::lowest();
            for (long i = 0; i < ranges.size(); ++i) {
                Dtype &range = ranges.data()[i];
                if (range < 0.0 || !std::isfinite(range)) { continue; }
                min_range = std::min(min_range, range);
                max_range = std::max(max_range, range);
                ranges_img.data()[i] = range;
            }
            ranges_img = (ranges_img.array() - min_range) / (max_range - min_range);
            cv::eigen2cv(ranges_img, depth_jet);
            depth_jet.convertTo(depth_jet, CV_8UC1, 255);
            cv::cvtColor(depth_jet, depth_jet, cv::COLOR_GRAY2BGR);
            cv::applyColorMap(depth_jet, depth_jet, cv::COLORMAP_JET);
        }
        cv::putText(depth_jet, fmt::format("update {:.2f} fps", gp_update_fps), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::imshow("ranges", depth_jet);
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
        for (const auto &point: gp.GetSensorGp()->GetSensorFrame()->GetHitPointsWorld()) { pcd_obs->points_.emplace_back(point.template cast<double>()); }
        pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        /// update the surface point cloud and normals
        pcd_surf_points->points_.clear();
        line_set_surf_normals->points_.clear();
        line_set_surf_normals->lines_.clear();
        if (const auto octree = gp.GetTree(); octree != nullptr) {
            for (auto it = octree->BeginLeaf(), end = octree->EndLeaf(); it != end; ++it) {
                if (!it->HasSurfaceData()) { continue; }
                auto &surface_data = gp.GetSurfaceDataManager()[it->surface_data_index];
                const Vector3 &position = surface_data.position;
                const Vector3 &normal = surface_data.normal;
                ERL_ASSERTM(std::abs(normal.norm() - 1.0) < 1.e-5, "normal.norm() = {:.6f}", normal.norm());
                pcd_surf_points->points_.emplace_back(position.template cast<double>());
                line_set_surf_normals->points_.emplace_back(position.template cast<double>());
                line_set_surf_normals->points_.emplace_back((position + 0.1 * normal).template cast<double>());
                line_set_surf_normals->lines_.emplace_back(line_set_surf_normals->points_.size() - 2, line_set_surf_normals->points_.size() - 1);
            }
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const auto duration_total = std::chrono::duration<Dtype, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && options.test_io) {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
            (void) timer;
            const auto filename = test_output_dir / fmt::format("gp_occ_surface_mapping_3d_{}.bin", type_name<Dtype>());
            ERL_ASSERTM(gp.Write(filename), "Failed to write to file: {}", filename);
            GpOccSurfaceMapping gp_load(std::make_shared<typename GpOccSurfaceMapping::Setting>());
            ERL_ASSERTM(gp_load.Read(filename), "Failed to read from file: {}", filename);
            ERL_ASSERTM(gp == gp_load, "gp != gp_load");
        }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();

    auto drawer_setting = std::make_shared<typename SurfaceMappingOctree::Drawer::Setting>();
    drawer_setting->area_min = area_min.template cast<double>();
    drawer_setting->area_max = area_max.template cast<double>();
    drawer_setting->occupied_only = true;
    typename SurfaceMappingOctree::Drawer octree_drawer(drawer_setting, gp.GetTree());
    auto mesh = geometries[0];
    geometries = octree_drawer.GetBlankGeometries();
    geometries.push_back(mesh);
    octree_drawer.DrawLeaves(geometries);
    visualizer.Reset();
    visualizer.AddGeometries(geometries);
    visualizer.Show();
}

TEST(GpOccSurfaceMapping, 3Dd) { TestImpl3D<double>(); }

TEST(GpOccSurfaceMapping, 3Df) { TestImpl3D<float>(); }

template<typename Dtype>
void
TestImpl2D() {
    GTEST_PREPARE_OUTPUT_DIR();
    using namespace erl::common;
    using GpOccSurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 2>;
    using SurfaceMappingQuadtree = erl::sdf_mapping::SurfaceMappingQuadtree<Dtype>;
    using QuadtreeDrawer = typename SurfaceMappingQuadtree::Drawer;
    using Lidar2D = erl::geometry::Lidar2D;
    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    // load setting from command line
    struct Options {
        std::string gazebo_train_file = kProjectRootDir / "data" / "gazebo_train.dat";
        std::string house_expo_map_file = kProjectRootDir / "data" / "house_expo_room_1451.json";
        std::string house_expo_traj_file = kProjectRootDir / "data" / "house_expo_room_1451.csv";
        std::string ucsd_fah_2d_file = kProjectRootDir / "data" / "ucsd_fah_2d.dat";
        std::string surface_mapping_config_file = kProjectRootDir / "config" / fmt::format("gp_occ_mapping_2d_{}.yaml", type_name<Dtype>());
        bool use_gazebo_room_2d = false;
        bool use_house_expo_lidar_2d = false;
        bool use_ucsd_fah_2d = false;
        bool visualize = false;
        bool test_io = false;
        bool hold = false;
        int stride = 1;
        int init_frame = 0;
        Dtype map_resolution = 0.025;
        Dtype surf_normal_scale = 0.35;
    };

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
            ("stride", po::value<int>(&options.stride)->default_value(options.stride), "stride for running the sequence")
            ("map-resolution", po::value<Dtype>(&options.map_resolution)->default_value(options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale), "Surface normal scale")
            ("init-frame", po::value<int>(&options.init_frame)->default_value(options.init_frame), "Initial frame index")
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

    ASSERT_TRUE(options.use_gazebo_room_2d || options.use_house_expo_lidar_2d || options.use_ucsd_fah_2d)
        << "Please specify one of --use-gazebo-data, --use-house-expo-data, --use-ucsd-fah-2d.";
    if (options.use_gazebo_room_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.gazebo_train_file)) << "Gazebo train data file " << options.gazebo_train_file << " does not exist.";
    }
    if (options.use_house_expo_lidar_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_map_file)) << "HouseExpo map file " << options.house_expo_map_file << " does not exist.";
        ASSERT_TRUE(std::filesystem::exists(options.house_expo_traj_file))
            << "HouseExpo trajectory file " << options.house_expo_traj_file << " does not exist.";
    }
    if (options.use_ucsd_fah_2d) {
        ASSERT_TRUE(std::filesystem::exists(options.ucsd_fah_2d_file)) << "ROS bag dat file " << options.ucsd_fah_2d_file << " does not exist.";
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
        auto train_data_loader = erl::geometry::GazeboRoom2D::TrainDataLoader(options.gazebo_train_file);
        max_update_cnt = static_cast<long>(train_data_loader.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = options.init_frame; i < static_cast<int>(train_data_loader.size()); i += options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles.cast<Dtype>());
            train_ranges.push_back(df.ranges.cast<Dtype>());
            train_poses.emplace_back(df.rotation.cast<Dtype>(), df.translation.cast<Dtype>());
            cur_traj.col(cnt) << df.translation.cast<Dtype>();
            const Dtype &x = cur_traj(0, cnt);
            const Dtype &y = cur_traj(1, cnt);
            if (x < map_min[0]) { map_min[0] = x; }
            if (x > map_max[0]) { map_max[0] = x; }
            if (y < map_min[1]) { map_min[1] = y; }
            if (y > map_max[1]) { map_max[1] = y; }
            map_min[0] -= 0.15;
            map_min[1] -= 0.4;
            map_max[0] += 0.2;
            map_max[1] += 0.15;
            map_resolution.array() = 0.02;
            map_padding.setZero();
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_house_expo_lidar_2d) {
        erl::geometry::HouseExpoMap house_expo_map(options.house_expo_map_file, 0.2);
        map_min = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().minCoeff().cast<Dtype>();
        map_max = house_expo_map.GetMeterSpace()->GetSurface()->vertices.rowwise().maxCoeff().cast<Dtype>();
        auto lidar_setting = std::make_shared<Lidar2D::Setting>();
        lidar_setting->num_lines = 720;
        Lidar2D lidar(lidar_setting, house_expo_map.GetMeterSpace());
        auto trajectory = LoadAndCastCsvFile<double>(options.house_expo_traj_file.c_str(), [](const std::string &str) -> double { return std::stod(str); });
        max_update_cnt = static_cast<long>(trajectory.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = options.init_frame; i < trajectory.size(); i += options.stride, cnt++) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << static_cast<Dtype>(waypoint[0]), static_cast<Dtype>(waypoint[1]);

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            VectorX lidar_ranges = lidar.Scan(rotation, translation, scan_in_parallel).cast<Dtype>();
            lidar_ranges += GenerateGaussianNoise<Dtype>(lidar_ranges.size(), 0.0, 0.01);
            train_angles.push_back(lidar.GetAngles().cast<Dtype>());
            train_ranges.push_back(lidar_ranges);
            train_poses.emplace_back(rotation.cast<Dtype>(), translation.cast<Dtype>());
            bar->Update();
        }
        train_angles.resize(cnt);
        train_ranges.resize(cnt);
        train_poses.resize(cnt);
        cur_traj.conservativeResize(2, cnt);
    } else if (options.use_ucsd_fah_2d) {
        erl::geometry::UcsdFah2D ucsd_fah(options.ucsd_fah_2d_file);
        map_min = erl::geometry::UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = erl::geometry::UcsdFah2D::kMapMax.cast<Dtype>();
        // prepare buffer
        max_update_cnt = (ucsd_fah.Size() - options.init_frame) / options.stride + 1;
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
            auto [sequence_number, timestamp, header_timestamp, rotation, translation, angles, ranges] = ucsd_fah[i];
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
    } else {
        std::cerr << "Please specify one of --use-gazebo-data, --use-house-expo-data, --use-ucsd-fah-2d." << std::endl;
        return;
    }
    max_update_cnt = cur_traj.cols();

    // load setting
    const auto gp_setting = std::make_shared<typename GpOccSurfaceMapping::Setting>();
    ASSERT_TRUE(gp_setting->FromYamlFile(options.surface_mapping_config_file));
    gp_setting->sensor_gp->sensor_frame->angle_min = train_angles[0].minCoeff();
    gp_setting->sensor_gp->sensor_frame->angle_max = train_angles[0].maxCoeff();
    gp_setting->sensor_gp->sensor_frame->num_rays = train_angles[0].size();
    GpOccSurfaceMapping gp(gp_setting);

    // prepare the visualizer
    auto drawer_setting = std::make_shared<typename QuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    auto drawer = std::make_shared<QuadtreeDrawer>(drawer_setting);

    std::vector<std::pair<cv::Point, cv::Point>> arrowed_lines;
    auto &surface_data_manager = gp.GetSurfaceDataManager();
    drawer->SetDrawTreeCallback([&](const QuadtreeDrawer *self, cv::Mat &img, typename SurfaceMappingQuadtree::TreeIterator &it) {
        const uint32_t cluster_depth = gp.GetTree()->GetTreeDepth() - gp_setting->cluster_level;
        const auto grid_map_info = self->GetGridMapInfo();
        if (it->GetDepth() == cluster_depth) {
            Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(Vector2(it.GetX(), it.GetY()));
            const cv::Point position_px_cv(position_px[0], position_px[1]);
            cv::circle(img, position_px_cv, 2, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
            return;
        }
        if (!it->HasSurfaceData()) { return; }
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data.position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, cv::Point(position_px[0], position_px[1]), 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data.normal * options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });
    drawer->SetDrawLeafCallback([&](const QuadtreeDrawer *self, cv::Mat &img, typename SurfaceMappingQuadtree::LeafIterator &it) {
        if (!it->HasSurfaceData()) { return; }
        const auto grid_map_info = self->GetGridMapInfo();
        auto &surface_data = surface_data_manager[it->surface_data_index];
        Eigen::Vector2i position_px = grid_map_info->MeterToPixelForPoints(surface_data.position);
        cv::Point position_px_cv(position_px[0], position_px[1]);
        cv::circle(img, position_px_cv, 1, cv::Scalar(0, 0, 255, 255), -1);  // draw surface point
        Eigen::Vector2i normal_px = grid_map_info->MeterToPixelForVectors(surface_data.normal * options.surf_normal_scale);
        cv::Point arrow_end_px(position_px[0] + normal_px[0], position_px[1] + normal_px[1]);
        arrowed_lines.emplace_back(position_px_cv, arrow_end_px);
    });

    cv::Scalar trajectory_color(0, 0, 0, 255);
    cv::Mat img;
    bool drawer_connected = false;
    const bool update_occupancy = gp_setting->update_occupancy;
    if (options.visualize) {
        if (update_occupancy) {
            drawer->DrawLeaves(img);
        } else {
            drawer->DrawTree(img);
        }
    }
    if (options.hold) {
        std::cout << "Press any key to start" << std::endl;
        cv::waitKey();  // wait for any key
    }

    // start the mapping
    auto grid_map_info = drawer->GetGridMapInfo();
    const std::string bin_file = test_output_dir / fmt::format("gp_occ_surface_mapping_2d_{}.bin", type_name<Dtype>());
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
                drawer->SetQuadtree(gp.GetTree());
                drawer_connected = true;
            }
            img.setTo(cv::Scalar(128, 128, 128, 255));
            arrowed_lines.clear();
            if (update_occupancy) {
                drawer->DrawLeaves(img);
            } else {
                drawer->DrawTree(img);
            }
            for (auto &[position_px_cv, arrow_end_px]: arrowed_lines) {
                cv::arrowedLine(img, position_px_cv, arrow_end_px, cv::Scalar(0, 0, 255, 255), 1, 8, 0, 0.1);
            }

            // draw trajectory
            DrawTrajectoryInplace<Dtype>(img, cur_traj.block(0, 0, 2, i), grid_map_info, trajectory_color, 2, pixel_based);

            // draw fps
            cv::putText(img, std::to_string(1000.0 / dt), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0, 255), 2);

            cv::imshow("GP Occ Surface Mapping", img);
            int key = cv::waitKey(1);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
        double avg_dt = t_ms / static_cast<double>(i + 1);
        double fps = 1000.0 / avg_dt;
        ERL_INFO("Average update time: {:f} ms, Average fps: {:f}", avg_dt, fps);
        std::cout << "=====================================" << std::endl;
    }

    if (options.test_io) {
        ASSERT_TRUE(gp.Write(bin_file)) << "Failed to write to " << bin_file;
        GpOccSurfaceMapping gp_load(std::make_shared<typename GpOccSurfaceMapping::Setting>());
        ASSERT_TRUE(gp_load.Read(bin_file)) << "Failed to read from " << bin_file;
        ASSERT_TRUE(gp == gp_load) << "Loaded GP is not equal to the original GP.";
    }

    if (options.hold) {
        std::cout << "Press any key to exit." << std::endl;
        cv::waitKey();  // wait for any key
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();
        double wait_time = 10.0;
        while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() < wait_time) {
            int key = cv::waitKey(10);
            if (key == 27) { break; }  // ESC
            if (key == 'q') { break; }
        }
    }
}

TEST(GpOccSurfaceMapping, 2Dd) { TestImpl2D<double>(); }

TEST(GpOccSurfaceMapping, 2Df) { TestImpl2D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
