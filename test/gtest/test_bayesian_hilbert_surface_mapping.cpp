#include "erl_common/block_timer.hpp"
#include "erl_common/csv.hpp"
#include "erl_common/progress_bar.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/depth_frame_3d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/lidar_frame_3d.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"
#include "erl_geometry/occupancy_quadtree_drawer.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_geometry/ucsd_fah_2d.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

#include <boost/program_options.hpp>

int g_argc = 0;
char **g_argv = nullptr;
const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
const std::filesystem::path kDataDir = kProjectRootDir / "data";
const std::filesystem::path kConfigDir = kProjectRootDir / "config";

template<typename Dtype>
cv::Mat
ConvertToImage(int xs, int ys, const Eigen::VectorX<Dtype> &prob_occupied) {
    cv::Mat prob_occupied_img(
        ys,
        xs,
        sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
        const_cast<Dtype *>(prob_occupied.data()));
    prob_occupied_img = prob_occupied_img.t();
    cv::normalize(prob_occupied_img, prob_occupied_img, 0, 255, cv::NORM_MINMAX);
    prob_occupied_img.convertTo(prob_occupied_img, CV_8UC1);
    cv::applyColorMap(prob_occupied_img, prob_occupied_img, cv::COLORMAP_JET);
    return prob_occupied_img;
}

template<typename Dtype>
void
ConvertToVoxelGrid(
    const cv::Mat &img_sdf,
    const Eigen::Matrix3X<Dtype> &positions,
    const std::shared_ptr<open3d::geometry::VoxelGrid> &voxel_grid_sdf) {
    voxel_grid_sdf->voxels_.clear();
    for (int j = 0; j < img_sdf.cols; ++j) {  // column major
        for (int i = 0; i < img_sdf.rows; ++i) {
            Eigen::Vector3d position = positions.col(i + j * img_sdf.rows).template cast<double>();
            const auto &color = img_sdf.at<cv::Vec3b>(i, j);
            voxel_grid_sdf->AddVoxel(
                {voxel_grid_sdf->GetVoxel(position),
                 Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)});
        }
    }
}

template<typename Dtype, int Dim>
void
TestIo(const erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, Dim> *bhm_surf_mapping) {
    ERL_BLOCK_TIMER_MSG("IO");
    using Mapping = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, Dim>;
    using Serializer = erl::common::Serialization<Mapping>;
    GTEST_PREPARE_OUTPUT_DIR();
    std::string filename = fmt::format("bhm_surf_mapping_{}d_{}.bin", Dim, type_name<Dtype>());
    filename = test_output_dir / filename;
    ASSERT_TRUE(Serializer::Write(filename, bhm_surf_mapping));
    Mapping bhm_surf_mapping_read(std::make_shared<typename Mapping::Setting>());
    ASSERT_TRUE(Serializer::Read(filename, &bhm_surf_mapping_read));
    ASSERT_TRUE(*bhm_surf_mapping == bhm_surf_mapping_read);
}

template<typename Dtype>
void
TestImpl3D() {
    GTEST_PREPARE_OUTPUT_DIR();
    using Octree = erl::geometry::OccupancyOctree<Dtype>;
    using OctreeDrawer = erl::geometry::OccupancyOctreeDrawer<Octree>;
    using BayesianHilbertSurfaceMapping = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 3>;
    using RangeSensor3D = erl::geometry::RangeSensor3D<Dtype>;
    using RangeSensorFrame3D = erl::geometry::RangeSensorFrame3D<Dtype>;
    using Lidar3D = erl::geometry::Lidar3D<Dtype>;
    using LidarFrame3D = erl::geometry::LidarFrame3D<Dtype>;
    using DepthCamera3D = erl::geometry::DepthCamera3D<Dtype>;
    using DepthFrame3D = erl::geometry::DepthFrame3D<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

#pragma region options

    struct Options {
        bool use_cow_and_lady = false;
        std::string cow_and_lady_dir;
        std::string mesh_file = kDataDir / "replica-hotel-0.ply";
        std::string traj_file = kDataDir / "replica-hotel-0-traj.txt";
        std::string surface_mapping_config_file =
            kConfigDir / "template" /
            fmt::format("bayesian_hilbert_mapping_3d_{}.yaml", type_name<Dtype>());
        std::string sensor_frame_type = type_name<LidarFrame3D>();
        std::string sensor_frame_config_file = kConfigDir / "sensors" / "lidar_frame_3d_271.yaml";
        std::string o3d_view_status_file = kConfigDir / "template" / "open3d_view_status.json";
        long seq_stride = 1;
        long scan_stride = 1;
        Dtype surf_normal_scale = 0.25;
        Dtype test_res = 0.02;
        Dtype test_z = 0.0;
        Dtype test_x_min = 0.0f;
        Dtype test_x_max = 0.0f;
        Dtype test_y_min = 0.0f;
        Dtype test_y_max = 0.0f;
        long test_xs = 150;
        long test_ys = 100;
        bool test_io = false;
        bool test_whole_map_at_end = false;
        bool hold = false;
        bool no_visualize = false;
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
                "surface mapping config file"
            )(
                "sensor-frame-type",
                po::value<std::string>(&options.sensor_frame_type)->default_value(options.sensor_frame_type)->value_name("sensor frame type"),
                fmt::format(
                        "sensor frame type, options: {}, {}, {}, {}",
                        type_name<erl::geometry::LidarFrame3Dd>(),
                        type_name<erl::geometry::DepthFrame3Dd>(),
                        type_name<erl::geometry::LidarFrame3Df>(),
                        type_name<erl::geometry::DepthFrame3Df>()).c_str()
            )(
                "sensor-frame-config-file",
                po::value<std::string>(&options.sensor_frame_config_file)->default_value(options.sensor_frame_config_file)->value_name("file"),
                "sensor frame config file"
            )
            (
                "o3d-view-status-file",
                po::value<std::string>(&options.o3d_view_status_file)->default_value(options.o3d_view_status_file)->value_name("file"),
                "Open3D view status file, used to set the view of the visualization window"
            )
            ("seq-stride", po::value<long>(&options.seq_stride)->default_value(options.seq_stride)->value_name("stride"), "stride")
            ("scan-stride", po::value<long>(&options.scan_stride)->default_value(options.scan_stride)->value_name("stride"), "scan stride")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale)->value_name("scale"), "surface normal scale")
            ("test-res", po::value<Dtype>(&options.test_res)->default_value(options.test_res)->value_name("res"), "test resolution")
            ("test-z", po::value<Dtype>(&options.test_z)->default_value(options.test_z)->value_name("z"), "test z")
            ("test-x-min", po::value<Dtype>(&options.test_x_min)->default_value(options.test_x_min)->value_name("x_min"), "test x min")
            ("test-x-max", po::value<Dtype>(&options.test_x_max)->default_value(options.test_x_max)->value_name("x_max"), "test x max")
            ("test-y-min", po::value<Dtype>(&options.test_y_min)->default_value(options.test_y_min)->value_name("y_min"), "test y min")
            ("test-y-max", po::value<Dtype>(&options.test_y_max)->default_value(options.test_y_max)->value_name("y_max"), "test y max")
            ("test-xs", po::value<long>(&options.test_xs)->default_value(options.test_xs)->value_name("xs"), "test xs")
            ("test-ys", po::value<long>(&options.test_ys)->default_value(options.test_ys)->value_name("ys"), "test ys")
            ("test-io", po::bool_switch(&options.test_io), "test IO")
            ("test-whole-map-at-end", po::bool_switch(&options.test_whole_map_at_end), "test the whole map at the end")
            ("hold", po::bool_switch(&options.hold), "hold the window")
            ("no-visualize", po::bool_switch(&options.no_visualize), "do not visualize");
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
    ASSERT_TRUE(options.scan_stride > 0);

    // prepare the data
    using namespace erl::common;
    using namespace erl::geometry;

    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::shared_ptr<RangeSensorFrame3D> range_sensor_frame = nullptr;
    std::shared_ptr<RangeSensor3D> range_sensor = nullptr;
    bool is_lidar = false;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Vector3 area_min, area_max;
    if (options.use_cow_and_lady) {
        cow_and_lady = std::make_shared<CowAndLady>(options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        area_min = cow_and_lady->GetMapMin().cast<Dtype>();
        area_max = cow_and_lady->GetMapMax().cast<Dtype>();
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
    } else {
        const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options.mesh_file);
        area_min = mesh->GetMinBound().template cast<Dtype>();
        area_max = mesh->GetMaxBound().template cast<Dtype>();
        geometries.push_back(mesh);

        if (options.sensor_frame_type == type_name<LidarFrame3D>()) {
            const auto lidar_frame_setting = std::make_shared<typename LidarFrame3D::Setting>();
            ASSERT_TRUE(lidar_frame_setting->FromYamlFile(options.sensor_frame_config_file));
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
            const auto depth_frame_setting = std::make_shared<typename DepthFrame3D::Setting>();
            ASSERT_TRUE(depth_frame_setting->FromYamlFile(options.sensor_frame_config_file));
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
        poses = Trajectory<Dtype>::LoadSe3(options.traj_file, false);
    }
    (void) is_lidar, (void) area_min, (void) area_max, (void) range_sensor_frame;

    // prepare the mapping
    const auto bhsm_setting = std::make_shared<typename BayesianHilbertSurfaceMapping::Setting>();
    ASSERT_TRUE(bhsm_setting->FromYamlFile(options.surface_mapping_config_file));
    BayesianHilbertSurfaceMapping bhsm(bhsm_setting);

    if (options.test_io) { TestIo<Dtype, 3>(&bhsm); }  // test IO of empty mapping

    // prepare the visualizer
    const auto visualizer_setting = std::make_shared<Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    visualizer_setting->translate_step = 0.01;
    Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto o3d_mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    o3d_mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    const auto o3d_mesh_sensor_xyz = open3d::geometry::TriangleMesh::CreateCoordinateFrame(0.1);
    const auto o3d_pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
    const auto o3d_pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
    const auto o3d_line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
    const auto o3d_voxel_grid = std::make_shared<open3d::geometry::VoxelGrid>();
    o3d_voxel_grid->origin_.setZero();
    o3d_voxel_grid->voxel_size_ = options.test_res;
    geometries.push_back(o3d_mesh_sensor);
    geometries.push_back(o3d_mesh_sensor_xyz);
    geometries.push_back(o3d_pcd_obs);
    geometries.push_back(o3d_pcd_surf_points);
    geometries.push_back(o3d_line_set_surf_normals);
    geometries.push_back(o3d_voxel_grid);
    visualizer.AddGeometries(geometries);

    // prepare the test positions
    Eigen::MatrixX<Vector3> positions(options.test_xs, options.test_ys);
    const Vector3 offset(
        -0.5f * options.test_res * static_cast<Dtype>(options.test_xs),
        -0.5f * options.test_res * static_cast<Dtype>(options.test_ys),
        0.0f);
    for (long j = 0; j < positions.cols(); ++j) {
        for (long i = 0; i < positions.rows(); ++i) {
            positions(i, j) = Vector3(
                                  static_cast<Dtype>(i) * options.test_res,
                                  static_cast<Dtype>(j) * options.test_res,
                                  0) +
                              offset;
        }
    }

    long wp_idx = 0;
    bool animation_ended = false;
    double bhsm_update_dt = 0.0;
    double bhsm_update_fps = 0.0;
    double cnt = 0.0;
    (void) bhsm_update_dt, (void) animation_ended, (void) bhsm_update_fps, (void) cnt;
    const long max_wp_idx =
        options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size());
    Matrix4 last_pose = Matrix4::Identity();
    cv::Mat ranges_img;
    MatrixX ranges;
    Matrix3 rotation_sensor, rotation;
    Vector3 translation_sensor, translation;

    auto run_update = [&]() {
        if (options.use_cow_and_lady) {
            const auto frame = (*cow_and_lady)[wp_idx];
            rotation = frame.rotation.cast<Dtype>();
            translation = frame.translation.cast<Dtype>();
            Matrix4 pose = Matrix4::Identity();
            pose.template topLeftCorner<3, 3>() = rotation;
            pose.template topRightCorner<3, 1>() = translation;
            pose = pose * DepthCamera3D::cTo;
            ranges = frame.depth.cast<Dtype>();
            ranges_img = frame.depth_jet;
            std::tie(rotation_sensor, translation_sensor) =
                CameraBase3D<Dtype>::ComputeCameraPose(rotation, translation);
        } else {
            std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
            ranges = range_sensor->Scan(rotation_sensor, translation_sensor);
            std::tie(rotation, translation) =
                range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);
        }
        wp_idx += options.seq_stride;

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
        double dt;
        {
            ERL_BLOCK_TIMER_MSG_TIME("bhm_mapping.Update", dt);
            Eigen::Map<const Matrix3X> points(
                range_sensor_frame->GetHitPointsWorld().data()->data(),
                3,
                range_sensor_frame->GetNumHitRays());
            bhsm.Update(rotation_sensor, translation_sensor, points, true /*parallel*/);
        }

        bhsm_update_dt = (bhsm_update_dt * cnt + dt) / (cnt + 1.0);
        cnt += 1.0;
        bhsm_update_fps = 1000.0 / bhsm_update_dt;
        ERL_INFO(
            "bhsm_update_dt: {:.3f} ms, bhsm_update_fps: {:.3f} fps",
            bhsm_update_dt,
            bhsm_update_fps);
        ERL_TRACY_PLOT("bhsm_update (ms)", bhsm_update_dt);
        ERL_TRACY_PLOT("bhsm_update (fps)", bhsm_update_fps);
    };

    auto vis_bhm = [&](open3d::visualization::Visualizer *vis) {
        auto map_boundary = bhsm.GetMapBoundary();
        if (options.test_x_min == options.test_x_max || options.test_y_min == options.test_y_max) {
            ERL_INFO("Map boundary is not fully defined, using surface mapping boundary.");
            options.test_x_min = map_boundary.min()[0];
            options.test_x_max = map_boundary.max()[0];
            options.test_y_min = map_boundary.min()[1];
            options.test_y_max = map_boundary.max()[1];
        }
        erl::common::GridMapInfo2D<Dtype> grid_map_info(
            Eigen::Vector2<Dtype>(options.test_x_min, options.test_y_min),
            Eigen::Vector2<Dtype>(options.test_x_max, options.test_y_max),
            Eigen::Vector2<Dtype>(options.test_res, options.test_res),
            Eigen::Vector2i(10, 10));
        Matrix3X test_positions(3, grid_map_info.Size());
        test_positions.topRows(2) =
            grid_map_info.GenerateMeterCoordinates(false).template cast<Dtype>();
        test_positions.row(2).setConstant(options.test_z + map_boundary.center[2]);
        VectorX prob_occupied;
        {
            ERL_BLOCK_TIMER_MSG("bhsm.Predict");
            Matrix3X gradient;
            bhsm.Predict(
                test_positions,
                false /*logodd*/,
                true /*faster*/,
                false /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                true /*parallel*/,
                prob_occupied,
                gradient);
        }
        const cv::Mat prob_occupied_img =
            ConvertToImage(grid_map_info.Shape(0), grid_map_info.Shape(1), prob_occupied);
        const cv::Mat occupancy_img = ConvertToImage<Dtype>(
            grid_map_info.Shape(0),
            grid_map_info.Shape(1),
            (prob_occupied.array() > 0.5).template cast<Dtype>());
        ConvertToVoxelGrid<Dtype>(prob_occupied_img, test_positions, o3d_voxel_grid);
        vis->UpdateGeometry(o3d_voxel_grid);
        cv::imshow("prob_occupied", prob_occupied_img);
        cv::imshow("occupancy", occupancy_img);
        cv::waitKey(1);
    };

#pragma region animation_callback
    auto callback = [&](Open3dVisualizerWrapper *wrapper,
                        open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();
        // options.hold is true, so the window is not closed yet
        if (animation_ended) {
            if (options.test_z != static_cast<Dtype>(visualizer_setting->z)) {
                options.test_z = static_cast<Dtype>(visualizer_setting->z);
                vis_bhm(vis);
            }
            cv::waitKey(1);
            return false;
        }

#pragma region end_of_animation
        if (wp_idx >= max_wp_idx) {
            animation_ended = true;
            if (options.test_whole_map_at_end) { vis_bhm(vis); }
            if (!options.hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }
#pragma endregion

        const auto t_start = std::chrono::high_resolution_clock::now();
        ERL_INFO("wp_idx: {}", wp_idx);

        run_update();

        // update visualization
        /// update the image
        if (!options.use_cow_and_lady) {
            for (long i = 0; i < ranges.size(); ++i) {
                Dtype &range = ranges.data()[i];
                if (range < 0.0 || range > 1000.0) { range = 0.0; }
            }
            if (is_lidar) {
                ranges_img = cv::Mat(
                    ranges.cols(),
                    ranges.rows(),
                    sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
                    ranges.data());
                cv::flip(ranges_img, ranges_img, 0);
                cv::resize(ranges_img, ranges_img, {0, 0}, 2.5, 2.5);
            } else {
                cv::eigen2cv(ranges, ranges_img);
            }
            cv::normalize(ranges_img, ranges_img, 0, 255, cv::NORM_MINMAX);
            ranges_img.convertTo(ranges_img, CV_8UC1);
            cv::applyColorMap(ranges_img, ranges_img, cv::COLORMAP_JET);
        }
        cv::putText(
            ranges_img,
            fmt::format("update {:.2f} fps", bhsm_update_fps),
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
        o3d_mesh_sensor->Transform(delta_pose.template cast<double>());
        o3d_mesh_sensor_xyz->Transform(delta_pose.template cast<double>());
        /// update the observation point cloud
        o3d_pcd_obs->points_.clear();
        o3d_pcd_obs->colors_.clear();
        o3d_pcd_obs->points_.reserve(range_sensor_frame->GetHitPointsWorld().size());
        for (const auto &point: range_sensor_frame->GetHitPointsWorld()) {
            o3d_pcd_obs->points_.emplace_back(point.template cast<double>());
        }
        o3d_pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        /// update the surface point cloud and normals
        o3d_pcd_surf_points->points_.clear();
        o3d_line_set_surf_normals->points_.clear();
        o3d_line_set_surf_normals->lines_.clear();
        const auto &surf_data_buffer = bhsm.GetSurfaceDataBuffer();
        for (auto &[key, local_bhm]: bhsm.GetLocalBayesianHilbertMaps()) {
            for (auto &[idx, surf_idx]: local_bhm->surface_indices) {
                auto &pt = surf_data_buffer[surf_idx];
                o3d_pcd_surf_points->points_.emplace_back(pt.position.template cast<double>());
                o3d_line_set_surf_normals->points_.emplace_back(
                    pt.position.template cast<double>());
                o3d_line_set_surf_normals->points_.emplace_back(
                    (pt.position + options.surf_normal_scale * pt.normal).template cast<double>());
                o3d_line_set_surf_normals->lines_.emplace_back(
                    o3d_line_set_surf_normals->points_.size() - 2,
                    o3d_line_set_surf_normals->points_.size() - 1);
            }
        }
        o3d_line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        /// update the voxel grid
        Matrix3X positions_test(3, positions.size());
        for (long j = 0; j < positions.cols(); ++j) {
            for (long i = 0; i < positions.rows(); ++i) {
                const Vector3 &position = positions(i, j);
                positions_test.col(i + j * positions.rows()) =  // sensor frame to world frame
                    rotation_sensor * position + translation_sensor;
            }
        }
        VectorX prob_occupied;
        {
            ERL_BLOCK_TIMER_MSG("bhsm.Predict");
            Matrix3X gradient;
            bhsm.Predict(positions_test, false, true, false, false, true, prob_occupied, gradient);
        }
        cv::Mat prob_occupied_img = ConvertToImage(options.test_xs, options.test_ys, prob_occupied);
        ConvertToVoxelGrid<Dtype>(prob_occupied_img, positions_test, o3d_voxel_grid);

        const auto t_end = std::chrono::high_resolution_clock::now();
        auto duration_total = std::chrono::duration<Dtype, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && options.test_io) { TestIo<Dtype, 3>(&bhsm); }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };
#pragma endregion

    // start the mapping
    if (options.no_visualize) {
        while (wp_idx < max_wp_idx) { run_update(); }
    } else {
        visualizer_setting->z = options.test_z;
        visualizer.SetAnimationCallback(callback);
        visualizer.SetViewStatus(options.o3d_view_status_file);
        visualizer.Show();
    }

    if (options.test_io) { TestIo<Dtype, 3>(&bhsm); }

    auto drawer_setting = std::make_shared<typename OctreeDrawer::Setting>();
    drawer_setting->area_min = area_min.template cast<double>();
    drawer_setting->area_max = area_max.template cast<double>();
    drawer_setting->occupied_only = true;
    OctreeDrawer octree_drawer(drawer_setting, bhsm.GetTree());
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
    using Lidar2D = erl::geometry::Lidar2D;
    using BayesianHilbertSurfaceMapping = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 2>;
    using Quadtree = erl::geometry::OccupancyQuadtree<Dtype>;
    using QuadtreeDrawer = erl::geometry::OccupancyQuadtreeDrawer<Quadtree>;
    using Matrix2 = Eigen::Matrix2<Dtype>;
    using Matrix2X = Eigen::Matrix2X<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Vector2 = Eigen::Vector2<Dtype>;
    using VectorX = Eigen::VectorX<Dtype>;

    // load setting from the command line
    struct Options {
        std::string gazebo_train_file = kDataDir / "gazebo";
        std::string house_expo_map_file = kDataDir / "house_expo_room_1451.json";
        std::string house_expo_traj_file = kDataDir / "house_expo_room_1451.csv";
        std::string ucsd_fah_2d_file = kDataDir / "ucsd_fah_2d.dat";
        std::string surface_mapping_config_file =
            kConfigDir / fmt::format("bayesian_hilbert_mapping_2d_{}.yaml", type_name<Dtype>());
        bool use_gazebo_room_2d = false;
        bool use_house_expo_lidar_2d = false;
        bool use_ucsd_fah_2d = false;
        bool visualize = false;
        bool test_io = false;
        bool hold = false;
        int stride = 5;
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
            ("map-resolution", po::value<Dtype>(&options.map_resolution)->default_value(options.map_resolution), "Map resolution")
            ("surf-normal-scale", po::value<Dtype>(&options.surf_normal_scale)->default_value(options.surf_normal_scale), "Surface normal scale")
            ("visualize", po::bool_switch(&options.visualize)->default_value(options.visualize), "Visualize the mapping")
            ("test-io", po::bool_switch(&options.test_io)->default_value(options.test_io), "Test IO")
            ("hold", po::bool_switch(&options.hold)->default_value(options.hold), "Hold the test until a key is pressed")
            ("stride", po::value<int>(&options.stride)->default_value(options.stride), "stride for running the sequence")
            ("init-frame", po::value<int>(&options.init_frame)->default_value(options.init_frame), "Initial frame index")
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

    using namespace erl::geometry;

    if (options.use_gazebo_room_2d) {
        auto train_data_loader = GazeboRoom2D::TrainDataLoader(options.gazebo_train_file);
        max_update_cnt =
            static_cast<long>(train_data_loader.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (int i = options.init_frame; i < static_cast<int>(train_data_loader.size());
             i += options.stride, ++cnt) {
            auto &df = train_data_loader[i];
            train_angles.push_back(df.angles.cast<Dtype>());
            train_ranges.push_back(df.ranges.cast<Dtype>());
            train_poses.emplace_back(df.rotation.cast<Dtype>(), df.translation.cast<Dtype>());
            cur_traj.col(cnt) << df.translation.cast<Dtype>();
            bar->Update();
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
            static_cast<long>(trajectory.size() - options.init_frame) / options.stride + 1;
        train_angles.reserve(max_update_cnt);
        train_ranges.reserve(max_update_cnt);
        train_poses.reserve(max_update_cnt);
        cur_traj.resize(2, max_update_cnt);
        auto bar_setting = std::make_shared<ProgressBar::Setting>();
        bar_setting->total = max_update_cnt;
        const auto bar = ProgressBar::Open(bar_setting, std::cout);
        int cnt = 0;
        for (std::size_t i = options.init_frame; i < trajectory.size();
             i += options.stride, cnt++) {
            bool scan_in_parallel = true;
            std::vector<double> &waypoint = trajectory[i];
            cur_traj.col(cnt) << static_cast<Dtype>(waypoint[0]), static_cast<Dtype>(waypoint[1]);

            Eigen::Matrix2d rotation = Eigen::Rotation2Dd(waypoint[2]).toRotationMatrix();
            Eigen::Vector2d translation(waypoint[0], waypoint[1]);
            VectorX lidar_ranges =
                lidar.Scan(rotation, translation, scan_in_parallel).cast<Dtype>();
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
        UcsdFah2D ucsd_fah(options.ucsd_fah_2d_file);
        map_min = UcsdFah2D::kMapMin.cast<Dtype>();
        map_max = UcsdFah2D::kMapMax.cast<Dtype>();
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
    } else {
        return;
    }
    max_update_cnt = cur_traj.cols();

    // load setting
    const auto bhm_mapping_setting =
        std::make_shared<typename BayesianHilbertSurfaceMapping::Setting>();
    ASSERT_TRUE(bhm_mapping_setting->FromYamlFile(options.surface_mapping_config_file));
    BayesianHilbertSurfaceMapping bhsm(bhm_mapping_setting);

    if (options.test_io) { TestIo<Dtype, 2>(&bhsm); }

    // prepare the visualizer
    auto drawer_setting = std::make_shared<typename QuadtreeDrawer::Setting>();
    drawer_setting->area_min = map_min;
    drawer_setting->area_max = map_max;
    drawer_setting->resolution = map_resolution[0];
    drawer_setting->padding = map_padding[0];
    drawer_setting->border_color = cv::Scalar(255, 0, 0, 255);
    auto drawer = std::make_shared<QuadtreeDrawer>(drawer_setting);

    std::filesystem::path tree_img_dir = test_output_dir / "tree_images";
    std::filesystem::path iter_cnt_dir = test_output_dir / "iter_cnt_images";
    std::filesystem::path logodd_dir = test_output_dir / "logodd_images";
    std::filesystem::path prob_occupied_dir = test_output_dir / "prob_occupied_images";
    std::filesystem::path gradient_norms_dir = test_output_dir / "gradient_norms_images";
    std::filesystem::create_directories(tree_img_dir);
    std::filesystem::create_directories(iter_cnt_dir);
    std::filesystem::create_directories(logodd_dir);
    std::filesystem::create_directories(prob_occupied_dir);
    std::filesystem::create_directories(gradient_norms_dir);

    long img_cnt = 0;
    cv::Scalar trajectory_color(255, 255, 255, 255);
    cv::Mat tree_img;
    drawer->SetQuadtree(bhsm.GetTree());
    if (options.visualize) {
        drawer->DrawLeaves(tree_img);
        cv::imshow("tree", tree_img);
        cv::imshow("iter_cnt", tree_img);
        cv::imshow("logodd", tree_img);
        cv::imshow("prob_occupied", tree_img);
        cv::imshow("gradient_norms", tree_img);
        cv::waitKey(10);

        std::string filename = fmt::format("{:04d}.png", img_cnt++);
        cv::imwrite(tree_img_dir / filename, tree_img);
        cv::imwrite(iter_cnt_dir / filename, tree_img);
        cv::imwrite(logodd_dir / filename, tree_img);
        cv::imwrite(prob_occupied_dir / filename, tree_img);
        cv::imwrite(gradient_norms_dir / filename, tree_img);

        constexpr int x_space = 10;
        constexpr int y_space = 120;
        cv::moveWindow("tree", 0, 0);
        cv::moveWindow("iter_cnt", tree_img.cols + x_space, 0);
        cv::moveWindow("logodd", 2 * (tree_img.cols + x_space), 0);
        cv::moveWindow("prob_occupied", 0, tree_img.rows + y_space);
        cv::moveWindow("gradient_norms", tree_img.cols + x_space, tree_img.rows + y_space);
    }
    if (options.hold) {
        std::cout << "Press any key to start" << std::endl;
        if (options.visualize) {  // wait for any key
            cv::waitKey(0);
        } else {
            std::cin.get();
        }
    }

    auto grid_map_info = drawer->GetGridMapInfo();
    Matrix2X grid_points = grid_map_info->GenerateMeterCoordinates(false /*c_stride*/);
    VectorX logodd_values;
    VectorX prob_occupied;
    Matrix2X gradients;
    std::vector<cv::Point2i> surface_points_cv;

    double bhsm_update_dt_ms = 0.0;
    double bhsm_update_fps = 0.0;
    double bhsm_predict_dt_ms = 0.0;
    double bhsm_predict_fps = 0.0;

    // start the mapping
    for (long i = 0; i < max_update_cnt; ++i) {
        const auto &[rotation, translation] = train_poses[i];
        const VectorX &angles = train_angles[i];
        const VectorX &ranges = train_ranges[i];
        // convert ranges to points
        Matrix2X points(2, angles.size());
        surface_points_cv.clear();
        for (long j = 0; j < angles.size(); ++j) {
            points.col(j) =
                rotation *
                    Vector2(ranges[j] * std::cos(angles[j]), ranges[j] * std::sin(angles[j])) +
                translation;
            Eigen::Vector2i px = grid_map_info->MeterToPixelForPoints(points.col(j));
            surface_points_cv.emplace_back(px[0], px[1]);
        }
        double dt;
        {
            ERL_BLOCK_TIMER_MSG_TIME("bhm_mapping.Update", dt);
            bhsm.Update(rotation, translation, points, true /*parallel*/);
        }
        bhsm_update_dt_ms =
            (bhsm_update_dt_ms * static_cast<double>(i) + dt) / static_cast<double>(i + 1);
        bhsm_update_fps = 1000.0 / bhsm_update_dt_ms;

        // save local bhms
        auto local_bhms = bhsm.GetLocalBayesianHilbertMaps();
        // save results
        const long size = local_bhms.begin()->second->bhm.GetHingedPoints().cols();
        MatrixX mu(size, static_cast<long>(local_bhms.size()));
        MatrixX sigma(size, static_cast<long>(local_bhms.size()));
        long idx = 0;
        for (const auto &[key, local_bhm]: local_bhms) {
            mu.col(idx) = local_bhm->bhm.GetWeights();
            sigma.col(idx) = local_bhm->bhm.GetWeightsCovariance().col(0);
            ++idx;
        }
        SaveEigenMatrixToTextFile<Dtype>(test_output_dir / "mu.txt", mu, EigenTextFormat::kCsvFmt);
        SaveEigenMatrixToTextFile<Dtype>(
            test_output_dir / "sigma.txt",
            sigma,
            EigenTextFormat::kCsvFmt);

        // visualize the result
        if (!options.visualize) { continue; }
        // predict the occupancy and gradient
        {
            ERL_BLOCK_TIMER_MSG_TIME("bhm_mapping.Predict", dt);
            bhsm.Predict(  //
                grid_points,
                true /*logodd*/,
                true /*faster*/,
                true /*compute gradient*/,
                false /*gradient with sigmoid*/,
                true /*parallel*/,
                logodd_values,
                gradients);
        }
        bhsm_predict_dt_ms =
            (bhsm_predict_dt_ms * static_cast<double>(i) + dt) / static_cast<double>(i + 1);
        bhsm_predict_fps = 1000.0 / bhsm_predict_dt_ms;

        prob_occupied.resize(logodd_values.size());
        for (long j = 0; j < logodd_values.size(); ++j) {
            prob_occupied[j] = logodd::Probability(logodd_values[j]);
        }

        /// draw the tree
        tree_img.setTo(cv::Scalar(128, 128, 128, 255));
        drawer->DrawLeaves(tree_img);
        DrawTrajectoryInplace<Dtype>(
            tree_img,
            cur_traj.block(0, 0, 2, i),
            grid_map_info,
            trajectory_color,
            2,
            true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(tree_img, px, cv::Scalar(0, 0, 255, 255), cv::MARKER_CROSS, 10, 2);
        }

        /// draw the iteration count
        Eigen::VectorXi iter_cnt = Eigen::VectorXi::Zero(prob_occupied.size());
        auto tree = bhsm.GetTree();
        for (long j = 0; j < grid_points.cols(); ++j) {
            QuadtreeKey key;
            if (!tree->CoordToKeyChecked(grid_points.col(j), bhm_mapping_setting->bhm_depth, key)) {
                continue;
            }
            if (!local_bhms.contains(key)) { continue; }
            iter_cnt[j] = local_bhms.at(key)->bhm.GetIterationCount();
        }
        cv::Mat iter_cnt_img(
            grid_map_info->Shape(1),
            grid_map_info->Shape(0),
            CV_32SC1,
            iter_cnt.data());
        cv::normalize(iter_cnt_img, iter_cnt_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        iter_cnt_img.convertTo(iter_cnt_img, CV_8UC1);
        cv::applyColorMap(iter_cnt_img, iter_cnt_img, cv::COLORMAP_JET);
        cv::flip(iter_cnt_img, iter_cnt_img, 0);
        DrawTrajectoryInplace<Dtype>(
            iter_cnt_img,
            cur_traj.block(0, 0, 2, i),
            grid_map_info,
            trajectory_color,
            2,
            true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(
                iter_cnt_img,
                px,
                cv::Scalar(255, 255, 255, 255),
                cv::MARKER_CROSS,
                10,
                2);
        }
        for (const auto &[key, bhm]: local_bhms) {
            const auto &boundary = bhm->bhm.GetMapBoundary();
            Eigen::Vector2i px1 = grid_map_info->MeterToPixelForPoints(boundary.min());
            Eigen::Vector2i px2 = grid_map_info->MeterToPixelForPoints(boundary.max());
            cv::rectangle(
                iter_cnt_img,
                cv::Point(px1[0], px1[1]),
                cv::Point(px2[0], px2[1]),
                cv::Scalar(0, 0, 0, 255),
                2);
        }

        /// draw the log odds
        cv::Mat logodd_img(
            grid_map_info->Shape(1),
            grid_map_info->Shape(0),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            logodd_values.data());
        cv::normalize(logodd_img, logodd_img, 0, 255, cv::NORM_MINMAX);
        logodd_img.convertTo(logodd_img, CV_8UC1);
        cv::applyColorMap(logodd_img, logodd_img, cv::COLORMAP_JET);
        cv::flip(logodd_img, logodd_img, 0);
        DrawTrajectoryInplace<Dtype>(
            logodd_img,
            cur_traj.block(0, 0, 2, i),
            grid_map_info,
            trajectory_color,
            2,
            true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(logodd_img, px, cv::Scalar(255, 255, 255, 255), cv::MARKER_CROSS, 10, 2);
        }

        /// draw the occupancy probability
        cv::Mat prob_occupied_img(
            grid_map_info->Shape(1),
            grid_map_info->Shape(0),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            prob_occupied.data());
        cv::normalize(prob_occupied_img, prob_occupied_img, 0, 255, cv::NORM_MINMAX);
        prob_occupied_img.convertTo(prob_occupied_img, CV_8UC1);
        cv::applyColorMap(prob_occupied_img, prob_occupied_img, cv::COLORMAP_JET);
        cv::flip(prob_occupied_img, prob_occupied_img, 0);
        DrawTrajectoryInplace<Dtype>(
            prob_occupied_img,
            cur_traj.block(0, 0, 2, i),
            grid_map_info,
            trajectory_color,
            2,
            true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(
                prob_occupied_img,
                px,
                cv::Scalar(255, 255, 255, 255),
                cv::MARKER_CROSS,
                10,
                2);
        }

        /// draw the gradient
        VectorX gradient_norms = gradients.colwise().norm();
        cv::Mat gradient_norms_img(
            grid_map_info->Shape(1),
            grid_map_info->Shape(0),
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            gradient_norms.data());
        cv::normalize(gradient_norms_img, gradient_norms_img, 0, 255, cv::NORM_MINMAX);
        gradient_norms_img.convertTo(gradient_norms_img, CV_8UC1);
        cv::applyColorMap(gradient_norms_img, gradient_norms_img, cv::COLORMAP_JET);
        cv::flip(gradient_norms_img, gradient_norms_img, 0);
        DrawTrajectoryInplace<Dtype>(
            gradient_norms_img,
            cur_traj.block(0, 0, 2, i),
            grid_map_info,
            trajectory_color,
            2,
            true);
        for (const auto &px: surface_points_cv) {
            cv::drawMarker(
                gradient_norms_img,
                px,
                cv::Scalar(255, 255, 255, 255),
                cv::MARKER_CROSS,
                10,
                2);
        }
        //// draw the surface normals
        const auto &surf_data_buffer = bhsm.GetSurfaceDataBuffer();
        for (auto &[key, local_bhm]: local_bhms) {
            for (auto &[idx, surf_idx]: local_bhm->surface_indices) {
                const auto &surf = surf_data_buffer[surf_idx];
                auto px1 = grid_map_info->MeterToPixelForPoints(surf.position);
                auto px2 = grid_map_info->MeterToPixelForPoints(surf.position + surf.normal);
                cv::arrowedLine(
                    gradient_norms_img,
                    cv::Point(px1(0, 0), px1(1, 0)),
                    cv::Point(px2(0, 0), px2(1, 0)),
                    cv::Scalar(255, 255, 255, 255),
                    2,
                    cv::LINE_AA);
            }
        }

        // draw fps
        cv::putText(
            tree_img,
            fmt::format("{:.2f}/{:.2f}", bhsm_update_fps, bhsm_predict_fps),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX,
            1,
            cv::Scalar(0, 255, 0, 255),
            2);

        cv::imshow("tree", tree_img);
        cv::imshow("iter_cnt", iter_cnt_img);
        cv::imshow("logodd", logodd_img);
        cv::imshow("prob_occupied", prob_occupied_img);
        cv::imshow("gradient_norms", gradient_norms_img);

        std::string filename = fmt::format("{:04d}.png", img_cnt++);
        cv::imwrite(tree_img_dir / filename, tree_img);
        cv::imwrite(iter_cnt_dir / filename, iter_cnt_img);
        cv::imwrite(logodd_dir / filename, logodd_img);
        cv::imwrite(prob_occupied_dir / filename, prob_occupied_img);
        cv::imwrite(gradient_norms_dir / filename, gradient_norms_img);

        cv::waitKey(10);
    }
    if (options.hold) { cv::waitKey(0); }
    if (options.test_io) { TestIo<Dtype, 2>(&bhsm); }
}

// Update FPS:
// Gazebo Room: 1500-1700 fps (float), 1400-1600 fps (double)
// Replica Lidar-271: 40-70 fps (float/double)

TEST(BayesianHilbertSurfaceMapping, 3Dd) { TestImpl3D<double>(); }

TEST(BayesianHilbertSurfaceMapping, 3Df) { TestImpl3D<float>(); }

TEST(BayesianHilbertSurfaceMapping, 2Dd) { TestImpl2D<double>(); }

TEST(BayesianHilbertSurfaceMapping, 2Df) { TestImpl2D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
