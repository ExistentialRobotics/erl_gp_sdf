#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/gazebo_room_2d.hpp"
#include "erl_geometry/house_expo_map.hpp"
#include "erl_geometry/lidar_2d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_helper.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;
const std::filesystem::path kDataDir = kProjectRootDir / "data";
const std::filesystem::path kConfigDir = kProjectRootDir / "config";
int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype>
cv::Mat
ConvertDepthToImage(const Eigen::MatrixX<Dtype> &ranges) {
    cv::Mat depth_jet;
    Eigen::MatrixX<Dtype> ranges_img = Eigen::MatrixX<Dtype>::Zero(ranges.rows(), ranges.cols());
    Dtype min_range = std::numeric_limits<Dtype>::max();
    Dtype max_range = std::numeric_limits<Dtype>::lowest();
    for (long i = 0; i < ranges.size(); ++i) {
        const Dtype &range = ranges.data()[i];
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
    return depth_jet;
}

template<typename Dtype>
std::pair<cv::Mat, cv::Mat>
ConvertSdfToImage(Eigen::VectorX<Dtype> &distances, const int rows, const int cols) {
    cv::Mat img_sdf(cols, rows, sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1, distances.data());
    img_sdf = img_sdf.t();
    cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);

    Eigen::MatrixX8U sdf_sign = (distances.array() >= 0.0f).template cast<uint8_t>() * 255;
    cv::Mat img_sdf_sign(cols, rows, CV_8UC1, sdf_sign.data());
    img_sdf_sign = img_sdf_sign.t();

    // for a zero pixel in img_sdf_sign fill the pixel in img_sdf with zero
    const cv::Mat mask = img_sdf_sign == 0;
    img_sdf.setTo(0, mask);
    return {std::move(img_sdf), std::move(img_sdf_sign)};
}

template<typename Dtype>
void
ConvertSdfToVoxelGrid(
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

template<typename Dtype>
void
TestImpl3D() {
    GTEST_PREPARE_OUTPUT_DIR();

    using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping<Dtype, 3>;
    using SdfMapping = erl::sdf_mapping::GpSdfMapping<Dtype, 3, SurfaceMapping>;
    using RangeSensor = erl::geometry::RangeSensor3D<Dtype>;
    using DepthFrame = erl::geometry::DepthFrame3D<Dtype>;
    using LidarFrame = erl::geometry::LidarFrame3D<Dtype>;
    using CowAndLady = erl::geometry::CowAndLady;
    using DepthCamera = erl::geometry::DepthCamera3D<Dtype>;
    using Lidar = erl::geometry::Lidar3D<Dtype>;

    using VectorX = Eigen::VectorX<Dtype>;
    using MatrixX = Eigen::MatrixX<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix4 = Eigen::Matrix4<Dtype>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;
    using Matrix4X = Eigen::Matrix4X<Dtype>;
    using Matrix6X = Eigen::Matrix<Dtype, 6, Eigen::Dynamic>;

    struct Options {
        bool use_cow_and_lady = false;  // use Cow and Lady dataset
        std::string cow_and_lady_dir;   // directory containing the Cow and Lady dataset
        std::string mesh_file = kDataDir / "replica-hotel-0.ply";       // mesh file
        std::string traj_file = kDataDir / "replica-hotel-0-traj.txt";  // trajectory file
        std::string surface_mapping_config_file = kConfigDir / "surface_mapping_3d_lidar.yaml";
        std::string sdf_mapping_config_file = kConfigDir / "sdf_mapping_3d_lidar.yaml";
        std::string sdf_mapping_bin_file;
        long stride = 1;
        long max_frames = std::numeric_limits<long>::max();
        Dtype test_res = 0.02;
        long test_xs = 150;
        long test_ys = 100;
        bool test_whole_map_at_end = false;  // test the whole map at the end
        Dtype test_z = 0.0;                  // test z for the whole map
        Dtype image_resize_scale = 10;       // image resize scale
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
                "surface mapping config file"
            )
            (
                "sdf-mapping-config-file",
                po::value<std::string>(&options.sdf_mapping_config_file)->default_value(options.sdf_mapping_config_file)->value_name("file"),
                "SDF mapping config file"
            )
            (
                "sdf-mapping-bin-file",
                po::value<std::string>(&options.sdf_mapping_bin_file)->default_value(options.sdf_mapping_bin_file)->value_name("file"),
                "SDF mapping bin file"
            )
            ("stride", po::value<long>(&options.stride)->default_value(options.stride)->value_name("stride"), "stride")
            ("max-frames", po::value<long>(&options.max_frames)->default_value(options.max_frames)->value_name("frames"), "max number of frames to process")
            ("test-res", po::value<Dtype>(&options.test_res)->default_value(options.test_res)->value_name("res"), "test resolution")
            ("test-xs", po::value<long>(&options.test_xs)->default_value(options.test_xs)->value_name("xs"), "test xs")
            ("test-ys", po::value<long>(&options.test_ys)->default_value(options.test_ys)->value_name("ys"), "test ys")
            ("test-whole-map-at-end", po::bool_switch(&options.test_whole_map_at_end), "test the whole map at the end")
            ("test-z", po::value<Dtype>(&options.test_z)->default_value(options.test_z)->value_name("z"), "test z for the whole map")
            ("image-resize-scale", po::value<Dtype>(&options.image_resize_scale)->default_value(options.image_resize_scale)->value_name("scale"), "image resize scale")
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
    const auto surface_mapping_setting = std::make_shared<typename SurfaceMapping::Setting>();
    ERL_ASSERTM(
        surface_mapping_setting->FromYamlFile(options.surface_mapping_config_file),
        "Failed to load surface_mapping_config_file: {}",
        options.surface_mapping_config_file);
    const auto sdf_mapping_setting = std::make_shared<typename SdfMapping::Setting>();
    ERL_ASSERTM(
        sdf_mapping_setting->FromYamlFile(options.sdf_mapping_config_file),
        "Failed to load sdf_mapping_config_file: {}",
        options.sdf_mapping_config_file);

    // prepare the scene
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<RangeSensor> range_sensor = nullptr;
    std::shared_ptr<CowAndLady> cow_and_lady = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Vector3 map_min, map_max;
    if (options.use_cow_and_lady) {
        ERL_INFO("Using Cow and Lady dataset.");
        ERL_INFO("Using depth.");
        cow_and_lady = std::make_shared<CowAndLady>(options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        map_min = cow_and_lady->GetMapMin().cast<Dtype>();
        map_max = cow_and_lady->GetMapMax().cast<Dtype>();
        const auto depth_frame_setting = std::make_shared<typename DepthFrame::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = CowAndLady::kCameraCy;
    } else {
        ERL_INFO("Using mesh file: {}", options.mesh_file);
        const auto mesh = open3d::io::CreateMeshFromFile(options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", options.mesh_file);
        map_min = mesh->GetMinBound().template cast<Dtype>();
        map_max = mesh->GetMaxBound().template cast<Dtype>();
        if (surface_mapping_setting->sensor_gp->sensor_frame_type ==
            demangle(typeid(LidarFrame).name())) {
            ERL_INFO("Using LiDAR.");
            const auto lidar_frame_setting =
                std::dynamic_pointer_cast<typename LidarFrame::Setting>(
                    surface_mapping_setting->sensor_gp->sensor_frame);
            const auto lidar_setting = std::make_shared<typename Lidar::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<Lidar>(lidar_setting);
        } else if (
            surface_mapping_setting->sensor_gp->sensor_frame_type == type_name<DepthFrame>()) {
            ERL_INFO("Using depth.");
            const auto depth_frame_setting =
                std::dynamic_pointer_cast<typename DepthFrame::Setting>(
                    surface_mapping_setting->sensor_gp->sensor_frame);
            const auto depth_camera_setting = std::make_shared<typename DepthCamera::Setting>();
            *depth_camera_setting = depth_frame_setting->camera_intrinsic;
            range_sensor = std::make_shared<DepthCamera>(depth_camera_setting);
        } else {
            ERL_FATAL(
                "Unknown sensor_frame_type: {}. Expected either {} or {}",
                surface_mapping_setting->sensor_gp->sensor_frame_type,
                demangle(typeid(LidarFrame).name()),
                type_name<DepthFrame>());
        }
        range_sensor->AddMesh(options.mesh_file);
        geometries.push_back(mesh);
        poses = erl::geometry::Trajectory<Dtype>::LoadSe3(options.traj_file, false);
    }
    (void) map_min, (void) map_max;

    // prepare the mapping
    const auto surf_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);
    SdfMapping sdf_mapping(sdf_mapping_setting, surf_mapping);

    // prepare the visualizer
    const auto vis_setting = std::make_shared<erl::geometry::Open3dVisualizerWrapper::Setting>();
    vis_setting->window_name = test_info->name();
    vis_setting->mesh_show_back_face = false;
    erl::geometry::Open3dVisualizerWrapper visualizer(vis_setting);
    const auto mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    const auto mesh_sensor_xyz = erl::geometry::CreateAxisMesh(Eigen::Matrix4d::Identity(), 0.1);
    const auto pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
    const auto pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
    const auto line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
    const auto voxel_grid_sdf = std::make_shared<open3d::geometry::VoxelGrid>();
    voxel_grid_sdf->origin_.setZero();
    voxel_grid_sdf->voxel_size_ = options.test_res;
    geometries.push_back(mesh_sensor);
    geometries.push_back(mesh_sensor_xyz);
    // geometries.push_back(pcd_obs);
    // geometries.push_back(pcd_surf_points);
    // geometries.push_back(line_set_surf_normals);
    geometries.push_back(voxel_grid_sdf);
    visualizer.AddGeometries(geometries);

    // prepare the test positions
    Eigen::MatrixX<Vector3> positions(options.test_xs, options.test_ys);
    const Vector3 offset(
        static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_xs),
        static_cast<Dtype>(-0.5f) * options.test_res * static_cast<Dtype>(options.test_ys),
        0.0);
    for (long j = 0; j < positions.cols(); ++j) {
        for (long i = 0; i < positions.rows(); ++i) {
            positions(i, j) = Vector3(
                                  static_cast<Dtype>(i) * options.test_res,
                                  static_cast<Dtype>(j) * options.test_res,
                                  0) +
                              offset;
        }
    }

    // animation callback
    long wp_idx = 0;
    bool animation_ended = false;
    (void) wp_idx, (void) animation_ended;
    const long max_wp_idx = std::min(
        options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size()),
        options.max_frames);
    Matrix4 last_pose = Matrix4::Identity();

    auto test_io = [&]() {
        ERL_BLOCK_TIMER_MSG("IO");
        std::string bin_file = fmt::format("gp_sdf_mapping_3d_{}.bin", type_name<Dtype>());
        bin_file = test_output_dir / bin_file;
        ERL_ASSERTM(
            erl::common::Serialization<SdfMapping>::Write(bin_file, &sdf_mapping),
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
        ERL_ASSERTM(sdf_mapping == sdf_mapping_read, "gp != gp_load");
    };

    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper,
                        open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();
        if (animation_ended) {  // options.hold is true, so the window is not closed yet
            cv::waitKey(1);
            return false;
        }

        if (wp_idx >= max_wp_idx) {
            animation_ended = true;
            if (options.test_whole_map_at_end) {
                erl::common::GridMapInfo2D<Dtype> grid_map_info(
                    map_min.template head<2>(),
                    map_max.template head<2>(),
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
                        sdf_mapping
                            .Test(test_positions, distances, gradients, variances, covairances));
                }
                auto [img_sdf, img_sdf_sign] =
                    ConvertSdfToImage(distances, grid_map_info.Shape(0), grid_map_info.Shape(1));
                ConvertSdfToVoxelGrid(img_sdf, test_positions, voxel_grid_sdf);
                Dtype resize_scale = options.image_resize_scale;
                resize_scale = std::min(
                    resize_scale,
                    static_cast<Dtype>(1920.0f) / static_cast<Dtype>(img_sdf.cols));
                resize_scale = std::min(
                    resize_scale,
                    static_cast<Dtype>(1920.0f) / static_cast<Dtype>(img_sdf.rows));
                cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
                cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
                cv::imshow("sdf_whole_map", img_sdf);
                cv::imshow("sdf_sign_whole_map", img_sdf_sign);
                cv::waitKey(1);
            }
            if (options.test_io) { test_io(); }
            if (!options.hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        Matrix3 rotation_sensor, rotation;
        Vector3 translation_sensor, translation;
        ERL_INFO("wp_idx: {}", wp_idx);

        cv::Mat depth_jet;
        MatrixX ranges;
        double dt;
        {
            ERL_BLOCK_TIMER_MSG_TIME("data loading", dt);
            if (options.use_cow_and_lady) {
                const auto frame = (*cow_and_lady)[wp_idx];
                rotation = frame.rotation.cast<Dtype>();
                translation = frame.translation.cast<Dtype>();
                Matrix4 pose = Matrix4::Identity();
                pose.template topLeftCorner<3, 3>() = rotation;
                pose.template topRightCorner<3, 1>() = translation;
                pose = pose * DepthCamera::cTo;
                rotation_sensor = pose.template topLeftCorner<3, 3>();
                translation_sensor = pose.template topRightCorner<3, 1>();
                ranges = frame.depth.cast<Dtype>();
                depth_jet = frame.depth_jet;
            } else {
                std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
                ranges = range_sensor->Scan(rotation_sensor, translation_sensor);
                std::tie(rotation, translation) =
                    range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);
                depth_jet = ConvertDepthToImage(ranges);
            }
        }
        ERL_TRACY_PLOT("data loading (ms)", dt);
        wp_idx += options.stride;

        {
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Update", dt);
            // provide ranges and frame pose in the world frame
            ERL_WARN_COND(
                !sdf_mapping.Update(rotation, translation, ranges),
                "sdf_mapping.Update failed.");
        }
        double gp_update_fps = 1000.0 / dt;
        ERL_TRACY_PLOT("sdf_mapping_update (ms)", dt);
        ERL_TRACY_PLOT("sdf_mapping_update (fps)", gp_update_fps);

        // test
        Matrix3X positions_test(3, positions.size());
        for (long j = 0; j < positions.cols(); ++j) {
            for (long i = 0; i < positions.rows(); ++i) {
                const Vector3 &position = positions(i, j);
                positions_test.col(i + j * positions.rows()) =
                    rotation_sensor * position + translation_sensor;  // sensor frame to world frame
            }
        }
        VectorX distances(positions_test.cols());
        {
            Matrix3X gradients(3, positions_test.cols());
            Matrix4X variances;
            Matrix6X covairances;
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Test", dt);
            EXPECT_TRUE(
                sdf_mapping.Test(positions_test, distances, gradients, variances, covairances));
        }
        double gp_test_fps = 1000.0 / dt;
        ERL_TRACY_PLOT("gp_test (ms)", dt);
        ERL_TRACY_PLOT("gp_test (fps)", gp_test_fps);

        // update visualization
        /// update the images and voxel grid
        if (depth_jet.rows > depth_jet.cols) {
            depth_jet = depth_jet.t();
            cv::flip(depth_jet, depth_jet, 0);
            if (depth_jet.rows < 256) {
                double scale = 256.0 / depth_jet.rows;
                cv::resize(depth_jet, depth_jet, cv::Size(), scale, scale);
            }
        }
        cv::putText(
            depth_jet,
            fmt::format("frame {}", wp_idx),
            cv::Point(10, 30),
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        cv::putText(
            depth_jet,
            fmt::format("update {:.2f} fps", gp_update_fps),
            cv::Point(10, 60),
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        auto [img_sdf, img_sdf_sign] =
            ConvertSdfToImage(distances, positions.rows(), positions.cols());
        ConvertSdfToVoxelGrid(img_sdf, positions_test, voxel_grid_sdf);
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
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        cv::putText(
            img_sdf,
            fmt::format("update {:.2f} fps", gp_update_fps),
            cv::Point(10, 60),
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        cv::putText(
            img_sdf,
            fmt::format("test {:.2f} fps", gp_test_fps),
            cv::Point(10, 90),
            cv::FONT_HERSHEY_PLAIN,
            1.5,
            cv::Scalar(255, 255, 255),
            2);
        cv::imshow("ranges", depth_jet);
        cv::imshow("sdf", img_sdf);
        cv::imshow("sdf_sign", img_sdf_sign);
        cv::waitKey(1);
        /// update the sensor mesh
        Matrix4 last_pose_inv = last_pose.inverse();
        Matrix4 cur_pose = Matrix4::Identity();
        cur_pose.template topLeftCorner<3, 3>() = rotation_sensor;
        cur_pose.template topRightCorner<3, 1>() = translation_sensor;
        Matrix4 delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose.template cast<double>());
        mesh_sensor_xyz->Transform(delta_pose.template cast<double>());
        /// update the observation point cloud
        if (std::find(geometries.begin(), geometries.end(), pcd_obs) != geometries.end()) {
            pcd_obs->points_.clear();
            pcd_obs->colors_.clear();
            const auto &hit_points =
                surf_mapping->GetSensorGp()->GetSensorFrame()->GetHitPointsWorld();
            pcd_obs->points_.reserve(hit_points.size());
            for (const auto &hit_point: hit_points) {
                pcd_obs->points_.emplace_back(hit_point.template cast<double>());
            }
            pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        }
        /// update the surface point cloud and normals
        auto it1 = std::find(geometries.begin(), geometries.end(), pcd_surf_points);
        auto it2 = std::find(geometries.begin(), geometries.end(), line_set_surf_normals);
        if (const auto tree = surf_mapping->GetTree();
            (it1 != geometries.end() || it2 != geometries.end()) && tree != nullptr) {
            pcd_surf_points->points_.clear();
            line_set_surf_normals->points_.clear();
            line_set_surf_normals->lines_.clear();
            auto end = surf_mapping->EndSurfaceData();
            for (auto it = surf_mapping->BeginSurfaceData(); it != end; ++it) {
                erl::sdf_mapping::SurfaceData<Dtype, 3> &surface_data = *it;
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

        const auto t_end = std::chrono::high_resolution_clock::now();
        const auto duration_total =
            std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && options.test_io) { test_io(); }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();
}

TEST(GpSdfMapping, 3Dd) { TestImpl3D<double>(); }

TEST(GpSdfMapping, 3Df) { TestImpl3D<float>(); }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
