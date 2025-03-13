#include "erl_common/eigen.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/occupancy_octree_drawer.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"

#include <boost/program_options.hpp>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;
using Dtype = float;
using GpOccSurfaceMapping3D = erl::sdf_mapping::GpOccSurfaceMapping3D<Dtype>;
using RangeSensor3D = erl::geometry::RangeSensor3D<Dtype>;
using Lidar3D = erl::geometry::Lidar3D<Dtype>;
using DepthCamera3D = erl::geometry::DepthCamera3D<Dtype>;
using DepthFrame3D = erl::geometry::DepthFrame3D<Dtype>;
using LidarFrame3D = erl::geometry::LidarFrame3D<Dtype>;
using SurfaceMappingOctree = erl::sdf_mapping::SurfaceMappingOctree<Dtype>;
using Matrix3 = Eigen::Matrix3<Dtype>;
using Matrix4 = Eigen::Matrix4<Dtype>;
using Vector3 = Eigen::Vector3<Dtype>;

struct Options {
    bool use_cow_and_lady = false;                                                  // use Cow and Lady dataset, otherwise use mesh_file and traj_file
    std::string cow_and_lady_dir;                                                   // directory containing the Cow and Lady dataset
    std::string mesh_file = kProjectRootDir / "data" / "replica-hotel-0.ply";       // mesh file
    std::string traj_file = kProjectRootDir / "data" / "replica-hotel-0-traj.txt";  // trajectory file
    std::string surface_mapping_config_file = kProjectRootDir / "config" / fmt::format("surface_mapping_3d_lidar_{}.yaml", type_name<Dtype>());
    long stride = 1;
    Dtype test_res = 0.02;
    Dtype test_z = 0.0;
    long test_xs = 150;
    long test_ys = 100;
    bool test_io = false;
    bool hold = false;
};

Options g_options;

TEST(GpOccSurfaceMapping3D, Build_Save_Load) {
    GTEST_PREPARE_OUTPUT_DIR();

    // load setting
    const auto gp_surf_setting = std::make_shared<GpOccSurfaceMapping3D::Setting>();
    ERL_ASSERTM(gp_surf_setting->FromYamlFile(g_options.surface_mapping_config_file), "Failed to load config file: {}.", g_options.surface_mapping_config_file);

    // prepare the scene
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<RangeSensor3D> range_sensor = nullptr;
    std::shared_ptr<erl::geometry::CowAndLady> cow_and_lady = nullptr;
    std::vector<std::pair<Matrix3, Vector3>> poses;
    Vector3 area_min, area_max;
    if (g_options.use_cow_and_lady) {
        cow_and_lady = std::make_shared<erl::geometry::CowAndLady>(g_options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        area_min = cow_and_lady->GetMapMin().cast<Dtype>();
        area_max = cow_and_lady->GetMapMax().cast<Dtype>();
        const auto depth_frame_setting = std::make_shared<DepthFrame3D::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = erl::geometry::CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = erl::geometry::CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = erl::geometry::CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = erl::geometry::CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = erl::geometry::CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = erl::geometry::CowAndLady::kCameraCy;
    } else {
        const auto mesh = open3d::io::CreateMeshFromFile(g_options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", g_options.mesh_file);
        area_min = mesh->GetMinBound().cast<Dtype>();
        area_max = mesh->GetMaxBound().cast<Dtype>();
        if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<LidarFrame3D>()) {
            const auto lidar_frame_setting = std::dynamic_pointer_cast<LidarFrame3D::Setting>(gp_surf_setting->sensor_gp->sensor_frame);
            const auto lidar_setting = std::make_shared<Lidar3D::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<Lidar3D>(lidar_setting);
            range_sensor->AddMesh(g_options.mesh_file);
        } else if (gp_surf_setting->sensor_gp->sensor_frame_type == type_name<DepthFrame3D>()) {
            const auto depth_frame_setting = std::dynamic_pointer_cast<DepthFrame3D::Setting>(gp_surf_setting->sensor_gp->sensor_frame);
            const auto depth_camera_setting = std::make_shared<DepthCamera3D::Setting>();
            *depth_camera_setting = depth_frame_setting->camera_intrinsic;
            range_sensor = std::make_shared<DepthCamera3D>(depth_camera_setting);
            range_sensor->AddMesh(g_options.mesh_file);
        } else {
            ERL_FATAL("Unknown sensor_frame_type: {}", gp_surf_setting->sensor_gp->sensor_frame_type);
        }
        geometries.push_back(mesh);
        poses = erl::geometry::Trajectory<Dtype>::LoadSe3(g_options.traj_file, false);
    }

    // prepare the mapping
    GpOccSurfaceMapping3D gp(gp_surf_setting);

    // prepare the visualizer
    const auto visualizer_setting = std::make_shared<erl::geometry::Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    erl::geometry::Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    const auto mesh_sensor_xyz = erl::geometry::CreateAxisMesh(Matrix4::Identity().cast<double>(), 0.1);
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
    const long max_wp_idx = g_options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size());
    Matrix4 last_pose = Matrix4::Identity();
    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();

        if (wp_idx >= max_wp_idx) {
            if (g_options.test_io) {
                const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
                (void) timer;
                const auto filename = test_output_dir / "gp_occ_surface_mapping_3d.bin";
                ERL_ASSERTM(gp.Write(filename), "Failed to write to file: {}", filename);
                GpOccSurfaceMapping3D gp_load(std::make_shared<GpOccSurfaceMapping3D::Setting>());
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
            if (g_options.use_cow_and_lady) {
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
        wp_idx += g_options.stride;

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
        if (!g_options.use_cow_and_lady) {
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
        cur_pose.topLeftCorner<3, 3>() = rotation;
        cur_pose.topRightCorner<3, 1>() = translation;
        Matrix4 delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose.cast<double>());
        mesh_sensor_xyz->Transform(delta_pose.cast<double>());
        /// update the observation point cloud
        pcd_obs->points_.clear();
        pcd_obs->colors_.clear();
        pcd_obs->points_.reserve(gp.GetSensorGp()->GetSensorFrame()->GetHitPointsWorld().size());
        for (const auto &point: gp.GetSensorGp()->GetSensorFrame()->GetHitPointsWorld()) { pcd_obs->points_.emplace_back(point.cast<double>()); }
        pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        /// update the surface point cloud and normals
        pcd_surf_points->points_.clear();
        line_set_surf_normals->points_.clear();
        line_set_surf_normals->lines_.clear();
        if (const auto octree = gp.GetOctree(); octree != nullptr) {
            for (auto it = octree->BeginLeaf(), end = octree->EndLeaf(); it != end; ++it) {
                if (!it->HasSurfaceData()) { continue; }
                auto &surface_data = gp.GetSurfaceDataManager()[it->surface_data_index];
                const Vector3 &position = surface_data.position;
                const Vector3 &normal = surface_data.normal;
                ERL_ASSERTM(std::abs(normal.norm() - 1.0) < 1.e-5, "normal.norm() = {:.6f}", normal.norm());
                pcd_surf_points->points_.emplace_back(position.cast<double>());
                line_set_surf_normals->points_.emplace_back(position.cast<double>());
                line_set_surf_normals->points_.emplace_back((position + 0.1 * normal).cast<double>());
                line_set_surf_normals->lines_.emplace_back(line_set_surf_normals->points_.size() - 2, line_set_surf_normals->points_.size() - 1);
            }
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const auto duration_total = std::chrono::duration<Dtype, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && g_options.test_io) {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
            (void) timer;
            const auto filename = test_output_dir / "gp_occ_surface_mapping_3d.bin";
            ERL_ASSERTM(gp.Write(filename), "Failed to write to file: {}", filename);
            GpOccSurfaceMapping3D gp_load(std::make_shared<GpOccSurfaceMapping3D::Setting>());
            ERL_ASSERTM(gp_load.Read(filename), "Failed to read from file: {}", filename);
            ERL_ASSERTM(gp == gp_load, "gp != gp_load");
        }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();

    auto drawer_setting = std::make_shared<SurfaceMappingOctree::Drawer::Setting>();
    drawer_setting->area_min = area_min.cast<double>();
    drawer_setting->area_max = area_max.cast<double>();
    drawer_setting->occupied_only = true;
    SurfaceMappingOctree::Drawer octree_drawer(drawer_setting, gp.GetOctree());
    auto mesh = geometries[0];
    geometries = octree_drawer.GetBlankGeometries();
    geometries.push_back(mesh);
    octree_drawer.DrawLeaves(geometries);
    visualizer.Reset();
    visualizer.AddGeometries(geometries);
    visualizer.Show();
}

int
main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        namespace po = boost::program_options;
        po::options_description desc;
        // clang-format off
        desc.add_options()
            ("help", "produce help message")
            (
                "use-cow-and-lady",
                po::bool_switch(&g_options.use_cow_and_lady),
                "use Cow and Lady dataset, otherwise use the mesh file and trajectory file"
            )(
                "cow-and-lady-dir",
                po::value<std::string>(&g_options.cow_and_lady_dir)->default_value(g_options.cow_and_lady_dir)->value_name("dir"),
                "directory containing the Cow and Lady dataset"
            )
            ("mesh-file", po::value<std::string>(&g_options.mesh_file)->default_value(g_options.mesh_file)->value_name("file"), "mesh file")
            ("traj-file", po::value<std::string>(&g_options.traj_file)->default_value(g_options.traj_file)->value_name("file"), "trajectory file")
            (
                "surface-mapping-config-file",
                po::value<std::string>(&g_options.surface_mapping_config_file)->default_value(g_options.surface_mapping_config_file)->value_name("file"),
                "SDF mapping config file"
            )
            ("stride", po::value<long>(&g_options.stride)->default_value(g_options.stride)->value_name("stride"), "stride")
            ("test-res", po::value<Dtype>(&g_options.test_res)->default_value(g_options.test_res)->value_name("res"), "test resolution")
            ("test-z", po::value<Dtype>(&g_options.test_z)->default_value(g_options.test_z)->value_name("z"), "test z")
            ("test-xs", po::value<long>(&g_options.test_xs)->default_value(g_options.test_xs)->value_name("xs"), "test xs")
            ("test-ys", po::value<long>(&g_options.test_ys)->default_value(g_options.test_ys)->value_name("ys"), "test ys")
            ("test-io", po::bool_switch(&g_options.test_io), "test IO")
            ("hold", po::bool_switch(&g_options.hold), "hold the window");
        // clang-format on

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return RUN_ALL_TESTS();
}
