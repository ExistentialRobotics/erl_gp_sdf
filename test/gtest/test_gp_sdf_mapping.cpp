#include "erl_common/test_helper.hpp"
#include "erl_geometry/cow_and_lady.hpp"
#include "erl_geometry/depth_camera_3d.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_geometry/trajectory.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
#include <erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

const std::filesystem::path kProjectRootDir = ERL_SDF_MAPPING_ROOT_DIR;
using Dtype = double;
using EVectorX = Eigen::VectorX<Dtype>;
using EMatrixX = Eigen::MatrixX<Dtype>;
using EMatrix3 = Eigen::Matrix3<Dtype>;
using EMatrix4 = Eigen::Matrix4<Dtype>;
using EVector3 = Eigen::Vector3<Dtype>;
using EMatrix3X = Eigen::Matrix3X<Dtype>;
using EMatrix4X = Eigen::Matrix4X<Dtype>;
using EMatrix6X = Eigen::Matrix<Dtype, 6, Eigen::Dynamic>;

struct Options {
    bool use_cow_and_lady = false;                                                  // use Cow and Lady dataset, otherwise use mesh_file and traj_file
    std::string cow_and_lady_dir;                                                   // directory containing the Cow and Lady dataset
    std::string mesh_file = kProjectRootDir / "data" / "replica-hotel-0.ply";       // mesh file
    std::string traj_file = kProjectRootDir / "data" / "replica-hotel-0-traj.txt";  // trajectory file
    std::string surface_mapping_config_file = kProjectRootDir / "config" / "surface_mapping_3d_lidar.yaml";
    std::string sdf_mapping_config_file = kProjectRootDir / "config" / "sdf_mapping_3d_lidar.yaml";
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

Options g_options;

cv::Mat
ConvertDepthToImage(const EMatrixX &ranges) {
    cv::Mat depth_jet;
    EMatrixX ranges_img = EMatrixX::Zero(ranges.rows(), ranges.cols());
    Dtype min_range = std::numeric_limits<Dtype>::max();
    Dtype max_range = std::numeric_limits<Dtype>::lowest();
    for (long i = 0; i < ranges.size(); ++i) {
        const double &range = ranges.data()[i];
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

std::pair<cv::Mat, cv::Mat>
ConvertSdfToImage(const EVectorX &distances, const long rows, const long cols) {
    EMatrixX sdf = distances.reshaped(rows, cols);
    const Eigen::MatrixX8U sdf_sign = (sdf.array() >= 0.0).cast<uint8_t>() * 255;
    // std::cout << "sdf.minCoeff(): " << sdf.minCoeff() << std::endl << "sdf.maxCoeff(): " << sdf.maxCoeff() << std::endl;
    sdf = (sdf.array() - sdf.minCoeff()) / (sdf.maxCoeff() - sdf.minCoeff()) * 255.0;
    const Eigen::MatrixX8U sdf_uint8 = sdf.cast<uint8_t>();
    cv::Mat img_sdf, img_sdf_sign;
    cv::eigen2cv(sdf_uint8, img_sdf);
    cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
    cv::eigen2cv(sdf_sign, img_sdf_sign);
    // for zero pixel in img_sdf_sign fill sdf_sign with zero
    const cv::Mat mask = img_sdf_sign == 0;
    img_sdf.setTo(0, mask);
    return {std::move(img_sdf), std::move(img_sdf_sign)};
}

void
ConvertSdfToVoxelGrid(const cv::Mat &img_sdf, const Eigen::Matrix3Xd &positions, const std::shared_ptr<open3d::geometry::VoxelGrid> &voxel_grid_sdf) {
    voxel_grid_sdf->voxels_.clear();
    for (int j = 0; j < img_sdf.cols; ++j) {  // column major
        for (int i = 0; i < img_sdf.rows; ++i) {
            auto position = positions.col(i + j * img_sdf.rows);
            const auto &color = img_sdf.at<cv::Vec3b>(i, j);
            voxel_grid_sdf->AddVoxel({voxel_grid_sdf->GetVoxel(position), Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)});
        }
    }
}

TEST(GPSdfMapping, Build_Save_Load_3D) {
    GTEST_PREPARE_OUTPUT_DIR();
    using SurfaceMapping = erl::sdf_mapping::GpOccSurfaceMapping3D;
    using GpSdfMapping3D = erl::sdf_mapping::GpSdfMapping<3, Dtype, SurfaceMapping>;
    // load setting
    const auto surface_mapping_setting = std::make_shared<SurfaceMapping::Setting>();
    ERL_ASSERTM(
        surface_mapping_setting->FromYamlFile(g_options.surface_mapping_config_file),
        "Failed to load surface_mapping_config_file: {}",
        g_options.surface_mapping_config_file);
    const auto sdf_mapping_setting = std::make_shared<GpSdfMapping3D::Setting>();
    ERL_ASSERTM(
        sdf_mapping_setting->FromYamlFile(g_options.sdf_mapping_config_file),
        "Failed to load sdf_mapping_config_file: {}",
        g_options.sdf_mapping_config_file);
    // create surface mapping
    const auto surface_mapping = std::make_shared<SurfaceMapping>(surface_mapping_setting);
    // create sdf mapping
    GpSdfMapping3D sdf_mapping(sdf_mapping_setting, surface_mapping);

    // prepare the scene
    std::vector<std::shared_ptr<open3d::geometry::Geometry>> geometries;  // for visualization
    std::shared_ptr<erl::geometry::RangeSensor3D> range_sensor = nullptr;
    std::shared_ptr<erl::geometry::CowAndLady> cow_and_lady = nullptr;
    std::vector<std::pair<EMatrix3, EVector3>> poses;
    EVector3 map_min, map_max;
    if (g_options.use_cow_and_lady) {
        ERL_INFO("Using Cow and Lady dataset.");
        ERL_INFO("Using depth.");
        cow_and_lady = std::make_shared<erl::geometry::CowAndLady>(g_options.cow_and_lady_dir);
        geometries.push_back(cow_and_lady->GetGroundTruthPointCloud());
        map_min = cow_and_lady->GetMapMin();
        map_max = cow_and_lady->GetMapMax();
        const auto depth_frame_setting = std::make_shared<erl::geometry::DepthFrame3D::Setting>();
        depth_frame_setting->camera_intrinsic.image_height = erl::geometry::CowAndLady::kImageHeight;
        depth_frame_setting->camera_intrinsic.image_width = erl::geometry::CowAndLady::kImageWidth;
        depth_frame_setting->camera_intrinsic.camera_fx = erl::geometry::CowAndLady::kCameraFx;
        depth_frame_setting->camera_intrinsic.camera_fy = erl::geometry::CowAndLady::kCameraFy;
        depth_frame_setting->camera_intrinsic.camera_cx = erl::geometry::CowAndLady::kCameraCx;
        depth_frame_setting->camera_intrinsic.camera_cy = erl::geometry::CowAndLady::kCameraCy;
    } else {
        ERL_INFO("Using mesh file: {}", g_options.mesh_file);
        const auto mesh = open3d::io::CreateMeshFromFile(g_options.mesh_file);
        ERL_ASSERTM(!mesh->vertices_.empty(), "Failed to load mesh file: {}", g_options.mesh_file);
        map_min = mesh->GetMinBound();
        map_max = mesh->GetMaxBound();
        if (surface_mapping_setting->sensor_gp->range_sensor_frame_type == demangle(typeid(erl::geometry::LidarFrame3D).name())) {
            ERL_INFO("Using LiDAR.");
            const auto lidar_frame_setting =
                std::dynamic_pointer_cast<erl::geometry::LidarFrame3D::Setting>(surface_mapping_setting->sensor_gp->range_sensor_frame);
            const auto lidar_setting = std::make_shared<erl::geometry::Lidar3D::Setting>();
            lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
            lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
            lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
            lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
            lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
            lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
            range_sensor = std::make_shared<erl::geometry::Lidar3D>(lidar_setting);
        } else if (surface_mapping_setting->sensor_gp->range_sensor_frame_type == demangle(typeid(erl::geometry::DepthFrame3D).name())) {
            ERL_INFO("Using depth.");
            const auto depth_frame_setting =
                std::dynamic_pointer_cast<erl::geometry::DepthFrame3D::Setting>(surface_mapping_setting->sensor_gp->range_sensor_frame);
            const auto depth_camera_setting = std::make_shared<erl::geometry::DepthCamera3D::Setting>();
            *depth_camera_setting = depth_frame_setting->camera_intrinsic;
            range_sensor = std::make_shared<erl::geometry::DepthCamera3D>(depth_camera_setting);
        } else {
            ERL_FATAL("Unknown range_sensor_frame_type: {}", surface_mapping_setting->sensor_gp->range_sensor_frame_type);
        }
        range_sensor->AddMesh(mesh->vertices_, mesh->triangles_);
        geometries.push_back(mesh);
        poses = erl::geometry::Trajectory::LoadSe3(g_options.traj_file, false);
    }

    // prepare the visualizer
    const auto visualizer_setting = std::make_shared<erl::geometry::Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    erl::geometry::Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto mesh_sensor = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    mesh_sensor->PaintUniformColor({1.0, 0.5, 0.0});
    const auto mesh_sensor_xyz = erl::geometry::CreateAxisMesh(EMatrix4::Identity(), 0.1);
    const auto pcd_obs = std::make_shared<open3d::geometry::PointCloud>();
    const auto pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
    const auto line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
    const auto voxel_grid_sdf = std::make_shared<open3d::geometry::VoxelGrid>();
    voxel_grid_sdf->origin_.setZero();
    voxel_grid_sdf->voxel_size_ = g_options.test_res;
    geometries.push_back(mesh_sensor);
    geometries.push_back(mesh_sensor_xyz);
    // geometries.push_back(pcd_obs);
    // geometries.push_back(pcd_surf_points);
    // geometries.push_back(line_set_surf_normals);
    geometries.push_back(voxel_grid_sdf);
    visualizer.AddGeometries(geometries);

    // prepare the test positions
    Eigen::MatrixX<EVector3> positions(g_options.test_xs, g_options.test_ys);  // test position in the sensor frame
    const EVector3 offset(
        -0.5 * g_options.test_res * static_cast<Dtype>(g_options.test_xs),
        -0.5 * g_options.test_res * static_cast<Dtype>(g_options.test_ys),
        0.0);
    for (long j = 0; j < positions.cols(); ++j) {
        for (long i = 0; i < positions.rows(); ++i) {
            positions(i, j) = EVector3(static_cast<Dtype>(i) * g_options.test_res, static_cast<Dtype>(j) * g_options.test_res, 0) + offset;
        }
    }

    // animation callback
    long wp_idx = 0;
    bool animation_ended = false;
    const long max_wp_idx = std::min(g_options.use_cow_and_lady ? cow_and_lady->Size() : static_cast<long>(poses.size()), g_options.max_frames);
    EMatrix4 last_pose = EMatrix4::Identity();
    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) -> bool {
        ERL_TRACY_FRAME_MARK_START();
        if (animation_ended) {  // g_options.hold is true, so the window is not closed yet
            cv::waitKey(1);
            return false;
        }

        if (wp_idx >= max_wp_idx) {
            animation_ended = true;
            if (g_options.test_whole_map_at_end) {
                erl::common::GridMapInfo2D grid_map_info(
                    map_min.head<2>(),
                    map_max.head<2>(),
                    Eigen::Vector2d(g_options.test_res, g_options.test_res),
                    Eigen::Vector2i(10, 10));
                EMatrix3X test_positions(3, grid_map_info.Size());
                test_positions.topRows(2) = grid_map_info.GenerateMeterCoordinates(false);
                test_positions.row(2).setConstant(g_options.test_z);
                EVectorX distances;
                {
                    EMatrix3X gradients(3, test_positions.cols());
                    EMatrix4X variances;
                    EMatrix6X covairances;
                    ERL_BLOCK_TIMER_MSG("sdf_mapping.Test");
                    EXPECT_TRUE(sdf_mapping.Test(test_positions, distances, gradients, variances, covairances));
                }
                auto [img_sdf, img_sdf_sign] = ConvertSdfToImage(distances, grid_map_info.Shape(0), grid_map_info.Shape(1));
                ConvertSdfToVoxelGrid(img_sdf, test_positions, voxel_grid_sdf);
                double resize_scale = g_options.image_resize_scale;
                resize_scale = std::min(resize_scale, 1920.0 / img_sdf.cols);
                resize_scale = std::min(resize_scale, 1920.0 / img_sdf.rows);
                cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
                cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
                cv::imshow("sdf_whole_map", img_sdf);
                cv::imshow("sdf_sign_whole_map", img_sdf_sign);
                cv::waitKey(1);
            }
            if (g_options.test_io) {
                // TODO: test io
                // const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
                // (void) timer;
                // const auto filename = test_output_dir / "gp_sdf_mapping_3d.bin";
                // ERL_ASSERTM(sdf_mapping.Write(filename), "Failed to write to file: {}", filename);
                // GpSdfMapping3D sdf_mapping_load(std::make_shared<GpSdfMapping3D::Setting>());
                // ERL_ASSERTM(sdf_mapping_load.Read(filename), "Failed to read from file: {}", filename);
                // ERL_ASSERTM(sdf_mapping == sdf_mapping_load, "gp != gp_load");
            }
            if (!g_options.hold) {
                wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
                vis->Close();                            // close the window
            }
            return true;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        EMatrix3 rotation_sensor, rotation;
        EVector3 translation_sensor, translation;
        ERL_INFO("wp_idx: {}", wp_idx);

        cv::Mat depth_jet;
        EMatrixX ranges;
        double dt;
        {
            ERL_BLOCK_TIMER_MSG_TIME("data loading", dt);
            if (g_options.use_cow_and_lady) {
                const auto frame = (*cow_and_lady)[wp_idx];
                rotation = frame.rotation;
                translation = frame.translation;
                EMatrix4 pose = EMatrix4::Identity();
                pose.topLeftCorner<3, 3>() = rotation;
                pose.topRightCorner<3, 1>() = translation;
                pose = pose * erl::geometry::DepthCamera3D::cTo;
                rotation_sensor = pose.topLeftCorner<3, 3>();
                translation_sensor = pose.topRightCorner<3, 1>();
                ranges = frame.depth;
                depth_jet = frame.depth_jet;
            } else {
                std::tie(rotation_sensor, translation_sensor) = poses[wp_idx];
                ranges = range_sensor->Scan(rotation_sensor, translation_sensor);
                std::tie(rotation, translation) = range_sensor->GetOpticalPose(rotation_sensor, translation_sensor);
                depth_jet = ConvertDepthToImage(ranges);
            }
        }
        ERL_TRACY_PLOT("data loading (ms)", dt);
        wp_idx += g_options.stride;

        {
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Update", dt);
            // provide ranges and frame pose in the world frame
            ERL_WARN_COND(!sdf_mapping.Update(rotation, translation, ranges), "sdf_mapping.Update failed.");
        }
        double gp_update_fps = 1000.0 / dt;
        ERL_TRACY_PLOT("sdf_mapping_update (ms)", dt);
        ERL_TRACY_PLOT("sdf_mapping_update (fps)", gp_update_fps);

        // test
        EMatrix3X positions_test(3, positions.size());
        for (long j = 0; j < positions.cols(); ++j) {
            for (long i = 0; i < positions.rows(); ++i) {
                const EVector3 &position = positions(i, j);
                positions_test.col(i + j * positions.rows()) = rotation_sensor * position + translation_sensor;  // sensor frame to world frame
            }
        }
        EVectorX distances(positions_test.cols());
        {
            EMatrix3X gradients(3, positions_test.cols());
            EMatrix4X variances;
            EMatrix6X covairances;
            ERL_BLOCK_TIMER_MSG_TIME("sdf_mapping.Test", dt);
            EXPECT_TRUE(sdf_mapping.Test(positions_test, distances, gradients, variances, covairances));
        }
        double gp_test_fps = 1000.0 / dt;
        ERL_TRACY_PLOT("gp_test (ms)", dt);
        ERL_TRACY_PLOT("gp_test (fps)", gp_test_fps);

        // update visualization
        /// update the images and voxel grid
        cv::putText(depth_jet, fmt::format("frame {}", wp_idx), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(depth_jet, fmt::format("update {:.2f} fps", gp_update_fps), cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        auto [img_sdf, img_sdf_sign] = ConvertSdfToImage(distances, positions.rows(), positions.cols());
        ConvertSdfToVoxelGrid(img_sdf, positions_test, voxel_grid_sdf);
        double resize_scale = g_options.image_resize_scale;
        resize_scale = std::min(resize_scale, 1920.0 / img_sdf.cols);
        resize_scale = std::min(resize_scale, 1920.0 / img_sdf.rows);
        cv::resize(img_sdf, img_sdf, cv::Size(), resize_scale, resize_scale);
        cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), resize_scale, resize_scale);
        cv::putText(img_sdf, fmt::format("frame {}", wp_idx), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_sdf, fmt::format("update {:.2f} fps", gp_update_fps), cv::Point(10, 60), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(img_sdf, fmt::format("test {:.2f} fps", gp_test_fps), cv::Point(10, 90), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 255), 2);
        cv::imshow("ranges", depth_jet);
        cv::imshow("sdf", img_sdf);
        cv::imshow("sdf_sign", img_sdf_sign);
        cv::waitKey(1);
        /// update the sensor mesh
        EMatrix4 last_pose_inv = last_pose.inverse();
        EMatrix4 cur_pose = EMatrix4::Identity();
        cur_pose.topLeftCorner<3, 3>() = rotation_sensor;
        cur_pose.topRightCorner<3, 1>() = translation_sensor;
        EMatrix4 delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;
        mesh_sensor->Transform(delta_pose);
        mesh_sensor_xyz->Transform(delta_pose);
        /// update the observation point cloud
        if (std::find(geometries.begin(), geometries.end(), pcd_obs) != geometries.end()) {
            pcd_obs->points_.clear();
            pcd_obs->colors_.clear();
            pcd_obs->points_ = surface_mapping->GetSensorGp()->GetRangeSensorFrame()->GetHitPointsWorld();
            pcd_obs->PaintUniformColor({0.0, 1.0, 0.0});
        }
        /// update the surface point cloud and normals
        if (const auto octree = surface_mapping->GetOctree();  //
            (std::find(geometries.begin(), geometries.end(), pcd_surf_points) != geometries.end() ||
             std::find(geometries.begin(), geometries.end(), line_set_surf_normals) != geometries.end()) &&
            octree != nullptr) {
            pcd_surf_points->points_.clear();
            line_set_surf_normals->points_.clear();
            line_set_surf_normals->lines_.clear();
            for (auto it = octree->BeginLeaf(), end = octree->EndLeaf(); it != end; ++it) {
                if (!it->HasSurfaceData()) { continue; }
                const auto &surface_data = surface_mapping->GetSurfaceDataManager()[it->surface_data_index];
                const EVector3 &position = surface_data.position;
                const EVector3 &normal = surface_data.normal;
                ERL_ASSERTM(std::abs(normal.norm() - 1.0) < 1.e-6, "normal.norm() = {:.6f}", normal.norm());
                pcd_surf_points->points_.emplace_back(position);
                line_set_surf_normals->points_.emplace_back(position);
                line_set_surf_normals->points_.emplace_back(position + 0.1 * normal);
                line_set_surf_normals->lines_.emplace_back(line_set_surf_normals->points_.size() - 2, line_set_surf_normals->points_.size() - 1);
            }
            line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const auto duration_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);
        ERL_TRACY_PLOT("gui_update (ms)", duration_total);
        ERL_TRACY_PLOT("gui_update (fps)", 1000.0 / duration_total);

        if (wp_idx == 1 && g_options.test_io) {
            // TODO: test io
            // const erl::common::BlockTimer<std::chrono::milliseconds> timer("IO");
            // (void) timer;
            // const auto filename = test_output_dir / "gp_sdf_mapping_3d.bin";
            // ERL_ASSERTM(sdf_mapping.Write(filename), "Failed to write to file: {}", filename);
            // GpSdfMapping3D sdf_mapping_load(std::make_shared<GpSdfMapping3D::Setting>());
            // ERL_ASSERTM(sdf_mapping_load.Read(filename), "Failed to read from file: {}", filename);
            // ERL_ASSERTM(sdf_mapping == sdf_mapping_load, "gp != gp_load");
        }

        ERL_TRACY_FRAME_MARK_END();
        return true;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();

    long max_gp_size = 0;
    long mean_gp_size = 0;
    long gp_cnt = 0;
    for (const auto &[key, local_gp]: sdf_mapping.GetGpMap()) {
        max_gp_size = std::max(local_gp->num_edf_samples, max_gp_size);
        mean_gp_size += local_gp->num_edf_samples;
        ++gp_cnt;
    }
    mean_gp_size /= gp_cnt;
    ERL_INFO("max_gp_size: {}, mean_gp_size: {}", max_gp_size, mean_gp_size);
}
