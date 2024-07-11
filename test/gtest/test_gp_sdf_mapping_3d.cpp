#include "erl_common/test_helper.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

#include <erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/VoxelGrid.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

#define STRIDE             1
#define ANIMATION_INTERVAL 2

TEST(GpSdfMapping3D, LiDAR) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_surf_setting = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping3D::Setting>();
    gp_surf_setting->FromYamlFile(gtest_src_dir / "../../config/surface_mapping_3d_lidar.yaml");
    const auto gp_surf = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping3D>(gp_surf_setting);
    const auto gp_sdf_setting = std::make_shared<erl::sdf_mapping::GpSdfMapping3D::Setting>();
    gp_sdf_setting->FromYamlFile(gtest_src_dir / "../../config/sdf_mapping_3d_lidar.yaml");
    erl::sdf_mapping::GpSdfMapping3D gp(gp_surf, gp_sdf_setting);

    const Eigen::MatrixXd traj_matrix = erl::common::LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "replica-hotel-0-traj.txt").transpose();

    const auto mesh = open3d::io::CreateMeshFromFile(gtest_src_dir / "replica-hotel-0.ply");
    const auto lidar_setting = std::make_shared<erl::geometry::Lidar3D::Setting>();
    const auto lidar_frame_setting = std::dynamic_pointer_cast<erl::geometry::LidarFrame3D::Setting>(gp_surf_setting->sensor_gp->range_sensor_frame);
    lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
    lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
    lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
    lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
    lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
    lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
    erl::geometry::Lidar3D lidar(lidar_setting, mesh->vertices_, mesh->triangles_);

    const auto visualizer_setting = std::make_shared<erl::geometry::Open3dVisualizerWrapper::Setting>();
    visualizer_setting->window_name = test_info->name();
    visualizer_setting->mesh_show_back_face = false;
    erl::geometry::Open3dVisualizerWrapper visualizer(visualizer_setting);
    const auto mesh_lidar = open3d::geometry::TriangleMesh::CreateSphere(0.05);
    const auto mesh_lidar_xyz = erl::geometry::Open3dVisualizerWrapper::CreateAxisMesh(Eigen::Matrix4d::Identity());
    mesh_lidar->PaintUniformColor({1.0, 0.5, 0.0});
    const auto pcd_lidar = std::make_shared<open3d::geometry::PointCloud>();
    const auto pcd_surf_points = std::make_shared<open3d::geometry::PointCloud>();
    // const auto line_set_lidar = std::make_shared<open3d::geometry::LineSet>();
    const auto line_set_surf_normals = std::make_shared<open3d::geometry::LineSet>();
    const auto voxel_grid_sdf = std::make_shared<open3d::geometry::VoxelGrid>();
    visualizer.AddGeometries(
        {mesh,
         mesh_lidar,
         mesh_lidar_xyz,
         // pcd_lidar,
         // pcd_surf_points,
         // line_set_surf_normals,
         voxel_grid_sdf});

    constexpr long test_xs = 150;
    constexpr long test_ys = 100;
    Eigen::MatrixX<Eigen::Vector3d> positions(test_xs, test_ys);
    constexpr double voxel_size = 0.02;
    voxel_grid_sdf->origin_.setZero();
    voxel_grid_sdf->voxel_size_ = voxel_size;
    const double z = traj_matrix(11, 0);
    const Eigen::Vector3d center(-0.5 * voxel_size * static_cast<double>(test_xs), -0.5 * voxel_size * static_cast<double>(test_ys), z);
    for (long j = 0; j < positions.cols(); ++j) {
        for (long i = 0; i < positions.rows(); ++i) {
            positions(i, j) = Eigen::Vector3d(static_cast<double>(i) * voxel_size, static_cast<double>(j) * voxel_size, 0) + center;
        }
    }

    long wp_idx = 0;
    long animation_cnt = 0;
    const Eigen::MatrixX<Eigen::Vector3d> &ray_dirs_frame = lidar.GetRayDirectionsInFrame();
    Eigen::Matrix4d last_pose = Eigen::Matrix4d::Identity();
    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *vis) -> bool {
        // if (wp_idx >= traj_matrix.cols()) {
        if (wp_idx >= 100) {
            wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
            vis->Close();                            // close the window
            return false;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        const Eigen::Matrix4d pose = traj_matrix.col(wp_idx).reshaped(4, 4).transpose();
        const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
        const Eigen::Vector3d translation = pose.topRightCorner<3, 1>();
        ERL_INFO("wp_idx: {}", wp_idx);
        wp_idx += STRIDE;

        Eigen::MatrixXd ranges;
        {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("lidar.Scan");
            (void) timer;
            ranges = lidar.Scan(rotation, translation);
        }

        {
            const erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Update");
            (void) timer;
            if (!gp.Update(rotation, translation, ranges)) {
                ERL_WARN("gp.Update failed.");
                wrapper->SetAnimationCallback(nullptr);
                return false;
            }
        }

        Eigen::Matrix4d last_pose_inv = last_pose.inverse();
        Eigen::Matrix4d cur_pose = Eigen::Matrix4d::Identity();
        cur_pose.topLeftCorner<3, 3>() = rotation;
        cur_pose.topRightCorner<3, 1>() = translation;
        Eigen::Matrix4d delta_pose = cur_pose * last_pose_inv;
        last_pose = cur_pose;

        {
            Eigen::Matrix3Xd positions_test(3, positions.size());
            for (long j = 0; j < positions.cols(); ++j) {
                for (long i = 0; i < positions.rows(); ++i) {
                    Eigen::Vector3d &position = positions(i, j);
                    positions_test.col(i + j * positions.rows()) = rotation * position + translation;
                }
            }
            Eigen::VectorXd distances(positions_test.cols());
            {
                Eigen::Matrix3Xd gradients(3, positions_test.cols());
                Eigen::Matrix4Xd variances;
                Eigen::Matrix6Xd covairances;
                const erl::common::BlockTimer<std::chrono::milliseconds> timer("gp.Test");
                (void) timer;
                gp.Test(positions_test, distances, gradients, variances, covairances);
            }
            Eigen::MatrixXd sdf = distances.reshaped(positions.rows(), positions.cols());
            Eigen::MatrixX8U sdf_sign = (sdf.array() > 0.0).cast<uint8_t>() * 255;
            std::cout << "sdf.minCoeff(): " << sdf.minCoeff() << std::endl << "sdf.maxCoeff(): " << sdf.maxCoeff() << std::endl;
            sdf = (sdf.array() - sdf.minCoeff()) / (sdf.maxCoeff() - sdf.minCoeff()) * 255.0;
            Eigen::MatrixX8U sdf_uint8 = sdf.cast<uint8_t>();
            cv::Mat img_sdf, img_sdf_sign;
            cv::eigen2cv(sdf_uint8, img_sdf);
            cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);
            cv::eigen2cv(sdf_sign, img_sdf_sign);

            voxel_grid_sdf->voxels_.clear();
            for (long j = 0; j < positions.cols(); ++j) {
                for (long i = 0; i < positions.rows(); ++i) {
                    const Eigen::Vector3d &position_test = positions_test.col(i + j * positions.rows());
                    const cv::Vec3b &color = img_sdf.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
                    voxel_grid_sdf->AddVoxel({voxel_grid_sdf->GetVoxel(position_test), Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)});
                }
            }

            cv::resize(img_sdf, img_sdf, cv::Size(), 10, 10);
            cv::resize(img_sdf_sign, img_sdf_sign, cv::Size(), 10, 10);
            cv::imshow("sdf", img_sdf);
            cv::imshow("sdf_sign", img_sdf_sign);
            cv::waitKey(1);
        }

        // update visualization
        mesh_lidar->Transform(delta_pose);
        mesh_lidar_xyz->Transform(delta_pose);
        pcd_lidar->points_.clear();
        pcd_lidar->colors_.clear();
        pcd_lidar->points_.reserve(ranges.size());
        for (long i = 0; i < ranges.size(); ++i) {
            const double &range = ranges.data()[i];
            const Eigen::Vector3d &dir = ray_dirs_frame.data()[i];
            pcd_lidar->points_.emplace_back(rotation * (range * dir) + translation);
        }
        pcd_lidar->PaintUniformColor({0.0, 1.0, 0.0});

        pcd_surf_points->points_.clear();
        line_set_surf_normals->points_.clear();
        line_set_surf_normals->lines_.clear();
        const auto octree = gp.GetSurfaceMapping()->GetOctree();
        for (auto it = octree->BeginLeaf(), end = octree->EndLeaf(); it != end; ++it) {
            const auto surface_data = it->GetSurfaceData();
            if (surface_data == nullptr) { continue; }
            const Eigen::Vector3d &position = surface_data->position;
            const Eigen::Vector3d &normal = surface_data->normal;
            ERL_ASSERTM(std::abs(normal.norm() - 1.0) < 1.e-6, "normal.norm() = {:.6f}", normal.norm());
            pcd_surf_points->points_.emplace_back(position);
            line_set_surf_normals->points_.emplace_back(position);
            line_set_surf_normals->points_.emplace_back(position + 0.1 * normal);
            line_set_surf_normals->lines_.emplace_back(line_set_surf_normals->points_.size() - 2, line_set_surf_normals->points_.size() - 1);
        }
        // pcd_surf_points->PaintUniformColor({1.0, 0.0, 0.0});
        line_set_surf_normals->PaintUniformColor({1.0, 0.0, 0.0});

        const auto t_end = std::chrono::high_resolution_clock::now();
        const auto duration_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        ERL_INFO("duration_total: {:.3f} ms", duration_total);

        return animation_cnt++ % ANIMATION_INTERVAL == 0;
    };

    visualizer.SetAnimationCallback(callback);
    visualizer.Show();

    long max_gp_size = 0;
    long mean_gp_size = 0;
    long gp_cnt = 0;
    for (const auto &[key, local_gp]: gp.GetGpMap()) {
        max_gp_size = std::max(local_gp->num_train_samples, max_gp_size);
        mean_gp_size += local_gp->num_train_samples;
        ++gp_cnt;
    }
    mean_gp_size /= gp_cnt;
    ERL_INFO("max_gp_size: {}, mean_gp_size: {}", max_gp_size, mean_gp_size);
}
