#include "erl_common/eigen.hpp"
#include "erl_common/test_helper.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp"

#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

#define STRIDE             1
#define ANIMATION_INTERVAL 2

TEST(GpOccSurfaceMapping3D, Build) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_setting = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping3D::Setting>();
    const auto lidar_frame_setting = std::make_shared<erl::geometry::LidarFrame3D::Setting>();
    lidar_frame_setting->valid_range_min = 0.2;
    lidar_frame_setting->valid_range_max = 30.0;
    lidar_frame_setting->azimuth_min = -M_PI * 3 / 4;
    lidar_frame_setting->azimuth_max = M_PI * 3 / 4;
    lidar_frame_setting->num_azimuth_lines = 271;
    lidar_frame_setting->elevation_min = -M_PI / 2;
    lidar_frame_setting->elevation_max = M_PI / 2;
    lidar_frame_setting->num_elevation_lines = 91;
    gp_setting->sensor_gp->range_sensor_frame_type = "lidar";
    gp_setting->sensor_gp->range_sensor_frame = lidar_frame_setting;
    gp_setting->sensor_gp->row_group_size = 10;
    gp_setting->sensor_gp->row_overlap_size = 3;
    gp_setting->sensor_gp->row_margin = 0;
    gp_setting->sensor_gp->col_group_size = 10;
    gp_setting->sensor_gp->col_overlap_size = 3;
    gp_setting->sensor_gp->col_margin = 0;
    gp_setting->update_occupancy = false;
    erl::sdf_mapping::GpOccSurfaceMapping3D gp(gp_setting);

    const Eigen::MatrixXd traj_matrix = erl::common::LoadEigenMatrixFromTextFile<double>(gtest_src_dir / "replica-hotel-0-traj.txt").transpose();

    const auto mesh = open3d::io::CreateMeshFromFile(gtest_src_dir / "replica-hotel-0.ply");
    const auto lidar_setting = std::make_shared<erl::geometry::Lidar3D::Setting>();
    lidar_setting->azimuth_min = lidar_frame_setting->azimuth_min;
    lidar_setting->azimuth_max = lidar_frame_setting->azimuth_max;
    lidar_setting->num_azimuth_lines = lidar_frame_setting->num_azimuth_lines;
    lidar_setting->elevation_min = lidar_frame_setting->elevation_min;
    lidar_setting->elevation_max = lidar_frame_setting->elevation_max;
    lidar_setting->num_elevation_lines = lidar_frame_setting->num_elevation_lines;
    erl::geometry::Lidar3D lidar(lidar_setting, mesh->vertices_, mesh->triangles_);

    // the trajectory we use is for depth camera, so we need to convert it for lidar
    Eigen::Matrix4d trans_depth_to_lidar;
    // clang-format off
    trans_depth_to_lidar << 0,  0, 1, 0,
                           -1,  0, 0, 0,
                            0, -1, 0, 0,
                            0,  0, 0, 1;
    // clang-format on
    trans_depth_to_lidar = Eigen::Matrix4d(trans_depth_to_lidar).inverse();

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
    visualizer.AddGeometries({mesh, mesh_lidar, mesh_lidar_xyz, pcd_lidar, pcd_surf_points, line_set_surf_normals});

    long wp_idx = 0;
    long animation_cnt = 0;
    const Eigen::MatrixX<Eigen::Vector3d> &ray_dirs_frame = lidar.GetRayDirectionsInFrame();
    Eigen::Matrix3d last_rotation = Eigen::Matrix3d::Identity();
    auto callback = [&](erl::geometry::Open3dVisualizerWrapper *wrapper, open3d::visualization::Visualizer *) -> bool {
        if (wp_idx >= traj_matrix.cols()) {
            wrapper->SetAnimationCallback(nullptr);  // stop calling this callback
            return false;
        }

        const auto t_start = std::chrono::high_resolution_clock::now();
        const Eigen::Matrix4d pose = traj_matrix.col(wp_idx).reshaped(4, 4).transpose() * trans_depth_to_lidar;
        const Eigen::Matrix3d rotation = pose.topLeftCorner<3, 3>();
        const Eigen::Vector3d translation = pose.topRightCorner<3, 1>();
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

        // update visualization
        mesh_lidar->Translate(translation - mesh_lidar->GetCenter());
        mesh_lidar_xyz->Translate(-mesh_lidar_xyz->GetCenter());
        mesh_lidar_xyz->Rotate(rotation * last_rotation.transpose(), Eigen::Vector3d::Zero());
        mesh_lidar_xyz->Translate(mesh_lidar->GetCenter());
        last_rotation = rotation;
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
        const auto octree = gp.GetOctree();
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
}
