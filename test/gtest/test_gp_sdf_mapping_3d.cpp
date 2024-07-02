#include "erl_common/test_helper.hpp"
#include "erl_geometry/lidar_3d.hpp"
#include "erl_geometry/open3d_visualizer_wrapper.hpp"
#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

#include <erl_sdf_mapping/gp_occ_surface_mapping_3d.hpp>
#include <open3d/geometry/LineSet.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

TEST(GpSdfMapping3D, LiDAR) {
    GTEST_PREPARE_OUTPUT_DIR();

    const auto gp_surf_setting = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping3D::Setting>();
    gp_surf_setting->FromYamlFile(gtest_src_dir / "../../config/surface_mapping_3d_lidar.yaml");
    const auto gp_surf = std::make_shared<erl::sdf_mapping::GpOccSurfaceMapping3D>(gp_surf_setting);
    const auto gp_sdf_setting = std::make_shared<erl::sdf_mapping::GpSdfMapping3D::Setting>();
    gp_sdf_setting->FromYamlFile(gtest_src_dir / "../../config/sdf_mapping_3d_lidar.yaml");
    erl::sdf_mapping::GpSdfMapping3D gp(gp_surf, gp_sdf_setting);
}
