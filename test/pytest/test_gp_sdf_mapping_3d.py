from erl_sdf_mapping import GpSdfMapping3D
import argparse
import numpy as np
import open3d as o3d
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-mapping-config-file", type=str, required=True)
    parser.add_argument("--sdf-mapping-bin-file", type=str, required=True)
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("-z", type=float, default=0.0)
    args = parser.parse_args()

    sdf_mapping_config_file = args.sdf_mapping_config_file
    sdf_mapping_bin_file = args.sdf_mapping_bin_file

    sdf_mapping_setting = GpSdfMapping3D.Setting()
    sdf_mapping_setting.from_yaml_file(sdf_mapping_config_file)
    sdf_mapping_setting.log_timing = False  # Disable timing log

    sdf_mapping = GpSdfMapping3D(sdf_mapping_setting)
    sdf_mapping.read(sdf_mapping_bin_file)  # setting will be overwritten by the file
    sdf_mapping_setting.log_timing = False  # Disable timing log again

    octree = sdf_mapping.surface_mapping.octree
    map_min, map_max = octree.metric_min_max
    xs, ys = np.meshgrid(
        np.arange(map_min[0], map_max[0], args.resolution),
        np.arange(map_min[1], map_max[1], args.resolution),
    )
    nx = xs.shape[0]
    ny = xs.shape[1]
    xs = xs.flatten()
    ys = ys.flatten()
    zs = np.full_like(xs, args.z)
    positions = np.array([xs, ys, zs])  # 3 x N
    sdf = sdf_mapping.test(positions)[0].reshape(nx, ny)
    sdf_min = sdf.min()
    sdf_max = sdf.max()
    sdf_uint8 = ((sdf - sdf_min) / (sdf_max - sdf_min) * 255).astype(np.uint8)
    sdf_jet = cv2.applyColorMap(sdf_uint8, cv2.COLORMAP_JET)
    sdf_jet = cv2.cvtColor(sdf_jet, cv2.COLOR_BGR2RGB)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions.T)
    pcd.colors = o3d.utility.Vector3dVector(sdf_jet.reshape(-1, 3).astype(np.float64) / 255)
    o3d.visualization.draw_geometries([mesh, pcd])


if __name__ == "__main__":
    main()
