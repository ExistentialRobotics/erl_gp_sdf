import os.path

import matplotlib.pyplot as plt
import numpy as np
from erl_sdf_mapping import GpOccSurfaceMapping2D
from erl_sdf_mapping import GpSdfMapping2D
from tqdm import tqdm
import time
import cv2


def draw_lidar_scans(filename: str):
    data = np.load(filename)
    surf_pts = []
    path = []
    n = data["time_stamps"].shape[0]
    lidar_angles = data["lidar_angles"]
    lidar_ranges = data["lidar_ranges"]
    lidar_poses = data["lidar_poses"]
    for i in range(n):
        angles = lidar_angles[i]
        ranges = lidar_ranges[i]
        pose = lidar_poses[i]

        mask = ~(np.isinf(ranges) | np.isnan(ranges))
        angles = angles[mask]
        ranges = ranges[mask]

        sin = np.sin(angles)
        cos = np.cos(angles)
        pts = np.array([ranges * cos, ranges * sin])
        pts = pose[:2, :2] @ pts + pose[:2, [2]]
        surf_pts.append(pts)
        path.append(pose[:2, [2]])
    surf_pts = np.concatenate(surf_pts, axis=1)
    path = np.concatenate(path, axis=1)

    plt.figure(figsize=(10, 10))
    plt.scatter(surf_pts[0], surf_pts[1], s=0.1)
    plt.plot(path[0], path[1], "r-")
    plt.axis("equal")
    plt.xlabel("x (meter)")
    plt.ylabel("y (meter)")
    plt.xlim([surf_pts[0].min(), surf_pts[0].max()])
    plt.ylim([surf_pts[1].min(), surf_pts[1].max()])
    plt.tight_layout()
    plt.show()


def run_sdf_mapping(filename):
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")

    surf_mapping_setting = GpOccSurfaceMapping2D.Setting()
    surf_mapping_setting.from_yaml_file(os.path.join(config_dir, "surface_mapping_2d.yaml"))
    surface_mapping = GpOccSurfaceMapping2D(surf_mapping_setting)

    sdf_mapping_setting = GpSdfMapping2D.Setting()
    sdf_mapping_setting.from_yaml_file(os.path.join(config_dir, "sdf_mapping.yaml"))
    sdf_mapping = GpSdfMapping2D(surface_mapping, sdf_mapping_setting)

    data = np.load(filename)
    n = data["time_stamps"].shape[0]
    lidar_angles = data["lidar_angles"]
    lidar_ranges = data["lidar_ranges"]
    lidar_poses = data["lidar_poses"]
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    for i in tqdm(range(n), ncols=80, desc="Mapping"):
        angles = lidar_angles[i]
        ranges = lidar_ranges[i]
        pose = lidar_poses[i]
        t0 = time.time()
        sdf_mapping.update(angles, ranges, pose)
        t_update = time.time()
        tqdm.write(f"[{i:06d}] time(sec): {t_update - t0:.6e}")

        # visualize
        sdf_mapping.surface_mapping


if __name__ == "__main__":
    draw_lidar_scans("long_succ4_till_no_path.npz")
