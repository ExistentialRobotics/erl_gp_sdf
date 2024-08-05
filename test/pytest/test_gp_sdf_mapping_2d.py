import argparse

import matplotlib.pyplot as plt
import numpy as np

from erl_sdf_mapping import GpSdfMapping2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-mapping-bin-file", type=str, required=True)
    parser.add_argument("--resolution", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="sdf.png")
    args = parser.parse_args()

    sdf_mapping_bin_file = args.sdf_mapping_bin_file
    resolution = args.resolution
    output = args.output

    sdf_mapping_setting = GpSdfMapping2D.Setting()
    sdf_mapping_setting.log_timing = False  # Disable timing log
    sdf_mapping = GpSdfMapping2D(sdf_mapping_setting)
    sdf_mapping.read(sdf_mapping_bin_file)  # setting will be overwritten by the file
    sdf_mapping_setting.log_timing = False  # Disable timing log again

    quadtree = sdf_mapping.surface_mapping.quadtree
    map_min, map_max = quadtree.metric_min_max
    xs, ys = np.meshgrid(
        np.arange(map_min[0], map_max[0], resolution),
        np.arange(map_min[1], map_max[1], resolution),
    )
    nx = xs.shape[0]
    ny = xs.shape[1]
    positions = np.array([xs.flatten(), ys.flatten()])  # 2 x N
    sdf = sdf_mapping.test(positions)[0].reshape(nx, ny)

    drawer_setting = quadtree.Drawer.Setting()
    drawer_setting.resolution = resolution
    drawer_setting.area_min = map_min
    drawer_setting.area_max = map_max
    drawer_setting.border_color = drawer_setting.occupied_color
    quadtree_drawer = quadtree.Drawer(drawer_setting, quadtree)
    quadtree_img = quadtree_drawer.draw_leaves()

    fig, ax = plt.subplots(dpi=300)
    ax.imshow(quadtree_img, extent=[map_min[0], map_max[0], map_min[1], map_max[1]], alpha=0.5)
    c = ax.pcolormesh(xs, ys, sdf, cmap="jet", alpha=0.5)
    fig.colorbar(c, ax=ax, shrink=0.9, pad=0.01)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output)
    plt.show()


if __name__ == "__main__":
    main()
