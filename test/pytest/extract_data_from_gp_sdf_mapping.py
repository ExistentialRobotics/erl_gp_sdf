import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches
import yaml

from erl_covariance import Covariance

from erl_sdf_mapping import GpSdfMapping2D
from erl_sdf_mapping import GpSdfMapping3D

from tqdm import tqdm
from erl_sdf_mapping import LogSdfGaussianProcess

# from erl_sdf_mapping import GpOccSurfaceMapping2D
# from erl_sdf_mapping import GpOccSurfaceMapping3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", required=True, type=int, choices=[2, 3])
    parser.add_argument("--sdf-mapping-bin-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    dim = args.dim
    sdf_mapping_bin_file = args.sdf_mapping_bin_file
    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if dim == 2:
        GpSdfMapping = GpSdfMapping2D
    elif dim == 3:
        GpSdfMapping = GpSdfMapping3D
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    tree_type = "quadtree" if dim == 2 else "octree"

    sdf_mapping_setting = GpSdfMapping.Setting()
    sdf_mapping_setting.log_timing = False  # Disable timing log
    sdf_mapping = GpSdfMapping(sdf_mapping_setting)
    sdf_mapping.read(sdf_mapping_bin_file)  # setting will be overwritten by the file
    sdf_mapping_setting.log_timing = False  # Disable timing log again

    # dump the setting to a YAML file
    sdf_mapping_setting.as_yaml_file(os.path.join(output_dir, "sdf_mapping_setting.yaml"))

    # surface_mapping_setting = sdf_mapping_setting.surface_mapping
    # surface_mapping_setting: Union[GpOccSurfaceMapping2D.Setting, GpOccSurfaceMapping3D.Setting]
    # cluster_depth = space_tree.tree_depth - surface_mapping_setting.cluster_level

    space_tree = getattr(sdf_mapping.surface_mapping, tree_type)
    map_min, map_max = space_tree.metric_min_max

    fig, ax = plt.subplots(dpi=300)

    gps_info = list()
    for tree_key, gp in tqdm(sdf_mapping.gps.items(), desc="Extracting GP data"):
        sdf_gp: LogSdfGaussianProcess = gp.gp
        sdf_gp_setting: LogSdfGaussianProcess.Setting = sdf_gp.setting
        kernel = sdf_gp.kernel
        log_kernel: Covariance = sdf_gp.log_kernel

        # some important information about the SDF GP, most fields should be the same across all GPs
        tree_key_tuple = tuple(tree_key[i] for i in range(dim))
        tree_key_str = "_".join(str(x) for x in tree_key_tuple)
        gp_info = dict(
            tree_key=list(tree_key_tuple),
            active=gp.active,
            locked_for_test=gp.locked_for_test,  # only useful when training and testing happens in parallel in C++
            num_train_samples=gp.num_train_samples,
            position=list(gp.position.tolist()),  # =list(space_tree.key_to_coord(tree_key, cluster_depth))
            half_size=gp.half_size,  # half size of the area used to collect training samples
            gp=dict(
                num_train_samples=sdf_gp.num_train_samples,
                num_train_samples_with_grad=sdf_gp.num_train_samples_with_grad,
                max_num_samples=sdf_gp_setting.max_num_samples,  # maximum number of training samples
                log_lambda=sdf_gp_setting.log_lambda,
                edf_threshold=sdf_gp_setting.edf_threshold,  # threshold of being near the surface
                unify_scale=sdf_gp_setting.unify_scale,  # kernel and log_kernel share the same scale if True
                kernel=dict(  # kernel used to compute SDF near the surface
                    type=kernel.type,
                    scale=kernel.setting.scale,
                    alpha=kernel.setting.alpha,
                    x_dim=kernel.setting.x_dim,  # may not match `dim`
                ),
                log_kernel=dict(  # kernel used to compute SDF far away from the surface
                    type=log_kernel.type,
                    scale=log_kernel.setting.scale,
                    alpha=log_kernel.setting.alpha,
                    x_dim=log_kernel.setting.x_dim,  # may not match `dim`
                ),
            ),
            datafile=os.path.join(output_dir, f"{tree_key_str}_gp_data.npz"),
        )
        gps_info.append(gp_info)

        # draw the area used to collect training samples
        if dim == 2:
            rect = patches.Rectangle(
                (gp.position[0] - gp.half_size, gp.position[1] - gp.half_size),
                2 * gp.half_size,
                2 * gp.half_size,
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(rect)
            circ = patches.Circle((gp.position[0], gp.position[1]), radius=0.1, edgecolor="r", facecolor="r")
            ax.add_patch(circ)

        # yaml_str = yaml.dump(gp_info)
        # print(yaml_str)

        k_rows, k_cols = kernel.get_minimum_ktrain_size(
            sdf_gp.num_train_samples,
            sdf_gp.num_train_samples_with_grad,
            dim,
        )
        log_k_rows, log_k_cols = log_kernel.get_minimum_ktrain_size(
            sdf_gp.num_train_samples,
            0,  # we don't use any gradient for the log kernel
            dim,
        )
        # if unify_scale is True, log_k_train should be the same as the top-left block of k_train

        gp_data = dict(
            x_train=sdf_gp.x_train[:dim, : sdf_gp.num_train_samples],  # (dim, num_train_samples)
            y_train=sdf_gp.y_train[: sdf_gp.num_train_samples],
            grad_train=sdf_gp.grad_train[:dim, : sdf_gp.num_train_samples_with_grad],
            # flags indicate if gradient is available for each training sample
            grad_flag=sdf_gp.grad_flag[: sdf_gp.num_train_samples],
            var_x_train=sdf_gp.var_x_train[: sdf_gp.num_train_samples],
            var_y_train=sdf_gp.var_y_train[: sdf_gp.num_train_samples],
            var_grad_train=sdf_gp.var_grad_train[: sdf_gp.num_train_samples],
            k_train=sdf_gp.k_train[:k_rows, :k_cols],  # K: covariance matrix of training samples by `kernel`
            alpha=sdf_gp.alpha[:k_cols],  # (K + sigma^2 I)^{-1} y
            cholesky_k_train=sdf_gp.cholesky_k_train[:k_rows, :k_cols],  # Cholesky decomposition of K
            # K_log: covariance matrix of training samples by `log_kernel`
            log_k_train=sdf_gp.log_k_train[:log_k_rows, :log_k_cols],
            log_alpha=sdf_gp.log_alpha[:log_k_cols],  # (K_log + sigma^2 I)^{-1} exp(-log_lambda * y)
            # Cholesky decomposition of K_log
            log_cholesky_k_train=sdf_gp.log_cholesky_k_train[:log_k_rows, :log_k_cols],
        )

        np.savez(gp_info["datafile"], **gp_data)

    with open(os.path.join(output_dir, "gps_info.yaml"), "w") as f:
        yaml.dump(gps_info, f)

    print(len(gps_info), "GP data files saved in", output_dir)

    ax.add_patch(
        patches.Rectangle(
            map_min,
            map_max[0] - map_min[0],
            map_max[1] - map_min[1],
            linewidth=1,
            linestyle="--",
            edgecolor="b",
            facecolor="none",
            label="tree boundary",
        ),
    )
    ax.set_xlim(map_min[0] - gps_info[0]["half_size"], map_max[0] + gps_info[0]["half_size"])
    ax.set_ylim(map_min[1] - gps_info[0]["half_size"], map_max[1] + gps_info[0]["half_size"])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
