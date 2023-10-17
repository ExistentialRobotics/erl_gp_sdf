import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.backend_bases import MouseEvent
from tqdm import tqdm

from erl_common.storage import GridMapInfo2D
from erl_geometry import Space2D
from erl_geometry.gazebo.sequence import GazeboSequence
from erl_geometry.house_expo.list_data import get_map_and_traj_files
from erl_geometry.house_expo.sequence import HouseExpoSequence
from erl_sdf_mapping import GpOccSurfaceMapping2D
from erl_sdf_mapping import GpSdfMapping2D
from erl_sdf_mapping.gpis import GpisMap2D
from erl_sdf_mapping.gpis import LogGpisMap2D


# import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--house-expo-index", type=int)
    parser.add_argument("--use-log", action="store_true")
    parser.add_argument("--use-log-v2", action="store_true")
    parser.add_argument("--skip", default=50, type=int)
    parser.add_argument("--add-offset-points", action="store_true")
    parser.add_argument("--map-resolution", default=0.05, type=float)
    parser.add_argument("--padding", default=5, type=int)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--draw-all", action="store_true")
    parser.add_argument("--draw-boundary", action="store_true")
    parser.add_argument("--draw-normal", action="store_true")
    parser.add_argument("--normal-scale", default=1.0, type=float)
    parser.add_argument("--draw-var-mask", action="store_true")
    parser.add_argument("--draw-robot", action="store_true")
    parser.add_argument("--draw-rays", action="store_true")
    parser.add_argument("--draw-traj", action="store_true")
    parser.add_argument("--draw-quadtree", action="store_true")
    parser.add_argument("--draw-abs-err", action="store_true")
    parser.add_argument("--draw-sddf-v2", action="store_true")
    parser.add_argument("--draw-sdf-variance", action="store_true")
    parser.add_argument("--draw-gradx-variance", action="store_true")
    parser.add_argument("--draw-grady-variance", action="store_true")
    parser.add_argument("--dump-quadtree-structure", action="store_true")
    args = parser.parse_args()

    if args.draw_all:
        args.draw_boundary = True
        args.draw_normal = True
        args.draw_normal = True
        args.draw_var_mark = True
        args.draw_robot = True
        args.draw_rays = True
        args.draw_traj = True
        args.draw_quadtree = True
        args.draw_abs_err = True
        args.draw_sddf_v2 = True

    tqdm.write(f"Current directory: {os.path.realpath(os.curdir)}")
    tqdm.write(f"Script path: {__file__}")

    if args.use_log:
        setting = LogGpisMap2D.Setting()
    elif args.use_log_v2:
        setting = GpSdfMapping2D.Setting()
    else:
        setting = GpisMap2D.Setting()
    print(setting.as_yaml_string())

    if args.house_expo_index is None:
        init_frame = 100  # first 100 frames are almost static
        sequence = GazeboSequence()
    else:
        if args.use_log_v2:
            pass
        else:
            # the wall thickness in HouseExpoMap is typically 10cm
            setting.quadtree.min_half_area_size = 0.1
            setting.node_container.min_squared_distance = 0.01
            setting.update_gp_sdf.search_area_scale = 4  # smaller -> faster, lower accuracy
            setting.gp_theta.train_buffer.valid_angle_min = -np.pi
            setting.gp_theta.train_buffer.valid_angle_max = np.pi
            setting.gp_theta.train_buffer.valid_range_min = 0.0
            setting.gp_theta.train_buffer.valid_range_max = np.inf
            setting.test_query.search_area_half_size = 4.8  # smaller -> faster

        if args.use_log:
            setting.update_gp_sdf.add_offset_points = False
            if args.add_offset_points:
                tqdm.write("--add-offset-points is ignored when --use-log is used.")
            setting.update_gp_sdf.offset_distance = 0.0
            setting.gp_sdf.log_lambda = 40.0
        elif args.use_log_v2:
            setting.offset_distance = 0.0
            setting.gp_sdf.log_lambda = 40.0
            setting.test_query.softmax_temperature = 1.0
        else:
            setting.update_gp_sdf.add_offset_points = args.add_offset_points

        json_file, path_file = get_map_and_traj_files()[args.house_expo_index]
        init_frame = 0
        sequence = HouseExpoSequence(json_file, path_file, lidar_mode="kDdf")

    if args.use_log:
        gpis_map = LogGpisMap2D(setting)
    elif args.use_log_v2:
        surface_mapping = GpOccSurfaceMapping2D()
        gpis_map = GpSdfMapping2D(surface_mapping, setting)
    else:
        gpis_map = GpisMap2D(setting)

    sdf_gt = None
    if args.house_expo_index is None:  # Gazebo
        xmin = -5
        xmax = 20
        ymin = -15
        ymax = 5
    else:
        vertices = sequence.map.meter_space.surface.vertices
        xmin = np.min(vertices[0])
        xmax = np.max(vertices[0])
        ymin = np.min(vertices[1])
        ymax = np.max(vertices[1])
    grid_map_info = GridMapInfo2D(
        min=np.array([xmin, ymin]),
        max=np.array([xmax, ymax]),
        resolution=np.array([args.map_resolution, args.map_resolution]),
        padding=np.array([args.padding, args.padding]),
    )
    extent = (
        grid_map_info.min_at(0),
        grid_map_info.max_at(0),
        grid_map_info.min_at(1),
        grid_map_info.max_at(1),
    )
    xy_coords = grid_map_info.generate_meter_coordinates(c_stride=True)
    # x down, y right -> x right, y up
    xg = xy_coords[0].reshape(grid_map_info.shape).T[::-1]
    yg = xy_coords[1].reshape(grid_map_info.shape).T[::-1]
    height = grid_map_info.height
    width = grid_map_info.width
    if args.house_expo_index is not None:
        sdf_gt = sequence.map.meter_space.compute_sdf_image(grid_map_info, sign_method=Space2D.SignMethod.kPolygon)

    ############
    # TRAINING #
    ############
    cnt = 0
    path = []
    fig = None
    if args.save_video:
        fig = plt.figure(figsize=(15, 10))
        os.makedirs("images", exist_ok=True)
    for frame in tqdm(
        sequence[init_frame :: args.skip],
        ncols=80,
        desc=os.path.splitext(os.path.basename(__file__))[0],
    ):
        t0 = time.time()
        gpis_map.update(frame.angles_in_frame, frame.ranges, frame.pose_matrix[:2])
        t_update = time.time()
        tqdm.write(f"[{cnt}] time(sec): {t_update - t0:.6e}")
        cnt += 1

        if args.save_video:
            distances, gradients, distance_variances, gradient_variances = gpis_map.test(xy_coords)
            distance_image = distances.reshape([height, width], order="F")[::-1]
            quadtree_image = gpis_map.quadtree.plot(
                grid_map_info, node_types=[0], node_type_colors={0: np.array([0, 255, 0])}, node_type_radius={0: 3}
            )
            plt.clf()
            path.append(frame.translation_vector)
            plt.imshow(quadtree_image, extent=extent)
            plt.pcolormesh(xg, yg, distance_image, edgecolors="none", alpha=0.9)
            plt.colorbar()
            plt.clim(np.min(sdf_gt), np.max(sdf_gt))
            path_np = np.array(path).T
            plt.plot(path_np[0], path_np[1], "r-")
            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            plt.imsave(f"images/frame_{cnt:04d}.png", image)
            plt.pause(0.001)

    if args.save_video:
        import ffmpeg

        (
            ffmpeg.input("images/*.png", pattern_type="glob", framerate=15)
            .output("log_gpis_map.mp4" if args.use_log else "gpis_map.mp4")
            .run()
        )

    if args.dump_quadtree_structure:
        if args.use_log:
            filename = "QuadTree-GPIS-LOG.txt"
        else:
            filename = "QuadTree-GPIS.txt"
        with open(filename, "w") as file:
            file.write(gpis_map.dump_quadtree_structure())
        tqdm.write(f"QuadTree structure is saved at {filename}.")

    ########
    # TEST #
    ########
    t0 = time.time()
    if args.use_log_v2:
        pass
        distances, gradients, variances, covariances = gpis_map.test(xy_coords)
        distance_variances = variances[0]
        gradient_variances = variances[1:]
    else:
        distances, gradients, distance_variances, gradient_variances = gpis_map.test(xy_coords)
    assert distances is not None, "gpis_map.test failed."

    t_test = time.time()
    tqdm.write(f"time(sec): test = {t_test - t0}")

    if args.no_render:
        return

    #############
    # visualize #
    #############
    if sdf_gt is not None:
        plt.figure()
        plt.imshow(
            sequence.map.meter_space.generate_map_image(grid_map_info),
            cmap=cm.binary_r,
            extent=extent,
        )
        plt.pcolormesh(
            xg,
            yg,
            sdf_gt,
            alpha=0.8,
            edgecolors="none",
        )
        plt.colorbar()
        plt.clim(np.min(sdf_gt), np.max(sdf_gt))
        plt.title("HouseExpo Ground Truth Map")
        plt.tight_layout()

    if args.draw_quadtree:
        plt.figure()
        plt.imshow(
            gpis_map.quadtree.plot(
                grid_map_info=grid_map_info,
                node_types=[0],
                node_type_colors={0: np.array([0, 0, 255])},
                node_type_radius={0: 1},
            ),
            extent=extent,
        )
        plt.title("QuadTree")
        plt.tight_layout()
        plt.show()

    head = 0.5 * np.array([[0, 1, 0, 0], [0.25, 0, -0.25, 0.25]])
    last_frame = sequence[-1]
    sensor_pose = last_frame.pose_matrix
    head = sensor_pose[:2, :2] @ head + sensor_pose[:2, [2]]
    # x down, y right, along y first (row major) <--> y down, x right, along y first (column major)
    distances = distances.reshape([height, width], order="F")[::-1]  # x down, y right -> x right, y up
    # (2, height x width) -> (2, height, width)
    # reshape will flatten the array by the specified order, then reshape it by the same order
    gradients = gradients.reshape([2, height, width], order="F")[:, ::-1]  # x down, y right -> x right, y up
    distance_variances = distance_variances.reshape([height, width], order="F")[::-1]
    gradient_variances = gradient_variances.reshape([2, height, width], order="F")[:, ::-1]

    plt.figure(figsize=(11, 8))

    # draw the distance field
    mask: npt.NDArray[np.float64] = distance_variances
    var_min = np.min(mask)
    var_max = np.max(mask)
    mask: float = 1 - (mask - var_min) / (var_max - var_min)
    # distance variance as the alpha channel
    # alpha: higher value -> higher transparency
    if args.draw_var_mask:
        plt.pcolormesh(
            xg,
            yg,
            distances,
            alpha=mask,
            edgecolors="none",
        )
    else:
        plt.pcolormesh(xg, yg, distances, edgecolors="none")

    plt.colorbar()
    plt.clim(np.min(distances), np.max(distances))

    # extract the zero-set, i.e. surface
    space2d_meters = Space2D(
        map_image=distances,
        grid_map_info=grid_map_info,
        free_threshold=0,
        delta=0.01,
        parallel=True,
    )
    surf_pts = []
    surf_normals = []
    pts_var = []
    normals_var = []
    for obj_idx in range(space2d_meters.surface.num_objects):
        obj_metric_vertices = space2d_meters.surface.get_object_vertices(obj_idx)
        xs = obj_metric_vertices[0].copy()
        ys = obj_metric_vertices[1].copy()
        us = grid_map_info.meter_to_grid_for_values(xs, 0)
        vs = height - grid_map_info.meter_to_grid_for_values(ys, 1)

        if args.draw_boundary:
            plt.plot(xs, ys, "r-")
            plt.scatter(xs, ys, c="k", s=2)

        quiver_data = None
        if args.draw_normal:
            s = 2
            normals = gradients[:, vs[::s], us[::s]]
            nx = normals[0]
            ny = normals[1]
            plt.quiver(
                xs[::s],
                ys[::s],
                nx,
                ny,
                scale=args.normal_scale,
                scale_units="xy",
                color="magenta",
            )
            quiver_data = plt.quiver(
                xs[0], ys[0], nx[0], ny[0], scale=args.normal_scale, scale_units="xy", color="magenta"
            )

        # save data to list
        surf_pts.append(obj_metric_vertices)
        surf_normals.append(gradients[:, vs, us])  # type: ignore
        pts_var.append(distance_variances[vs, us])  # type: ignore
        normals_var.append(gradient_variances[:, vs, us])  # type: ignore

    if args.use_log:
        filename = "room-LogGPIS.pkl"
    else:
        filename = "room-GPIS.pkl"
    if args.save_results:
        with open(filename, "wb") as file:  # type: ignore
            pickle.dump(
                dict(
                    surf_pts=surf_pts,
                    surf_normals=surf_normals,
                    pts_var=pts_var,
                    normals_var=normals_var,
                ),
                file,  # type: ignore
            )

    if args.draw_rays:
        # draw the rays at the last frame
        mask = (last_frame.ranges < 3e1) & (last_frame.ranges > 0.2)  # type: ignore
        mask_sum = np.sum(mask)
        if mask_sum > 0:
            rays = np.empty((mask_sum * 2, 2))
            rays[1::2, :] = last_frame.end_points_in_world[:, mask]
            rays[::2, :] = sensor_pose[:2, 2]
            plt.plot(rays[:, 0], rays[:, 1], color=[0.0, 1.0, 0.0], linewidth=0.5)

    if args.draw_traj:
        # draw the robot's trajectory
        plt.plot(sequence.path[:, 0], sequence.path[:, 1], "k-")

    if args.draw_robot:
        # draw the robot
        plt.plot(head[0], head[1], "r-")

    if args.use_log:
        plt.title("LogGPIS Demo: Room")
    else:
        plt.title("GPIS Demo: Room")
    plt.tight_layout()

    line_data = plt.plot(sequence.path[-1, 0], sequence.path[-1, 1], "r-")[0]
    angles = np.linspace(-np.pi, np.pi, 180)

    def on_move(event: MouseEvent) -> None:
        # get the x and y pixel coords
        # x, y = event.x, event.y
        if event.inaxes:
            # ax = event.inaxes  # the axes instance
            xdata = event.xdata
            ydata = event.ydata
            v = distances.shape[0] - int(round((ydata - ymin) / args.map_resolution))
            u = int(round((xdata - xmin) / args.map_resolution))

            v = min(max(v, 0), distances.shape[0] - 1)  # type: ignore
            u = min(max(u, 0), distances.shape[1] - 1)  # type: ignore

            t = 0
            if args.draw_normal:
                quiver_data.set_offsets([xdata, ydata])
                quiver_data.set_UVC(gradients[0, v, u], gradients[1, v, u])

            if args.draw_sddf_v2:
                positions = np.array([xdata, ydata]).reshape(2, 1)
                positions = np.tile(positions, (1, angles.shape[0]))
                t_start = time.time()
                sddf = gpis_map.compute_sddf_v2(
                    positions, angles, threshold=0.01, max_distance=-1, max_marching_steps=-1
                )
                t = time.time() - t_start
                pts = np.empty((2, sddf.shape[0] * 2))
                pts[0, ::2] = xdata
                pts[1, ::2] = ydata
                pts[0, 1::2] = sddf * np.cos(angles) + xdata
                pts[1, 1::2] = sddf * np.sin(angles) + ydata
                line_data.set_data(pts)
            else:
                line_data.set_data(xdata, ydata)
            plt.gcf().canvas.draw_idle()

            tqdm.write(
                f"[{xdata:.6f}, {ydata:.6f}]:\tdist = {distances[v, u]:.6f},\t"  # type: ignore
                f"dist_var = {distance_variances[v, u]:.6f}\t"
                f"grad = {gradients[:, v, u]}\t"
                f"grad_norm = {np.linalg.norm(gradients[:, v, u]):.6f}\t"
                f"grad_var = {gradient_variances[:, v, u]}\t"
                f"sddf_t = {t:.6f} sec"
            )

    plt.connect("motion_notify_event", on_move)

    if args.draw_abs_err and sdf_gt is not None:
        fig = plt.figure(figsize=(11, 8))
        ax = plt.axes(projection="3d")
        abs_err = np.abs(sdf_gt - distances)
        surf = ax.plot_surface(
            xg,
            yg,
            abs_err,
            linewidth=0,
            antialiased=True,
            cmap=cm.viridis,
        )
        mae = np.mean(abs_err)
        mae_std = np.std(abs_err)
        ax.set_zlim(np.min(abs_err), mae * 2)
        fig.colorbar(surf)
        surf.set_clim([np.min(abs_err), mae * 2])

        trust = mae <= 3 * distance_variances  # type: ignore

        plt.title(f"MAE = {mae:.3e} +- {mae_std:.3e}, trust: {np.sum(trust) / trust.size * 100.:.2f} %")
        plt.tight_layout()

    if args.draw_sdf_variance:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots(1, 1)
        surf = ax.pcolormesh(
            xg,
            yg,
            distance_variances,
            edgecolors="none",
        )
        fig.colorbar(surf)
        plt.title("SDF Variance")
        plt.tight_layout()

    if args.draw_gradx_variance:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots(1, 1)
        surf = ax.pcolormesh(
            xg,
            yg,
            gradient_variances[0],
            edgecolors="none",
        )
        fig.colorbar(surf)
        plt.title("Gradient X Variance")
        plt.tight_layout()

    if args.draw_grady_variance:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.subplots(1, 1)
        surf = ax.pcolormesh(
            xg,
            yg,
            gradient_variances[1],
            edgecolors="none",
        )
        fig.colorbar(surf)
        plt.title("Gradient Y Variance")
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
