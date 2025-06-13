import os
import time

import matplotlib.pyplot as plt
import numpy as np
from erl_geometry import Lidar2D
from erl_geometry import Space2D
from tqdm import tqdm
import torch
import matplotlib.style as mplstyle
import matplotlib

matplotlib.use("TkAgg")
mplstyle.use("fast")


class CpuTimer:

    def __init__(self, message, repeats: int = 1, warmup: int = 0):
        self.message = message
        self.repeats = repeats
        self.warmup = warmup
        self.cnt = 0
        self.t = 0
        self.average_t = 0
        self._total_t = 0
        self.total_t = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        if self.cnt < self.warmup:
            self.cnt += 1
            return
        self.cnt += 1
        assert self.cnt <= self.repeats
        self.t = self.end - self.start
        self._total_t += self.t
        self.average_t = self._total_t / (self.cnt - self.warmup)
        self.total_t = self.average_t * self.cnt
        tqdm.write(f"{self.message}: {self.t:.6f}(cur)/{self.average_t:.6f}(avg)/{self.total_t:.6f}(total) seconds")


def rbf_kernel(x: torch.Tensor, grid: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Radial basis function kernel.
    :param x: (N, d) input points
    :param grid: (M, d) grid points
    :param gamma: float
    :return:
    """
    return torch.exp(-gamma * torch.square(x[:, None, :] - grid[None, :, :]).sum(dim=-1))  # (N, M)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid function.
    :param x: (N,) input points
    :return:
    """
    x = torch.clip(x, -100, 100)
    return 1.0 / (1.0 + torch.exp(-x))


def calculate_posterior(
    phi: torch.Tensor,
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    xi: torch.Tensor,
    sigma_scheme: str = "full",
):
    """
    Calculate the posterior.
    :param phi: (N, M) feature matrix
    :param y: (N,) target values
    :param mu: (M,) mean of the weights
    :param sigma: (M, M) covariance of the weights
    :param xi: (N, ) EM parameters
    :param sigma_scheme: str, "diagonal" or "full"
    :return:
    """
    lambda_xi = sigmoid(xi)
    lambda_xi = (lambda_xi - 0.5) / (2 * xi)  # (N,)
    lambda_xi[torch.abs(xi) < 1e-6] = 0.125
    assert torch.all(lambda_xi >= 0)

    if sigma_scheme == "diagonal":
        sig0 = torch.diag(sigma)
        sig = 1.0 / (1.0 / sig0 + 2 * ((phi**2).T * lambda_xi).sum(dim=1))
        new_sigma = torch.diag(sig)
        new_mu = sig * (mu / sig0 + phi.T @ (y - 0.5).ravel())
    else:
        sigma_inv = torch.linalg.inv(sigma)
        new_sigma_inv = sigma_inv + 2 * (phi.T * lambda_xi) @ phi
        new_sigma = torch.linalg.inv(new_sigma_inv)
        new_mu = new_sigma @ (sigma_inv @ mu + phi.T @ (y - 0.5))
    return new_mu, new_sigma


def calculate_xi(phi, mu, sigma, sigma_scheme: str = "full"):
    """
    Calculate xi.
    :param phi: (N, M) feature matrix
    :param mu: (M,) mean of the weights
    :param sigma: (M, M) covariance of the weights
    :param sigma_scheme: str, "diagonal" or "full"
    :return:
    """
    if sigma_scheme == "diagonal":
        xi_sq = (phi**2 * torch.diag(sigma)).sum(dim=1) + (phi @ mu) ** 2
    else:
        xi_sq = (phi * (phi @ sigma)).sum(axis=1) + (phi @ mu) ** 2
    return torch.sqrt(xi_sq)


def predict_fast(x, grid, mu, sigma, gamma, sigma_scheme: str = "full", compute_grad: bool = False):
    diff = x[:, None, :] - grid[None, :, :]  # (N, M, d)
    phi = torch.exp(-gamma * torch.square(diff).sum(dim=-1))  # (N, M)
    t1 = phi @ mu  # (N,)
    if sigma_scheme == "diagonal":
        t2 = 1 + (phi**2 * torch.diag(sigma)).sum(dim=1) * torch.pi / 8
    else:
        t2 = 1 + (phi * (phi @ sigma)).sum(dim=1) * torch.pi / 8
    t3 = torch.sqrt(t2)
    h = t1 / t3  # (N,)
    y = sigmoid(h)
    grad = None
    if compute_grad:
        t4 = mu.view(1, -1) / t3.view(-1, 1)  # (N, M)
        t5 = (torch.pi / 8) * (h / t2).view(-1, 1)  # (N, 1)
        if sigma_scheme == "diagonal":
            grad_phi_h = t4 - t5 * (phi * torch.diag(sigma))  # (N, M)
        else:
            grad_phi_h = t4 - t5 * (phi @ sigma)  # (N, M)
        grad_x_h = -2 * gamma * diff * (phi * grad_phi_h)[:, :, torch.newaxis]  # (N, M, d)
        grad_x_h = grad_x_h.sum(dim=1)  # (N, d)
        grad = (y * (1 - y)).view(-1, 1) * grad_x_h  # (N, d)
    return y, grad


def predict_faster(x, grid, mu, sigma, gamma, sigma_scheme: str = "full", compute_grad: bool = False):
    diff = x[:, None, :] - grid[None, :, :]
    phi = torch.exp(-gamma * torch.square(diff).sum(dim=-1))  # (N, M)
    t1 = phi @ mu  # (N,)
    h = t1
    y = sigmoid(h)
    grad = None
    if compute_grad:
        grad_x_phi = -2 * gamma * diff * phi[:, :, torch.newaxis]  # (N, M, d)
        grad = (grad_x_phi * mu.reshape(1, -1, 1)).sum(dim=1)  # (N, d)
    return y, grad


def predict_grad_fast(x, grid, mu, sigma, gamma, sigma_scheme: str = "full"):
    diff = x[:, None, :] - grid[None, :, :]  # (N, M, d)
    phi = torch.exp(-gamma * torch.square(diff).sum(dim=-1))  # (N, M)
    t1 = phi @ mu  # (N,)
    if sigma_scheme == "diagonal":
        t2 = 1 + (phi**2 * torch.diag(sigma)).sum(dim=1) * torch.pi / 8
    else:
        t2 = 1 + (phi * (phi @ sigma)).sum(dim=1) * torch.pi / 8
    t3 = torch.sqrt(t2)
    h = t1 / t3  # (N,)

    t4 = mu.view(1, -1) / t3.view(-1, 1)  # (N, M)
    t5 = (torch.pi / 8) * (h / t2).view(-1, 1)  # (N, 1)
    if sigma_scheme == "diagonal":
        grad_phi_h = t4 - t5 * (phi * torch.diag(sigma))  # (N, M)
    else:
        grad_phi_h = t4 - t5 * (phi @ sigma)  # (N, M)
    grad_x_h = -2 * gamma * diff * (phi * grad_phi_h)[:, :, torch.newaxis]  # (N, M, d)
    grad_x_h = grad_x_h.sum(dim=1)  # (N, d)
    # y = sigmoid(h)
    # grad = (y * (1 - y)).view(-1, 1) * grad_x_h  # (N, d)
    grad = grad_x_h
    return grad


def predict_grad_faster(x, grid, mu, sigma, gamma, sigma_scheme: str = "full"):
    diff = x[:, None, :] - grid[None, :, :]
    phi = torch.exp(-gamma * torch.square(diff).sum(dim=-1))  # (N, M)
    grad_x_phi = -2 * gamma * diff * phi[:, :, torch.newaxis]  # (N, M, d)
    grad = (grad_x_phi * mu.reshape(1, -1, 1)).sum(dim=1)  # (N, d)
    return grad


def predict(x, grid, mu, sigma, gamma, sigma_scheme: str = "full", compute_grad: bool = False, faster: bool = False):
    if faster:
        return predict_faster(x, grid, mu, sigma, gamma, sigma_scheme, compute_grad)
    return predict_fast(x, grid, mu, sigma, gamma, sigma_scheme, compute_grad)


def predict_grad(x, grid, mu, sigma, gamma, sigma_scheme: str = "full", faster: bool = False):
    if faster:
        return predict_grad_faster(x, grid, mu, sigma, gamma, sigma_scheme)
    return predict_grad_fast(x, grid, mu, sigma, gamma, sigma_scheme)


def generate_space():
    angles = np.linspace(0, 2 * np.pi, 50)
    radius1 = 0.3
    pts_circle1 = np.stack([np.cos(angles) * radius1 - 1, np.sin(angles) * radius1 + 0.2])
    angles = np.linspace(0, 2 * np.pi, 100)
    radius2 = 0.8
    pts_circle2 = np.stack([np.cos(angles) * radius2 + 0.3, np.sin(angles) * radius2])
    half_length = 2.0
    n = 40
    pts_box = np.vstack(
        [
            np.stack([np.full(n, -half_length), np.linspace(-half_length, half_length, n)], axis=-1),
            np.stack([np.linspace(-half_length, half_length, n), np.full(n, half_length)], axis=-1),
            np.stack([np.full(n, half_length), np.linspace(half_length, -half_length, n)], axis=-1),
            np.stack([np.linspace(half_length, -half_length, n), np.full(n, -half_length)], axis=-1),
        ]
    ).T
    space = Space2D([pts_circle1, pts_circle2, pts_box], outside_flags=[True, True, False])
    return space


def generate_trajectory(repeats: int = 1):
    angles = np.linspace(0, 2 * np.pi, 50)
    a = 1.6
    b = 1.2
    xy = np.stack([a * np.cos(angles), b * np.sin(angles)], axis=-1)
    diff = np.diff(xy, axis=0)
    angles: np.ndarray = np.arctan2(diff[:, 1], diff[:, 0])
    xy = xy[1:]
    if repeats > 1:
        xy = np.concatenate([xy] * repeats, axis=0)  # repeat the trajectory
        angles = np.concatenate([angles] * repeats)
    return xy, angles


def generate_dataset(
    sensor_pos: torch.Tensor,
    ray_dirs_world: torch.Tensor,
    dists: torch.Tensor,
    max_dist: float = 30.0,
    free_pts_per_meter: float = 1.0,
    margin: float = 0.05,
):
    """
    Generate a dataset.
    :param sensor_pos: (2,) sensor position
    :param ray_dirs_world: (2, N) ray directions in the world frame
    :param dists: (N,) distances
    :param max_dist: float
    :param free_pts_per_meter: float
    :param margin: float
    :return:
    """
    mask = dists < max_dist
    num_hit_pts = mask.sum()
    num_free_pts = (torch.clip(dists, 0, max_dist) * free_pts_per_meter).to(torch.int64)
    num_pts = num_free_pts.sum() + num_hit_pts
    pts = torch.empty((num_pts, 2))
    labels = torch.zeros(num_pts, dtype=dists.dtype)
    cnt = 0
    device = sensor_pos.device
    for i in range(len(dists)):
        r = (torch.rand(num_free_pts[i], device=device) * (1 - 2 * margin) + margin) * dists[i]
        pts[cnt : cnt + num_free_pts[i]] = (sensor_pos.reshape(-1, 1) + ray_dirs_world[:, [i]] * r[torch.newaxis]).T
        cnt += num_free_pts[i]
        if mask[i]:
            pts[cnt] = sensor_pos + ray_dirs_world[:, i] * dists[i]
            labels[cnt] = 1
            cnt += 1
    return pts, labels


def get_hit_pts(sensor_pos, ray_dirs_world, dists):
    pts = sensor_pos[:, torch.newaxis] + ray_dirs_world * dists
    return pts


def getTrainingData(data, robot_pos, max_laser_distance, unoccupied_points_per_meter=1, margin=0.01):
    distances = np.sqrt(np.sum((data - robot_pos) ** 2, axis=1))

    # parametric filling
    for n in range(len(distances)):
        dist = distances[n]
        laser_endpoint = data[n, :3]
        para = np.sort(np.random.random(np.int16(dist * unoccupied_points_per_meter)) * (1 - 2 * margin) + margin)[
            :, np.newaxis
        ]
        points_scan_i = robot_pos + para * (
            laser_endpoint - robot_pos
        )  # y = <x0, y0, z0> + para <x, y, z>; para \in [0, 1]
        # print('points_scan_i', points_scan_i)

        if n == 0:  # first data point
            if dist >= max_laser_distance:  # there's no laser reflection
                points = points_scan_i
                labels = np.zeros((points_scan_i.shape[0], 1))
            else:  # append the arrays with laser end-point
                points = np.vstack((points_scan_i, laser_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))
        else:
            if dist >= max_laser_distance:  # there's no laser reflection
                points = np.vstack((points, points_scan_i))
                labels = np.vstack((labels, np.zeros((points_scan_i.shape[0], 1))))
            else:  # append the arrays with laser end-point
                points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
                labels = np.vstack(
                    (labels, np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis])))
                )

    return np.hstack((points, labels))


def demo():
    np.random.seed(0)
    # Let's generate a toy dataset wit 30 LIDAR beams (note: this disregards circular geometry for simplicity)
    nBeams = 30
    robotPos = np.array([[5, 1]])
    laserHitpoints = np.hstack((np.linspace(-40, 40, nBeams).reshape(-1, 1), 50 * np.ones((nBeams, 1))))
    laserHitpoints[:8, 1] = 40

    # Get the training set
    trainingData = getTrainingData(laserHitpoints, robotPos, 100, 0.3, 0.03)

    # Plot
    plt.subplot(121)
    for i in range(nBeams):
        plt.plot(
            np.concatenate((robotPos[:, 0], np.array([laserHitpoints[i, 0]]))),
            np.concatenate((robotPos[:, 1], np.array([laserHitpoints[i, 1]]))),
            c="b",
        )
    plt.scatter(laserHitpoints[:, 0], laserHitpoints[:, 1], c="r", s=10, marker="*")
    plt.scatter(robotPos[:, 0], robotPos[:, 1], c="k", marker="s")
    plt.title("LiDAR beams")
    plt.subplot(122)
    plt.scatter(trainingData[:, 0], trainingData[:, 1], c=trainingData[:, 2], marker="x", cmap="jet")
    plt.colorbar()
    plt.title("Training datasets")
    plt.axis("equal")
    plt.tight_layout()

    # Step 0 - data
    X3, y3 = trainingData[:, :2], trainingData[:, 2]

    # Step 1 - define hinde points
    xx, yy = np.meshgrid(np.linspace(-60, 60, 60), np.linspace(0, 60, 30))
    grid = np.hstack((xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)))

    # Step 2 - compute features
    gamma = 0.7
    Phi = rbf_kernel(X3, grid, gamma=gamma)
    print("feature map size: ", Phi.shape)

    # Step 3 - estimate the parameters
    # Let's define the prior
    N, D = Phi.shape[0], Phi.shape[1]
    epsilon = np.ones(N)
    mu = np.zeros(D)
    # sig = 10000 * np.ones(D)
    sig = 10000 * np.eye(D)
    sigma_scheme = "diagonal"
    for i in range(3):
        # E-step
        mu, sig = calculate_posterior(Phi, y3, mu, sig, epsilon, sigma_scheme)

        # M-step
        # epsilon = np.sqrt(np.sum((Phi**2) * sig, axis=1) + (Phi.dot(mu.reshape(-1, 1)) ** 2).ravel())
        epsilon = calculate_xi(Phi, mu, sig, sigma_scheme)

    # Step 4 - predict
    qxx, qyy = np.meshgrid(np.linspace(-60, 60, 360), np.linspace(0, 60, 180))
    qX = np.hstack((qxx.ravel().reshape(-1, 1), qyy.ravel().reshape(-1, 1)))
    qPhi = rbf_kernel(qX, grid, gamma=gamma)
    qw = np.random.multivariate_normal(mu, sig, 1000)
    occ = sigmoid(qw.dot(qPhi.T))
    occMean = np.mean(occ, axis=0)
    occStdev = np.std(occ, axis=0)

    # occ = predictLap(qPhi, mu, sig)
    occMean = predict(qX, grid, mu, sig, gamma, "diagonal")[0]

    # Plot
    plt.figure(figsize=(30, 6))
    plt.subplot(131)
    # tpl.scatter(grid[:,0], grid[:,1], c='k', marker='o')
    plt.scatter(X3[:, 0], X3[:, 1], c=y3, marker="x", cmap="jet")
    plt.colorbar()
    plt.title("Hinge points and dataset")
    plt.xlim([-60, 60])
    plt.ylim([0, 60])
    plt.axis("equal")
    plt.subplot(132)
    plt.scatter(qX[:, 0], qX[:, 1], c=occMean, s=4, cmap="jet", vmin=0, vmax=1)
    plt.colorbar()
    plt.scatter(grid[:, 0], grid[:, 1], c=mu, marker="o", cmap="jet", s=1)
    plt.colorbar()
    plt.title("Occupancy probability - mean")
    plt.xlim([-60, 60])
    plt.ylim([0, 60])
    plt.axis("equal")
    plt.subplot(133)
    plt.scatter(qX[:, 0], qX[:, 1], c=occStdev, s=4, cmap="jet", vmin=0)
    plt.colorbar()
    plt.title("Occupancy probability - stdev")
    plt.xlim([-60, 60])
    plt.ylim([0, 60])
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def print_sigma(sigma):
    tqdm.write(
        f"diag(sigma).min(): {torch.diag(sigma).min().item():.6f}, det(sigma): {np.linalg.det(sigma.cpu().numpy())}"
    )


def main():
    # numpy warnings as errors
    # np.seterr(all="raise")

    img_dir = "/home/daizhirui/results/bhm"
    # device = "cpu"
    device = "cuda"
    dtype = torch.float32
    torch.set_grad_enabled(False)

    space = generate_space()
    space_vertices_cpu = space.surface.vertices
    space_vertices = torch.tensor(space_vertices_cpu.T, dtype=dtype).to(device)
    lidar_setting = Lidar2D.Setting()
    lidar_setting.max_angle = np.deg2rad(135)
    lidar_setting.min_angle = np.deg2rad(-135)
    lidar_setting.num_lines = 135
    lidar = Lidar2D(lidar_setting, space)
    trajectory = generate_trajectory(1)

    # generate the grid
    x_min = y_min = -5.0
    x_max = y_max = 5.0
    # grid_size = 51
    grid_size = 31
    grid_cpu = torch.meshgrid(
        torch.linspace(x_min, x_max, grid_size, dtype=dtype),
        torch.linspace(y_min, y_max, grid_size, dtype=dtype),
    )
    grid_cpu = torch.stack(grid_cpu, dim=-1).reshape(-1, 2)  # (121, 2)
    grid = grid_cpu.to(device)
    m = grid.shape[0]
    # gamma = 45
    gamma = 20
    mu = torch.zeros(m, dtype=dtype).to(device)
    sigma = 10000 * torch.eye(m, dtype=dtype).to(device)
    print_sigma(sigma)

    n_iter = 3
    # sigma_scheme = "full"
    sigma_scheme = "diagonal"
    faster_predict = True
    faster_grad_predict = True

    h = w = 100
    x_test = torch.meshgrid(
        torch.linspace(x_min, x_max, w, dtype=dtype),
        torch.linspace(y_min, y_max, h, dtype=dtype),
        indexing="xy",
    )
    x_test = torch.stack(x_test, dim=-1).reshape(-1, 2).to(device)  # (10000, 2)

    fig = plt.figure(figsize=(20, 9))

    plt.subplot(121)
    ax1 = plt.gca()
    occ = predict(x_test, grid, mu, sigma, gamma, sigma_scheme, False, faster_predict)[0]
    grad = predict_grad(x_test, grid, mu, sigma, gamma, sigma_scheme, faster_grad_predict)
    grad2 = predict_grad(space_vertices, grid, mu, sigma, gamma, sigma_scheme, faster_grad_predict)
    occ = occ.cpu().numpy()
    grad = grad.cpu().numpy()
    grad2 = grad2.cpu().numpy()
    img_occ = plt.imshow(
        occ.reshape(h, w),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="jet",
        interpolation="bilinear",
    )
    cbar_img_occ = plt.colorbar()
    grid_pts = plt.scatter(grid_cpu[:, 0], grid_cpu[:, 1], c=mu.cpu().numpy(), s=2, marker="o", cmap="jet")
    cbar_grid_pts = plt.colorbar(grid_pts)
    traj = plt.plot(trajectory[0][0], trajectory[0][1], c="white")[0]
    plt.title(f"gamma={gamma}, sigma_scheme={sigma_scheme}, grid={grid_size}x{grid_size}, faster={faster_predict}")
    plt.xlabel("x")
    plt.tight_layout()

    plt.subplot(122)
    ax2 = plt.gca()
    # grad = -grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    img_grad = plt.imshow(
        np.linalg.norm(grad, axis=-1).reshape(h, w),
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        cmap="jet",
        interpolation="bilinear",
    )
    cbar_img_grad = plt.colorbar()
    grad2 = -grad2 / np.linalg.norm(grad2, axis=-1, keepdims=True)
    quiver_grad = plt.quiver(
        space_vertices_cpu[0],
        space_vertices_cpu[1],
        grad2[:, 0],
        grad2[:, 1],
        color="white",
        scale=2,
        scale_units="xy",
        width=0.003,
        headwidth=2.5,
        headlength=3.5,
    )
    plt.scatter(space_vertices_cpu[0], space_vertices_cpu[1], c="g", s=5)
    plt.title(f"Gradient, faster={faster_grad_predict}")
    plt.xlabel("x")
    plt.tight_layout()
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    plt.pause(0.001)

    traj_idx = 0

    bhm_update_timer = CpuTimer("BHM Update", n_iter * len(trajectory[0]))
    bhm_predict_timer = CpuTimer("BHM Predict", n_iter * len(trajectory[0]))
    plt_timer = CpuTimer("Plotting", n_iter * len(trajectory[0]))
    plt.savefig(os.path.join(img_dir, f"{0:06d}.png"))

    ray_dirs = torch.tensor(lidar.ray_directions_in_frame, device=device).to(device)
    for xy, theta in zip(*trajectory):
        pts_file = f"/home/daizhirui/D/Dev/erl_sddf/cmake-build-debug/src/erl_gp_sdf/dataset_points_{traj_idx}.txt"
        y_file = f"/home/daizhirui/D/Dev/erl_sddf/cmake-build-debug/src/erl_gp_sdf/dataset_labels_{traj_idx}.txt"
        traj_idx += 1

        scan = lidar.scan(theta, xy)
        rotation = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).to(device)

        ray_dirs_world = rotation @ ray_dirs
        if os.path.exists(pts_file):
            import pandas as pd

            pts = pd.read_csv(pts_file).to_numpy().astype(np.float32).T
            y = pd.read_csv(y_file).to_numpy().astype(np.float32).flatten()
            pts = torch.tensor(pts, device=device, dtype=dtype)
            y = torch.tensor(y, device=device, dtype=dtype)
        else:
            pts, y = generate_dataset(
                torch.tensor(xy, device=device, dtype=dtype),
                ray_dirs_world,
                torch.tensor(scan, device=device, dtype=dtype),
                max_dist=30.0,
                free_pts_per_meter=3,
                margin=0.05,
            )
        pts = pts.to(device)
        y = y.to(device)

        # plt.scatter(pts[:, 0], pts[:, 1], c=y, marker="x", cmap="jet")

        lidar_pts = get_hit_pts(xy, ray_dirs_world.cpu().numpy(), scan)
        ax1.scatter(lidar_pts[0][::2], lidar_pts[1][::2], c="g", s=1)
        traj.set_data(trajectory[0][:traj_idx, 0], trajectory[0][:traj_idx, 1])

        xi = torch.ones(pts.shape[0], dtype=dtype).to(device)
        phi = rbf_kernel(pts, grid, gamma)
        # torch.diag(sigma).clip_(0.0001)
        print_sigma(sigma)
        for i in range(n_iter):
            with bhm_update_timer:
                mu, sigma = calculate_posterior(phi, y, mu, sigma, xi, sigma_scheme)
                xi = calculate_xi(phi, mu, sigma, sigma_scheme)
            print_sigma(sigma)

            with bhm_predict_timer:
                occ = predict(x_test, grid, mu, sigma, gamma, sigma_scheme, False, faster_predict)[0]
                grad = predict_grad(x_test, grid, mu, sigma, gamma, sigma_scheme, faster_grad_predict)
                grad2 = predict_grad(space_vertices, grid, mu, sigma, gamma, sigma_scheme, faster_grad_predict)
                occ = occ.cpu().numpy()
                grad = grad.cpu().numpy()
                grad2 = grad2.cpu().numpy()

            with plt_timer:
                img_occ.set_data(occ.reshape(h, w))
                img_occ.set_clim(occ.min(), occ.max())
                cbar_img_occ.update_normal(img_occ)

                grid_pts.set_array(mu.cpu().numpy())
                grid_pts.set_clim(mu.min(), mu.max())
                cbar_grid_pts.update_normal(grid_pts)

                # grad = -grad / np.linalg.norm(grad, axis=-1, keepdims=True)
                # grad_theta = np.arctan2(grad[:, 1], grad[:, 0]).reshape(h, w)
                grad_norm = np.linalg.norm(grad, axis=-1)
                img_grad.set_data(grad_norm.reshape(h, w))
                img_grad.set_clim(0, grad_norm.max())
                cbar_img_grad.update_normal(img_grad)
                grad2 = -grad2 / (np.linalg.norm(grad2, axis=-1, keepdims=True) + 1e-6)
                quiver_grad.set_UVC(grad2[:, 0], grad2[:, 1])

                ax1.set_xlabel(f"{device}, Update time: {bhm_update_timer.t:.6f}/{bhm_update_timer.average_t:.6f} sec")
                ax2.set_xlabel(
                    f"{device}, Predict time: {bhm_predict_timer.t:.6f}/{bhm_predict_timer.average_t:.6f} sec"
                )

                fig.canvas.blit(fig.bbox)
                fig.canvas.flush_events()
                # plt.pause(0.001)
                plt.savefig(os.path.join(img_dir, f"{(traj_idx - 1) * n_iter + i:06d}.png"))
        # break
    tqdm.write("Done")
    plt.show()


if __name__ == "__main__":
    # demo()
    main()
