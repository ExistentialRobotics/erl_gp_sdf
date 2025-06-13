import os

from bayesian_hilbert_map import calculate_xi
from bayesian_hilbert_map import calculate_posterior
from bayesian_hilbert_map import rbf_kernel
from bayesian_hilbert_map import predict
from bayesian_hilbert_map import sigmoid

import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.patches as patches
import matplotlib

matplotlib.use("TkAgg")
mplstyle.use("fast")


# xi = torch.tensor(np.linspace(-2, 2, 100))
# y = (sigmoid(xi) - 0.5) / (2 * xi)
# y[torch.abs(xi) < 1e-6] = 0.125
# print(xi)
# print(y)
# plt.plot(xi.numpy(), y.numpy())
# plt.xlabel("xi")
# plt.ylabel("y")
# plt.show()


data_folder = "/home/daizhirui/results/erl_gp_sdf/bhm_mapping/32764,32768"

dtype = torch.float32

hinged_points_file = f"{data_folder}/hinged_points.txt"
hinged_points = pd.read_csv(hinged_points_file, header=None).to_numpy().T  # (m, 2)
hinged_points = torch.tensor(hinged_points, dtype=dtype).cuda()

map_min_file = f"{data_folder}/map_min.txt"
map_min = pd.read_csv(map_min_file, header=None).to_numpy().flatten().tolist()
map_max_file = f"{data_folder}/map_max.txt"
map_max = pd.read_csv(map_max_file, header=None).to_numpy().flatten().tolist()

step = 0
n_itr = 3
gamma = 40
sigma_scheme = "diagonal"
faster_predict = True
faster_grad_predict = True


h = w = 200
x_test = torch.meshgrid(
    torch.linspace(map_min[0], map_max[0], w, dtype=dtype),
    torch.linspace(map_min[1], map_max[1], h, dtype=dtype),
    indexing="xy",
)
x_test = torch.stack(x_test, dim=-1).reshape(-1, 2).cuda()  # (h * w, 2)

m = hinged_points.shape[0]
mu = torch.zeros(m, dtype=dtype).cuda()
sigma = 10000 * torch.eye(m, dtype=dtype).cuda()

torch.set_grad_enabled(False)

plt_img_occ = None
plt_img_grad = None
plt_scatters = []

for step in range(50):
    points_file = f"{data_folder}/{step}_dataset_points.txt"
    if not os.path.exists(points_file):
        continue
    labels_file = f"{data_folder}/{step}_dataset_labels.txt"
    sensor_origin_file = f"{data_folder}/{step}_sensor_origin.txt"
    sensor_points_file = f"{data_folder}/{step}_sensor_points.txt"
    iter_cnt_file = f"{data_folder}/{step}_iteration_count.txt"
    print(points_file)
    points = pd.read_csv(points_file, header=None).to_numpy().T  # (n, 2)
    labels = pd.read_csv(labels_file, header=None).to_numpy().flatten()  # (n, )
    sensor_origin = pd.read_csv(sensor_origin_file, header=None).to_numpy().flatten()  # (2, )
    sensor_points = pd.read_csv(sensor_points_file, header=None).to_numpy().T  # (m, 2)
    iter_cnt = pd.read_csv(iter_cnt_file, header=None).to_numpy().flatten()[0]

    points = torch.tensor(points, dtype=dtype).cuda()
    labels = torch.tensor(labels, dtype=dtype).cuda()

    xi = torch.ones(points.shape[0], dtype=dtype).cuda()
    phi = rbf_kernel(points, hinged_points, gamma)

    for i in range(n_itr):
        mu, sigma = calculate_posterior(phi, labels, mu, sigma, xi, sigma_scheme)
        xi = calculate_xi(phi, mu, sigma, sigma_scheme)

        occ, grad = predict(
            x_test,
            hinged_points,
            mu,
            sigma,
            gamma,
            sigma_scheme,
            True,
            faster_predict,
        )

        occ = occ.cpu().numpy()
        grad = grad.cpu().numpy()
        for scatter in plt_scatters:
            scatter.remove()
        plt_scatters.clear()
        if plt_img_occ is None:
            extent = (map_min[0], map_max[0], map_min[1], map_max[1])
            plt_img_occ = plt.imshow(
                occ.reshape(h, w), cmap="jet", origin="lower", extent=extent, interpolation="bilinear"
            )
            plt_img_grad = plt.imshow(
                np.linalg.norm(grad, axis=-1).reshape(h, w),
                cmap="jet",
                origin="lower",
                extent=extent,
                interpolation="bilinear",
            )
            plt_scatters.append(
                plt.scatter(sensor_origin[0], sensor_origin[1], marker="o", color="r", label="Sensor Origin")
            )
            plt_scatters.append(
                plt.scatter(sensor_points[:, 0], sensor_points[:, 1], marker="*", color="y", label="Sensor Points")
            )
            plt_scatters.append(
                plt.scatter(
                    points[:, 0].cpu().numpy(),
                    points[:, 1].cpu().numpy(),
                    marker="x",
                    c=1 - labels.cpu().numpy(),
                    cmap="jet",
                    label="Points",
                )
            )
            plt.title(f"step {step}, iter_cnt {iter_cnt}")
            plt.pause(0.01)
        else:
            plt_img_occ.set_array(occ.reshape(h, w))
            plt_img_grad.set_array(np.linalg.norm(grad, axis=-1).reshape(h, w))
            plt_scatters.append(
                plt.scatter(sensor_origin[0], sensor_origin[1], marker="o", color="r", label="Sensor Origin")
            )
            plt_scatters.append(
                plt.scatter(sensor_points[:, 0], sensor_points[:, 1], marker="*", color="y", label="Sensor Points")
            )
            plt_scatters.append(
                plt.scatter(
                    points[:, 0].cpu().numpy(),
                    points[:, 1].cpu().numpy(),
                    marker="x",
                    c=1 - labels.cpu().numpy(),
                    cmap="jet",
                    label="Points",
                )
            )
            plt.title(f"step {step}, iter_cnt {iter_cnt}")
            fig = plt.gcf()
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
        plt.savefig(f"{data_folder}/vis_{step:03d}.png")
    print(labels)
plt.show()
