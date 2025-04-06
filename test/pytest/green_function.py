import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# mixture of gaussians
def mixture_of_gaussians(x, means, sigma, t):
    """
    Mixture of Gaussians function.
    :param x: (N, 2) input points
    :param means: (num_gaussians, 2) means of the gaussians
    :param sigma: float, standard deviation of the gaussians
    :param t: float
    :return:
    """
    num_gaussians = means.shape[0]
    dists = np.square(x[:, None, :] - means[None, :, :]).sum(axis=-1)  # (N, num_gaussians)
    s = sigma**2 + t
    # alpha = 1.0 / (2 * np.pi * s) / num_gaussians
    # alpha = 1.0 / (2 * np.pi * s)
    # alpha = 1.5 / num_gaussians
    alpha = 1
    print(alpha)
    y = alpha * np.exp(-(0.5 / s) * dists).sum(axis=-1)  # (N,)
    print(y)
    return y


def edf_estimation(x, means, sigma, t):
    return np.sqrt(np.abs(-2 * t * np.log(mixture_of_gaussians(x, means, sigma, t))))


def edf_gt_circle(x, circle_center, radius):
    """
    EDF for a circle.
    :param x: (N, 2) input points
    :param circle_center: (2,) center of the circle
    :param radius: float, radius of the circle
    :return:
    """
    dists = np.linalg.norm(x - circle_center, axis=-1)
    return np.abs(dists - radius)


def plot_img(ax, img, x_min, x_max, y_min, y_max, circle_center, radius, title):
    mappable = ax.imshow(img, extent=[x_min, x_max, y_min, y_max], origin="lower", cmap="jet", interpolation="bilinear")
    plt.colorbar(mappable, ax=ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.add_patch(patches.Circle(circle_center, radius, facecolor="none", edgecolor="black"))


def main():
    circle_center = np.array([0.0, 0.0])
    radius = 1.0
    num_gaussians = 100
    sigma = 0.001

    # generate means
    angles = np.linspace(0, 2 * np.pi, num_gaussians, endpoint=False)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)  # (num_gaussians, 2)
    means += circle_center.reshape(1, 2)  # center the means

    # generate points
    x_min = y_min = -2.0
    x_max = y_max = 2.0
    n = 100
    points = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    points = np.stack(points, axis=-1).reshape(-1, 2)  # (n*n, 2)
    # compute edf
    t = 0.005
    edf = edf_estimation(points, means, sigma, t)
    # compute edf error
    edf_gt = edf_gt_circle(points, circle_center, radius)
    edf_error = edf - edf_gt
    edf_error = np.abs(edf_error)

    # visualize
    edf = edf.reshape(n, n)
    edf_gt = edf_gt.reshape(n, n)
    edf_error = edf_error.reshape(n, n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    plot_img(axes[0], edf, x_min, x_max, y_min, y_max, circle_center, radius, f"EDF (t={t}, sigma={sigma})")
    plot_img(axes[1], edf_error, x_min, x_max, y_min, y_max, circle_center, radius, "EDF Error")
    plot_img(axes[2], edf_gt, x_min, x_max, y_min, y_max, circle_center, radius, "Ground Truth EDF")
    plt.show()


if __name__ == "__main__":
    main()
