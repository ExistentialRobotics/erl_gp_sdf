import numpy as np
import matplotlib.pyplot as plt


def gaussian_distance(x, y, sigma, n=1000000):
    # a point p1 ~ N((0, 0), sigma^2 I)
    # what's the distribution of the distance from p2=(x, y) to p1?

    p1_samples = np.random.normal(0, sigma, (n, 2))
    p2 = np.array([x, y])
    distances = np.linalg.norm(p1_samples - p2, axis=1)

    d0 = np.linalg.norm(p2)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # draw histogram
    plt.hist(distances, bins=200, density=True)
    plt.axvline(d0, color="r", label="d0")
    plt.axvline(sigma, color="g", label="sigma")
    plt.axvline(mean_distance, color="b", label="mean")
    plt.xlabel("distance")
    plt.ylabel("density")
    sigma_square = sigma ** 2
    plt.title(f"distance from ({x}, {y}) to N((0, 0), {sigma_square} I), std={std_distance}")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"gaussian_distance_{x}_{y}_{sigma}.png")
    plt.show()


def sigma_vs_distance_std(x, y):
    sigmas = np.concatenate(
        [
            np.linspace(0.01, 0.1, 10, endpoint=False),
            np.linspace(0.1, 1, 10, endpoint=False),
            # np.linspace(1, 10, 10, endpoint=False),
            # np.linspace(10, 100, 10, endpoint=False),
        ]
    )
    n = 1000000
    p2 = np.array([x, y])
    std_values = []
    for sigma in sigmas:
        p1_samples = np.random.normal(0, sigma, (n, 2))
        distances = np.linalg.norm(p1_samples - p2, axis=1)
        std_distance = np.std(distances)
        std_values.append(std_distance)
    plt.plot(sigmas, std_values, label="std of distance")
    plt.plot(sigmas, sigmas, label="sigma")
    plt.xlabel("sigma")
    plt.ylabel("std of distance")
    plt.title(f"std of distance from ({x}, {y}) to N((0, 0), sigma^2 I)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    sigma_vs_distance_std(1, 1)
    sigma_vs_distance_std(1, 10)

    # gaussian_distance(1, 1, 0.1)
    # gaussian_distance(1, 1, 1)
    # gaussian_distance(1, 1, 2)
    # gaussian_distance(1, 1, 4)
    # gaussian_distance(1, 1, 10)
    # gaussian_distance(1, 1, 20)

    gaussian_distance(10, 10, 0.1)
    gaussian_distance(10, 10, 1)
    gaussian_distance(10, 10, 2)
    gaussian_distance(10, 10, 4)
    gaussian_distance(10, 10, 10)
    gaussian_distance(10, 10, 20)


if __name__ == "__main__":
    main()
