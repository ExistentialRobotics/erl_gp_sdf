import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    mu = pd.read_csv("mu-2d.txt", header=None).to_numpy()
    sigma = pd.read_csv("sigma-2d.txt", header=None).to_numpy()
    sigma = 10000 - sigma

    mu = np.log(np.abs(mu) + 1)
    sigma = np.log(np.abs(sigma) + 1)

    fig_size = (30, 7)

    plt.figure(figsize=fig_size)
    plt.imshow(mu, cmap="jet")
    plt.colorbar(orientation="horizontal")
    plt.title("Mu")
    plt.tight_layout()

    plt.figure(figsize=fig_size)
    plt.imshow(sigma, cmap="jet")
    plt.colorbar(orientation="horizontal")
    plt.title("Sigma")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
