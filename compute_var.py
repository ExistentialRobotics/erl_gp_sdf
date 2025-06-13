import numpy as np
import struct
import yaml
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad as get_grad


def read_array_from_bin(filename):
    with open(filename, "rb") as f:
        data = f.read()
    size = struct.unpack("l", data[:8])[0]
    rows = struct.unpack("l", data[8:16])[0]
    cols = struct.unpack("l", data[16:24])[0]
    assert size == rows * cols, "size != rows * cols"
    array = np.frombuffer(data[24:], dtype=np.float32, count=rows * cols).reshape((cols, rows)).T
    return array


def read_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return data


def softmin(z, alpha: float):
    """
    :param z: (N, )
    :param alpha:
    :return:
    """
    s = np.exp(-alpha * z)
    s = s / np.sum(s, axis=0, keepdims=True)
    return s


def softmin_torch(z: torch.Tensor, alpha: float) -> torch.Tensor:
    s = torch.exp(-alpha * z)
    s = s / torch.sum(s, dim=0, keepdim=True)
    return s


def compute_var_numpy(x: np.ndarray, test_position: np.ndarray, temperature: float = 10.0, z0: float = 0.0):
    diff = test_position.reshape(2, 1) - x  # (2, N)
    z = np.linalg.norm(diff, axis=0)  # (N, )
    s = softmin(z - z0, temperature)  # (N, )
    h = np.dot(s, z)  # scalar

    diff /= z.reshape(1, -1)  # (2, N)
    grad = -(s * (temperature * h + 1.0 - temperature * z)).reshape(1, -1) * diff  # (2, N)

    l = temperature * s * (h - z) + s  # (N, )
    grad_star = (diff @ l).reshape(2, 1)  # (2, 1)
    grad_star_norm = np.linalg.norm(grad_star)
    grad_star_normalized = grad_star / grad_star_norm
    grad_normalization = (np.eye(2) - grad_star_normalized @ grad_star_normalized.T) / grad_star_norm
    sv = (diff @ s).reshape(2, 1)  # (2, 1)
    grad_g0 = np.zeros_like(grad)
    grad_g1 = np.zeros_like(grad)
    for j in range(grad.shape[1]):
        # lj = temperature * s[j] * (h - z[j]) + s[j]
        vj = diff[:, j].reshape(2, 1)
        g = (temperature * l[j]) * (vj - sv) + (temperature * s[j]) * (vj - grad_star)
        g = vj @ g.T + l[j] / z[j] * (vj @ vj.T - np.eye(2))

        g = g @ grad_normalization

        # g = s[j] * (diff @ (s * (z[j] - z))) - (lj + s[j]) * vj
        # g = temperature * vj.reshape(2, 1) @ g.reshape(1, 2)
        # g += lj / z[j] * (vj.reshape(2, 1) @ vj.reshape(1, 2) - np.eye(2))
        # g = l / z[i] * (diff[:, i].reshape(2, 1) @ diff[:, i].reshape(1, 2) - np.eye(2))
        grad_g0[:, j] = g[:, 0]
        grad_g1[:, j] = g[:, 1]
    return s, h, grad, grad_g0, grad_g1


def compute_var_torch(x: torch.Tensor, test_position: torch.Tensor, temperature: float = 10.0, z0: float = 0.0):
    with torch.enable_grad():
        x = x.requires_grad_(True)
        test_position = test_position.requires_grad_(True)
        diff = test_position.reshape(2, 1) - x  # (2, N)
        z = diff.norm(dim=0)
        s = softmin_torch(z - z0, temperature)
        h = s.dot(z)

        # h.backward()
        # grad = x.grad
        grad = get_grad(h, x, create_graph=True, retain_graph=True)[0]

        # V = diff / z.reshape(1, -1)  # (2, N)
        # l = temperature * (s.reshape(-1, 1) * s.reshape(1, -1) - torch.diag(s)) @ z + s
        # l = temperature * s * (h - z) + s
        # j = 0
        # i = 1

        # grad(l[j], x[:, j])
        # g0 = get_grad(l[j], x, create_graph=True, retain_graph=True, allow_unused=True)[0][:, j]
        # g1 = temperature * (l[j] + s[j] - 2 * l[j] * s[j]) * V[:, j]

        # grad(l[i], x[:, j])
        # g0 = get_grad(l[i], x, create_graph=True, retain_graph=True)[0][:, j]
        # sisj = s[i] * s[j]
        # g1 = -temperature * (temperature * sisj * (2 * h - z[i] - z[j]) + 2 * sisj) * V[:, j]

        g = get_grad(h, test_position, create_graph=True, retain_graph=True)[0]
        # l = temperature * s * (h - z) + s
        # V = diff / z.reshape(1, -1)  # (2, N)
        # g = V @ l
        g = g / torch.linalg.norm(g, dim=0)
        grad_g0 = get_grad(g[0], x, retain_graph=True)[0]
        grad_g1 = get_grad(g[1], x, retain_graph=True)[0]
        # g[0].backward()
        # grad_g0 = x.grad
    return s, h, grad, grad_g0, grad_g1


def grad_v():
    x_star = torch.rand((2, 1), requires_grad=True)
    xi = torch.rand((2, 1), requires_grad=True)
    diff = x_star - xi
    z = diff.norm(dim=0)
    v = diff / z
    grad0 = [
        get_grad(v[0], xi, create_graph=True, retain_graph=True)[0],
        get_grad(v[1], xi, create_graph=True, retain_graph=True)[0],
    ]
    grad1 = (v @ v.T - torch.eye(2)) / z
    print(grad0)
    print(grad1)


def main():
    build_type = "release"
    sdf_gp_id = "100431781586896"
    folder = f"/home/daizhirui/D/Dev/erl_sddf/cmake-build-{build_type}/src/erl_gp_sdf/sdf_gp_debug_{sdf_gp_id}"
    x = read_array_from_bin(f"{folder}/train_set.x.bin")
    y = read_array_from_bin(f"{folder}/train_set.y.bin")
    var_x = read_array_from_bin(f"{folder}/train_set.var_x.bin")
    config = read_yaml(f"{folder}/test_covariance.yaml")
    test_position = np.array(config["test_position"])
    num_samples = config["num_samples"]
    # softmin_temperature = config["softmin_temperature"]
    softmin_temperature = 20.0
    f0 = config["f0"]
    sz_sdf = config["sz_sdf"]
    var_sdf0 = config["var_sdf"]
    var_grad = config["var_grad"]
    x = x[:, :num_samples]  # (2, N)
    y = y[:num_samples]  # (N, 3)
    var_x = var_x[:num_samples]  # (N, )
    # f0 = 0

    s_numpy, h_numpy, grad_numpy, grad_g0_numpy, grad_g1_numpy = compute_var_numpy(
        x, test_position, softmin_temperature, f0
    )
    s_torch, h_torch, grad_torch, grad_g0_torch, grad_g1_torch = compute_var_torch(
        torch.tensor(x), torch.tensor(test_position), softmin_temperature, f0
    )
    w_grad_numpy = np.sum(grad_numpy**2, axis=0)
    var_sdf = w_grad_numpy.dot(var_x)

    print(f"h_numpy: {h_numpy}")
    print(f"h_torch: {h_torch.item()}")
    print(f"var_sdf: {var_sdf}")
    print(f"std_sdf: {np.sqrt(var_sdf)}")
    print(f"f0: {f0}")
    print(f"sz_sdf: {sz_sdf}")
    print(f"var_sdf0: {var_sdf0}")
    print(f"var_grad: {var_grad}")

    # print(s_numpy.flatten())
    # print(w_grad_numpy.flatten())
    # print(var_x.flatten())
    # plt.plot(np.arange(num_samples), s_numpy.flatten(), label="weight")
    # plt.plot(np.arange(num_samples), w_grad_numpy.flatten(), label="weight_grad")
    # plt.plot(np.arange(num_samples), var_x.flatten(), label="var_x")
    # plt.xlabel("sample index")
    # plt.ylabel("value")
    # plt.legend()
    # plt.show()

    assert np.allclose(h_numpy, h_torch.detach().numpy())
    assert np.allclose(grad_numpy, grad_torch.detach().numpy())

    # print(f"grad_g0_numpy: {grad_g0_numpy}")
    # print(f"grad_g0_torch: {grad_g0_torch.detach().numpy()}")
    # print(f"grad_g1_numpy: {grad_g1_numpy}")
    # print(f"grad_g1_torch: {grad_g1_torch.detach().numpy()}")

    assert np.allclose(grad_g0_numpy, grad_g0_torch.detach().numpy())
    assert np.allclose(grad_g1_numpy, grad_g1_torch.detach().numpy())


if __name__ == "__main__":
    # grad_v()
    main()
