# import sympy as sp
#
# sp.init_printing(wrap_line=False)
# x = sp.symbols("x")
# xi = sp.symbols("xi")
#
#
# def sigmoid(v):
#     return 1.0 / (1.0 + sp.exp(-v))
#
#
# h = sp.exp((x - xi) / 2.0 + (0.5 - sigmoid(xi)) / (2.0 * xi) * (x**2 - xi**2))
# dhdxi = sp.diff(h, xi)
# dhdxi = sp.simplify(dhdxi)
# # sp.pretty_print(dhdxi)
# print(dhdxi)

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


x = np.linspace(-5, 5, 200)
# y = (0.5 - sigmoid(x)) / x
# plt.plot(x, y)
# plt.show()

xs = 3
ys = (xs - x) + (0.5 - sigmoid(x)) / x * (xs**2 - x**2)
zs = sigmoid(x) * np.exp(ys / 2.0)
# plt.plot(x, ys)
plt.plot(x, zs)
plt.grid()
plt.show()
