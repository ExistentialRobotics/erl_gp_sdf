import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

# ------------------------
# Define Surfaces
# ------------------------
domain_size = 5.0
N_plot = 101
x = np.linspace(0, domain_size, N_plot)
y = np.linspace(0, domain_size, N_plot)
Xg, Yg = np.meshgrid(x, y, indexing='ij')
points_grid = np.column_stack([Xg.ravel(), Yg.ravel()])

center = np.array([domain_size / 2.0, domain_size / 2.0])
R = 1.0
num_points = 200
thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

X_circle = center + np.column_stack([R * np.cos(thetas), R * np.sin(thetas)])
y_circle = np.ones(num_points)
grad_circle = np.column_stack([np.cos(thetas), np.sin(thetas)])

xs = np.linspace(0, domain_size, num_points, endpoint=False)
ys = xs.copy()

sq_left   = np.column_stack([np.zeros(num_points), ys])
sq_right  = np.column_stack([domain_size*np.ones(num_points), ys])
sq_bottom = np.column_stack([xs, np.zeros(num_points)])
sq_top    = np.column_stack([xs, domain_size*np.ones(num_points)])

X_square = np.vstack([sq_left, sq_right, sq_bottom, sq_top])
y_square = np.ones(X_square.shape[0])

n_left   = np.tile([1.0,  0.0], (num_points,1))
n_right  = np.tile([-1.0, 0.0], (num_points,1))
n_bottom = np.tile([0.0,  1.0], (num_points,1))
n_top    = np.tile([0.0, -1.0], (num_points,1))

grad_square = np.vstack([n_left, n_right, n_bottom, n_top])

X_train = np.vstack([X_circle, X_square])
y_train = np.concatenate([y_circle, y_square])
grad_train = np.vstack([grad_circle, grad_square])

# ------------------------
# Define Matérn 3/2 kernel and its gradient
# ------------------------
sigma_f = 1.0
lambda_param = 50.0
ell = np.sqrt(3) / lambda_param

def matern32_kernel(X1, X2):
    dists = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=-1)
    sqrt3_r_l = np.sqrt(3) * dists / ell
    return sigma_f**2 * (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

def matern32_kernel_grad(X1, X2):
    diff = X1[:, None, :] - X2[None, :, :]
    dists = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-12
    sqrt3_r_l = np.sqrt(3) * dists / ell
    common = - (3.0 / ell**2) * np.exp(-sqrt3_r_l)
    return sigma_f**2 * common * diff

# ------------------------
# Pre-compute inverses
# ------------------------
noise_var = 1e-6
K = matern32_kernel(X_train, X_train) + noise_var * np.eye(len(X_train))
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# ------------------------
# Predict GP mean and gradient on grid
# ------------------------
k_star = matern32_kernel(X_train, points_grid)
f_mean = k_star.T @ alpha
grad_k = matern32_kernel_grad(points_grid, X_train)
alpha_vec = alpha.reshape(1, -1, 1)
f_grad = np.sum(grad_k * alpha_vec, axis=1)

# ------------------------
# Log-GPIS distance and gradient
# ------------------------
d_field = -np.log(f_mean) / lambda_param
grad_d = - (1.0 / (lambda_param * f_mean[:, None])) * f_grad

# ------------------------
# Determine sign using nearest surface normal
# ------------------------
kdtree = cKDTree(X_train)
_, nn_idx = kdtree.query(points_grid, k=1)
surface_norm = grad_train[nn_idx]
dot_prod = np.einsum('ij,ij->i', grad_d, surface_norm)
sign_field = np.where(dot_prod <= 0.0, -1.0, 1.0)
d_signed = sign_field * d_field
# ------------------------
# Visualization
# ------------------------
grad_x  = grad_d[:, 0].reshape(N_plot, N_plot)
grad_y  = grad_d[:, 1].reshape(N_plot, N_plot)


# ------------------------
# Gradient Diffusion via connection Laplacian on regular grid
# ------------------------
N = N_plot
h = domain_size / (N - 1)
N2 = N * N

# 1D negative-definite Laplacian
main = -2 * np.ones(N)
off = np.ones(N - 1)
D1d = diags([main, off, off], [0, -1, 1], shape=(N, N), format='csr')

# 2D Laplacian via sparse Kronecker sums
I = identity(N, format='csr')
L2d = kron(D1d, I, format='csr') + kron(I, D1d, format='csr')
L2d = L2d / (h * h)

# Backward Euler: (I - t Δ) X = X0, choose t = h^2
t = h**2
A = identity(N2, format='csr') - t * L2d

# Build initial vector measure X0 on grid
X0 = np.zeros((N2, 2))
tree = cKDTree(points_grid)
_, indices = tree.query(X_train, k=1)
for i, idx in enumerate(indices):
    X0[idx] += grad_train[i]

# Solve for diffused vectors
X_diffused = np.zeros_like(X0)
X_diffused[:, 0] = spsolve(A, X0[:, 0])
X_diffused[:, 1] = spsolve(A, X0[:, 1])

# Normalize to unit vectors
norms = np.linalg.norm(X_diffused, axis=1)
Y_diffused = np.zeros_like(X_diffused)
nonzero = norms > 0
Y_diffused[nonzero] = X_diffused[nonzero] / norms[nonzero, None]

# Reshape for plotting
Xq = X_diffused.reshape((N, N, 2))
Yq = Y_diffused.reshape((N, N, 2))

# Compute dot product between diffused Yq and log_gpis gradient field
dotted_result = np.einsum('ij,ij->i', Y_diffused, grad_d)
# Get sign of dotted_result to determine sign
sign_field2 = np.sign(dotted_result)

d_field_diffused = sign_field2 * d_field
d_field_diffused = d_field_diffused.reshape(N_plot, N_plot)

# Plotting: quiver plots for non-normalized and normalized fields
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
stride = 3

# 1. Log-GPIS Gradient Field
axes[0][0].quiver(Xg[::stride, ::stride], Yg[::stride, ::stride],
                  -grad_x[::stride, ::stride], -grad_y[::stride, ::stride],
                  scale=40, width=0.002, alpha=0.7)
axes[0][0].quiver(X_train[:,0], X_train[:,1],
                  grad_train[:,0], grad_train[:,1],
                  scale=25, width=0.003, color='lime')
axes[0][0].set_title('Log-GPIS Gradient Field (subsampled)')
axes[0][0].set_xlim(0, domain_size)
axes[0][0].set_ylim(0, domain_size)
axes[0][0].set_aspect('equal')

# 2. Log-GPIS Distance Field
im1 = axes[1][0].pcolormesh(Xg, Yg, d_signed.reshape(N_plot,N_plot), shading='auto',
                            cmap='inferno',
                            vmin=-np.max(np.abs(d_signed)),
                            vmax=np.max(np.abs(d_signed)))
axes[1][0].quiver(X_train[:,0], X_train[:,1],
                  grad_train[:,0], grad_train[:,1],
                  scale=25, width=0.003, color='lime')
axes[1][0].set_title('Log-GPIS Distance Field (Signed)')
axes[1][0].set_xlim(0, domain_size)
axes[1][0].set_ylim(0, domain_size)
axes[1][0].set_aspect('equal')
fig.colorbar(im1, ax=axes[1][0], fraction=0.046, pad=0.04)

# 3. Diffused Gradient Field
axes[0][1].quiver(Xg[::stride, ::stride], Yg[::stride, ::stride],
                  Yq[::stride, ::stride, 0], Yq[::stride, ::stride, 1],
                  scale=40, width=0.002)
axes[0][1].quiver(X_train[:,0], X_train[:,1],
                  grad_train[:,0], grad_train[:,1],
                  scale=25, width=0.003, color='lime')
axes[0][1].set_title('Diffused Gradient Field (subsampled)')
axes[0][1].set_xlim(0, domain_size)
axes[0][1].set_ylim(0, domain_size)
axes[0][1].set_aspect('equal')

# 4. Signed Distance Field (diffused)
im2 = axes[1][1].pcolormesh(Xg, Yg, d_field_diffused, shading='auto',
                            cmap='inferno',
                            vmin=-np.max(np.abs(d_signed)),
                            vmax=np.max(np.abs(d_signed)))
axes[1][1].quiver(X_train[:,0], X_train[:,1],
                  grad_train[:,0], grad_train[:,1],
                  scale=25, width=0.003, color='lime')
axes[1][1].set_title('Signed Distance Field (diffused)')
axes[1][1].set_xlim(0, domain_size)
axes[1][1].set_ylim(0, domain_size)
axes[1][1].set_aspect('equal')
fig.colorbar(im2, ax=axes[1][1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
