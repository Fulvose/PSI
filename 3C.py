import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]
rv1 = multivariate_normal(mean1, cov1)

# 2
mean2 = [0, 0]
cov2 = [[2, 0], [0, 1]]
rv2 = multivariate_normal(mean2, cov2)

# 3
mean3 = [0, 0]
cov3 = [[1, 0], [0, 2]]
rv3 = multivariate_normal(mean3, cov3)
# siatka punktów
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
# gęstości
pdf1 = rv1.pdf(pos)
pdf2 = rv2.pdf(pos)
pdf3 = rv3.pdf(pos)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].contourf(x, y, pdf1, cmap='viridis')
ax[1].contourf(x, y, pdf2, cmap='viridis')
ax[2].contourf(x, y, pdf3, cmap='viridis')
ax[0].set_title('Rozkład 1')
ax[1].set_title('Rozkład 2')
ax[2].set_title('Rozkład 3')
plt.show()



# parametry
params = [
    (np.array([0, 0]), np.array([[1, 0], [0, 1]])),
    (np.array([0, 0]), np.array([[2, 0], [0, 1]])),
    (np.array([0, 0]), np.array([[1, 0], [0, 2]]))
]
# siatka
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, (mean, cov) in enumerate(params):
    rv = multivariate_normal(mean, cov)
    axs[i].contour(x, y, rv.pdf(pos), levels=10)
    axs[i].set_aspect('equal')
    axs[i].set_title(f"n={mean}, cov={cov}")
plt.show()

params = [
    (np.array([0, 0]), np.array([[1, 0], [0, 1]])),
    (np.array([0, 0]), np.array([[2, 0], [0, 1]])),
    (np.array([0, 0]), np.array([[1, 0], [0, 2]]))
]
# siatka 
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
for i, (mean, cov) in enumerate(params):
    rv = multivariate_normal(mean, cov)
    axs[i].contour(x, y, rv.pdf(pos), levels=10)
    axs[i].set_aspect('equal')
    axs[i].set_title(f"n={mean}, cov={cov}")
plt.show()