mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])

X_rv = multivariate_normal(mean, cov)
# próbka z rozkładu normalnego
X_sample = X_rv.rvs(1000)
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)
z = rv.pdf(pos)

fig, ax = plt.subplots()
ax.scatter(X_sample[:, 0], X_sample[:, 1], alpha=0.2)
ax.contour(x, y, z)
plt.show()




mean1 = np.array([0, 0])
cov1 = np.array([[4.40, -2.75], [-2.75,  5.50]])
X1_rv = multivariate_normal(mean1, cov1)
X = X1_rv.rvs(1000)

mean_sample = np.mean(X, axis=0)
cov_sample = np.cov(X.T)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(X[:, 0], X[:, 1], alpha=0.5)

x, y = np.mgrid[-5:5:.01, -5:5:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean_sample, cov_sample)
ax.contour(x, y, rv.pdf(pos), levels=10)

eig_vals, eig_vecs = np.linalg.eig(cov_sample)
for i in range(len(eig_vals)):
    eig_val, eig_vec = eig_vals[i], eig_vecs[:, i]
    ax.quiver(mean_sample[0], mean_sample[1], eig_val * eig_vec[0], eig_val * eig_vec[1], color='r', scale=5)

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
plt.show()





n = 1000
sample = np.random.rand(n, 2)
mean = np.mean(sample, axis=0)
covariance = np.cov(sample.T)

print("Średnia próbki:")
print(mean)
print("Macierz kowariancji próbki:")
print(covariance)

plt.figure(figsize=(6,6))
plt.plot(sample[:,0], sample[:,1], '.', ms=3)
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, covariance)
plt.figure(figsize=(6,6))
plt.contourf(x, y, rv.pdf(pos), cmap=plt.cm.Blues)
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

eigenvalues, eigenvectors = np.linalg.eig(covariance)
plt.figure(figsize=(6,6))
plt.quiver(mean[0], mean[1], eigenvectors[0,0]*np.sqrt(eigenvalues[0]), eigenvectors[1,0]*np.sqrt(eigenvalues[0]), angles='xy', scale_units='xy', scale=1, color='r')
plt.quiver(mean[0], mean[1], eigenvectors[0,1]*np.sqrt(eigenvalues[1]), eigenvectors[1,1]*np.sqrt(eigenvalues[1]), angles='xy', scale_units='xy', scale=1, color='b')
plt.xlim([0,1])
plt.ylim([0,1])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()





