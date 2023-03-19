n_samples = 100
data, data_y = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

d = 2
h = (4/(d+2)*n_samples)**(-1/(d+4)) * np.std(data)

kde = st.gaussian_kde(data.T, bw_method=h)


x_min, y_min = data.min(axis=0) - 0.1
x_max, y_max = data.max(axis=0) + 0.1
xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])

density = np.reshape(kde(positions).T, xx.shape)

fig, ax = plt.subplots()
ax.imshow(density, cmap='Blues', origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.scatter(data[:,0], data[:,1], color='black', alpha=0.5)
ax.set_title("Kernel")
plt.show()