X= X2_rv.rvs(1000) 
fig, ax = plt.subplots(figsize= (5,5))
ax.scatter(X[:,0],X[:,1])

x, y = np.mgrid[-10:10:.01,-10:10:.01]
ax.contour(x,y,X2_rv.pdf(np.dstack((x,y))))
plt.show()


fig, ax = plt.subplots(figsize= (5,5))
ax.scatter(X[:,0],X[:,1])

x, y = np.mgrid[-10:10:.01,-10:10:.01]
ax.contour(x,y,X2_rv.pdf(np.dstack((x,y))))


val =np.linalg.eigvals(cov)
val, vect =np.linalg.eig(cov)

ax.plot([0,vect[0][0]],[0,vect[1][0]],color="black")
ax.plot([0,vect[0][1]],[0,vect[1][1]],color="black")

print(val)
print(vect)
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





