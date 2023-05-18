from sklearn.svm import SVC

#1

c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

for current_c in c_vals:
    svc = SVC(kernel="linear", C=current_c)
    svc.fit(X, y)
    plot_decision_regions(X,y,svc)
    plt.show()

#2
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0, cluster_std=1.5)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

#3
from sklearn.svm import SVC

c_vals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

for current_c in c_vals:
    svc = SVC(kernel="linear", C=current_c)
    svc.fit(X, y)
    plot_decision_regions(X,y,svc)
    plt.show()