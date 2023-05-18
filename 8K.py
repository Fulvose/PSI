hyperparameters = [[0.02, 1.0],
                  [0.1, 1.0],
                  [1, 1.0],
                  [10, 1.0],
                  [100, 1.0],
                  [100, 10],
                  [100, 100]]
for hp in hyperparameters:
    clf = SVC(gamma=hp[0], C=hp[1], kernel='rbf')
    clf.fit(X, y)
    y_hat = clf.predict(X)
    plot_decision_regions(X, y_hat, clf)
    plt.show()
    
  from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0, cluster_std=1.25)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

hyperparameters = [[0.02, 1.0],
                  [0.1, 1.0],
                  [1, 1.0],
                  [10, 1.0],
                  [50, 1.0],
                  [100, 10],
                  [100, 100]]
for hp in hyperparameters:
    clf = SVC(gamma=hp[0], C=hp[1], kernel='rbf')
    clf.fit(X, y)
    y_hat = clf.predict(X)
    plot_decision_regions(X, y_hat, clf)
    plt.show()
