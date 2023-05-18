poly_kernel_svm_clf = SVC(kernel="poly", degree=1, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=2, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0, cluster_std=1.25)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=1, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=2, coef0=1, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=4, coef0=1, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()


poly_kernel_svm_clf = SVC(kernel="poly", degree=4, coef0=2, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()

poly_kernel_svm_clf = SVC(kernel="poly", degree=5, coef0=2, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()
