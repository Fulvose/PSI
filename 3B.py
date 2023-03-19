import numpy as np
cov = np.array([[1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
means = np.array([1, 1, 1, 3, 3, 3, 5, 5, 7, 8])
data = np.random.multivariate_normal(mean=means, cov=cov, size=1000)
print(data.shape)


n = 1000
means = np.array([1, 1, 1, 3, 3, 3, 5, 5, 7, 8])
cov = np.eye(len(means))
x = np.random.multivariate_normal(means, cov, n)
x[:, 1] += np.sin(2 * np.pi * x[:, 1] / 3)
x[:, 4] += np.sin(2 * np.pi * x[:, 4] / 5)
sns.heatmap(np.corrcoef(x.T), cmap='coolwarm', annot=True)
plt.show()


boston = datasets.load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
sns.pairplot(data)
sns.heatmap(data.corr(), annot=True)