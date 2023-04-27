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


n=1000
x1=stats.norm(0,1).rvs(n)
x2=stats.norm(0,1).rvs(n)
x4=stats.norm(0,1).rvs(n)
x6=stats.norm(0,1).rvs(n)
x7=stats.norm(0,1).rvs(n)
x9=stats.norm(0,1).rvs(n)
x10=stats.norm(0,1).rvs(n)
X=np.stack((x1,x2, 2*x1, x4,3*x1,x6,x7,-2*x7,x9,x10,x9**2,1/x9 ),1)

df=pd.DataFrame(X)
sns.pairplot(df)
plt.show()


boston = datasets.load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
sns.pairplot(data)
sns.heatmap(data.corr(), annot=True)