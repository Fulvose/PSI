#1
# generowanie macierzy kowariancji
cov = np.eye(10)
cov[0,1] = cov[1,0] = cov[1,2] = cov[2,1] = cov[6,7] = cov[7,6] = -0.5
cov[0,2] = cov[2,0] = cov[1,3] = cov[3,1] = cov[3,4] = cov[4,3] = cov[5,6] = cov[6,5] = cov[8,9] = cov[9,8] = 0.0

# generowanie danych
n = 1000
mean = np.zeros(10)
X = np.random.multivariate_normal(mean, cov, n)


#2
# generowanie macierzy kowariancji
cov = np.eye(10)
cov[0,1] = cov[1,0] = cov[1,2] = cov[2,1] = cov[6,7] = cov[7,6] = -0.5
cov[0,2] = cov[2,0] = cov[1,3] = cov[3,1] = cov[3,4] = cov[4,3] = cov[5,6] = cov[6,5] = cov[8,9] = cov[9,8] = 0.0

# generowanie danych
n = 1000
mean = np.zeros(10)
X = np.random.multivariate_normal(mean, cov, n)

# dodawanie skorelowania nieliniowego
X[:,0] = X[:,0]**2
X[:,3] = X[:,3]**2

plt.figure(figsize=(10, 8))
sns.heatmap(np.cov(X.T), annot=True, cmap='coolwarm')
plt.show()
#Na heatmap widać zmianę


#3
boston = datasets.load_boston()
data = boston.data

# wykres sns.pairplot
sns.set(style='ticks')
sns.pairplot(sns.load_dataset("tips"))
plt.show()

# wykres sns.heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Macierz korelacji cech dla boston', fontsize=16)
plt.show()
