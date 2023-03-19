# Pierwsze zadania
corr = {}
corr['pearson'], _ = stats.pearsonr(x,y)
corr['spearman'], _ = stats.spearmanr(x,y)
corr['kendall'], _ = stats.kendalltau(x,y)

# Korelacja Pearsona
n = 100
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
x, y = np.random.multivariate_normal(mean, cov, n).T
plt.scatter(x, y)
plt.show()
print('Korelacja Pearsona:', pearsonr(x, y)[0])
# Korelacja Spearmana
n = 100
x = np.linspace(0, 1, n)
y = x
plt.scatter(x, y)
plt.show()
print('Korelacja Spearmana:', spearmanr(x, y)[0])
# Korelacja Kendall'a
n = 100
x = np.linspace(0, 1, n)
y = x
plt.scatter(x, y)
plt.show()
print('Korelacja Kendall\'a:', kendalltau(x, y)[0])