#1
import seaborn as sns

df_adv = pd.read_csv('Advertising.csv', index_col=0)
sns.pairplot(df_adv)
plt.show()

#2
df_adv = pd.read_csv('Advertising.csv', index_col=0)
sns.heatmap(df_adv.corr(), annot=True, cmap='red')
plt.show()