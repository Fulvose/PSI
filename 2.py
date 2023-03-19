import numpy as np
import numpy.linalg as alg

#1
A = np.random.rand(100, 10)
distances = np.array([alg.norm(r1 - r2) for r1 in A for r2 in A]).reshape(100, 100)
print(distances)

#2
mean = [0,0,0,0,0]
cov = np.eye(5)
X = np.random.multivariate_normal(mean, cov, 100)
print(np.std(X, axis=0))

np.mean(X, axis=0)
X_new = (X-np.mean(X, axis=0))/np.std(X, axis=0)

print(np.mean(X_new, axis=0))
print(np.std(X_new, axis=0))

#3
arr = np.random.randint(5,16,100)
counts = np.bincount(arr)
print(counts)
maxCount = np.argmax(counts)
print(maxCount, counts[maxCount])

#4
y = np.where(y==0, -1, 1)
#print(y)
X = (X- np.min(X, axis = 0))/(np.max(X, axis=0)- np.min(X, axis=0))
np.min(X, axis=0)

#5
df=pd.read_csv('data/airports.csv', header=None)
df.iloc[-12:, 3]
df.iloc[1,]
df[df.iloc[:,3]=="Poland"]
df[df.iloc[:,3] !==df.iloc[:,2]]
df.iloc[:,5] = df.iloc[:,5] * 0.3048
panstwa = df.iloc[:,5]
unikalne_panstwa = panstwa.unique()
for panstwo in unikalne_panstwa:
    if df[df.iloc[:,5] == panstwo]['lotnisko'].value_counts().iloc[0] == 1:
        print(panstwo)

#6
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

#7
df['HasCabin'] = np.where(df['Cabin'].isnull(), 0, 1)

#8
df.dropna(inplace=True)

#9
import matplotlib.pyplot as plt
%matplotlib inline
mean = 5
q = 2
x = np.linspace(mean-6*q, mean+6*q, 1000)
y = (1/(q*np.sqrt(2*np.pi))) * np.exp(-((x-mean)**2)/(2*q**2))

plt.plot(x, y, color='blue')
plt.xlim([mean-6*q, mean+6*q])

plt.fill_between(x, y, 0, where=((x>mean-3*q) & (x<mean+3*q)), color='lightblue')

area = np.trapz(y, x)

plt.text(mean-5*q, 0.05, 'Pole pod krzywą: {:.2f}'.format(area), fontsize=12)
plt.show()