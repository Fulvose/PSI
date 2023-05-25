from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, random_state=5) 
svm = SVC()
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print("Scaled test set accuracy: {:.2f}".format( svm.score(X_test_scaled, y_test)))