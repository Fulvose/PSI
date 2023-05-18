#1
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

clf = LogisticRegression()
clf.fit(X_train,y_train)
X_test
y_test_5
predected = clf.predict(X_test)
f1_score(predected,y_test_5,average=None)
print(precision_score(y_test,predected, average=None))
print(recall_score(y_test,predected, average= None))