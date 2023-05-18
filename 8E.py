from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

clf = LogisticRegression()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)

print(precision_score(y_test, predict))
print(recall_score(y_test, predict))
print(f1_score(y_test, predict))