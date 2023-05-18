#1
from sklearn.linear_model import LogisticRegression
lr_l1 = LogisticRegression(C=1, penalty="l2").fit(X_train, y_train)
print(lr_l1.coef_.T)

#2
from sklearn.linear_model import LogisticRegression
lr_l1 = LogisticRegression(C=100, penalty="l2").fit(X_train, y_train)
print(lr_l1.coef_.T)

#3
from sklearn.linear_model import LogisticRegression
lr_l1 = LogisticRegression(C=0.01, penalty="l2").fit(X_train, y_train)
print(lr_l1.coef_.T)

#4
for C, marker in zip([0.001], ['o-', '^-', 'v-']):
    lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train, y_train)

print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(wine.data.shape[1]), wine.feature_names, rotation=90)
plt.hlines(0, 0, wine.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()

for C, marker in zip([1], ['o-', '^-', 'v-']):
    lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train, y_train)

print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

#5

plt.xticks(range(wine.data.shape[1]), wine.feature_names, rotation=90)
plt.hlines(0, 0, wine.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()

#6
for C, marker in zip([100], ['o-', '^-', 'v-']):
    lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear').fit(X_train, y_train)

print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_train, y_train)))
print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test, y_test)))
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(wine.data.shape[1]), wine.feature_names, rotation=90)
plt.hlines(0, 0, wine.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()