tree_clf1 = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_clf1.fit(X, y)

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, tree_clf1)
plt.show()

from sklearn import tree
tree.plot_tree(tree_clf1)

#2
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
tree_clf2.fit(X, y)

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, tree_clf2)
plt.show()

from sklearn import tree
tree.plot_tree(tree_clf2)

#3
from sklearn import  metrics

X_test, y_test = make_moons(n_samples=20000, noise=.7, random_state=10)

print("precision_score: {}".format(metrics.precision_score(y_test, tree_clf1.predict(X_test)) ))
print("recall_score: {}".format( metrics.recall_score(y_test, tree_clf1.predict(X_test)) ))
print("f1_score: {}".format( metrics.f1_score(y_test, tree_clf1.predict(X_test)) ))
print("accuracy_score: {}".format( metrics.accuracy_score(y_test, tree_clf1.predict(X_test)) ))
print("precision_score: {}".format(metrics.roc_auc_score(y_test, tree_clf1.predict(X_test)) ))

y_pred_proba = tree_clf1.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()