ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), 
    n_estimators=2, learning_rate=1, 
    algorithm="SAMME.R", random_state=42)
ada_clf.fit(X_train, y_train)
plot_decision_regions(X, y, ada_clf)
plt.show()

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), 
    n_estimators=10, learning_rate=0.5, 
    algorithm="SAMME.R", random_state=42)
ada_clf.fit(X_train, y_train)
plot_decision_regions(X, y, ada_clf)
plt.show()

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), 
    n_estimators=10, learning_rate=1, 
    algorithm="SAMME.R", random_state=42)
ada_clf.fit(X_train, y_train)
plot_decision_regions(X, y, ada_clf)
plt.show()