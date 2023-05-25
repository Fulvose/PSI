from xgboost import XGBClassifier

xgb_clf = XGBClassifier  (n_estimators=10, learning_rate=1, random_state=42)
xgb_clf.fit(X_train, y_train)
plot_decision_regions(X_train, y_train, xgb_clf)
plt.show()

xgb_clf = XGBClassifier  (n_estimators=2, learning_rate=0.5, random_state=42)
xgb_clf.fit(X_train, y_train)
plot_decision_regions(X_train, y_train, xgb_clf)
plt.show()

xgb_clf = XGBClassifier  (n_estimators=2, learning_rate=1, random_state=42)
xgb_clf.fit(X_train, y_train)
plot_decision_regions(X_train, y_train, xgb_clf)
plt.show()

xgb_clf = XGBClassifier  (n_estimators=10, learning_rate=0.5, random_state=42)
xgb_clf.fit(X_train, y_train)
plot_decision_regions(X_train, y_train, xgb_clf)
plt.show()