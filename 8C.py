#1
log_reg.predict_proba(X)

#2
log_reg.intercept_
log_reg.coef_

#3
log_reg.predict_proba(X)

#4
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_reg.intercept_
log_reg.coef_
log_reg.predict(X_train)
log_reg.predict_proba(X_train)
accuracy_score(log_reg.predict(X_train),y_train)