#1
x=np.vstack(x)
x_plot = np.vstack(np.linspace(0, 10, 1000))
regr_rf = RandomForestRegressor(max_depth=10, max_features=1, n_estimators=10, random_state=3)
y_rbf = regr_rf.fit(x, y)
# Plot outputs
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
plt.plot(x_plot, regr_rf.predict(x_plot), color='blue',linewidth=3)
plt.show()

#2
grid = GridSearchCV(RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True),
                      param_grid={'max_depth': [100, 300, 500, 600],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'n_estimators': [100, 200, 300, 400]},
                      #cv=kfold,
                      refit=True)

grid.fit(x, y)
grid.best_params_

#3
grid = GridSearchCV(RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True),
                      param_grid={'max_depth': [100, 300, 500, 600],
                                  'max_features': ['auto', 'sqrt', 'log2'],
                                  'n_estimators': [100, 200, 300, 400]},
                      #cv=kfold,
                      refit=True)

grid.fit(X, y)
print(grid.best_params_)
grid.best_estimator_