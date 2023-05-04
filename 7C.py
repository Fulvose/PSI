#1
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [0.001, 0.01, 0.1, 1, 10 ,100, 1000],
                               "gamma": [0.001, 0.01, 0.1, 1, 10 ,100, 1000]})

svr.fit(X, y)
svr.best_params_