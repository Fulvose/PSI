from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [1e-5,0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [1e-5,0.001, 0.01, 0.1, 1, 10, 100]}

grid_1 = GridSearchCV(SVC(), param_grid, cv=kfold, return_train_score=True)

grid_1.fit(X_train, y_train)
grid_1.best_params_

grid_2 = GridSearchCV(SVC(kernel='linear'), param_grid, cv=kfold, return_train_score=True)

grid_2.fit(X_train, y_train)
grid_2.best_params_

param_grid = {'C': [1, 10]} 
grid_4 = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, return_train_score=True)

grid_4.fit(X_train, y_train)
grid_4.best_params_