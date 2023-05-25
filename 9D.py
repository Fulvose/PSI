pipe2 = Pipeline([('preprocessing', preprocess_pipeline), ('classifier', SVC(kernel='rbf'))])

param_grid2 = {
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_2 = GridSearchCV(pipe2, param_grid2, cv=kfold, return_train_score=True)

grid_2.fit(X_train, y_train)
grid_2.best_params_

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
pipe4 = Pipeline([('preprocessing', preprocess_pipeline), ('classifier', LogisticRegression())])

param_grid4 = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none']
}

grid_4 = GridSearchCV(pipe4, param_grid4, cv=kfold, return_train_score=True)

grid_4.fit(X_train, y_train)
grid_4.best_params_