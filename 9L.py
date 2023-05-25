from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

model = GradientBoostingClassifier()
parameters = {"learning_rate": sp_randFloat(),
                  "subsample"    : sp_randFloat(),
                  "n_estimators" : sp_randInt(100, 1000),
                  "max_depth"    : sp_randInt(4, 10)
                 }


randm = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                               cv = 2, n_iter = 10, n_jobs=-1)
randm.fit(X_train, y_train)

print(" Results from Random Search " )
print(" The best estimator across ALL searched params:", randm.best_estimator_)
print("The best score across ALL searched params:", randm.best_score_)
print("The best parameters across ALL searched params:", randm.best_params_)

from sklearn.linear_model import LogisticRegression
grid_values = {'penalty': ['l1', 'l2'],'C':[0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
clf = LogisticRegression()
grid = GridSearchCV(clf, grid_values, cv = 10, scoring='accuracy')
grid.fit(X_train, y_train) 
print(grid.best_params_) 