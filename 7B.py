#1
grid_2 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), Lasso(alpha=1, tol=0.1)),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                      'lasso__alpha': [1., 2., 3.]},
                      scoring='r2',
                      cv=kfold, 
                      n_jobs=-1)
scores_2 = cross_val_score(grid_2, X, y, scoring='r2', cv=5)
print('CV Lasso R2: %.3f +/- %.3f' % (np.mean(scores_2), np.std(scores_2)))
mean_r2.append(np.mean(scores_2))
var_r2.append(np.std(scores_2))

grid_3 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1, tol=0.1)),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4],
                      'ridge__alpha': [1., 2., 3.]},
                      scoring='r2',
                      cv=kfold, 
                      n_jobs=-1)
scores_3 = cross_val_score(grid_3, X, y, scoring='r2', cv=5)
print('CV Ridge R2: %.3f +/- %.3f' % (np.mean(scores_3), np.std(scores_3)))
mean_r2.append(np.mean(scores_3))
var_r2.append(np.std(scores_3))

grid_4 = GridSearchCV(estimator=make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                      param_grid={'polynomialfeatures__degree': [1, 2, 3, 4]},
                      scoring='r2',
                      cv=kfold, 
                      n_jobs=-1)
scores_4 = cross_val_score(grid_4, X, y, scoring='r2', cv=5)
print('CV LR R2: %.3f +/- %.3f' % (np.mean(scores_4), np.std(scores_4)))
mean_r2.append(np.mean(scores_4))
var_r2.append(np.std(scores_4))\


#2
import pandas as pd
d = {'mean r2': mean_r2, 
     'var r2': var_r2, 
    }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=['ElasticNet','Lasso','Ridge','LR'])
df