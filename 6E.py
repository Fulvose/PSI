#1
models = []
predicts = []
names=[]
for i in [ 1, 10000, 0.0001 ]:
    models.append((f'alpha {i}', make_pipeline(PolynomialFeatures(20), Ridge(alpha=i)) ))
x_plot = np.vstack(np.linspace(-3, 3, 1000))
for name, model in models:
    print(name)
    model.fit(x, y)
    predicts.append(model.predict(x_plot))
    names.append(name)
x_plot = np.vstack(np.linspace(-3, 3, 1000))
plt.plot(x, y, 'ok');
for i in range(len(models)):
    plt.plot(x_plot, predicts[i],linewidth=3,label=names[i])
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
plt.legend()    
plt.show()  

#2
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge()),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7], 
                                "ridge__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
                    cv=5,
                    refit=False)
grid.fit(x, y)

#3
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7,25]},
                    cv=5,
                    refit=False)
grid.fit(X, y)

grid2 = GridSearchCV(Ridge(), param_grid={'alpha': [0.00001, 0.001, 0.0002, 0.01, 1, 2, 3, 10, 1000, 100000]}, cv=5)
grid2.fit(X, y)
grid2.best_estimator_.alpha

from sklearn import metrics
model = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model.fit(X, y)
metrics.r2_score(y, model.predict(X))