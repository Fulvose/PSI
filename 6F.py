#1
models = []
predicts = []
names=[]
for i in [ 1, 10000, 0.0001 ]:
    models.append((f'alpha {i}', make_pipeline(PolynomialFeatures(20), Lasso(alpha=i)) ))

x_plot = np.vstack(np.linspace(-3, 3, 1000))
for name, model in models:
    print(name)
    model.fit(x, y)
    predicts.append(model.predict(x_plot))
    names.append(name)
    
x_plot = np.vstack(np.linspace(-3, 3, 1000))
plt.plot(x, y, 'ok');
for i in range(len(models)):
    #print(i)
    plt.plot(x_plot, predicts[i],linewidth=3,label=names[i])
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
plt.legend()    
plt.show()    

#2
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Lasso()),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7,20]},
                    cv=5,
                    refit=False)
grid.fit(x, y)
grid.best_params_
grid = GridSearchCV(Lasso(), param_grid={'alpha': [0.00001, 0.001, 0.0002, 0.01,0.1, 1, 2, 3, 10, 1000, 100000]}, cv=5)
grid.fit(x, y)
grid.best_estimator_.alpha

#3
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Lasso(alpha=1)),
                    param_grid={'polynomialfeatures__degree': [i for i in range(1,15,1)],
                    'lasso__alpha' : [10**i for i in range(-5,5,1)]},
                    refit=True)
grid.fit(X_train,y_train)
print(grid.best_params_)
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, grid.predict(X_test)) ))
print("r2 score: {}".format( metrics.r2_score(y_test, grid.predict(X_test)) ))