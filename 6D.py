#1
from sklearn.model_selection import train_test_split
from sklearn import  metrics
x = stats.uniform(0,3).rvs(100)
f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
y = f(x) + stats.norm(0,0.3).rvs(len(x))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
X_train=np.vstack(X_train)
X_test=np.vstack(X_test)
model1 = make_pipeline(PolynomialFeatures(1), linear_model.LinearRegression())
model1.fit(X_train, y_train)
model1.predict(X_test)

print("model1")
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model1.predict(X_test)) ))

model3 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model3.fit(X_train, y_train)
model3.predict(X_test)
print("model3")
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model3.predict(X_test)) ))

model4 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
model4.fit(X_train, y_train)
model4.predict(X_test)
print("model4")
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model4.predict(X_test)) ))

model5 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
model5.fit(X_train, y_train)
model5.predict(X_test)
print("model5")
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model5.predict(X_test)) ))

model25 = make_pipeline(PolynomialFeatures(25), linear_model.LinearRegression())
model25.fit(X_train, y_train)
model25.predict(X_test)
print("model25")
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, model25.predict(X_test)) ))

#2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge()),
                    param_grid={'polynomialfeatures__degree': [i for i in range(1,10,1)],
                    'ridge__alpha' : [10**i for i in range(-5,5,1)]},
                    refit=True)
grid.fit(X_train,y_train)
print(grid.best_params_)
print("Mean squared error: {}".format( metrics.mean_squared_error(y_test, grid.predict(X_test)) ))