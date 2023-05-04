#1
model = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model.fit(x, y)
# Plot outputs
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model.predict(x_plot), color='blue',linewidth=3)
plt.show()

#2
model = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
model.fit(x, y)
# Plot outputs
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model.predict(x_plot), color='blue',linewidth=3)
plt.show()

#3
model = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
model.fit(x, y)
# Plot outputs
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model.predict(x_plot), color='blue',linewidth=3)
plt.show()

#4
from sklearn import metrics
print("model2")
error1 = np.average( np.abs(model2.predict(x) -y) )
print("Mean absolute errors: {}".format(error1))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y, model2.predict(x))))

error2 = np.average( (model2.predict(x) -y) **2 )
print("Mean squared error: {}".format(error2))
print("Mean squared error: {}".format( metrics.mean_squared_error(y, model2.predict(x)) ))

error3 = np.median( np.abs(model2.predict(x) -y) )
print("Median absolute error: {}".format( error3 ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y, model2.predict(x)) ))

print("R^2: {}".format(metrics.r2_score(y, model2.predict(x))))
ss_res=np.sum( (y-model2.predict(x))**2 )
ss_tot=np.sum( (y-np.mean(y))**2 )
R=1-ss_res/ss_tot
print("R^2: {}".format(R))

error4 = 1-np.var(y - model2.predict(x) )/np.var(y)
print("Explained variance score: {}".format( error4 ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y, model2.predict(x)) ))

print("model3")
error1 = np.average( np.abs(model3.predict(x) -y) )
print("Mean absolute errors: {}".format(error1))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y, model3.predict(x))))

error2 = np.average( (model3.predict(x) -y) **2 )
print("Mean squared error: {}".format(error2))
print("Mean squared error: {}".format( metrics.mean_squared_error(y, model3.predict(x)) ))

error3 = np.median( np.abs(model3.predict(x) -y) )
print("Median absolute error: {}".format( error3 ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y, model3.predict(x)) ))

print("R^2: {}".format(metrics.r2_score(y, model3.predict(x))))
ss_res=np.sum( (y-model3.predict(x))**2 )
ss_tot=np.sum( (y-np.mean(y))**2 )
R=1-ss_res/ss_tot
print("R^2: {}".format(R))

error4 = 1-np.var(y - model3.predict(x) )/np.var(y)
print("Explained variance score: {}".format( error4 ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y, model3.predict(x)) ))


print("model4")
error1 = np.average( np.abs(model4.predict(x) -y) )
print("Mean absolute errors: {}".format(error1))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y, model4.predict(x))))

error2 = np.average( (model4.predict(x) -y) **2 )
print("Mean squared error: {}".format(error2))
print("Mean squared error: {}".format( metrics.mean_squared_error(y, model4.predict(x)) ))

error3 = np.median( np.abs(model4.predict(x) -y) )
print("Median absolute error: {}".format( error3 ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y, model4.predict(x)) ))

print("R^2: {}".format(metrics.r2_score(y, model4.predict(x))))
ss_res=np.sum( (y-model4.predict(x))**2 )
ss_tot=np.sum( (y-np.mean(y))**2 )
R=1-ss_res/ss_tot
print("R^2: {}".format(R))

error4 = 1-np.var(y - model4.predict(x) )/np.var(y)
print("Explained variance score: {}".format( error4 ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y, model4.predict(x)) ))


print("model5")
error1 = np.average( np.abs(model5.predict(x) -y) )
print("Mean absolute errors: {}".format(error1))
print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y, model5.predict(x))))

error2 = np.average( (model5.predict(x) -y) **2 )
print("Mean squared error: {}".format(error2))
print("Mean squared error: {}".format( metrics.mean_squared_error(y, model5.predict(x)) ))

error3 = np.median( np.abs(model5.predict(x) -y) )
print("Median absolute error: {}".format( error3 ))
print("Median absolute error: {}".format( metrics.median_absolute_error(y, model5.predict(x)) ))

print("R^2: {}".format(metrics.r2_score(y, model5.predict(x))))
ss_res=np.sum( (y-model5.predict(x))**2 )
ss_tot=np.sum( (y-np.mean(y))**2 )
R=1-ss_res/ss_tot
print("R^2: {}".format(R))

error4 = 1-np.var(y - model5.predict(x) )/np.var(y)
print("Explained variance score: {}".format( error4 ))
print("Explained variance score: {}".format( metrics.explained_variance_score(y, model5.predict(x)) ))