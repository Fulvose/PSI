#1
model1 = linear_model.LinearRegression()
x1=np.vstack(X1)
model1.fit(x1, y1)

model2 = linear_model.LinearRegression()
x2=np.vstack(X2)
model2.fit(x2, y2)

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(x1, y1,  color='black')
plt.plot(x1, model1.predict(x1), color='blue',linewidth=3)
plt.plot(x2, model2.predict(x2), color='red',linewidth=3)

plt.scatter(X1, y1,  color='blue')
plt.scatter(X2, y2,  color='red')
x_tr = np.linspace(0, 1, 200)
plt.show()


#2
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_ylim([-1.5,1.5])
axes.set_xlim([0,1])

model3 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model3.fit(x1, y1)
plt.scatter(x1, y1,  color='blue')
x_plot = np.vstack(np.linspace(0, 10, 1000))
plt.plot(x_plot, model3.predict(x_plot), color='blue',linewidth=3)

model2x2 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model2x2.fit(x2, y2)
plt.scatter(x2, y2,  color='red')
x_plot = np.vstack(np.linspace(0, 10, 1000))
plt.plot(x_plot, model2x2.predict(x_plot), color='red',linewidth=3)
plt.show()