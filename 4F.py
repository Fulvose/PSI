#3
def compute_error(arguments):
    a,b = arguments
    return np.sum((y-(a*x+b))**2)

#4
optimize.fmin_cg(compute_error,np.array((0, 0)) )


#5
def compute_error_1(arguments):
    a,b = arguments
    return np.sum(np.abs(y-(a*x+b)))

#6
optimize.fmin_cg(compute_error_1,np.array((0, 0)) )

#7
lr = lm.LinearRegression()
lr.fit(x[:, np.newaxis], y);
print(lr.coef_)
print(lr.intercept_)
f_lr = lambda x: lr.coef_ * x +lr.intercept_
x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr, y_tr, '--k');
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()