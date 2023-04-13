#1
M2 = np.vstack( (np.ones_like(x), x, x**2,x**3) ).T
p2 = np.linalg.lstsq(M2, y, rcond=None)

f_lr_2 = lambda x: p2[0][3]*pow(x,3) +p2[0][2]*pow(x,2) + p2[0][1] * x + p2[0][0] 

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_2(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()

#2
M2 = np.vstack( (np.ones_like(x), x, x**2,x**3,x**4) ).T
p2 = np.linalg.lstsq(M2, y, rcond=None)

f_lr_2 = lambda x: p2[0][4]*pow(x,4) +p2[0][3]*pow(x,3) +p2[0][2]*pow(x,2) + p2[0][1] * x + p2[0][0] 

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_2(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()

#3
M2 = np.vstack( (np.ones_like(x), x, x**2,x**3,x**4,x**5) ).T
p2 = np.linalg.lstsq(M2, y, rcond=None)

f_lr_2 = lambda x: p2[0][5]*pow(x,5)+p2[0][4]*pow(x,4) +p2[0][3]*pow(x,3) +p2[0][2]*pow(x,2) + p2[0][1] * x + p2[0][0] 

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_2(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()

#4
import statsmodels.formula.api as smf
df = pd.DataFrame({'x':x, 'y':y})

Res1F = smf.ols('y~x', df).fit()
Res2F = smf.ols('y ~ x+I(x**2)', df).fit()