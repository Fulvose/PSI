#1
N = 100000
u=stats.uniform()
data = u.rvs(size=N)
mu= np.sum(data)/len(data)
sigma = np.sqrt(np.sum((data-mu)**2) /len(data))

t = np.arange(-2, 2, 0.05)
num_bins = 50
fig, ax = plt.subplots(1, 1)
ax.hist(data, density=True, histtype='stepfilled', alpha=0.5, label='histogram')
ax.legend(loc='best', frameon=False)
ax.plot(t, stats.norm.pdf(t,mu, sigma), 'k-', lw=2, label='a=-1, b=1')
ax.legend()
plt.show()


#2
mu, sigma = stats.norm.fit(data)

t = np.arange(-2, 2, 0.05)
num_bins = 50
fig, ax = plt.subplots(1, 1)
ax.hist(data, density=True, histtype='stepfilled', alpha=0.5, label='histogram')
ax.legend(loc='best', frameon=False)
ax.plot(t, stats.norm.pdf(t,mu, sigma), 'k-', lw=2, label='a=-1, b=1')
ax.legend()
plt.show()


#4
def MLE(x):
  mu, sigma = x
  return np.sum(np.log(Gpdf(data, mu, sigma)))

#5
print(MLE((0, 1)))
print(MLE((0, 2)))
print(MLE((1, 1)))
print(MLE((0.5, 0.2)))


#7
def MLE(x):
    mu, sigma = x
    return -np.sum(np.log(Gpdf(data, mu, np.abs(sigma) )))

x0 = np.asarray((1, 1))  # Initial guess.
res1 = optimize.fmin_cg(MLE, x0)
print(res1)


#9
def gauss_split(x, mu, sigma,tau):
        pivot=np.searchsorted(x,mu)

        left= x[:pivot]
        right=x[pivot:]

        return np.sqrt(2/np.pi) * 1/(1 + tau) *(1/sigma) * np.concatenate([
           np.e ** (-(left-mu)**2/(2 * sigma**2)),
           np.e ** (-(right-mu)**2/(2 * tau**2 * sigma**2))])

t = np.arange(-2, 2, 0.05)
 
mu = 0
sigma = 1
tau = 1

fig, ax = plt.subplots(1, 1)
ax.plot(t, gauss_split(t,mu, sigma, tau), 'g-', label='mu = 0, sigma = 1, tau = 1')
ax.legend(loc='best', frameon=False)
 
mu = 0
sigma = 1
tau = 0.5
 
ax.plot(t, gauss_split(t,mu, sigma, tau), 'r-', label='mu = 0, sigma = 1, tau = 0.5')
 

mu = 0
sigma = 0.5
tau = 1
 
ax.plot(t, gauss_split(t,mu, sigma, tau), 'b-', label='mu = 1, sigma = 0.5, tau = 1')
ax.legend()
plt.show()


#10
def mle_2(x):
    mu, sigma, tau = x
    return np.sum(np.log(gauss_split(x,mu,np.abs(sigma),np.abs(tau))))


#11
x = np.asarray((1, 1, 1))  # Initial guess.
res = optimize.fmin_cg(mle_2, x)
print(res)


