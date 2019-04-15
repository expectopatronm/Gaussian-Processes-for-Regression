import numpy as np
import math
import matplotlib.pyplot as pl
from scipy.optimize import minimize

N = 30  # number of training points.
M = 50  # number of test points.
sigma_n = 0.18  # noise variance.
sigma_w = 300*sigma_n

# This is the true unknown function we are trying to approximate
def f(x):
    return np.sin(2*np.pi*x).flatten()

# def f(x, mu=-1, sig=1):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))).flatten()

# f = lambda x: (0.25*(x**2)).flatten()

def periodic_kernel(x, xprime, sigma=1, length_scale=0.31622, period=2):
    m = np.subtract(x,xprime.T)
    return sigma**2 * np.exp(-(2*np.sin((np.pi * m)/period)**2)/ (length_scale**2))

# GP squared exponential rbf_kernel
def rbf_kernel(a, b, sigma_f=1, l = 0.31622776601):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return sigma_f**2 * np.exp(-.5 * (1/l**2) * sqdist)

def rq_kernel(a, b, sigma_f=1, l = 0.31622776601, alpha = 1.0):
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return sigma_f**2 * (1 + sqdist * 0.5 * (1/l**2))**(-alpha)

# Negative log-likelihood for training data
def log_likelihood(X_train, Y_train, noise):
    def step(theta):
        K = rbf_kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        return np.sum(np.log(np.diagonal(np.linalg.cholesky(K)))) + \
               0.5 * Y_train.T.dot(np.linalg.inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
    return step

# Sample some input points and noisy versions of the function evaluated at these points.
X_train = np.random.uniform(-5, 5, size=(N,1))
noise = sigma_n*np.random.randn(N)
y_train = f(X_train) + noise

# Apply the rbf_kernel function to our training points
K = periodic_kernel(X_train, X_train)
L = np.linalg.cholesky(K + sigma_n*np.eye(N))

# points we're going to make predictions at.
X_test = np.linspace(-5, 5, M).reshape(-1,1)
noise = sigma_n*np.random.randn(M)
y_test = f(X_test) + noise

# compute the mean at our test points.
Lk = np.linalg.solve(L, periodic_kernel(X_train, X_test))
mu = np.dot(Lk.T, np.linalg.solve(L, y_train))

# compute the variance at our test points.
K_test = periodic_kernel(X_test, X_test)
s2 = np.diag(K_test) - np.sum(Lk**2, axis=0)
sigma = np.sqrt(s2)

# res = minimize(log_likelihood(X_train, y_train, sigma_n), [1, 1],
#                bounds=((1e-5, None), (1e-5, None)),
#                method='L-BFGS-B')
#
# l_opt, sigma_f_opt = res.x
#
# # compute the mean at our test points.
# Lk = np.linalg.solve(L, rbf_kernel(X_train, X_test, sigma_f=sigma_f_opt,l=l_opt))
# mu = np.dot(Lk.T, np.linalg.solve(L, y_train))
#
# # compute the variance at our test points.
# K_test = rbf_kernel(X_test, X_test, sigma_f=sigma_f_opt,l=l_opt)
# s2 = np.diag(K_test) - np.sum(Lk**2, axis=0)
# sigma = np.sqrt(s2)

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X_train, y_train, 'r+', ms=20, label='Samples')
pl.plot(X_test, f(X_test), 'b-')
# pl.gca().fill_between(X_test.flat, mu-2*sigma, mu+2*sigma, color="#dddddd")
pl.plot(X_test, mu, 'r--', lw=2,  label='Mean')
pl.title('Samples and their Mean predictions')
pl.legend()
pl.show()

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_test + 1/sigma_w*np.eye(M))
f_prior = np.dot(L, np.random.normal(size=(M,3)))
pl.figure(2)
pl.clf()
pl.plot(X_test, f_prior)
# pl.gca().fill_between(X_test.flat, 2, -2, color="#dddddd")
pl.title('Three samples from the GP prior')
pl.show()

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_test + 1/sigma_w*np.eye(M) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(M,3)))
pl.figure(3)
pl.clf()
# pl.plot(X_test, y_test, 'r+', ms=20, label='Samples')
pl.plot(X_test, f_post)
# pl.gca().fill_between(X_test.flat, mu-2*sigma, mu+2*sigma, color="#dddddd")
pl.plot(X_test, mu, 'r--', lw=2,  label='Mean')
pl.title('Three samples from the GP posterior')
pl.legend()
pl.show()
