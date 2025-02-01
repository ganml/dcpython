# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:48:13 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_t
from sklearn.datasets import make_blobs
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def kmtd(X, k=3, nu=3, sigma=None, tol=1e-8, maxit=100):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    mixingProp = np.ones(k) / k
    ind = np.random.choice(n, k, replace=False)
    clusterCenters = X[ind,:]
    if sigma is None:
        sigma = np.sqrt(np.mean(np.var(X, axis=0))*(nu-2)/nu)
    density = np.zeros((n, k))
    for i in range(k):
        density[:,i] = mixingProp[i]*multivariate_t(loc=clusterCenters[i,:], shape=sigma**2*np.eye(d), df=nu).pdf(X)
    conditionalProb = density / np.sum(density, axis=1, keepdims=True)
    logLikelihood = np.sum(np.log(np.sum(density, axis=1))).item()
    dm = np.zeros((n,k))
    numIter = 1
    while numIter < maxit:
        # M-step
        dv = np.sum(conditionalProb, axis=0)
        mixingProp = dv / np.sum(dv)
        for i in range(k):
            dm[:,i] = np.sum(np.square(X-clusterCenters[i,:]), axis=1)
        for i in range(k):
            vw = conditionalProb[:,i]/(nu*sigma**2 + dm[:,i])
            clusterCenters[i,:] = np.sum(X*vw.reshape((n,1)), axis=0)/np.sum(vw)
        # E-step
        density = np.zeros((n, k))
        for i in range(k):
            density[:,i] = mixingProp[i]*multivariate_t(loc=clusterCenters[i,:], shape=sigma**2*np.eye(d), df=nu).pdf(X)
        conditionalProb = density / np.sum(density, axis=1, keepdims=True)
        logLikelihood_ = logLikelihood
        logLikelihood = np.sum(np.log(np.sum(density, axis=1))).item()
        numIter += 1
        if np.abs(logLikelihood - logLikelihood_) < tol:
            break;
    return conditionalProb, clusterCenters, mixingProp, logLikelihood, numIter

def kmtd2(X, k=3, nu=3, sigma=None, numrun=10, maxit=100):
    bestCP, bestCC, bestMP, bestLL, bestIters = kmtd(X, k=k, nu=nu, sigma=sigma, maxit=maxit)
    vLL = np.zeros(numrun)
    vLL[0] = bestLL
    for i in range(numrun-1):
        cp, cc, mp, ll, iters = kmtd(X, k=k, nu=nu, sigma=sigma, maxit=maxit)
        vLL[i+1] =ll
        if ll > bestLL:
            bestCP, bestCC, bestMP, bestLL, bestIters = cp, cc, mp, ll, iters
    return bestCP, bestCC, bestMP, bestLL, bestIters, vLL

# examples

# synthetic data
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

bcp, bcc, bmp, bll, biters, vLL = kmtd2(X, k=3, nu=3, numrun=100)
yhat = np.argmax(bcp, axis=1)
cm1 = createCM(y, yhat)
print(cm1)
print([bll, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vLL)), vLL, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("log-likelihood")
fig.savefig("300llkmtd.pdf", bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(3):
    members = yhat == i
    center = bcc[i,:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("kmtd1.pdf", bbox_inches='tight')

# iris data
iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

bcp, bcc, bmp, bll, biters, vLL = kmtd2(X, k=3, nu=3, numrun=100)
yhat = np.argmax(bcp, axis=1)
cm1 = createCM(y, yhat)
print(cm1)
print([bll, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vLL)), vLL, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("log-likelihood")
fig.savefig("irisllkmtd.pdf", bbox_inches='tight')