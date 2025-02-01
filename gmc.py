# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:08:00 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def gmc(X, k=3, tol=1e-8, maxit=100):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    mixingProp = np.ones(k) / k
    ind = np.random.choice(n, k, replace=False)
    clusterCenters = X[ind,:]
    varianceMatrices = [np.diag(np.var(X, axis=0))] * k
    density = np.zeros((n, k))
    for i in range(k):
        density[:,i] = mixingProp[i]*multivariate_normal(mean=clusterCenters[i,:], cov=varianceMatrices[i]).pdf(X)
    conditionalProb = density / np.sum(density, axis=1, keepdims=True)
    logLikelihood = np.sum(np.log(np.sum(density, axis=1))).item()
    numIter = 1
    while numIter < maxit:
        # M-step
        dv = np.sum(conditionalProb, axis=0)
        mixingProp = dv / np.sum(dv)
        for i in range(k):
            clusterCenters[i,:] = np.sum(X*conditionalProb[:,i].reshape((n,1)), axis=0)/np.sum(conditionalProb[:,i])
            Xc = X - clusterCenters[i,:]
            varianceMatrices[i] = np.matmul(np.transpose(Xc), Xc * conditionalProb[:,i].reshape((n,1)))/np.sum(conditionalProb[:,i])
        # E-step
        density = np.zeros((n, k))
        for i in range(k):
            density[:,i] = mixingProp[i]*multivariate_normal(mean=clusterCenters[i,:], cov=varianceMatrices[i]).pdf(X)
        conditionalProb = density / np.sum(density, axis=1, keepdims=True)
        logLikelihood_ = logLikelihood
        logLikelihood = np.sum(np.log(np.sum(density, axis=1))).item()
        numIter += 1
        if np.abs(logLikelihood - logLikelihood_) < tol:
            break;
    return conditionalProb, clusterCenters, mixingProp, logLikelihood, numIter

def gmc2(X, k=3, numrun=10, maxit=100):
    bestCP, bestCC, bestMP, bestLL, bestIters = gmc(X, k, maxit)
    vLL = np.zeros(numrun)
    vLL[0] = bestLL
    for i in range(numrun-1):
        cp, cc, mp, ll, iters = gmc(X, k, maxit)
        vLL[i+1] =ll
        if ll > bestLL:
            bestCP, bestCC, bestMP, bestLL, bestIters = cp, cc, mp, ll, iters
    return bestCP, bestCC, bestMP, bestLL, bestIters, vLL

# examples

# synthetic data
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

bcp, bcc, bmp, bll, biters, vLL = gmc2(X, k=3, numrun=100)
yhat = np.argmax(bcp, axis=1)
cm1 = createCM(y, yhat)
print(cm1)
print([bll, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vLL)), vLL, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("log-likelihood")
fig.savefig("300ll.pdf", bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(3):
    members = yhat == i
    center = bcc[i,:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("gmc1.pdf", bbox_inches='tight')

# iris data
iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

bcp, bcc, bmp, bll, biters, vLL = gmc2(X, k=3, numrun=100)
yhat = np.argmax(bcp, axis=1)
cm1 = createCM(y, yhat)
print(cm1)
print([bll, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vLL)), vLL, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("log-likelihood")
fig.savefig("irisll.pdf", bbox_inches='tight')
