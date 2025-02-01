# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:02:12 2025

@author: gjgan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from ucimlrepo import fetch_ucirepo 

def createCM(y, yhat):
    y = np.ascontiguousarray(y)
    yhat = np.ascontiguousarray(yhat)
    labelr = np.unique(y)
    labelc = np.unique(yhat)
    nrow = len(labelr)
    ncol = len(labelc)
    cm = pd.DataFrame(np.zeros((nrow, ncol), dtype=np.int32), index=labelr, columns=labelc)
    for i in range(len(y)):
        cm.loc[y[i], yhat[i]] += 1
    return cm

def cmeans(X, k=3, alpha=2, tol=1e-6, maxit=100):
    if alpha <= 1:
        raise ValueError("Invalid alpha")
    X = np.ascontiguousarray(X)
    n, d = X.shape
    epsilon = 1e-8
    ind = np.random.choice(n, k, replace=False)
    clusterCenters = X[ind,:]
    dm = np.zeros((n,k))
    for i in range(k):
        dm[:,i] = np.sum(np.square(X-clusterCenters[i,:]), axis=1)
    dma = np.pow(dm + epsilon, -1/(alpha-1))
    clusterMembership = dma / np.sum(dma, axis=1, keepdims=True)
    objectiveValue = np.sum(np.multiply(np.pow(clusterMembership, alpha), dm))
    numIter = 1
    while numIter < maxit:
        # update cluster centers
        ua = np.pow(clusterMembership, alpha)
        for i in range(k):
            clusterCenters[i,:] = np.sum(X*ua[:,i].reshape((n,1)), axis=0)/np.sum(ua[:,i])
        # update cluster membership
        objectiveValue_ = objectiveValue
        for i in range(k):
            dm[:,i] = np.sum(np.square(X-clusterCenters[i,:]), axis=1)
        dma = np.pow(dm + epsilon, -1/(alpha-1))
        clusterMembership = dma / np.sum(dma, axis=1, keepdims=True)
        objectiveValue = np.sum(np.multiply(np.pow(clusterMembership, alpha), dm))
        numIter += 1
        if np.abs(objectiveValue - objectiveValue_) < tol:
            break;    
    return clusterMembership, clusterCenters, objectiveValue.item(), numIter

def cmeans2(X, k=3, alpha=2, numrun=10, maxit=100):
    bestFM, bestCC, bestOV, bestIters = cmeans(X, k=k, alpha=alpha, maxit=maxit)
    print([bestOV, bestIters])
    for i in range(numrun-1):
        fm, cc, ov, iters = cmeans(X, k=k, alpha=alpha, maxit=maxit)
        print([ov, iters])
        if ov < bestOV:
            bestFM, bestCC, bestOV, bestIters = fm, cc, ov, iters
    return bestFM, bestCC, bestOV, bestIters

# examples

# synthetic data
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

fm, cc, ov, iters = cmeans2(X, k=3)
yhat = np.argmax(fm, axis=1)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fm[::30,:], precision=4, suppress_small=True))

fm, cc, ov, iters = cmeans2(X, k=3, alpha=8)
yhat = np.argmax(fm, axis=1)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fm[::30,:], precision=4, suppress_small=True))


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(3):
    members = yhat == i
    center = cc[i,:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("cmeans1.pdf", bbox_inches='tight')

# iris data

iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

fm, cc, ov, iters = cmeans2(X, k=3)
yhat = np.argmax(fm, axis=1)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fm[::30,:], precision=4, suppress_small=True))

x = [10*i for i in range(fm.shape[0]//10)]
df = pd.DataFrame(fm[x,:], index=x, columns=["C1", "C2", "C3"])
ax = df.plot( 
    kind = 'barh', 
    stacked = True, 
    colormap='Greys',
    title = 'Fuzzy membership', 
    mark_right = True) 
fig = ax.get_figure()
fig.savefig('irisfm2.pdf')

fm, cc, ov, iters = cmeans2(X, k=3, alpha=8)
yhat = np.argmax(fm, axis=1)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fm[::30,:], precision=4, suppress_small=True))

x = [10*i for i in range(fm.shape[0]//10)]
df = pd.DataFrame(fm[x,:], index=x, columns=["C1", "C2", "C3"])
ax = df.plot( 
    kind = 'barh', 
    stacked = True, 
    colormap='Greys',
    title = 'Fuzzy membership', 
    mark_right = True) 
fig = ax.get_figure()
fig.savefig('irisfm8.pdf')
