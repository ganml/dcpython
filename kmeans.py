# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:14:56 2025

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

def kmeans(X, k=3, maxit=100):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    ind = np.random.choice(n, k, replace=False)
    clusterCenters = X[ind,:]
    dm = np.zeros((n,k))
    for i in range(k):
        dm[:,i] = np.sum(np.square(X-clusterCenters[i,:]), axis=1)
    clusterMembership = np.argmin(dm, axis=1)
    numIter = 1
    while numIter < maxit:
        # update cluster centers
        for i in range(k):
            bInd = clusterMembership==i
            if np.any(bInd):
                clusterCenters[i,:] = np.mean(X[bInd,:], axis=0)
            else:
                clusterCenters[i,:] = X[np.random.randint(0, n),:]
        # update cluster membership
        clusterMembership_ = clusterMembership.copy()
        for i in range(k):
            dm[:,i] = np.sum(np.square(X-clusterCenters[i,:]), axis=1)
        clusterMembership = np.argmin(dm, axis=1)
        nChanges = np.count_nonzero(clusterMembership - clusterMembership_)
        numIter += 1
        if nChanges == 0:
            break;
    objectiveValue = np.sum(dm[list(range(n)), clusterMembership]).item()
    return clusterMembership, clusterCenters, objectiveValue, numIter
    
def kmeans2(X, k=3, numrun=10, maxit=100):
    bestCM, bestCC, bestOV, bestIters = kmeans(X, k, maxit)
    print([bestOV, bestIters])
    for i in range(numrun-1):
        cm, cc, ov, iters = kmeans(X, k, maxit)
        print([ov, iters])
        if ov < bestOV:
            bestCM, bestCC, bestOV, bestIters = cm, cc, ov, iters
    return bestCM, bestCC, bestOV, bestIters
        
# examples

# synthetic data
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

yhat, cc, ov, iters = kmeans(X, 3)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)

bcm, bcc, bov, biters = kmeans2(X)
cm2 = createCM(y, bcm)
print([bov, biters])
print(cm2)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(3):
    members = bcm == i
    center = bcc[i,:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("kmeans1.pdf", bbox_inches='tight')

# iris data
iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

bcm, bcc, bov, biters = kmeans2(X, numrun=20)
cm1 = createCM(y, bcm)
print(cm1)
print([bov, biters])

bcm, bcc, bov, biters = kmeans2(X, k=6, numrun=20)
cm1 = createCM(y, bcm)
print(cm1)
print([bov, biters])

# kmeans from scikit-learn
from sklearn.cluster import KMeans
res = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(X)
cm2 = createCM(y, res.labels_)
print(cm2)
print([res.inertia_, res.n_iter_])

print(createCM(bcm, res.labels_))
