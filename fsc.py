# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:38:19 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM
from kmeans import kmeans2

def fsc(X, k=3, alpha=2, tol=1e-6, maxit=100):
    if alpha <= 1:
        raise ValueError("Invalid alpha")
    X = np.ascontiguousarray(X)
    n, d = X.shape
    epsilon = 1e-8
    ind = np.random.choice(n, k, replace=False)
    clusterCenters = X[ind,:]
    featureWeight = np.ones((k, d)) / d
    dm = np.zeros((n, k))
    for i in range(k):
        dm[:,i] = np.sum(np.multiply(np.square(X-clusterCenters[i,:]), d**(-alpha)), axis=1)
    clusterMembership = np.argmin(dm, axis=1)     
    objectiveValue = np.sum(dm[list(range(n)), clusterMembership]).item()
    numIter = 1
    while numIter < maxit:
        # update feature weight
        for i in range(k):
            bInd = clusterMembership==i
            if np.any(bInd):
                dv = np.pow(np.sum(np.square(X[bInd,:]-clusterCenters[i,:])+epsilon, axis=0), -1/(alpha-1))
                featureWeight[i,:] = dv / np.sum(dv)
            else:
                featureWeight[i,:] = 1/d
        # update cluster centers
        for i in range(k):
            bInd = clusterMembership==i
            if np.any(bInd):
                clusterCenters[i,:] = np.mean(X[bInd], axis=0)
            else:
                clusterCenters[i,:] = X[np.random.randint(0, n),:]
        # update cluster membership
        for i in range(k):
            dm[:,i] = np.sum(np.multiply(np.square(X-clusterCenters[i,:]), featureWeight[clusterMembership,:]**alpha), axis=1)
        clusterMembership = np.argmin(dm, axis=1)     
        objectiveValue_ = objectiveValue
        objectiveValue = np.sum(dm[list(range(n)), clusterMembership]).item()
        numIter += 1
        if np.abs(objectiveValue-objectiveValue_) < tol:
            break;    
    return clusterMembership, clusterCenters, featureWeight, objectiveValue, numIter

def fsc2(X, k=3, alpha=2, numrun=10, maxit=100):
    bestCM, bestCC, bestFW, bestOV, bestIters = fsc(X, k=k, alpha=alpha, maxit=maxit)
    print([bestOV, bestIters])
    for i in range(numrun-1):
        cm, cc, fw, ov, iters = fsc(X, k=k, alpha=alpha, maxit=maxit)
        print([ov, iters])
        if ov < bestOV:
            bestCM, bestCC, bestFW, bestOV, bestIters = cm, cc, fw, ov, iters
    return bestCM, bestCC, bestFW, bestOV, bestIters

# examples

# synthetic data
np.random.seed(1)
X = np.zeros((300, 3))
y = np.zeros(300, dtype=int)
for i in range(3):
    ind = [100*i+j for j in range(100)]
    X[ind,:] = np.random.rand(100,3)*2 + 4*i
    X[ind,i] = np.random.rand(100)*12
    y[ind] = i
ind = np.random.permutation(list(range(300)))
X = X[ind,:]
y = y[ind]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1],X[:,2], color="k", s=12)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig("3d.pdf", bbox_inches='tight')

yhat, cc, fw, ov, iters = fsc2(X, k=3, alpha=2, numrun=100)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fw, precision=4, suppress_small=True))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
markers = ["s", "+", "."]
for i in range(3):
    ind = yhat==i
    ax.scatter(X[ind,0], X[ind,1],X[ind,2], color="black", s=16, marker=markers[i])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig("3dfsc.pdf", bbox_inches='tight')

yhat, bcc, bov, biters = kmeans2(X)
cm2 = createCM(y, yhat)
print([bov, biters])
print(cm2)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
markers = ["s", "+", "."]
for i in range(3):
    ind = yhat==i
    ax.scatter(X[ind,0], X[ind,1],X[ind,2], color="black", s=16, marker=markers[i])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig("3dkmeans.pdf", bbox_inches='tight')

# iris data

iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

yhat, cc, fw, ov, iters = fsc2(X, k=3, alpha=3)
cm1 = createCM(y, yhat)
print([ov, iters])
print(cm1)
print(np.array_str(fw, precision=4, suppress_small=True))

