# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:22:56 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy import stats
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def estBeta(Xn, Xc):
    numVar = np.mean(np.var(Xn, axis=0))
    vv = np.zeros(Xc.shape[1])
    for j in range(Xc.shape[1]):
        nv, nc = np.unique(Xi[:,j], return_counts=True)
        vp = nc/np.sum(nc)
        vv[j] = 1 - np.sum(np.square(vp))
    return numVar / np.mean(vv)


def kproto(Xn, Xc, k=3, beta=None, tol=1e-8, maxit=100):
    Xn = np.ascontiguousarray(Xn)
    Xc = np.ascontiguousarray(Xc)
    n, d1 = Xn.shape
    nc, d2 = Xc.shape
    if n != nc:
       raise ValueError("dimension mismatch") 
    if beta is None:
        beta = estBeta(Xn, Xc)
    ind = np.random.choice(n, k, replace=False)
    clusterCentersn = Xn[ind,:]
    clusterCentersc = Xc[ind,:]
    dm = np.zeros((n,k))
    for i in range(k):
        dm[:,i] = np.sum(np.square(Xn-clusterCentersn[i,:]), axis=1) + beta * np.count_nonzero(Xc-clusterCentersc[i,:], axis=1)
    clusterMembership = np.argmin(dm, axis=1)
    objectiveValue = np.sum(dm[list(range(n)), clusterMembership]).item()
    numIter = 1
    while numIter < maxit:
        # update cluster centers
        for i in range(k):
            bInd = clusterMembership==i
            if np.any(bInd):
                clusterCentersn[i,:] = np.mean(Xn[bInd], axis=0)
                clusterCentersc[i,:] = stats.mode(Xc[bInd])[0]
            else:
                clusterCentersn[i,:] = Xn[np.random.randint(0, n),:]
                clusterCentersc[i,:] = Xc[np.random.randint(0, n),:]
        # update cluster membership
        for i in range(k):
            dm[:,i] = np.sum(np.square(Xn-clusterCentersn[i,:]), axis=1) + beta * np.count_nonzero(Xc-clusterCentersc[i,:], axis=1)
        clusterMembership = np.argmin(dm, axis=1)
        objectiveValue_ = objectiveValue
        objectiveValue = np.sum(dm[list(range(n)), clusterMembership]).item()
        numIter += 1
        if np.abs(objectiveValue - objectiveValue_) < tol:
            break;
    return clusterMembership, clusterCentersn, clusterCentersc, objectiveValue, numIter

def kproto2(Xn, Xc, k=3, beta=None, numrun=10, maxit=100):
    bestCM, bestCCn, bestCCc, bestOV, bestIters = kproto(Xn, Xc, k=k, beta=beta, maxit=maxit)
    vOV = np.zeros(numrun)
    vOV[0] = bestOV
    for i in range(numrun-1):
        cp, ccn, ccc, ov, iters = kproto(Xn, Xc, k=k, beta=beta, maxit=maxit)
        vOV[i+1] =ov
        if ov < bestOV:
            bestCM, bestCCn, bestCCc, bestOV, bestIters = cp, ccn, ccc, ov, iters 
    return bestCM, bestCCn, bestCCc, bestOV, bestIters, vOV

# examples

heart_disease = fetch_ucirepo(id=45) 
  
X = heart_disease.data.features 
y = heart_disease.data.targets.copy() 
y[y>0] = 1

print(X.isnull().sum())

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(X)
Xi = imp.transform(X)

varNames = list(X.columns)
numNames = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
numInd = [varNames.index(s) for s in numNames]
catInd = [varNames.index(s) for s in set(varNames).difference(numNames)]
catInd.sort()
        
Xn = Xi[:, numInd]
Xc = Xi[:, catInd].astype(int)

vMin = np.min(Xn, axis=0)
vMax = np.max(Xn, axis=0)
Xn = (Xn-vMin)/(vMax-vMin)

print(np.min(Xn, axis=0))
print(np.max(Xn, axis=0))
print(np.min(Xc, axis=0))
print(np.max(Xc, axis=0))
print(estBeta(Xn, Xc))

# use default parameters
bcm, bccn, bccc, bov, biters, vOV = kproto2(Xn, Xc, k=2, numrun=100)
cm1 = createCM(y, bcm)
print(cm1)
print([bov, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vOV)), vOV, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("Objective function value")
fig.savefig("heartov.pdf", bbox_inches='tight')

# use beta=1
bcm, bccn, bccc, bov, biters, vOV = kproto2(Xn, Xc, k=2, beta=1, numrun=100)
cm1 = createCM(y, bcm)
print(cm1)
print([bov, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vOV)), vOV, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("Objective function value")
fig.savefig("heartov1.pdf", bbox_inches='tight')

# use beta=0
bcm, bccn, bccc, bov, biters, vOV = kproto2(Xn, Xc, k=2, beta=0, numrun=100)
cm1 = createCM(y, bcm)
print(cm1)
print([bov, biters])

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(range(len(vOV)), vOV, color="black")
ax.set_xlabel("Run number")
ax.set_ylabel("Objective function value")
fig.savefig("heartov1.pdf", bbox_inches='tight')
