# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 06:24:52 2025

@author: gjgan
"""

import time
import numpy as np
from scipy import stats
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def loss(X, gamma, k):
    dLoss = 0
    nNonempty = 0
    for i in range(k):
        bInd = gamma == i
        if np.any(bInd):
            center = stats.mode(X[bInd])[0]
            dLoss += np.count_nonzero(X-center)
            nNonempty += 1
    return (dLoss, nNonempty/k)
        
def mutation(X, gamma, k, cm, mprob):
    n = X.shape[0]
    dm = np.zeros((n, k))
    for i in range(k):
        bInd = gamma == i
        if np.any(bInd):
            center = stats.mode(X[bInd])[0]
        else:
            center = X[np.random.randint(0, n),:]
        dm[:,i] = np.count_nonzero(X-center, axis=1)
    dm = cm*np.max(dm, axis=1, keepdims=True) - dm
    dm = dm / np.sum(dm, axis=1, keepdims=True)
    ind = np.where( np.random.rand(n) < mprob )[0]
    clusterMembership = gamma
    for i in ind:
        clusterMembership[i] = np.random.choice(k, size=1, p=dm[i,:].flatten()).item()
    return clusterMembership

def kmode(X, gamma, k):
    n = X.shape[0]
    dm = np.zeros((n, k))
    for i in range(k):
        bInd = gamma == i
        if np.any(bInd):
            center = stats.mode(X[bInd])[0]
        else:
            center = X[np.random.randint(0, n),:]
        dm[:,i] = np.count_nonzero(X-center, axis=1)
    return np.argmin(dm, axis=1)

def gkmodes(X, k=3, pop=50, maxgen=100, c=1.5, cm=1.5, mprob=0.2):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    population = np.random.choice(k, pop*n).reshape(pop, n)
    vLoss = np.zeros((pop, 2))     
    vFit = np.zeros(pop)
    for g in range(maxgen):
        # calculate loss, legality ratio
        for i in range(pop):
            vLoss[i,:] = loss(X, population[i,:], k)
        # calculate fitness 
        bLegal = vLoss[:,1] >= 1
        maxL = np.max(vLoss[bLegal,0])
        vFit[bLegal] = c*maxL - vLoss[bLegal,0]
        minF = np.min(vFit[bLegal])
        bNotlegal = np.logical_not(bLegal)
        vFit[bNotlegal] = vLoss[bNotlegal, 1] * minF
        # selection
        vP = vFit / np.sum(vFit)
        ind = np.random.choice(pop, size=pop, p=vP)
        population = population[ind,:]
        # mutation
        for i in range(pop):
            population[i,:] = mutation(X, population[i,:], k, cm, mprob)
        # k-modes operator
        for i in range(pop):
            population[i,:] = kmode(X, population[i,:], k)
    for i in range(pop):
        vLoss[i,:] = loss(X, population[i,:], k)
    return population, vLoss
    
# examples

soybean_small = fetch_ucirepo(id=91) 
  
X = soybean_small.data.features 
y = soybean_small.data.targets 


begt = time.time()
cms, losses = gkmodes(X, k=4)
endt = time.time()
print(endt-begt)

bestInd = np.argmin(losses[:,0])
print(bestInd)
yhat = cms[bestInd,:]
cm1 = createCM(y, yhat)
print(cm1)


begt = time.time()
cms2, losses2 = gkmodes(X, k=3)
endt = time.time()
print(endt-begt)

bestInd2 = np.argmin(losses2[:,0])
print(bestInd2)
yhat2 = cms2[bestInd2,:]
cm1 = createCM(y, yhat2)
print(cm1)
