# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:44:09 2025

@author: gjgan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from ucimlrepo import fetch_ucirepo 

def getInd(ind, m):
    ind = np.ascontiguousarray(ind)
    if len(ind) == 2: # convert (i,j) -> h
        return int(m*ind[0] + ind[1] - (ind[0]+1)*(ind[0]+2)/2)
    else: # convert h -> (i, j)
        i = int(np.floor((2*m-1-np.sqrt((2*m-1)**2 - 8*ind[0]))/2))
        j = int((i+1)*(i+2)/2 -m*i+ind[0])
        return (i, j)

def single(dm):
    dm2 = squareform(dm)
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = dist[indMin]
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = np.minimum(dm2[ind, ij[0]], dm2[ind, ij[1]])
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res            

def complete(dm):
    dm2 = squareform(dm)
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = dist[indMin]
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = np.maximum(dm2[ind, ij[0]], dm2[ind, ij[1]])
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res        

def average(dm):
    dm2 = squareform(dm)
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = dist[indMin]
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = (clusterSize[s1]*dm2[ind, ij[0]] + clusterSize[s2]*dm2[ind, ij[1]])/size
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res   

def weighted(dm):
    dm2 = squareform(dm)
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = dist[indMin]
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = (dm2[ind, ij[0]] + dm2[ind, ij[1]])/2
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res  

def centroid(dm):
    dm2 = squareform(np.square(dm))
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = np.sqrt(dist[indMin])
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = (clusterSize[s1]*dm2[ind, ij[0]] + clusterSize[s2]*dm2[ind, ij[1]] - clusterSize[s1]*clusterSize[s2]*dm2[ij[0], ij[1]]/size)/size 
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res  

def median(dm):
    dm2 = squareform(np.square(dm))
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = np.sqrt(dist[indMin])
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        tmp[0:(m-1), m-1] = dm2[ind, ij[0]]/2 + dm2[ind, ij[1]]/2 - dm2[ij[0], ij[1]]/4 
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res  
    
def ward(dm):
    dm2 = squareform(np.square(dm))
    n = dm2.shape[0]
    unmergedClusters = list(range(0,n))
    clusterSize = [1] * n
    res = np.zeros((n-1, 4))
    for s in range(n-1):
        m = len(unmergedClusters)
        R, C = np.triu_indices(m, k=1)
        dist = dm2[R, C]  
        indMin = np.argmin(dist)
        dMin = np.sqrt(dist[indMin])
        ij = getInd(indMin, m)
        s1 = unmergedClusters[ij[0]]
        s2 = unmergedClusters[ij[1]]
        size = clusterSize[s1] + clusterSize[s2]
        res[s,:] = [s1, s2, dMin, size]
        unmergedClusters.remove(s1)
        unmergedClusters.remove(s2)
        unmergedClusters.append(n+s)
        clusterSize.append(size)
        # update distance matrix
        m -= 1
        tmp = np.zeros((m, m))
        ind = list(range(m+1))
        ind.remove(ij[0])
        ind.remove(ij[1])
        ind2 = unmergedClusters[0:-1]
        tmp[0:(m-1), 0:(m-1)] = dm2[np.ix_(ind, ind)]
        size2 = np.array([clusterSize[i] for i in ind2])
        tmp[0:(m-1), m-1] = np.divide( np.multiply(size2+clusterSize[s1], dm2[ind, ij[0]]) + np.multiply(size2+clusterSize[s2], dm2[ind, ij[1]]) - size2 * dm2[ij[0], ij[1]], size2+clusterSize[s1]+clusterSize[s2])
        tmp[m-1, 0:(m-1)] = tmp[0:(m-1), m-1]
        dm2 = tmp
    return res 

# examples
iris = fetch_ucirepo(id=53)   
X = iris.data.features 
dm = pdist(X)

# single linkage
Z1 = single(dm)
Z1b = hierarchy.linkage(dm, 'single')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("single.pdf", bbox_inches='tight')

# complete linkage
Z1 = complete(dm)
Z1b = hierarchy.linkage(dm, 'complete')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("complete.pdf", bbox_inches='tight')

# average linkage
Z1 = average(dm)
Z1b = hierarchy.linkage(dm, 'average')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("average.pdf", bbox_inches='tight')

# weighted average linkage
Z1 = weighted(dm)
Z1b = hierarchy.linkage(dm, 'weighted')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("weighted.pdf", bbox_inches='tight')

# centroid linkage
Z1 = centroid(dm)
Z1b = hierarchy.linkage(dm, 'centroid')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("centroid.pdf", bbox_inches='tight')

# median linkage
Z1 = median(dm)
Z1b = hierarchy.linkage(dm, 'median')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("median.pdf", bbox_inches='tight')

# ward's linkage
Z1 = ward(dm)
Z1b = hierarchy.linkage(dm, 'ward')

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
dn1 = hierarchy.dendrogram(Z1, ax=axes[0], no_labels=True, color_threshold=0, above_threshold_color="black")
dn1b = hierarchy.dendrogram(Z1b, ax=axes[1], no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("ward.pdf", bbox_inches='tight')


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

y = iris.data.targets
ms1 = hierarchy.fcluster(Z1, t=6.5, criterion='distance')
cm1 = createCM(y, ms1)
print(cm1)
