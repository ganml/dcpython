# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:40:38 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import pairwise_distances
from sklearn.cluster import MeanShift
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def msparabolic(X, sigma=1, tol=1e-8, maxit=300):
    X = np.ascontiguousarray(X)
    Z = X.copy()
    iters = 1
    while iters < maxit:
        W = (pairwise_distances(Z, X) < sigma).astype(float)
        E = np.sum(W, axis=1, keepdims=True)
        Z_ = Z
        Z = (W/E) @ X
        iters += 1
        if np.max(np.abs(Z-Z_)) < tol:
            break    
    n, d = X.shape
    clusterCenter = np.zeros((0,d))
    sind = np.argsort(E[:,0])[::-1]
    unassigned = [index.item() for index in sind if index > 0]  
    while len(unassigned) > 0:
        i = unassigned[0]        
        dist = pairwise_distances(Z[i,:].reshape(1, -1), Z[unassigned,:])
        ind = np.where(dist[0,:] < sigma)[0]
        clusterCenter = np.append(clusterCenter, Z[i,:].reshape(1,-1), axis=0)
        for j in ind:
            sind[sind==unassigned[j]] = -1
        unassigned = [index.item() for index in sind if index > 0]  
    dist = pairwise_distances(X, clusterCenter)
    clusterMembership = np.argmin(dist, axis=1)
    return clusterMembership, clusterCenter, Z, iters

def msgaussian(X, sigma=1, tol=1e-8, maxit=300):
    X = np.ascontiguousarray(X)
    Z = X.copy()
    iters = 1
    while iters < maxit:
        W = np.exp(-0.5*np.square(pairwise_distances(Z, X)/sigma))
        E = np.sum(W, axis=1, keepdims=True)
        Z_ = Z
        Z = (W/E) @ X
        iters += 1
        if np.max(np.abs(Z-Z_)) < tol:
            break
    n, d = X.shape
    clusterCenter = np.zeros((0,d))
    sind = np.argsort(E[:,0])[::-1]
    unassigned = [index.item() for index in sind if index > 0]  
    while len(unassigned) > 0:
        i = unassigned[0]        
        dist = pairwise_distances(Z[i,:].reshape(1, -1), Z[unassigned,:])
        ind = np.where(dist[0,:] < sigma)[0]
        clusterCenter = np.append(clusterCenter, Z[i,:].reshape(1,-1), axis=0)
        for j in ind:
            sind[sind==unassigned[j]] = -1
        unassigned = [index.item() for index in sind if index > 0]  
    dist = pairwise_distances(X, clusterCenter)
    clusterMembership = np.argmin(dist, axis=1)
    return clusterMembership, clusterCenter, Z, iters

# synthetic data 1
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

dm = pdist(X)
sigma = np.percentile(dm, 25)

yhat, cc, Z, iters = msparabolic(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X[:,0], X[:,1], color="white")
for i in range(cc.shape[0]):
    ax.plot(cc[i,0], cc[i,1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=20, markeredgewidth=2)
for i in range(X.shape[0]):
    ax.text(X[i,0], X[i, 1], str(yhat[i]), fontsize=12)
fig.savefig("blobms1.pdf", bbox_inches='tight')

clustering = MeanShift(bandwidth=sigma).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)

yhat, cc, Z, iters = msgaussian(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)



# circles

X, y = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=0)
X = np.round(X, decimals=4)

dm = pdist(X)
sigma = np.percentile(dm, 15)

yhat, cc, Z, iters = msparabolic(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X[:,0], X[:,1], color="white")
for i in range(cc.shape[0]):
    ax.plot(cc[i,0], cc[i,1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=20, markeredgewidth=2)
for i in range(X.shape[0]):
    ax.text(X[i,0], X[i, 1], str(yhat[i]), fontsize=12)
fig.savefig("circlems1.pdf", bbox_inches='tight')

clustering = MeanShift(bandwidth=sigma).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)

yhat, cc, Z, iters = msgaussian(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)
yhat, cc, Z, iters = msgaussian(X, sigma=sigma/2)
cm1 = createCM(y, yhat)
print(cm1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X[:,0], X[:,1], color="white")
for i in range(cc.shape[0]):
    ax.plot(cc[i,0], cc[i,1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=20, markeredgewidth=2)
for i in range(X.shape[0]):
    ax.text(X[i,0], X[i, 1], str(yhat[i]), fontsize=12)
fig.savefig("circlems2.pdf", bbox_inches='tight')

# iris data

iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

dist = pdist(X)
sigma = np.percentile(dist, 15)
yhat, cc, Z, iters = msparabolic(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)

clustering = MeanShift(bandwidth=sigma).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)


yhat, cc, Z, iters = msgaussian(X, sigma=sigma)
cm1 = createCM(y, yhat)
print(cm1)
yhat, cc, Z, iters = msgaussian(X, sigma=sigma/2)
cm1 = createCM(y, yhat)
print(cm1)
