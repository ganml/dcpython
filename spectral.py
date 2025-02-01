# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:41:54 2025

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import laplacian
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import SpectralClustering
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM
from kmeans import kmeans2

def epsilonneighbor(dm, epsilon=None):
    n = dm.shape[0]
    if epsilon is None:
        epsilon = np.mean(dm)/5
    S = np.zeros((n,n))
    S[dm < epsilon] = 1
    return S

def gaussian(dm, delta=None):
    if delta is None:
        delta = np.mean(dm)/5
    S = np.exp(-np.square(dm/delta)/2)
    return S

def spectral(S, h=3):
    L = laplacian(S, normed=True)
    ev = np.linalg.eigh(L)
    ind = np.argsort(ev[0])
    U = np.asanyarray(ev[1][:,ind[0:h]], float)
    U = U/np.sqrt(1e-8+np.sum(np.square(U), axis=1, keepdims=True))
    return U
    
# synthetic data 1
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

dm = squareform(pdist(X))

S = epsilonneighbor(dm)
U = spectral(S, h=4)
bcm, bcc, bov, biters = kmeans2(U, k=3)
cm2 = createCM(y, bcm)
print([bov, biters])
print(cm2)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0,0].scatter(X[:,0], X[:,1], color="black")
ax[0,1].scatter(X[:,0], X[:,1], color="black")
ind = np.where(S > 0)
for i in range(ind[0].shape[0]):
    ij = [ind[0][i], ind[1][i]]
    ax[0,1].plot(X[ij,0], X[ij,1], '-', color="grey")
ax[1,0].scatter(U[:,0], U[:,1], color="black")
for i in range(3):
    center = bcc[i,:]
    ax[1,0].plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
markers = ["x", "o", "+"]
for i in range(3):
    members = bcm == i
    ax[1,1].plot(X[members, 0], X[members, 1], markers[i], color="black")
ax[0,0].set_title("D")
ax[0,1].set_title("S")
ax[1,0].set_title("U")
ax[1,1].set_title("CM")
fig.savefig("blob1spectral.png", bbox_inches='tight', dpi=300)


S = gaussian(dm)
U = spectral(S, h=4)
bcm, bcc, bov, biters = kmeans2(U, k=3)
cm2 = createCM(y, bcm)
print([bov, biters])
print(cm2)

clustering = SpectralClustering(n_clusters=3,
        assign_labels='discretize',
        random_state=0).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)

# circles

X, y = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=0)

dm = squareform(pdist(X))

S = epsilonneighbor(dm)
U = spectral(S, h=4)
bcm, bcc, bov, biters = kmeans2(U, k=2)
cm2 = createCM(y, bcm)
print([bov, biters])
print(cm2)

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0,0].scatter(X[:,0], X[:,1], color="black")
ax[0,1].scatter(X[:,0], X[:,1], color="black")
ind = np.where(S > 0)
for i in range(ind[0].shape[0]):
    ij = [ind[0][i], ind[1][i]]
    ax[0,1].plot(X[ij,0], X[ij,1], '-', color="grey")
ax[1,0].scatter(U[:,0], U[:,1], color="black")
for i in range(2):
    center = bcc[i,:]
    ax[1,0].plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
markers = ["x", "o", "+"]
for i in range(2):
    members = bcm == i
    ax[1,1].plot(X[members, 0], X[members, 1], markers[i], color="black")
ax[0,0].set_title("D")
ax[0,1].set_title("S")
ax[1,0].set_title("U")
ax[1,1].set_title("CM")
fig.savefig("circle1spectral.png", bbox_inches='tight', dpi=300)

clustering = SpectralClustering(n_clusters=2,
        assign_labels='discretize', 
        gamma = 20,
        random_state=0).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)


# iris data

iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

dm = squareform(pdist(X))


S = gaussian(dm)
U = spectral(S, h=3)
bcm, bcc, bov, biters = kmeans2(U, k=3)
cm2 = createCM(y, bcm)
print([bov, biters])
print(cm2)

clustering = SpectralClustering(n_clusters=3,
        assign_labels='discretize',
        random_state=0).fit(X)
cm = createCM(y, clustering.labels_)
print(cm)

