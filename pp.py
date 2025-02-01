# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:06:18 2025

@author: gjgan
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import lil_array
from scipy.sparse.linalg import matrix_power
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from ucimlrepo import fetch_ucirepo 
from dcutil import createCM

def pp(X, delta=None, s=5, maxit=100):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    W = np.zeros((n,n))
    dm = squareform(pdist(X))
    if delta is None:
        delta = np.percentile(np.mean(dm, axis=0), 10)
    V = np.exp(-np.square(dm/delta)/2)
    ind = np.argsort(V)[:, -1:-s-1:-1]
    val = np.take_along_axis(V, ind, axis=-1)
    np.put_along_axis(W, ind, val, axis=-1)
    W = W / np.sum(W, axis=1, keepdims=True)
    numIter = 1
    attractors = set(np.argmax(W, axis=1))
    while numIter < maxit:
        W = np.linalg.matrix_power(W, 2)
        attractors_ = attractors.copy()
        attractors = set(np.argmax(W, axis=1))
        if attractors == attractors_:
            break
    return list(attractors), W

def pps(X, delta=None, s=5, maxit=100):
    X = np.ascontiguousarray(X)
    n, d = X.shape
    W = lil_array((n,n))
    dm = squareform(pdist(X))
    if delta is None:
        delta = np.percentile(np.mean(dm, axis=0), 10)
    V = np.exp(-np.square(dm/delta)/2)
    ind = np.argsort(V)[:, -1:-s-1:-1]
    val = np.take_along_axis(V, ind, axis=-1)
    np.put_along_axis(W, ind, val, axis=-1)
    W = W / np.sum(W, axis=1).reshape((n,1))
    numIter = 1
    attractors = set(np.argmax(W, axis=1))
    while numIter < maxit:
        W = matrix_power(W, 2)
        attractors_ = attractors.copy()
        attractors = set(np.argmax(W, axis=1))
        if attractors == attractors_:
            break
    return list(attractors), W

# examples

# synthetic data 1
centers = [[3, 3], [-3, -3], [3, -3]]
X, y = make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=1)

ci, W = pp(X)
yhat = np.argmax(W, axis=1)
cm1 = createCM(y, yhat)
print(cm1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(3):
    members = yhat == ci[i]
    center = X[ci[i],:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("pp1.pdf", bbox_inches='tight')

# circles

X, y = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=0)
ci, W = pp(X, s=10)
yhat = np.argmax(W, axis=1)
cm1 = createCM(y, yhat)
print(cm1)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.scatter(X[:,0], X[:,1], color="black")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
markers = ["x", "o", "+"]
for i in range(2):
    members = yhat == ci[i]
    center = X[ci[i],:]
    ax.plot(X[members, 0], X[members, 1], markers[i], color="black")
    ax.plot(center[0], center[1], "^", markerfacecolor="white", 
        markeredgecolor="black", markersize=15)
fig.savefig("pp2.pdf", bbox_inches='tight')

# iris data

iris = fetch_ucirepo(id=53)   
X = iris.data.features 
y = iris.data.targets

ci, W = pp(X)
yhat = np.argmax(W, axis=1)
cm1 = createCM(y, yhat)
print(cm1)

ci, W = pp(X, s=10)
yhat = np.argmax(W, axis=1)
cm1 = createCM(y, yhat)
print(cm1)

ci, W = pps(X)
yhat = np.argmax(W, axis=1)
cm1 = createCM(y, yhat)
print(cm1)
print(W.shape)
W
