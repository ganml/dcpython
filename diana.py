# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 14:25:25 2025

@author: gjgan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from ucimlrepo import fetch_ucirepo 

def split(dm, ms):
    dm2 = dm[np.ix_(ms, ms)]
    dist = np.sum(dm2, axis=0)
    indMax = np.argmax(dist)
    ms1 = [ms[indMax]]
    ms2 = ms.copy()
    ms2.remove(ms[indMax])
    bChanged = True
    while bChanged:
        bChanged = False
        if len(ms2) == 1:
            break
        for s in ms2:
            dist1 = np.sum(dm[s, ms1])/len(ms1)
            dist2 = np.sum(dm[s, ms2])/(len(ms2)-1)
            if dist1 < dist2:
                bChanged = True
                ms1.append(s)
                ms2.remove(s)
                break
    return ms1, ms2

def diana(dm):
    dm2 = squareform(dm)
    n = dm2.shape[0]
    clusterDiameter = [0] * (2*n-1)
    clusterMember = [[]] * (2*n-1)
    clusterDiameter[2*n-2] = np.max(dm)
    clusterMember[2*n-2] = [i for i in range(n)]
    unsplitClusters = [2*n-2]
    res = np.zeros((n-1, 4))
    res[n-2, 2:4] = [clusterDiameter[2*n-2], n]
    for s in range(n-2,-1,-1):
        diam =[clusterDiameter[i] for i in unsplitClusters]
        indMax = np.argmax(diam)
        s0 = unsplitClusters[indMax]
        ms0 = clusterMember[s0]
        ms1, ms2 = split(dm2, ms0)
        if len(ms1) > 1:
            s1 = 2*n - 2 - clusterMember[::-1].index([])
            clusterMember[s1] = ms1
            clusterDiameter[s1] = np.max(dm2[np.ix_(ms1, ms1)])
            res[s1-n, 2:4] = [clusterDiameter[s1], len(ms1)]
            unsplitClusters.append(s1)
        else:
            s1 = ms1[0]
        if len(ms2) > 1:
            s2 = 2*n - 2 - clusterMember[::-1].index([])
            clusterMember[s2] = ms2
            clusterDiameter[s2] = np.max(dm2[np.ix_(ms2, ms2)])
            res[s2-n, 2:4] = [clusterDiameter[s2], len(ms2)]
            unsplitClusters.append(s2)
        else:
            s2 = ms2[0]
        res[s0-n, 0:2] = [s1, s2]
        unsplitClusters.remove(s0)
    return res

# examples
iris = fetch_ucirepo(id=53)   
X = iris.data.features 
dm = pdist(X)

Z = diana(dm)

fig, axes = plt.subplots(1, 1, figsize=(8, 4))
dn1 = hierarchy.dendrogram(Z, ax=axes, no_labels=True, color_threshold=0, above_threshold_color="black")
fig.savefig("diana.pdf", bbox_inches='tight')

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
ms1 = hierarchy.fcluster(Z, t=3.5, criterion='distance')
cm1 = createCM(y, ms1)
print(cm1)
