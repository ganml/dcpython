# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:07:28 2024

@author: gjgan
"""

import numpy as np
import matplotlib.pyplot as plt

# overview

t = np.arange(1, 101)
y = np.cumsum(np.random.standard_normal(100))

# OO style
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(t, y, color="black")  
ax.set_xlabel('t')  
ax.set_ylabel('y') 
ax.set_title("Random walk")  

# Matlab style
plt.figure(figsize=(5, 3))
plt.plot(t, y, color="black") 
plt.xlabel('t')
plt.ylabel('y')
plt.title("Random walk")

plt.savefig("randomwalk.pdf", bbox_inches='tight')

# basic plotting
from ucimlrepo import fetch_ucirepo 
  
auto_mpg = fetch_ucirepo(id=9) 
  
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

# scatter plots
fig1, ax1 = plt.subplots(figsize=(4, 3))
ax1.scatter(X["cylinders"], y, color="black", s=8)  
ax1.set_xlabel('Cylinders')  
ax1.set_ylabel('mpg') 

fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.scatter(X["horsepower"], y, color="black", s=8)  
ax2.set_xlabel('Horse power')  
ax2.set_ylabel('mpg') 

# histograms

from matplotlib.ticker import PercentFormatter

fig3, ax3 = plt.subplots(figsize=(4, 3))
ax3.hist(X["displacement"], bins=50, color="black")
ax3.set_title("Displacement")

fig4, ax4 = plt.subplots(figsize=(4, 3))
ax4.hist(y, bins=50, color="black", density=True)
ax4.set_title("mpg")
ax4.yaxis.set_major_formatter(PercentFormatter(xmax=1))

# box plots
fig5, ax5 = plt.subplots(figsize=(4, 3))
ax5.boxplot(X["displacement"], patch_artist=True, 
                 boxprops=dict(color='black', facecolor='white'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(marker='o', color='black', markersize=5))
ax5.set_title("Displacement")

fig6, ax6 = plt.subplots(figsize=(4, 3))
ax6.boxplot(y, patch_artist=True, 
                 boxprops=dict(color='black', facecolor='white'),
                 whiskerprops=dict(color='black'),
                 capprops=dict(color='black'),
                 medianprops=dict(color='black'),
                 flierprops=dict(marker='o', color='black', markersize=5))
ax6.set_title("mpg")

# sub plots

iris = fetch_ucirepo(id=53) 
  
X = iris.data.features 
y = iris.data.targets 

varnames = list(X.columns)
fig7, ax7 = plt.subplots(3,2, figsize=(6,9))
fig7.tight_layout()
nCount = 0;
for i in range(3):
    for j in range(i+1, 4):
        rowInd = nCount // 2
        colInd = nCount - 2*rowInd
        nCount += 1
        ax7[rowInd][colInd].scatter(X[varnames[i]], X[varnames[j]], color='black', s=5)
        ax7[rowInd][colInd].set_xlabel(varnames[i])
        ax7[rowInd][colInd].set_ylabel(varnames[j])

# file io

fig1.savefig("cylinders.pdf", bbox_inches='tight')
fig2.savefig("horsepower.pdf", bbox_inches='tight')
fig3.savefig("histdisplacement.pdf", bbox_inches='tight')
fig4.savefig("histmpg.pdf", bbox_inches='tight')
fig5.savefig("boxplotdisplacement.pdf", bbox_inches='tight')
fig6.savefig("boxplotmpg.pdf", bbox_inches='tight')
fig7.savefig("iris.pdf", bbox_inches='tight')
