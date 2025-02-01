# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:08:58 2025

@author: gjgan
"""
import numpy as np
import pandas as pd

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