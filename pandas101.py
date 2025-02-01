# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:46:56 2024

@author: gjgan
"""

import pandas as pd
import numpy as np

s1 = pd.Series([3.14, "str", True, np.array([1,2]), np.nan])
print(s1)

s2 = pd.Series([3.14, "str", True, np.array([1,2]), np.nan], index=["a", "b", "c", "d", "e"])
print(s2)

s3 = pd.Series({0: 3.14, 1: "str", 2: True, 3: np.array([1,2]), 4: np.nan})
print(s3)

s4 = pd.Series(range(1, 9))

s1.values
s1.index

print(s1[0]) # get the first element
print(s1[:3]) # get the first three elements
print(s1[0:4:2]) # get elements at indices 0, 2
print(s1[-1:-4:-2]) # get elements at indices -1, -3

print(s2[s2.index[0]]) # get the first element
print(s2[s2.index[:3]]) # get the first three elements
print(s2[s2.index[0:4:2]]) # get elements at index[0], index[2]
print(s2[-1:-4:-2]) # get elements at positions -1, -3
print(s2[s2.index[-1:-4:-2]]) # get elements at index[0], index[2]

print(s2.iloc[0]) # get the first element
print(s2.iloc[:3]) # get the first three elements
print(s2.iloc[0:4:2]) # get elements at indices 0, 2
print(s2.iloc[-1:-4:-2]) # get elements at indices -1, -3

print(s2.loc["d"]) # get the element with index d
print(s2.loc[["a", "b"]]) # get the elements with indices a, b

# data frame
dic = {
       "V1": pd.Series([1, 2, 3], index=["r1", "r2", "r3"]),
       "V2": pd.Series(["str", [0, 1], np.nan], index=["r1", "r2", "r3"])
     }
df = pd.DataFrame(dic)
print(df)

dic2 = {
       "V1": pd.Series([1, 2, 3], index=["r1", "r2", "r3"]),
       "V2": pd.Series(["str", [0, 1], np.nan], index=["r4", "r5", "r6"])
     }
df2 = pd.DataFrame(dic2)
print(df2)

dat = np.array(range(1,9)).reshape((4,2))
rownames = ["r1", "r2", "r3", "r4"]
colnames = ["V1", "V2"]
df3 = pd.DataFrame(dat, index=rownames, columns=colnames)
print(df3)

print(df3.columns)
print(df3.index)

print(df3["V1"]) # get column V1
print(df3[["V1", "V1"]]) # get column V1 twice
print(df3.loc[["r1", "r3"], "V2"]) # get specified rows and columns by explicint indices
print(df3.iloc[[0,2], 1]) # get specified rows and columns by implicit indices


print(df3[df3["V1"] > 3])
print(df3[df3.V1 > 3])


# views and copies
pd.set_option("mode.copy_on_write", False)

m = pd.DataFrame({"a": range(1, 4), "b": range(4, 7)})

print(m)
v1 = m.iloc[0:2, ]
print(v1._is_view)
print(v1._is_copy)
v1.iloc[0,1] = -1
print(m)

pd.set_option("mode.copy_on_write", True)

m = pd.DataFrame({"a": range(1, 4), "b": range(4, 7)})

print(m)
v1 = m.iloc[0:2, ]
print(v1._is_view)
print(v1._is_copy)
v1.iloc[0,1] = -1
print(v1._is_view)
print(v1._is_copy)
print(m)

# data manipulation
from ucimlrepo import fetch_ucirepo 
  
auto_mpg = fetch_ucirepo(id=9) 
  
X = auto_mpg.data.features 
y = auto_mpg.data.targets 
  
print(type(X))
print(X.columns)
print(X.iloc[:,0:4])
print(y)

# display data frame, summary statistics
print(X.describe())
print(X.isna().sum())

# add, drop columns

X1 = X.copy()
print(X1.columns)
X1["mpg"] = y # add a column at the end
print(X1.columns)
X1.pop("mpg") # drop a column
print(X1.columns)
X1.insert(0, "mpg", y) # insert a column before the first column
print(X1.columns)
del X1["mpg"] # delete a column
print(X1.columns)

# create new columns from existing ones

X1 = X.assign(displacement2=np.sqrt(X1["displacement"]))
print(X1.head())

# add, drop rows

n, d = X.shape
r = pd.DataFrame(np.array([X.iloc[np.random.randint(0, n), j] for j in range(d)]).reshape((1,d)), columns=X.columns)

X1 = X._append(r) # add a row
X1.loc[len(X1)] = r.values[0] # add a row
X1 = pd.concat([X1, r]) # add a row
print(X1.index)
X2 = X1.reset_index()
print(X2.index)

X3 = X2.drop([398, 399, 400]) # drop rows
print(X3.tail())

# check, fill, drop missing values
X4 = X[X.isnull().any(axis=1)]
print(X4)

X5 = X.fillna(0)
print(X5.isna().sum())

# group, aggregate data
X["mpg"] = y
X.to_csv("autompg.csv")

dat = pd.read_csv("autompg.csv", index_col=0)
print(dat.head())
print(dat.tail())