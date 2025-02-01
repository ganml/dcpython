# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:03:04 2024

@author: gjgan
"""

import numpy as np

# vector
v1 = np.array([1, 2, 3])
v2 = np.array([[1, 2, 3]])
v3 = np.array([[1], [2], [3]])

print(v1.shape)
print(v2.shape)
print(v3.shape)

# 2-d array
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array(range(1, 5)).reshape((2,2))

print(m1)
print(m2)

# high dimensional array
t1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
t2 = np.array(range(1, 9)).reshape((2, 2, 2))

print(t1)
print(t1.shape)
print(t2)
print(t2.shape)

print(t1.flatten())

s1 = np.zeros((2, 2))
s2 = np.ones((2, 2))
s3 = np.full((2, 2), 3)

print(s1)
print(s2)
print(s3)

s4 = np.linspace(0, 1, 10)
s5 = np.linspace(1, 10, 10)
print(s4)
print(s5)

# indexing, slicing, manipulation
v1 = np.array(range(1, 9))
print(v1)
print(v1[:]) # select all elements
print(v1[:3]) # select elements with indices < 3
print(v1[3:]) # select elements with indices >= 3
print(v1[3]) # select the element with index 3
print(v1[[0,2,4]]) # select elments with indices in [0,2,4]
print(v1[-1]) # select the last element
print(v1[-3:-1]) # select elements with indices -3, -2
print(v1[-3:]) # select the last three elements

print(v1[::1]) # select elements with indices 0,1,2,...
print(v1[::2]) # select elements with indices 0, 2, 4, ...
print(v1[::-1]) # select elements with indices -1, -2, ...
print(v1[0::-1]) # select the element with index 0
print(v1[5::-1]) # select elements with indices 5,4,...,0
print(v1[-3::-1]) # select elements with indices -3,-4,...

m1 = np.array(range(1,9)).reshape((2,4))

print(m1)
print(m1[:,1]) # select the second column
print(m1[0,:]) # select the first row
print(m1[1,::-1]) # select the second row reversed
print(m1[0,[1,3]]) # select the first row and the second, fourth columns

# views and copies

m = np.array(range(1,7)).reshape((3,2))

print(m)
v1 = m[0:2,] # this is a view
v1[0, 1] = -1
print(np.may_share_memory(v1, m))
print(m)

v2 = m[0:3:2,] # this is a view
v2[0, 1] = -2
print(np.may_share_memory(v2, m))
print(m)

m = np.array(range(1,7)).reshape((3,2))

print(m)
c1 = m[[0,2],:] # this is a copy 
c1[0, 1] = -1
print(np.may_share_memory(v1, m))
print(m)

m = np.array(range(1,7)).reshape((3,2))
v1 = m[0:2,] 
c1 = m[[0,1],]
print(v1.base)
print(c1.base)

# array operations
m1 = np.array(range(1,5)).reshape((2,2))
m2 = np.array(range(5,9)).reshape((2,2))

print(m1 + m2)
print(m1 - m2)
print(m1 * m2)
print(m1 / m2)
print(m1 ** m2)
print(m1 + 2)
print(m1 ** 2)

m1 = np.array(range(1, 9)).reshape((2,4))
v1 = np.array(range(1, 5))

print(m1)
print(v1)
print(m1 + v1)

# numpy functions
rng = np.random.default_rng()

v1 = rng.random(10) # generate 10 random numbers from the uniform distribution on [0,1)
v2 = rng.standard_normal(10) # generate 10 random numbers from the standard normal distribution
v3 = rng.integers(0, 100, 10) # generate 10 random integers from {0,1,...,99} 

print(v1)
print(v2)
print(v3)

rng = np.random.default_rng(seed=2024)
print(rng.random(5))
rng = np.random.default_rng(seed=2024)
print(rng.random(5))

# statistical functions
rng = np.random.default_rng(seed=1)
y = rng.standard_normal(1000)
print("97.5% percentile:", np.percentile(y, 97.5))
print("0.975 quantile:", np.quantile(y, 0.975))
print("min:", np.min(y))
print("median:", np.median(y))
print("max:", np.max(y))
print("mean:", np.mean(y))
print("std:", np.std(y))

# sort and search
rng = np.random.default_rng(seed=1)
y = rng.standard_normal(5)

print(y)
print(np.sort(y))
print(np.argsort(y))

rng = np.random.default_rng(seed=2)
y = rng.standard_normal(5)

print(y)
print(np.max(y))
print(np.argmax(y))
print(np.min(y))
print(np.argmin(y))

# matrices
m1 = np.array([1, 0.1, 0.1, 1]).reshape((2,2))
m2 = np.matrix(m1)

print(type(m1))
print(type(m2))

print(m1 * m1) # element-wise multiplication
print(m2 * m2) # matrix multiplication
print(m1 @ m1) # matrix multiplication
print(m2 @ m2) # matrix multiplication


# norm
v = np.array(range(1,5))
m = v.reshape((2,2))

print("diag:", np.diag(m))
print("lower triangle:\n", np.tril(m))
print("upper triangle:\n", np.triu(m))
print("transpose:\n", np.transpose(m))
print("rank:", np.linalg.matrix_rank(m))
print("norm of v:", np.linalg.norm(v))
print("norm of m:", np.linalg.norm(m))
print("trace:", np.linalg.trace(m))

A = np.matrix(np.array([1, -2, 3, 2, 1, 1, -3, 2, -3]).reshape((3,3)))
b = np.matrix(np.array([7, 4, -10]).reshape((3,1)))
x = np.linalg.solve(A, b)

print(x)
print(A * x)

invA = np.linalg.inv(A)
print(invA * b)

# file IO
rng = np.random.default_rng(seed=1)
dat = np.array(rng.standard_normal(10000)).reshape((1000,10))

np.savetxt("dat.csv", dat, fmt="%.8f", delimiter=",")
dat2 = np.loadtxt("dat.csv", dtype=float, delimiter=",")
print(np.linalg.norm(dat2-dat))

np.save("dat.npy", dat)
dat3 = np.load("dat.npy")
print(np.linalg.norm(dat3-dat))


# code optimization
import timeit

mysetup = "from math import sqrt"
mycode = '''
dSum = 0
for i in range(1,1000001):
    dSum += 1/sqrt(i)
'''
print(timeit.timeit(setup=mysetup, stmt=mycode, number=100))

mysetup2 = "import numpy as np"
mycode2 = '''
dSum = np.sum(np.reciprocal(np.sqrt(np.r_[1:1000001])))
'''
print(timeit.timeit(setup=mysetup2, stmt=mycode2, number=100))

import math
dSum = 0
for i in range(1,1000001):
    dSum += 1/math.sqrt(i)
print(dSum)

dSum = np.sum(np.reciprocal(np.sqrt(np.r_[1:1000001])))
print(dSum)

# broadcasting
a1 = np.array(range(1,9)).reshape((2,2,2))
a2 = np.array(range(1,5)).reshape((2,2))

print(a1)
print(a2)
print(a1+a2)