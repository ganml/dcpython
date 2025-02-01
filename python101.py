# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math

pi = 3.1415926
Pi = 3.141592653589793

print(pi)
print(Pi)

import keyword
print(keyword.kwlist)

bA = True
iB = 2024
fC = 3.14
sD = 'data'
type(bA)
type(iB)
type(fC)
type(sD)

x, y = 1, 2
x
y
x, y = y, x
print(x)
print(y)
z = 3.14

x, y, z = 1, 2, 3
print(x)
print(y)
print(z)
x, y, z = y, z, x
print(x)
print(y)
print(z)

s1 = "a string"
len(s1)
s1[0:2]
s1[-1]
s1[-3:-1]
s1[-3:]

# string operators and functions
print("a " + "string")
print("a" * 5)
print("string"[0:3])
print("str" in "string")
print("string".replace("s", "S"))
print("1,2,3".split(","))
print("{} and {} are the results".format(1, 2))
print(",".join(["1", "2", "3"]))

# data structures
lst1 = ["a", 1, True]
print(lst1)
print(lst1[0])

lst1.append(3.14)
print(lst1)

lst1.insert(0, 3.14)
print(lst1)

print(lst1[0]) # get the first element
print(lst1[-1]) # get the last element
print(lst1[1:3]) # get the second and the third elemnts
print(lst1[-3:]) # get the last three elements
print(lst1[1:]) # get all elements except the first one
print(lst1[:3]) # get the first three elements
print(lst1[::-1]) # get elements with indices -1, -2, ...
print(lst1[0:3:2]) # get elements 0, 2
print(lst1[-2:-4:-2]) # get element -2

lst1[0:2] = [-1, -2]
print(lst1)

t1 = (1, 3)
t2 = ("a", True, 1)
print(t1)
print(t2)
print(t1[1])
print(t2[1:3])

m1 = {"a" : 1, "b" : 2}
print(m1)

print(m1["a"])
print(m1["b"])

print(m1.keys())
print(m1.values())
print(m1.items())

print(list(m1.keys()))
print(list(m1.values()))
print(list(m1.items()))

s1 = set()
s1.add(1)
s1.add("a")
s2 = {1, "a", "b", True, False}
print(s1)
print(s2)

s2.discard(2)
s2.remove(2)

# arithmetic operators
print("2 + 3 =", 2 + 3)
print("2 - 3 =", 2 - 3)
print("2 * 3 =", 2 * 3)
print("2 / 3 =", 2 / 3)
print("2 ** 3 =", 2 ** 3)
print("3.5 % 0.6 =", 3.5 % 0.6)
print("3.5 // 0.6 =", 3.5 // 0.6)

# assignment operators
x = 3.5
print("x =", x)
x += 0.6
print("x += 0.6:", x)
x -= 0.6
print("x -= 0.6:",x)
x *= 0.6
print("x *= 0.6:",x)
x /= 0.6
print("x /= 0.6:",x)
x %= 0.6
print("x %= 0.6:",x)
x //= 0.6
print("x //= 0.6:",x)

# comparison operators
print("1 == 2:", 1 == 2)
print("1 != 2:", 1 != 2)
print("1 < 2:", 1 < 2)
print("1 > 2:", 1 > 2)
print("1 <= 2:", 1 <= 2)
print("1 >= 2:", 1 >= 2)

print("3.5 % 0.6 > 0.5:", 3.5 % 0.6 > 0.5)
print("3.5 % 0.6 > 0.5:", 3.5 % 0.6 > 0.5 + 1e-10)

# logical operators
print("1 < 2 and 1 > 2: ", 1 < 2 and 1 > 2)
print("1 < 2 or 1 > 2: ", 1 < 2 or 1 > 2)
print("not 1 < 2:", not 1 < 2)

# identity and membership operators
x = 1
y = [1, 2, 3]
z = [1, 2, 3]
print("x is y:", x is y)
print("x is not y:", x is not y)
print("y is z:", y is z)
print("x in y:", x in y)
print("x not in y:", x not in y)
print("y in z", y in z)

# control statements and loops
x = 1
if x > 0:
    print("x is positive")

y = 2
if x > y:
    print("x is greather than y")
else:
    print("x is not greater than y")
    
if y > 2:
    print("y is greater than 2")
elif y == 2:
    print("y is equal to 2")
else:
    print("y is less than 2")
    
# for loop to calculate 1 + 2 + ... + 10
dSum = 0
for i in range(1, 11):
    dSum += i
print(dSum)

# while loop to calculate 1 + 2 + ... + 10
dSum = 0
i = 0
while i < 10:
    i += 1
    dSum += i
print(dSum)

# nest if in for loop
dSum = 0
for i in range(1, 1001):
    dSum += i
    if i % 100 == 0:
        print("Iteration {}: {}".format(i, dSum))
        
# while loop with break
dSum = 0
i = 0
while True:
    i += 1
    dSum += i
    if i >= 10:
        break
print(dSum)

# for loop with continue
dSum = 0
for i in range(1, 11):
    if i % 2 == 1:
        continue
    dSum += i
print(dSum)

# function
def sqrt(x):
    if x < 0:
        return("{} is negative".format(x))
    else:
        return(math.sqrt(x))

print(sqrt(-1))
print(sqrt(2))

# lambda function
add = lambda x, y: x+y
print(add(1,2))

x = [(lambda x: math.sqrt(x))(i) for i in range(1,11)]
print(x)

# read and write files
import os
fi = open(os.path.join(r"c:\users\gjgan\documents\researchu\book\dcpython\code", "file.txt"), "r") 
content = fi.read()
fi.close()
print(content)

with open(os.path.join(r"c:\users\gjgan\documents\researchu\book\dcpython\code", "file.txt"), "r") as fi:
    content = fi.read()
print(content)

with open(os.path.join(r"c:\users\gjgan\documents\researchu\book\dcpython\code", "file.txt"), "r") as fi:
    content = fi.readlines()
print(content)

with open(os.path.join(r"c:\users\gjgan\documents\researchu\book\dcpython\code", "integers.txt"), "w") as fi:
    fi.write(",".join([str(i) for i in range(1,6)]))
    fi.write("\n")
    fi.write(",".join([str(i) for i in range(6,11)]))

# error handling
x = -2
try:
    print(math.sqrt(x))
except ValueError as e:
    print(str(e))

def sqrt2(x):
    if not isinstance(x, (int, float)):
        raise ValueError("{} is not numeric".format(x))
    elif x < 0:
        raise ValueError("{} is negative".format(x))
    else:
        return(math.sqrt(x))

for x in [-2, 2, "a"]:    
    try:
        print(sqrt2(x))
    except ValueError as e:
        print(str(e))

# object-oriented programming
from abc import abstractmethod
class Algorithm:
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def fit(self):
        pass

class Kmean(Algorithm):
    def fit(self):
        return("Clustering by kmeans")
    
class Kmode(Algorithm):
    def fit(self):
        return("Clustering by kmodes")
    
alg1 = Kmean("algorithm 1")
alg2 = Kmode("algorithm 2")

print(alg1.fit())
print(alg2.fit())
print(alg1.name)
print(alg2.name)

# code optimization
def fibo(n):
    if n == 1 or n == 2:
        return(1)
    else:
        return(fibo(n-1) + fibo(n-2))
    
import time
beg = time.time()
print(fibo(40))
end = time.time()
print("Execution time: {:.4f} seconds".format(end - beg))

import fibonacci
beg = time.time()
print(fibonacci.fibo(40))
end = time.time()
print("Execution time: {:.4f} seconds".format(end - beg))
