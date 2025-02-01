# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:37:00 2024

@author: gjgan
"""

def fibo(n):
    if n == 1 or n == 2:
        return(1)
    else:
        return(fibo(n-1) + fibo(n-2))