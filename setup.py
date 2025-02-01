# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 22:38:01 2024

@author: gjgan
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fibonacci.py"),
)