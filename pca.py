#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:07:45 2020

@author: raghav
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal')
plt.show()

pca = PCA(n_components=2)
pca.fit(X)

def drawVector(vertexZero, vertexOne, ax = None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='->', linewidth=2, shrinkA = 0, shrinkB = 0)
    ax.annotate('', vertexOne, vertexZero, arrowprops = arrowprops)
    
# Plot Data
    
plt.scatter(X[:, 0], X[:, 1])
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    drawVector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.show()

pca = PCA(n_components = 1)
pca.fit(X)
xPca = pca.transform(X)
xNew = pca.inverse_transform(xPca)
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(xNew[:, 0], xNew[:, 1], alpha = 0.8)
plt.axis('equal')
plt.show()