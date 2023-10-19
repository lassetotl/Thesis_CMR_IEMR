# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:53:42 2023

@author: lassetot
"""

import numpy as np
from util import clockwise_angle, gaussian_2d
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 

#%%

X = np.random.randn(100, 2)
X[:,1] += 0.3 * X[:,0]
cov = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eig(cov)
'''
import matplotlib.pyplot as plt;
theta = np.linspace(0, 2*np.pi, 1000);
ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
plt.plot(ellipsis[0,:], ellipsis[1,:])
'''


angle = -clockwise_angle([1, 0], eigenvectors[0])  # degrees

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

plt.quiver(0, 0, eigenvectors[0][0], eigenvectors[0][1], scale = np.sqrt(10/eigenvalues[0]))
plt.quiver(0, 0, eigenvectors[1][0], eigenvectors[1][1], scale = np.sqrt(10/eigenvalues[1]))

ellipse = Ellipse((0, 0), eigenvalues[0], eigenvalues[1], angle=angle*180/np.pi, alpha=0.1)
ax.add_artist(ellipse)

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)

plt.show()
print(eigenvalues)

#%%

a = np.zeros((2,2))
a[0,1] = 2
print(a)

#%%

theta = (np.random.rand(1))*2*np.pi # radians

print(np.cos(theta))
print(np.cos(theta + np.pi))

#%%
z = np.zeros(3); z[:] = np.nan
z[0] = 1
print(z)

#%%

a = np.array([[0,0,1,1],
     [0,2,1,2],
     [2,3,3,3],
     [3,4,4,4]])

print(ndi.binary_erosion(a).astype(a.dtype)*a)

#%%

a = [0,1,2,3,4,5,6,7,8,9]

for i in a:
    if i in range(3, 5+1) or i in range(6, 9):
        print(i)