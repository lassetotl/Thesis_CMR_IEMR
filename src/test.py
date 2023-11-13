# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:53:42 2023

@author: lassetot
"""

import numpy as np
from util import clockwise_angle, theta_rad, draw_ellipsoid
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import plotly.graph_objects as go
from plotly.offline import plot

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
        
#%%

a = np.zeros(3)

a = np.append(a, 5)
print(a)

b = []
print(np.array(b))

a1_ = [[], [], [], []]

a1_[0].append(1)
a1_[0].append(3)

print(a1_)

#%%
range_ = range(36)
range1 = range(36, 0)
print(any((range1)))
print(any((range_)))

x = 6
if x not in range(10, 20):
    print('a')
    
#%%

myList = [1, 2, 3, 100, 5]
mylist_e = list(enumerate(myList))

mylist_sorted = sorted(mylist_e, key=lambda x: x[1])
print(mylist_sorted)

#%%

a, b, c = np.array([-0.61191069, 0.14524356, 0.46811077])
vec = np.array([[ 0.53578182, 0.74326574, 0.40061687],
 [ 0.5657862, -0.6682228, 0.48307791],
 [-0.62675659, 0.03216087, 0.77855113]])

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection='3d')
plt.quiver(0, 0, 0, a*vec[0][0], a*vec[0][1], a*vec[0][2], color='r')
plt.quiver(0, 0, 0, b*vec[1][0], b*vec[1][1], b*vec[1][2], color='g')
plt.quiver(0, 0, 0, c*vec[2][0], c*vec[2][1], c*vec[2][2], color='b')
lim = 1
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
plt.show()

# are the vectors orthoganal?
print(theta_rad(vec[0], vec[1])*180/np.pi)
print(theta_rad(vec[2], [0,0,1])*180/np.pi)
print(theta_rad(vec[2], [1,0,0])*180/np.pi)
print(theta_rad(vec[2], [0,1,0])*180/np.pi)
#%%
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
x1 = a * np.cos(u) * np.sin(v)
y1 = b * np.sin(u) * np.sin(v)
z1 = c * np.cos(v)
# points on the ellipsoid
points = np.stack([t.flatten() for t in [x1, y1, z1]])

v1, v2, v3 = vec
# 3x3 transformation matrix
T = np.array([v1, v2, v3]).T

# transform coordinates to the new orthonormal basis
new_points = T @ points
x2 = new_points[0, :]
y2 = new_points[1, :]
z2 = new_points[2, :]
x2, y2, z2 = [t.reshape(x1.shape) for t in [x2, y2, z2]]

# scale vector for better visualization
scale = 5
v1, v2, v3 = [scale * t for t in [v1, v2, v3]]

fig = go.Figure([
    # axis on the standard base
    go.Scatter3d(x=[0, 5], y=[0, 0], z=[0, 0], mode="lines", name="x1", line=dict(width=5, color="red")),
    go.Scatter3d(x=[0, 0], y=[0, 5], z=[0, 0], mode="lines", name="y1", line=dict(width=5, color="green")),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 5], mode="lines", name="z1", line=dict(width=5, color="blue")),
    # axis on the new orthonormal base
    go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]], mode="lines", name="x2", line=dict(width=2, color="red")),
    go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]], mode="lines", name="y2", line=dict(width=2, color="green")),
    go.Scatter3d(x=[0, v3[0]], y=[0, v3[1]], z=[0, v3[2]], mode="lines", name="z2", line=dict(width=2, color="blue")),
    # original ellipsoid aligned to the standard base
    go.Surface(x=x1, y=y1, z=z1, opacity=0.35, colorscale="plotly3", surfacecolor=y1, cmin=y1.min(), cmax=y1.max(), colorbar=dict(len=0.6, yanchor="bottom", y=0)),
    # final ellipsoid aligned to the new orthonormal base
    go.Surface(x=x2, y=y2, z=z2, opacity=1, colorscale="aggrnyl", surfacecolor=y1, cmin=y1.min(), cmax=y1.max(), colorbar=dict(len=0.6, yanchor="bottom", y=0, x=0.95))
])
fig.update_layout({"scene": {"aspectmode": "auto"}})
plot(fig, auto_open=True)

#%%

a = np.array([3,3,4])

phi = theta_rad(np.array([0,0,1]), a)
print(phi)