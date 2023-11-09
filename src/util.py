# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:41:56 2023

@author: lassetot
"""
#imports

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
import pandas
#import seaborn as sns; sns.set()
#import sklearn

import scipy.io as sio
import scipy.ndimage as ndi
from scipy.signal import wiener, convolve2d
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from plotly.offline import plot

#from numba import njit #  use this to compile?

#%%
#Strain rate tensor (xy plane, dim=2) incl mask
#(Selskog et al 2002, hentet 17.08.23)

#V[i, j, 0, t, i]
# xval, yval, zval (...), timepoint, axis


def D_ij_2D(x, y, V, M, t, sigma, mask): #Construct SR tensor for specific point
    L = np.zeros((2, 2), dtype = float) #Jacobian 2x2 matrix
    
    # calculate certainty matrix from normalized magnitude plot
    C = M/np.max(M)
    
    # velocity x and y components from combodata multiplied by C + gaussian convolution
    # should compensate border artifacts
    #vx = convolve2d(V[:, :, 0, t, 1]*C, g, 'same')*mask / convolve2d(C, g, 'same')
    #vy = convolve2d(V[:, :, 0, t, 0]*C, g, 'same')*mask / convolve2d(C, g, 'same')
    
    vx = ndi.gaussian_filter(V[:, :, 0, t, 0]*C, sigma)*mask / ndi.gaussian_filter(C, sigma)
    vy = ndi.gaussian_filter(V[:, :, 0, t, 1]*C, sigma)*mask / ndi.gaussian_filter(C, sigma)
    
    dy = dx = 1 # voxel length 1 in our image calculations
    
    # note!: the diagonal has been switched for script testing!
    L[0, 0] = (C[x+1,y]*(vx[x+1,y]-vx[x,y]) + C[x-1,y]*(vx[x,y]-vx[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[1, 0] = -(C[x,y+1]*(vx[x,y+1]-vx[x,y]) + C[x,y-1]*(vx[x,y]-vx[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
    
    L[0, 1] = -(C[x+1,y]*(vy[x+1,y]-vy[x,y]) + C[x-1,y]*(vy[x,y]-vy[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[1, 1] = (C[x,y+1]*(vy[x,y+1]-vy[x,y]) + C[x,y-1]*(vy[x,y]-vy[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
            
    D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
    return D_ij

def D_ij_3D(x, y, V, M, t, sigma, mask, slice_): #Construct SR tensor for specific point
    L = np.zeros((3, 3), dtype = float) #Jacobian 3x3 matrix
    
    # calculate certainty matrix from normalized magnitude plot
    C = M/np.max(M)
    
    vx = ndi.gaussian_filter(V[f'V{slice_+1}'][:, :, 0, t, 0]*C, sigma)*mask / ndi.gaussian_filter(C, sigma)
    vy = ndi.gaussian_filter(V[f'V{slice_+1}'][:, :, 0, t, 1]*C, sigma)*mask / ndi.gaussian_filter(C, sigma)
    vz = ndi.gaussian_filter(V[f'V{slice_+1}'][:, :, 0, t, 2]*C, sigma)*mask / ndi.gaussian_filter(C, sigma)
    
    #vza = ndi.gaussian_filter(V[f'V{slice_+2}'][:, :, 0, t, 2]*C, sigma)*mask / ndi.gaussian_filter(C, sigma) # z-velocity above 
    #vzb =  # z-velocity below
    
    dy = dx = 1 # voxel length 1 in our image calculations
    dz = 1  # slicethickness/(res*10)  # relative voxel height
    
    # note!: the diagonal has been switched for script testing!
    L[0, 0] = (C[x+1,y]*(vx[x+1,y]-vx[x,y]) + C[x-1,y]*(vx[x,y]-vx[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[1, 0] = -(C[x,y+1]*(vx[x,y+1]-vx[x,y]) + C[x,y-1]*(vx[x,y]-vx[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
    #L[2, 0] = 
    
    L[0, 1] = -(C[x+1,y]*(vy[x+1,y]-vy[x,y]) + C[x-1,y]*(vy[x,y]-vy[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[1, 1] = (C[x,y+1]*(vy[x,y+1]-vy[x,y]) + C[x,y-1]*(vy[x,y]-vy[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
    #L[2, 1] =
    
    L[0, 2] = -(C[x+1,y]*(vz[x+1,y]-vy[x,y]) + C[x-1,y]*(vz[x,y]-vz[x-1,y])) / (dz*(C[x+1,y]+C[x-1,y]))
    L[1, 2] = (C[x,y+1]*(vz[x,y+1]-vy[x,y]) + C[x,y-1]*(vz[x,y]-vz[x,y-1])) / (dz*(C[x,y+1]+C[x,y-1]))      
    #L[2, 2] =
    
    D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
    return D_ij


# Note: returns smallest angle in radians between vectors 
def theta_rad(v, w):
    theta_r = np.arccos(v.dot(w)/(norm(v)*norm(w)))
    if theta_r > np.pi/2:
        theta_r = np.pi - theta_r
    return theta_r
#https://stats.stackexchange.com/questions/9898/how-to-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-r


# Note: forces angle to be 0 or 90 degrees relative to radial unit vector
# !! not to be used for serious quantitative analysis !!
def theta_extreme(v, w):
    theta_r = np.arccos(v.dot(w)/(norm(v)*norm(w)))
    if theta_r > np.pi/2:
        theta_r = np.pi - theta_r
    # force extreme values
    if (theta_r > np.pi/4):
        theta_r = np.pi/2
    else:
        theta_r = 0
    return theta_r


# running average of array a with window size N
def running_average(a, N, mode = 'same'):
    return np.convolve(a, np.ones(N)/N, mode = mode)


#https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def clockwise_angle(p1, p2):  # redundant?
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2*np.pi) # clockwise angle between vectors to points
# modulo operator % transforms range from (-pi, pi) to (0, 2pi)


# insert point and eigenvalues/vectors + index of highest eigenval
# angle in radians, converts to degrees in function
def draw_ellipse(x, y, vec, val, max_i, min_i, angle, hx):
    ellipse = patches.Ellipse((x, y), val[max_i], val[min_i], angle=angle*180/np.pi, color = hx)
    return ellipse
    

# insert arrays / meshgrid x y?
def gaussian_2d(sigma):
    d = round(2*np.pi*sigma)  # g size based on sigma 
    z = np.zeros((d, d))
    # create gaussian, peak shifted to middle
    for i in range(-d, d):
        for j in range(-d, d):
            z[i, j] = np.exp(-0.5*((i - d/2)**2 + (j - d/2)**2)/sigma**2)/(2*np.pi*sigma**2)
    return z

# https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python
def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return np.ufunc.reduce(np.add, np.gradient(F))

# https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/
# interquartile range, returns outliers and df without outliers
# set high threshold to include all data
def drop_outliers_IQR(df, column_name, threshold = 1.5):

    q1 = df[column_name].quantile(0.25)

    q3 = df[column_name].quantile(0.75)

    IQR = q3 - q1
    
    # threshold of 1.5 is convention
    outliers = df[(df[column_name] < q1 - threshold*IQR) | (df[column_name] > q3 + threshold*IQR)]

    outliers_dropped = df.drop(outliers.index)
    
    # calculate linear fit for data withing treshhold
    a, b = np.polyfit(outliers_dropped['Day'], outliers_dropped[column_name], 1)

    return outliers, outliers_dropped, a, b

def draw_ellipsoid(vec, val):
    # compute ellipsoid coordinates on standard basis
    # https://stackoverflow.com/questions/72153185/plot-an-ellipsoid-from-three-orthonormal-vectors-and-the-magnitudes-using-matplo
    a, b, c = val
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
        # axis on the new orthonormal base
        go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0, v1[2]], mode="lines", name="x2", line=dict(width=2, color="red")),
        go.Scatter3d(x=[0, v2[0]], y=[0, v2[1]], z=[0, v2[2]], mode="lines", name="y2", line=dict(width=2, color="green")),
        go.Scatter3d(x=[0, v3[0]], y=[0, v3[1]], z=[0, v3[2]], mode="lines", name="z2", line=dict(width=2, color="blue")),
        
        # final ellipsoid aligned to the new orthonormal base
        go.Surface(x=x2, y=y2, z=z2, opacity=1, colorscale="aggrnyl", surfacecolor=y1, cmin=y1.min(), cmax=y1.max(), colorbar=dict(len=0.6, yanchor="bottom", y=0, x=0.95))
    ])
    fig.update_layout({"scene": {"aspectmode": "auto"}})
