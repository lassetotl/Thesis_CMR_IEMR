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
#import pandas as pd
#import seaborn as sns; sns.set()
#import sklearn

from astropy.convolution import Gaussian2DKernel

import scipy.io as sio
import scipy.ndimage as ndi
from scipy.signal import wiener, convolve2d
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
#Strain rate tensor (xy plane, dim=2) incl mask
#(Selskog et al 2002, hentet 17.08.23)

#V[i, j, 0, t, i]
# xval, yval, zval (...), timepoint, axis

def D_ij_2D(x, y, V, M, t, g): #Construct SR tensor for specific point
    L = np.zeros((2, 2), dtype = float) #Jacobian 2x2 matrix
    
    # calculate certainty matrix from normalized magnitude plot
    C = M/np.max(M)
    
    # velocity x and y components from combodata multiplied by C + gaussian convolution
    # should compensate border artifacts
    vx = convolve2d(V[:, :, 0, t, 1]*C, g) / convolve2d(C, g)
    vy = convolve2d(V[:, :, 0, t, 0]*C, g) / convolve2d(C, g)
    
    dx = dy = 1  # voxel length 1 in our image calculations
    L[0, 0] = (C[x+1,y]*(vx[x+1,y]-vx[x,y]) + C[x-1,y]*(vx[x,y]-vx[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[0, 1] = (C[x,y+1]*(vx[x,y+1]-vx[x,y]) + C[x,y-1]*(vx[x,y]-vx[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
    
    L[1, 0] = (C[x+1,y]*(vy[x+1,y]-vy[x,y]) + C[x-1,y]*(vy[x,y]-vy[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
    L[1, 1] = (C[x,y+1]*(vy[x,y+1]-vy[x,y]) + C[x,y-1]*(vy[x,y]-vy[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
            
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

#https://stackoverflow.com/questions/11435809/compute-divergence-of-vector-field-using-python
def divergence(F):
    """ compute the divergence of n-D scalar field `F` """
    return np.ufunc.reduce(np.add, np.gradient(F))