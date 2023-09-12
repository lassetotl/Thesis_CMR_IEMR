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

import scipy.io as sio
import scipy.ndimage as ndi
from scipy.signal import wiener
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%%
#Strain rate tensor (xy plane, dim=2) incl mask
#(Selskog et al 2002, hentet 17.08.23)

#V[i, j, 0, t, i]
# xval, yval, zval (...), timepoint, axis

def D_ij(V, t, f, mask_, dim = 2): #Construct SR tensor
    L = np.zeros((dim, dim), dtype = object) #Jacobian 2D velocity matrices
    
    v_i = 1; x_j = 0 #index 0 is y and 1 is x (?)
    for i in range(dim):
        for j in range(dim):
            #Gathering velocity data and applying gaussian smoothing
            V_ = ndi.gaussian_filter(V[:f, :f, 0, t, v_i]*mask_, sigma = 1)
            #V_[V_ == 0] = np.nan
            
            L[i, j] = np.gradient(V_, axis=x_j, edge_order = 1)
            x_j += 1
        v_i -= 1
        x_j = 0
    
    D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
    return D_ij


# Note: returns angle in radians between vectors 
def theta(v, w): return np.arccos(v.dot(w)/(norm(v)*norm(w)))
#https://stats.stackexchange.com/questions/9898/how-to-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-r

# running average of array a with window size N
def running_average(a, N, mode = 'same'):
    return np.convolve(a, np.ones(N)/N, mode = mode)


#https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def clockwise_angle(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang1 - ang2) % (2*np.pi) # clockwise angle between vectors to points
#what does the modulo operator % do?


# insert point and eigenvalues/vectors + index of highest eigenval
# angle in radians
def draw_ellipse(x, y, vec, val, max_i, min_i, angle, hx):
    ellipse = patches.Ellipse((x, y), val[max_i], val[min_i], angle=angle*180/np.pi, color = hx)
    return ellipse
    