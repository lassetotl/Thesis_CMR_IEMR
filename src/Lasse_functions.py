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
# xval, yval, ?, timepoint, axis

def D_ij(V, t, f, mask_, dim = 2): #Construct SR tensor
    L = np.zeros((dim, dim), dtype = object) #Jacobian 2D velocity matrices
    
    v_i = 1; x_j = 0 #index 0 is y and 1 is x (?)
    for i in range(dim):
        s = 1
        for j in range(dim):
            #Gathering velocity data and applying gaussian smoothing
            V_ = ndi.gaussian_filter(V[:f, :f, 0, t, v_i]*mask_, sigma = 2)
            #V_[V_ == 0] = np.nan
            
            if (j==1) is True: #negative sign on V_y (?)
                s = -1
            L[i, j] = s * np.gradient(V_, axis=x_j, edge_order = 1)
            x_j += 1
        v_i -= 1
        x_j = 0
    
    D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
    return D_ij

# Note: returns angle in radians between vectos 
def theta(v, w): return np.arccos(v.dot(w)/(norm(v)*norm(w)))

#https://stats.stackexchange.com/questions/9898/how-to-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-r