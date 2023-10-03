# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:14:22 2023

This script contains a class that implements all of the work in the scripts combodataV and combodata_ellipse in 2d. 
It lets us easily prepare many separate instances (different datasets) in one run, and refer to the methods needed
(f.ex. to get a velocity plot, create ellipse animation, or just collect analysis data)

Will be expanded to 3d eventually, but functionality will be similar.

@author: lassetot
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from lasse_functions import D_ij_2D, theta_rad, running_average, clockwise_angle
from lasse_functions import gaussian_2d, theta_extreme

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import convolve2d
import scipy.interpolate as scint
from scipy.integrate import cumtrapz
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable


# create instance for each dataset (of type combodata)
class ComboDataSR_2D:
    def __init__(  # performs when an instance is created
            self,
            filename,  # combodata file in 
            n = 2,  # every n'th voxel in mask sampled (n = 1 to sample all)
            sigma = 2,  # sigma of gaussian distribution that smoothes velocity data
            ):
        
        self.filename = filename
        self.n = n
        self.sigma = sigma
        
        # generalize: make filename be whole directory line
        self.data = sio.loadmat(f'R:\Lasse\combodata_shax\{filename}.mat')["ComboData_thisonly"]
        self.V = self.data['V'][0,0] #velocity field
        self.M = self.data['Magn'][0,0] #magnitudes
        self.mask = self.data['Mask'][0,0] #mask for non-heart tissue
        
        self.T = len(self.V[0,0,0,:,0]) #Total amount of time steps
        self.T_es = self.data['TimePointEndSystole'][0,0][0][0]
        self.T_ed = self.data['TimePointEndDiastole'][0,0][0][0]
        self.res = self.data['Resolution'][0,0][0][0]  # temporal resolution, need this for correct SR units?
        
        self.g = gaussian_2d(self.sigma)
        
    
    def overview(self):
        print(f'{self.filename} overview:')
        print(f'Velocity field shape: {np.shape(self.V)}')
        print(f'Magnitudes field shape: {np.shape(self.M)}')
        print(f'Mask shape: {np.shape(self.mask)}')
        print(f'End systole at t={self.T_es}, end diastole at t={self.T_ed}')
        
        
    # plots vector field over time, saves video, returns global radial velocity
    def velocity(self):
        # range of time-points
        self.range_ = range(self.T)
        
        # create meshgrid from data axis dimensions
        self.ax = len(self.mask[:,0,0,0])
        self.ay = len(self.mask[0,:,0,0])

        # global rad and circ velocity
        self.gr = np.zeros(self.T)
        self.gc = np.zeros(self.T)
        
        # center of mass at t=0
        self.cx_0, self.cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(self.mask[:, :, 0, 0]))
        
        for t in self.range_:
            
            frame1 = self.M[:, :, 0, t] #photon density at time t
            mask_t = self.mask[:, :, 0, t]
            
            plt.subplots(figsize=(10,10))
            #ax = plt.gca()
            
            
            plt.imshow(frame1.T/np.max(frame1), origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
            
            #find center of mass of filled mask (middle of the heart)
            cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
            
            plt.title(f'Velocity plot over proton density at timepoint t = {t} ({self.filename})', fontsize = 15)
            
            
            #certainty matrix
            C = frame1/np.max(frame1)
            
            #wiener noise reduction filter (?)
            vx = ndi.gaussian_filter(self.V[:, :, 0, t, 0]*C, sigma = 2)*mask_t #x components of velocity w mask
            vy = ndi.gaussian_filter(self.V[:, :, 0, t, 1]*C, sigma = 2)*mask_t #y components (negative?)
            
            # vector decomposition
            for x in range(0, self.ax, self.n):
                for y in range(0, self.ay, self.n):
                    if mask_t[x, y] == 1: 
                        
                        r = np.array([x - cx, y - cy])
                        
                        v_ = np.array([vx[x, y], vy[x, y]])
                        plt.quiver(x, y, v_[0], v_[1], color = 'w', scale = 10, minshaft = 1, minlength = 0, width = 0.005)
                        theta = clockwise_angle(r, v_) + np.pi
                        
                        self.gr[t] += np.linalg.norm(v_)*np.cos(theta) 
                        self.gc[t] += np.linalg.norm(v_)*np.sin(theta) 
            
            plt.scatter(cx, cy, marker = 'x', c = 'w')
          
            z = 25 # +- window from center of mass at t = 0
            plt.xlim(self.cx_0-z, self.cx_0+z); plt.ylim(self.cy_0-z, self.cy_0+z)
            plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
            plt.show()
            
        plt.figure(figsize=(10, 8))
        plt.title(f'Global velocity over time ({self.filename})', fontsize = 15)
        plt.axvline(self.T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(self.T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        plt.axhline(0, c = 'k', lw = 1)

        plt.plot(self.range_, self.gr, lw = 2, label = 'Radial')
        plt.legend()
        
        # save video in folder named after filename
        filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in self.range_]

        with imageio.get_writer(f'R:\Lasse\plots\MP4\{self.filename}\Velocity.mp4', fps=7) as writer:    # inputs: filename, frame per second
            for filename in filenames:
                image = imageio.imread(filename)                         # load the image file
                writer.append_data(image)
        
        return self.gr
    
    #when functional:
    #def strain_rate(self):
    #def strain(self):
            
        
        
#%%
#testing
if __name__ == "__main__":
    run1 = ComboDataSR_2D('sham_D11-1_1d')
    run1.overview()
    grv1 = run1.velocity()

#%%

        
        