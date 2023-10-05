# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:14:22 2023

This script contains a class that implements all of the work in the scripts combodataV and combodata_ellipse in 2d. 
It lets us easily prepare many separate instances (different datasets) in one run, and refer to the methods needed
(f.ex. to get a velocity plot, create ellipse animation, or just collect analysis data)

Will be expanded to 3d eventually, but functionality will be similar.

@author: lassetot
"""

import os
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
        
        # get data axis dimensions
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
            vy = ndi.gaussian_filter(self.V[:, :, 0, t, 1]*C, sigma = 2)*mask_t #y components 
            
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
          
            w = 25 # +- window from center of mass at t = 0
            plt.xlim(self.cx_0-w, self.cx_0+w); plt.ylim(self.cy_0-w, self.cy_0+w)
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
    
    # set plot = 0 to ignore plotting and just calculate global strain rate
    # set save = 0 to avoid overwriting current .mp4 and .npy files
    def strain_rate(self, plot = 1, save = 1):  
        # range of time-points
        self.range_ = range(self.T)
        
        # get data axis dimensions
        self.ax = len(self.mask[:,0,0,0])
        self.ay = len(self.mask[0,:,0,0])
        
        # create arrays
        self.r_sr = np.zeros(self.T); self.r_sr[:] = np.nan #Graph values
        self.c_sr = np.zeros(self.T); self.c_sr[:] = np.nan #Graph values
        
        # center of mass at t=0
        self.cx_0, self.cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(self.mask[:, :, 0, 0]))
        
        # custom colormap
        c_cmap = plt.get_cmap('plasma')
        norm_ = mpl.colors.Normalize(vmin = 0, vmax = 90)
        
        print(f'Calculating Global Strain rate for {self.filename} ()...')
        for t in self.range_:

            # combodata mask 
            mask_t = self.mask[:, :, 0, t] #mask at this timepoint
            
            #find center of mass of filled mask (middle of the heart)
            cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
            
            # erode mask 
            mask_e = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
            
            # certainty matrix from magnitude data
            M_norm = (self.M[:, :, 0, t]/np.max(self.M[:, :, 0, t]))
            
            if plot == 1:
                plt.subplots(figsize=(10,10))
                ax = plt.gca()
                # plot magnitude M plot, normalize for certainty values
                # transpose in imshow to allign with mask
                plt.imshow(M_norm.T, origin = 'lower', cmap = 'gray', alpha = 1)
            
            # reset radial and circumferential contributions from last frame / initialize
            rad_e = 0 #radial components of eigenvectors, sum will be saved every t
            circ_e = 0 #circumferential ...
            
            #calculate eigenvalues and vectors
            e_count = 0  # ellipse counter in this frame
            for x in range(0, self.ax, self.n):
                for y in range(0, self.ay, self.n): 
                    # search in eroded mask to avoid border artifacts
                    if mask_e[x, y] == 1:
                        # SR tensor for point xy
                        D_ = D_ij_2D(x, y, self.V, M_norm, t, self.g, self.sigma, mask_t)     
                        val, vec = np.linalg.eig(D_)
                        
                        e_count += 1
                        
                        # vector between center of mass and point (x, y) 
                        r = np.array([x - cx, y - cy])
                        #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                        
                        # index of eigenvalues
                        val_max_i = np.argmax(val)  # most positive value
                        val_min_i = np.argmin(val)  # most negative
                        
                        theta = theta_rad(r, vec[val_max_i])  # angle between highest eigenvector and r
                        theta_ = theta_rad(r, vec[val_min_i]) # angle between lowest eigenvector and r
                        
                        # radial/circumferential contributions from each eigenvector
                        # scaled with amount of ellipses, varies because of dynamic mask
                        
                        #higher eigenvalues weighted higher (abs to not affect direction)
                        r1 = (val[val_max_i])*abs(np.cos(theta))/e_count
                        r2 = (val[val_min_i])*abs(np.cos(theta_))/e_count
                        
                        c1 = (val[val_max_i])*abs(np.sin(theta))/e_count
                        c2 = (val[val_min_i])*abs(np.sin(theta_))/e_count
                        
                        # global contribution
                        rad_e += r1 + r2
                        circ_e += c1 + c2
                        
                        ## for class, skip this part if only data is requested ##
                        if plot == 1:
                            # hex code, inputs in range (0, 1) so theta is scaled
                            hx = mpl.colors.rgb2hex(c_cmap(theta/(np.pi/2)))  # code with
                            #hx = mpl.colors.rgb2hex(c_cmap(I))  # color code with invariant
                            
                            # angle between eigenvector and x-axis, converted to degrees anti-clockwise
                            # clockwise theta needed
                            theta_c = clockwise_angle(r, vec[val_max_i])
                            e_angle = -(clockwise_angle([1,0], r) + theta_c)*180/np.pi
                            
                            # draw ellipses that are spanned by eigenvectors
                            # eigenvalues are transformed (1 + tanh(val)) to have a circular unit ellipse
                            ellipse = patches.Ellipse((x, y), (1 + np.tanh(val[val_max_i])), (1 + np.tanh(val[val_min_i])), 
                                                      angle = e_angle, color = hx)
                            
                            #unit ellipse
                            #unit_ellipse = patches.Ellipse((x, y), 1, 1, color = 'k'); ax.add_artist(unit_ellipse)
                            
                            ax.add_artist(ellipse)
            
            # graph subplot values
            self.r_sr[t] = rad_e #global radial strain rate this frame
            self.c_sr[t] = circ_e #global circumferential strain rate
            
            if plot == 1:
                plt.scatter(cx, cy, marker = 'x', c = 'w', s = 210, linewidths = 3)
                #plt.scatter(mis[0], mis[1], marker = 'x', c = 'r')
             
                plt.title(f'Strain Rate at t = {t} ({self.filename})', fontsize = 15)
                
                w = 25  # +- window from center of mass at t = 0
                plt.xlim(self.cx_0-w, self.cx_0+w); plt.ylim(self.cy_0-w, self.cy_0+w)
                
                # informatic text on plot
                plt.text(self.cx_0 - w + 3, self.cy_0 - w + 3, f'Gaussian smoothing ($\sigma = {self.sigma}$)',
                         color = 'w', fontsize = 15)
                plt.text(self.cx_0 - w + 3, self.cy_0 - w + 6, f'{e_count} Ellipses', 
                         color = 'w', fontsize = 15)
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="6%", pad=0.09)
                
                sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
                cbar = plt.colorbar(sm, cax = cax)
                cbar.set_label('$\Theta$ (degrees)', fontsize = 15)
                plt.tight_layout()
                
                plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
                plt.show(); plt.close()
                
        if plot == 1:
            # plot global strain rate
            
            N = 4 #window
            grsr = running_average(self.r_sr, N)
            gcsr = running_average(self.c_sr, N)

            plt.figure(figsize=(10, 8))

            plt.title(f'Global Strain rate over time ({self.filename})', fontsize = 15)
            plt.axvline(self.T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axvline(self.T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
            plt.axhline(0, c = 'k', lw = 1)

            plt.xlim(0, self.T)#; plt.ylim(0, 50)
            plt.xlabel('Timepoints', fontsize = 15)
            plt.ylabel('$s^{-1}$', fontsize = 20)

            plt.plot(self.range_, self.r_sr, 'lightgrey')
            plt.plot(self.range_, self.r_sr, 'lightgrey')

            plt.plot(self.range_, grsr, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
            plt.plot(self.range_, gcsr, 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average

            plt.legend()

            plt.subplots_adjust(wspace=0.25)
            
            if save == 1:
                if os.path.exists(f'R:\Lasse\plots\MP4\{self.filename}') == False:
                    os.makedirs(f'R:\Lasse\plots\MP4\{self.filename}')
                    
                plt.savefig(f'R:\Lasse\plots\MP4\{self.filename}\{self.filename}_GSR.PNG')
            plt.show()
            
        return self.r_sr, self.c_sr
    
    #def strain(self):  # called after 'strain_rate', message?
            
        
        
#%%
# example of use
if __name__ == "__main__":
    # create instance for input combodata file
    run1 = ComboDataSR_2D('sham_D11-1_1d')
    
    # get info/generate data 
    #run1.overview()
    #grv1 = run1.velocity()
    r1, c1 = run1.strain_rate(plot = 0, save = 0)

#%%