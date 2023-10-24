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
from matplotlib.lines import Line2D

from numpy.linalg import norm
from util import D_ij_2D, theta_rad, running_average, clockwise_angle
from util import gaussian_2d, theta_extreme

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import convolve2d
import scipy.interpolate as scint
from scipy.integrate import cumtrapz
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns


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
        
        # generalize: make filename be whole directory line?
        self.data = sio.loadmat(f'R:\Lasse\combodata_shax\{filename}')["ComboData_thisonly"]
        self.V = self.data['V'][0,0] #velocity field
        self.M = self.data['Magn'][0,0] #magnitudes
        self.mask = self.data['Mask'][0,0] #mask for non-heart tissue
        self.mask_segment = self.data['MaskS_medium'][0,0]
        
        self.T = len(self.V[0,0,0,:,0]) #Total amount of time steps
        self.T_es = self.data['TimePointEndSystole'][0,0][0][0]
        self.T_ed = self.data['TimePointEndDiastole'][0,0][0][0]
        self.res = self.data['Resolution'][0,0][0][0]  # temporal resolution, need this for correct SR units?
        self.TR = self.data['TR'][0,0][0][0]     
        
        # infarct sector, arbitrary if no infarct sector in metadata
        self.infarct = 0
        self.mis = [4, 13]  # arbitrary choice
        l = self.filename.split('_')
        if l[0] == 'mi' and (any(np.isnan(self.data['InfarctSector'][0,0][0])) == False):
            self.mis = self.data['InfarctSector'][0,0][0]
            self.infarct = 1  
        
        # segment slices alloted to non-infarct sectors, rounded down to int
        if self.mis[0] < self.mis[1]:
            infarct_length = self.mis[1] - self.mis[0]  # length in nr of sectors
        else:
            infarct_length = self.mis[0] - 36 - self.mis[1]
        
        # amount of segments in each remaining slice
        self.sl = int(np.floor((36 - abs(infarct_length))/6))
    
    def overview(self):
        print(f'{self.filename} overview:')
        print(f'Velocity field shape: {np.shape(self.V)}')
        print(f'Magnitudes field shape: {np.shape(self.M)}')
        print(f'Mask shape: {np.shape(self.mask)}')
        print(f'End systole at t={self.T_es}, end diastole at t={self.T_ed}')
        
        if self.infarct == 1:
            print(f'Infarct sector at {self.mis}')
        else:
            print(f'No infarct sector found in this slice, sector 1 set as {self.mis}')
            
        
        
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
    
    # input array of strain rate data
    # (used internally by later methods)
    def _strain(self, strain_rate, weight = 10):
        # weighting for integrals in positive/flipped time directions
        # cyclic boundary conditions
        w = np.tanh((self.T_ed-self.range_)/weight)[:self.T_ed]
        w_f = np.tanh(self.range_/weight)[:self.T_ed]

        strain = cumtrapz(strain_rate[:self.T_ed], self.range_TR[:self.T_ed], initial=0)
        strain_flipped = np.flip(cumtrapz(strain_rate[:self.T_ed][::-1], self.range_TR[:self.T_ed][::-1], initial=0))
        return (w*strain + w_f*strain_flipped)/2
    
    # set plot = 0 to ignore plotting and just calculate global strain rate
    # set save = 0 to avoid overwriting current .mp4 and .npy files
    # set segment = 1 to calculate/plot strain rate over 4 sectors
    # (method always calculates in sectors, but sums sectors if segment = 0 after saving synchrony parameters)
    def strain_rate(self, plot = 1, save = 1, segment = 0):  
        # range of time-points
        self.range_ = np.array(range(self.T))
        self.range_TR = self.range_*self.TR
        
        # get data axis dimensions
        self.ax = len(self.mask[:,0,0,0])
        self.ay = len(self.mask[0,:,0,0])
        
        # each row contains strain rate data for sector 1, 2, 3, 4
        self.r_matrix = np.zeros((4, self.T)); self.r_matrix[:, :] = np.nan
        self.c_matrix = np.zeros((4, self.T)); self.c_matrix[:, :] = np.nan

        # for each segment, we store angles corresponding to positive/negative eigenvalues 
        self.a1 = np.zeros((4, self.T), dtype = 'object') # 'positive' angles (stretch direction)
        self.a2 = np.zeros((4, self.T), dtype = 'object') # 'negative' angles (compression direction)
        
        # center of mass at t=0
        self.cx_0, self.cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(self.mask[:, :, 0, 0]))
        
        # setup segmentation
        if segment == 1:  # segmentation plot, colors for each segment
            c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())
            norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)
            
        else:  # for global plot, color code the angle distribution
            c_cmap = plt.get_cmap('plasma')
            norm_ = mpl.colors.Normalize(vmin = 0, vmax = 90)
            
        if self.infarct == 1:
            legend_handles1 = [Line2D([0], [0], color = c_cmap(0), lw = 2, label = 'Infarct'),
                      Line2D([0], [0], color = c_cmap(1), lw = 2, label = 'Adjacent'),
                      Line2D([0], [0], color = c_cmap(2), lw = 2, label = 'Medial'),
                      Line2D([0], [0], color = c_cmap(3), lw = 2, label = 'Remote')]
        else:
            legend_handles1 = [Line2D([0], [0], color = c_cmap(0), lw = 2, label = 'Sector 1'),
                      Line2D([0], [0], color = c_cmap(1), lw = 2, label = 'Sector 2'),
                      Line2D([0], [0], color = c_cmap(2), lw = 2, label = 'Sector 3'),
                      Line2D([0], [0], color = c_cmap(3), lw = 2, label = 'Sector 4')]
        
        if save == 1:
            if os.path.exists(f'R:\Lasse\plots\MP4\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\plots\MP4\{self.filename}')
        
        print(f'Calculating Global Strain rate for {self.filename}...')
        for t in self.range_[:self.T_ed+1]:

            # combodata mask 
            mask_t = self.mask[:, :, 0, t] #mask at this timepoint
            mask_segment_t = self.mask_segment[:, :, 0, t] #mask at this timepoint
            
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
            
            sector = 0  # reset sector number
            
            # remove nan's
            self.r_matrix[:, t] = 0; self.c_matrix[:, t] = 0
            
            # angles from sectors appended here, reset every t
            a1_ = [[], [], [], []]; a2_ = [[], [], [], []]
            
            # amount of ellipses in this timepoint in each sector 1-4 is stored here
            e_count = np.zeros(4)
            
            #calculate eigenvalues and vectors
            for x in range(0, self.ax, self.n):
                for y in range(0, self.ay, self.n): 
                    # search in eroded mask to avoid border artifacts
                    if mask_e[x, y] == 1:
                        # SR tensor for point xy
                        D_ = D_ij_2D(x, y, self.V, M_norm, t, self.sigma, mask_t)     
                        val, vec = np.linalg.eig(D_)
                        
                        # skip this voxel if eigenvalue signs are equal
                        #if np.sign(val[0]) == np.sign(val[1]):
                            #continue
                        
                        # vector between center of mass and point (x, y) 
                        r = np.array([x - cx, y - cy])
                        #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                        
                        # index of eigenvalues
                        val_max_i = np.argmax(val)  # most positive value
                        val_min_i = np.argmin(val)  # most negative
                        
                        theta = theta_rad(r, vec[val_max_i])  # angle between highest eigenvector and r
                        theta_ = theta_rad(r, vec[val_min_i]) # angle between lowest eigenvector and r
                        
                        # local contribution
                        sect_xy = mask_segment_t[x, y]  # sector value in (x,y)
                        
                        # need two ranges for every segment to counter invalid ranges (f.ex. range(33, 17))
                        # all values that add/subtract risk becoming negative or >36, thus % application

                        mis = self.mis; sl = self.sl
                        range0 = range(mis[0], mis[1]); range0_ = range(mis[1], mis[0])
                        range1 = range((mis[0]-sl)%36, mis[0]); range1_ = range(mis[0], (mis[0]-sl)%36)
                        range11 = range(mis[1], (mis[1]+sl)%36); range11_ = range((mis[1]+sl)%36, mis[1])
                        range2 = range((mis[0]-2*sl)%36, (mis[0]-sl)%36); range2_ = range((mis[0]-sl)%36, (mis[0]-2*sl)%36)
                        range22 = range((mis[1]+sl)%36, (mis[1]+2*sl)%36); range22_ = range((mis[1]+2*sl)%36, (mis[1]+sl)%36)
                        
                        # sector 4 defined from the ends of sector 3
                        if any(range22):
                            p_end = range22[-1]%36
                        else:
                            p_end = range22_[0]%36
                            
                        if any(range2):
                            n_end = (range2[0] + 1)%36
                        else:
                            n_end = (range2_[-1] + 1)%36
                        
                        # infarct sector (red)
                        if sect_xy in range0 or (sect_xy not in range0_)*any(range0_):  
                            sector = 0
                            e_count[sector] += 1
                            self.r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) \
                                + (val[val_min_i])*abs(np.cos(theta_))
                            self.c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) \
                                + (val[val_min_i])*abs(np.sin(theta_))
                            
                        
                        # adjacent (green)
                        elif sect_xy in range1 or (sect_xy not in range1_)*any(range1_) \
                            or sect_xy in range11 or (sect_xy not in range11_)*any(range11_):   
                                
                            sector = 1
                            e_count[sector] += 1
                            self.r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) \
                                + (val[val_min_i])*abs(np.cos(theta_))
                            self.c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) \
                                + (val[val_min_i])*abs(np.sin(theta_))
                            
                            
                        # medial (blue)                
                        elif sect_xy in range2 or (sect_xy not in range2_)*any(range2_) \
                            or sect_xy in range22 or (sect_xy not in range22_)*any(range22_):   
                                
                            sector = 2
                            e_count[sector] += 1
                            self.r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) \
                                + (val[val_min_i])*abs(np.cos(theta_))
                            self.c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) \
                                + (val[val_min_i])*abs(np.sin(theta_))
                        
                        
                        # remote (purple)
                        elif sect_xy in range(p_end, n_end) \
                            or (sect_xy not in range(n_end, p_end))*any(range(n_end, p_end)):
                            sector = 3
                            
                            e_count[sector] += 1
                            self.r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) \
                                + (val[val_min_i])*abs(np.cos(theta_))
                            self.c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) \
                                + (val[val_min_i])*abs(np.sin(theta_))
                        
                        else:  # avoid plotting ellipses in invalid ranges
                            continue
                        
                        # angle sum collected, scaled to get average angle each t
                        # does not assume that each 2d tensor has a positive and negative eigenvector
                        if val[val_max_i] > 0:
                            a1_[sector].append(theta) 
                        if val[val_min_i] > 0:
                            a1_[sector].append(theta_)
                            
                        if val[val_max_i] < 0:
                            a2_[sector].append(theta) 
                        if val[val_min_i] < 0:
                            a2_[sector].append(theta_)
                        
                        
                        if plot == 1:
                            # hex code, inputs in range (0, 1) so theta is scaled
                            if segment == 1:
                                # color code after sector 1 to 4
                                hx = mpl.colors.rgb2hex(list(c_cmap(sector)))
                            else:
                                hx = mpl.colors.rgb2hex(c_cmap(theta/(np.pi/2)))  # code with
                                #hx = mpl.colors.rgb2hex(c_cmap(I))  # color code with invariant?
                            
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
            
        
            for sector in range(4):  # scaling for correct units 
                self.r_matrix[sector, t] = self.r_matrix[sector, t] / (e_count[sector]*self.res)
                self.c_matrix[sector, t] = self.c_matrix[sector, t] / (e_count[sector]*self.res)  
            
                # collect angles in degrees
                self.a1[sector, t] = np.array(a1_[sector])*180/np.pi
                self.a2[sector, t] = np.array(a2_[sector])*180/np.pi
            
            # ellipse plot
            if plot == 1: 
                plt.scatter(cx, cy, marker = 'x', c = 'w', s = 210, linewidths = 3)
                #plt.scatter(mis[0], mis[1], marker = 'x', c = 'r')
             
                plt.title(f'Strain Rate at t = {t} ({self.filename})', fontsize = 15)
                
                w = 25  # +- window from center of mass at t = 0
                plt.xlim(self.cx_0-w, self.cx_0+w); plt.ylim(self.cy_0-w, self.cy_0+w)
                
                # informatic text on plot
                plt.text(self.cx_0 - w + 3, self.cy_0 - w + 3, f'Gaussian smoothing ($\sigma = {self.sigma}$)',
                         color = 'w', fontsize = 15)
                res_ = round(self.res*w*2, 4)
                plt.text(self.cx_0 - w + 3, self.cy_0 - w + 9, f'{res_} x {res_} cm', 
                         color = 'w', fontsize = 15)
                
                if segment == 0:
                    plt.text(self.cx_0 - w + 3, self.cy_0 - w + 6, f'{int(sum(e_count))} Ellipses', 
                             color = 'w', fontsize = 15)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="6%", pad=0.09)
                    
                    sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
                    cbar = plt.colorbar(sm, cax = cax)
                    cbar.set_label('$\Theta$ (degrees)', fontsize = 15)
                    
                else:
                    plt.text(self.cx_0 - w + 3, self.cy_0 - w + 6, 'Ellipse count:', color = 'w', fontsize = 15)
                    plt.text(self.cx_0 - w + 12, self.cy_0 - w + 6, f'{int(e_count[0])}', color = c_cmap(0), fontsize = 15)
                    plt.text(self.cx_0 - w + 15, self.cy_0 - w + 6, f'{int(e_count[1])}', color = c_cmap(1), fontsize = 15)
                    plt.text(self.cx_0 - w + 18, self.cy_0 - w + 6, f'{int(e_count[2])}', color = c_cmap(2), fontsize = 15)
                    plt.text(self.cx_0 - w + 21, self.cy_0 - w + 6, f'{int(e_count[3])}', color = c_cmap(3), fontsize = 15)
                
                plt.tight_layout()
                
                plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
                plt.show(); plt.close()
        
        # mean stretch/compression (a1/a2) angles
        a1_mean = np.zeros((4, self.T)); a2_mean = np.zeros((4, self.T))
        for t in self.range_:
            for sector in range(4):
                a1_mean[sector, t] = np.mean(self.a1[sector, t])
                a2_mean[sector, t] = np.mean(self.a2[sector, t])
            
        if plot == 1:
            # plot global strain rate

            plt.figure(figsize=(10, 8))

            plt.title(f'Global Strain rate over time ({self.filename})', fontsize = 15)
            plt.axvline(self.T_es*self.TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axvline(self.T_ed*self.TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
            plt.axhline(0, c = 'k', lw = 1)

            plt.xlim(0, self.T_ed*self.TR)#; plt.ylim(0, 50)
            plt.xlabel('Time [s]', fontsize = 15)
            plt.ylabel('$s^{-1}$', fontsize = 20)

            rsr = np.sum(self.r_matrix, axis = 0)
            csr = np.sum(self.c_matrix, axis = 0)

            plt.plot(self.range_TR, rsr, 'lightgrey')
            plt.plot(self.range_TR, csr, 'lightgrey')

            plt.plot(self.range_TR, running_average(rsr, 4), 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
            plt.plot(self.range_TR, running_average(csr, 4), 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average

            plt.legend()

            plt.subplots_adjust(wspace=0.25)
            
            if save == 1:
                if os.path.exists(f'R:\Lasse\plots\MP4\{self.filename}') == False:
                    os.makedirs(f'R:\Lasse\plots\MP4\{self.filename}')
                    
                plt.savefig(f'R:\Lasse\plots\MP4\{self.filename}\{self.filename}_GSR.PNG')
                
                filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in self.range_]  
                  
                with imageio.get_writer(f'R:\Lasse\plots\MP4\{self.filename}\Ellipses.mp4', 
                                        fps=7, macro_block_size = 1) as writer:    # inputs: filename, frame per second
                    for filename in filenames:
                        image = imageio.imread(filename)                         # load the image file
                        writer.append_data(image)
                
            plt.show()
                
        if plot == 1:
            # plot strain over time

            # strain curve analysis, synchrony of sectors
            self.c_peakvals = np.zeros(4); self.r_peakvals = np.zeros(4)
            self.c_peaktime = np.zeros(4); self.r_peaktime = np.zeros(4)


            plt.figure(figsize=(8, 6))

            plt.title(f'Regional Strain over time ({self.filename})', fontsize = 15)
            plt.axvline(self.T_es*self.TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axvline(self.T_ed*self.TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
            plt.axhline(0, c = 'k', lw = 1)

            plt.xlim(0, self.T_ed*self.TR)#; plt.ylim(0, 50)
            plt.xlabel('Time [s]', fontsize = 15)
            plt.ylabel('%', fontsize = 15)

            for sector in range(4):
                rs = 100*self._strain(self.r_matrix[sector, :])
                cs = 100*self._strain(self.c_matrix[sector, :])
                
                plt.plot(self.range_TR[:self.T_ed], rs, c = c_cmap(sector), lw=2)
                plt.plot(self.range_TR[:self.T_ed], cs, c = c_cmap(sector), lw=2)
                
                self.r_peakvals[sector] = np.max(rs); self.r_peaktime[sector] = np.argmax(rs)*self.TR
                self.c_peakvals[sector] = np.min(cs); self.c_peaktime[sector] = np.argmin(cs)*self.TR
                
                plt.scatter(self.r_peaktime[sector], self.r_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                plt.scatter(self.c_peaktime[sector], self.c_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                
            plt.legend(handles = legend_handles1)

            plt.subplots_adjust(wspace=0.25)
            plt.savefig(f'R:\Lasse\plots\MP4\{self.filename}\{self.filename}_GS.PNG')
            plt.show()
            
            #angles over time

            plt.figure(figsize = (10, 8))
            plt.title(f'Regional radial concentration over time ({self.filename})', fontsize = 15)
            plt.axvline(self.T_es*self.TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axvline(self.T_ed*self.TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
            plt.xlim(0, self.T*self.TR)#; plt.ylim(0, 50)
            plt.xlabel('Timepoints', fontsize = 15)
            plt.ylabel('Degrees', fontsize = 20)

            # mean angles
            for sector in range(4):
                #plt.plot(range_TR, a1_std, color = c_cmap(sector), label = 'Positive eigenvectors (stretch)')
                #plt.plot(range_TR, a2_std, 'g', label = 'Negative eigenvectors (compression)')
                # difference
                plt.plot(self.range_TR, abs(a1_mean[sector, :] - a2_mean[sector, :]), color = c_cmap(sector))

            plt.legend(handles = legend_handles1, loc = 'lower right')
            plt.show()
            
        if segment == 0:  # turn all return arrays global
            self.r_matrix = np.sum(self.r_matrix, axis = 0)
            self.c_matrix = np.sum(self.c_matrix, axis = 0)
            
            self.a1 = np.sum(a1_mean, axis = 0)[:self.T_ed] / 4
            self.a2 = np.sum(a2_mean, axis = 0)[:self.T_ed] / 4
            
            r_strain = 100*self._strain(self.r_matrix)
            c_strain = 100*self._strain(self.c_matrix)
            
        else:
            self.a1 = a1_mean
            self.a2 = a2_mean
            
        if save == 1:
            # save strain/strain rate/angle dist npy files for analysis

            if os.path.exists(f'R:\Lasse\strain data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\strain data\{self.filename}')
                
            np.save(fr'R:\Lasse\strain data\{self.filename}\r_strain', r_strain)
            np.save(fr'R:\Lasse\strain data\{self.filename}\c_strain', c_strain)
                
            if os.path.exists(f'R:\Lasse\strain rate data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\strain rate data\{self.filename}')
            
            np.save(fr'R:\Lasse\strain rate data\{self.filename}\r_strain_rate', self.r_matrix)
            np.save(fr'R:\Lasse\strain rate data\{self.filename}\c_strain_rate', self.c_matrix)
            
            if os.path.exists(f'R:\Lasse\\angle distribution data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\\angle distribution data\{self.filename}')
            
            np.save(fr'R:\Lasse\\angle distribution data\{self.filename}\angle_distribution_pos', self.a1)
            np.save(fr'R:\Lasse\\angle distribution data\{self.filename}\angle_distribution_neg', self.a2)
                
        # if save = 0 the parameters can still be collected from return statement without overwriting 
        return self.r_matrix, self.c_matrix, self.a1, self.a2
            
        
        
#%%
# example of use
if __name__ == "__main__":
    # create instance for input combodata file
    run1 = ComboDataSR_2D('mi_D12-8_45d')
    
    # get info/generate data 
    run1.overview()
    #grv1 = run1.velocity()
    run1.strain_rate(plot = 1, save = 0, segment = 0)
    
    #print(run1.__dict__['TR'])  # example of dictionary functionality
