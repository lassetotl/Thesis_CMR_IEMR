# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2023

(WIP) Expansion of the ComboDataSR_2D class, but applied to series of combodata slices.
The final class will perform similar calculations but for a main slice (slice06?)
plus a range of slices above and below. Or just a given range?

@author: lassetot
"""

import os, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D

from numpy.linalg import norm
from util import theta_rad, running_average, clockwise_angle


import scipy.io as sio
import scipy.ndimage as ndi 
import scipy.interpolate as scint
from scipy.integrate import cumtrapz, simpson
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
import h5py

# create instance for each dataset (of type combodata)
class ComboDataSR_3D:
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
        self.data = h5py.File(f'R:\Lasse\combodata_3d_shax\{self.filename}.mat', 'r')['ComboData']
        
        # global parameters for this set
        self.T_es = int(self.data[self.data['TimePointEndSystole'][0,0]][0,0])
        self.res = float(self.data[self.data['Resolution'][0,0]][0,0])  # spatial resolution in cm
        self.slicethickness = float(self.data[self.data['SliceThickness'][0,0]][0,0])  # in mm
        self.dz = self.slicethickness/(self.res*10)  # relative voxel height (4.266666666666667)
        self.TR = float(self.data[self.data['TR'][0,0]][0,0])  # temporal resolution in s
        self.ShortDesc = self.data['ShortDesc']
        self.slices = len(self.ShortDesc)  # nr of slices in this file
        
        pss0 = [float(self.data[self.data['pss0'][i,0]][0,0]) for i in range(len(self.data['pss0']))]
        # sorted slice order and z positions
        idx, pss0 = zip(*sorted(list(enumerate(pss0)), reverse = True, key = lambda x: x[1]))
        
        # matrices and values that differ between slices collected here
        self.V = {}; self.M = {}; self.mask = {}  # velocity, magnitude, mask for all slices
        self.mask_segment = {}; self.slicenr = {}  # mask segments, order of slice numbers
        self.T_ed = {}; self.mis = {}  # timepoint end diastole, infarct sectors
        
        # dictionary keys for all slices
        for slice_ in range(self.slices):
            self.V[f'V{slice_ + 1}'] = np.array(self.data[self.data['V'][idx[slice_], 0]])  # velocity field for one slice
            self.M[f'M{slice_ + 1}'] = np.array(self.data[self.data['Magn'][idx[slice_], 0]])  #magnitudes
            self.mask[f'mask{slice_ + 1}'] = np.array(self.data[self.data['Mask'][idx[slice_], 0]])  #mask for non-heart tissue 
            self.mask_segment[f'mask_segment{slice_ + 1}'] = np.array(self.data[self.data['MaskS_medium'][idx[slice_], 0]])
            self.T_ed[f'T_ed{slice_ + 1}'] = int(np.array(self.data[self.data['TimePointEndDiastole'][idx[slice_], 0]]))
            self.mis[f'mis{slice_ + 1}'] = np.array(self.data[self.data['InfarctSector'][idx[slice_], 0]])  #mask for non-heart tissue 
        
            a = []  # construct ShortDescription
            for i in range(len(self.data[self.ShortDesc[0,0]])):
                try: 
                    self.data[self.ShortDesc[idx[slice_], 0]][i,0]
                except IndexError:
                    break
                else:
                    a.append(chr(self.data[self.ShortDesc[idx[slice_], 0]][i,0]))
            
            desc = ''.join(a).split(' ')[-2][-2:].lstrip('0')
            self.slicenr[f'slice {slice_ + 1}'] = desc
        
        a = ''.join(a).split(' ')[1:3]
        self.ID = a[0] + ' ' + a[1]
        self.T = len(self.V['V1'][0,:,0,0,0])  #Total amount of time steps     
        
        # infarct sector, arbitrary if no infarct sector in metadata
        self.infarct = 0  # true/false
        self.mis = [4, 13]  # arbitrary choice
        #l = self.filename.split('_')
        #if l[0] == 'mi' and (any(np.isnan(self.data['InfarctSector'][0,0][0])) == False):
        #    self.mis = self.data['InfarctSector'][0,0][0]
        #    self.infarct = 1  
        
        # segment slices alloted to non-infarct sectors, rounded down to int
        if self.mis[0] < self.mis[1]:
            infarct_length = self.mis[1] - self.mis[0]  # length in nr of sectors
        else:
            infarct_length = self.mis[0] - 36 - self.mis[1]
        
        # amount of segments in each remaining slice
        self.sl = int(np.floor((36 - abs(infarct_length))/6))
    
    ### internal functions (prefixed by '_') are called by the main methods ###    
    
    # calculate strain rate tensor for given point (x, y) and time t, 
    # and a mask for this timepoint t using Selskog method
    def _D_ij_3D(self, x, y, t): 
        L = np.zeros((3, 3), dtype = float) #Jacobian 3x3 matrix
        
        '''
        [L = [dvx/dx, dvx/dy, dvx/dz],
         [dvy/dx, dvy/dy, dvy/dz],
         [dvz/dx, dvz/dy, dvz/dz]]
        '''
        
        dy = dx = 1 # voxel length 1 in our image calculations
        dz = self.dz
        
        # collecting fields at t
        # fields 'a' above, 'b' below to get dz derivatives
        vx = self.vx; vy = self.vy; vz = self.vz 
        C = self.C; Ca = self.Ca; Cb = self.Cb
        vxa = self.vxa; vxb = self.vxb
        vya = self.vya; vyb = self.vyb
        vza = self.vza; vzb = self.vzb
        
        # linear interpolation to use dz = 1
        vxa_xy = np.interp(1, [0, dz], [vx[x, y], vxa[x, y]])
        vxb_xy = np.interp(1, [0, dz], [vx[x, y], vxb[x, y]])
        vya_xy = np.interp(1, [0, dz], [vy[x, y], vya[x, y]])
        vyb_xy = np.interp(1, [0, dz], [vy[x, y], vyb[x, y]])
        vza_xy = np.interp(1, [0, dz], [vz[x, y], vza[x, y]])
        vzb_xy = np.interp(1, [0, dz], [vz[x, y], vzb[x, y]]); dz = 1
        
        # equation 4 in Selskog et al., 2002
        L[0, 0] = (C[x+1,y]*(vx[x+1,y]-vx[x,y]) + C[x-1,y]*(vx[x,y]-vx[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
        L[1, 0] = (C[x,y+1]*(vx[x,y+1]-vx[x,y]) + C[x,y-1]*(vx[x,y]-vx[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
        L[2, 0] = (Ca[x,y]*(vxa_xy-vx[x,y]) + Cb[x,y]*(vxb_xy-vx[x,y])) / (dz*(Ca[x,y]+Cb[x,y]))
        
        L[0, 1] = (C[x+1,y]*(vy[x+1,y]-vy[x,y]) + C[x-1,y]*(vy[x,y]-vy[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
        L[1, 1] = (C[x,y+1]*(vy[x,y+1]-vy[x,y]) + C[x,y-1]*(vy[x,y]-vy[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
        L[2, 1] = (Ca[x,y]*(vya_xy-vy[x,y]) + Cb[x,y]*(vyb_xy-vy[x,y])) / (dz*(Ca[x,y]+Cb[x,y]))
        
        L[0, 2] = (C[x+1,y]*(vz[x+1,y]-vz[x,y]) + C[x-1,y]*(vz[x,y]-vz[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
        L[1, 2] = (C[x,y+1]*(vz[x,y+1]-vz[x,y]) + C[x,y-1]*(vz[x,y]-vz[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))      
        L[2, 2] = (Ca[x,y]*(vza_xy-vz[x,y]) + Cb[x,y]*(vzb_xy-vz[x,y])) / (dz*(Ca[x,y]+Cb[x,y]))
        
        D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
        return D_ij
    
    def _D_ij_2D(self, x, y, t):  # inherit from 2d class?
        L = np.zeros((2, 2), dtype = float) #Jacobian 2x2 matrix
        
        dy = dx = 1 # voxel length 1 in our image calculations
        vx = self.vx; vy = self.vy; C = self.C
        
        # note!: the diagonal has been switched for script testing!
        L[0, 0] = (C[x+1,y]*(vx[x+1,y]-vx[x,y]) + C[x-1,y]*(vx[x,y]-vx[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
        L[1, 0] = -(C[x,y+1]*(vx[x,y+1]-vx[x,y]) + C[x,y-1]*(vx[x,y]-vx[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
        
        L[0, 1] = -(C[x+1,y]*(vy[x+1,y]-vy[x,y]) + C[x-1,y]*(vy[x,y]-vy[x-1,y])) / (dx*(C[x+1,y]+C[x-1,y]))
        L[1, 1] = (C[x,y+1]*(vy[x,y+1]-vy[x,y]) + C[x,y-1]*(vy[x,y]-vy[x,y-1])) / (dy*(C[x,y+1]+C[x,y-1]))
                
        D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
        return D_ij
    
    # find nearest non-zero index in 2d array    
    def _nearest_nonzero_idx(self, a, x, y):
        idx = np.argwhere(a)
        return idx[((idx - [x,y])**2).sum(1).argmin()]
    
    # input array of strain rate data
    # (used internally by later methods)
    def _strain(self, strain_rate, T_ed, weight = 10):  # inherit from 2d class?
        # weighting for integrals in positive/flipped time directions
        # cyclic boundary conditions
        # (old weights)
        #w = np.tanh((self.T_ed - 1 - self.range_)/weight) 
        #w_f = np.tanh(self.range_/weight) 
        
        # linear weights
        w1 = self.range_[:T_ed]*T_ed; w1 = w1/np.max(w1)
        w2 = np.flip(w1); w2 = w2/np.max(w2)
        
        strain = cumtrapz(strain_rate[:T_ed], self.range_TR[:T_ed]/1000, initial = 0)
        strain_flipped = np.flip(cumtrapz(strain_rate[::-1][:T_ed], self.range_TR[::-1][:T_ed]/1000, initial = 0))
        return w2*strain + w1*strain_flipped
    
    ### methods 'overview', 'velocity' and 'strain_rate' are called from instances of the class ### 
    
    def overview(self):
        print(f'{self.filename} global overview:')
        print(f'Velocity field shape: {np.shape(self.V["V1"])}')
        print(f'Magnitudes field shape: {np.shape(self.M["M1"])}')
        print(f'Mask shape: {np.shape(self.mask["mask1"])}')
        print(f'End systole at t={self.T_es}, end diastole at t={self.T_ed}')
        print(f'Number of slices: {self.slices}')
        
        if self.infarct == 1:
            print(f'Infarct sector at {self.mis}')
        else:
            print(f'No infarct sector found in this slice, sector 1 set as {self.mis}')
            
    # plots vector field  in one slice, saves video, returns global radial velocity
    # plot dimensions, dim = '3D', '2D'
    def velocity(self, slice_, save = 0, dim = '3D'):
        # slice 6 should produce the same plot and radial vel curve as combodata
        
        # relevant matrices for this slice
        V = self.V[f'V{slice_}']
        M = self.M[f'M{slice_}']
        mask = self.mask[f'mask{slice_}']
        
        # range of time-points
        self.range_ = range(self.T)
        slicenr = self.slicenr[f"slice {slice_}"]
        w = 25  # +- window from center of mass at t = 0
        if dim == '3D':
            c_cmap = plt.get_cmap('plasma')
            norm_ = mpl.colors.Normalize(vmin = -1.2, vmax = 1.2)
            sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
            w -= 12
        
        # get data axis dimensions
        self.ax = len(mask[0,0,:,0])
        self.ay = len(mask[0,0,0,:])

        # global rad, circ and longitudinal velocity
        self.gr = np.zeros(self.T)
        self.gc = np.zeros(self.T)
        self.gl = np.zeros(self.T)
        
        # center of mass at t=0
        self.cx_0, self.cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[0, 0, :, :].T))
        
        for t in self.range_:
            
            frame1 = M[t, 0, :, :].T  #proton density at time t
            mask_t = mask[t, 0, :, :].T
            
            #find center of mass of filled mask (middle of the heart)
            cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
            
            if dim == '2D':
                fig, ax = plt.subplots(figsize=(10,10))
                plt.imshow(frame1.T/np.max(frame1), origin = 'lower', cmap = 'gray', vmin = 0, vmax = 1)
                ax.scatter(cx, cy, marker = 'x', c = 'w', s = 210, linewidths = 3)
                
            else:
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111, projection='3d')
                cbar = plt.colorbar(sm)
                cbar.set_label('$v_z \ [cm/s]$', fontsize = 15)
                ax.scatter(cx, cy, self.cx_0, marker = 'x', c = 'k', s = 210, linewidths = 3)
            
            plt.title(f'Velocity plot at t = {t} ({self.ID}, Slice {slicenr})', fontsize = 15)
            
            #certainty matrix
            C = frame1/np.max(frame1)
            
            #wiener noise reduction filter (?)
            vx = ndi.gaussian_filter(V[0, t, 0, :, :].T*C, sigma = 2)*mask_t #x components of velocity w mask
            vy = ndi.gaussian_filter(V[1, t, 0, :, :].T*C, sigma = 2)*mask_t #y components
            vz = ndi.gaussian_filter(V[2, t, 0, :, :].T*C, sigma = 2)*mask_t #y components
            
            # vector decomposition
            for x in range(0, self.ax, self.n):
                for y in range(0, self.ay, self.n):
                    if mask_t[x, y] == 1: 
                        
                        r = np.array([x - cx, y - cy])
                        v_ = np.array([vx[x, y], vy[x, y]])
                        theta = clockwise_angle(r, v_) + np.pi
                        
                        if dim == '2D':
                            ax.quiver(x, y, v_[0], v_[1], color = 'w', scale = 10, minshaft = 1, minlength = 0, width = 0.005)
                        else:
                            v_ = np.array([vx[x, y], vy[x, y], vz[x, y]])
                            hx = mpl.colors.rgb2hex(c_cmap(norm_(vz[x, y])))
                            
                            ax.quiver(x, y, self.cx_0, v_[0], v_[1], v_[2], color = hx, linewidths = 2.8, arrow_length_ratio=0.8, length = 1.3)
                            ax.set_zlim(self.cx_0-w, self.cx_0+w)
                        
                        self.gr[t] += np.linalg.norm(v_)*np.cos(theta) 
                        self.gc[t] += np.linalg.norm(v_)*np.sin(theta)
                        self.gl[t] += vz[x, y]
            
            plt.xlim(self.cx_0-w, self.cx_0+w); plt.ylim(self.cy_0-w, self.cy_0+w)
            plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
            plt.show()
            
        plt.figure(figsize=(10, 8))
        plt.title(f'Global longitudinal velocity ({self.ID}, Slice {slicenr})', fontsize = 15)
        plt.axvline(self.T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(self.T_ed[f'T_ed{slice_}'], c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        plt.axhline(0, c = 'k', lw = 1)

        plt.plot(self.range_, self.gl, lw = 2, label = 'Longitudinal')
        plt.plot(self.range_, self.gr, lw = 2, label = 'Radial')
        #plt.plot(self.range_, self.gc, lw = 2, label = 'Circumferential')
        plt.legend()
        
        if save == 1:
            # save video in folder named after filename
            filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in self.range_]
    
            with imageio.get_writer(f'R:\Lasse\plots\MP4\{self.filename}\Velocity.mp4', fps=7) as writer:    # inputs: filename, frame per second
                for filename in filenames:
                    image = imageio.imread(filename)                         # load the image file
                    writer.append_data(image)
        
        return self.gr
    
    # set plot = 0 to ignore plotting and just calculate global strain rate
    # set save = 0 to avoid overwriting current .mp4 and .npy files
    # set segment = 1 to calculate/plot strain rate over 4 sectors
    # (method always calculates in sectors, but sums sectors if segment = 0 after saving synchrony parameters)
    def strain_rate(self, slice_, plot = 1, save = 1, segment = 0):  
        # relevant matrices for this slice
        V = self.V[f'V{slice_}']
        M = self.M[f'M{slice_}']
        mask = self.mask[f'mask{slice_}']
        mask_segment = self.mask_segment[f'mask_segment{slice_}']
        ID = self.ID + f', slice {slice_}'
        T_ed = self.T_ed[f'T_ed{slice_}']
        
        try:
            Va = self.V[f'V{slice_+1}']  # above
            Vb = self.V[f'V{slice_-1}']  # below
            Ma = self.M[f'M{slice_+1}']
            Mb = self.M[f'M{slice_-1}']
            mask_a = self.mask[f'mask{slice_+1}']
            mask_b = self.mask[f'mask{slice_-1}']
            
        except KeyError:
            raise Exception(f'\nSlice {slice_} is missing a slice above or below. \
                            \nChoose a slice between slices: \n{self.V.keys()}')
        
        # range of time-points
        self.range_ = np.array(range(T_ed))
        self.range_TR = self.range_*self.TR*1000  # plots in milliseconds
        
        # get data axis dimensions
        self.ax = len(mask[0,0,:,0])
        self.ay = len(mask[0,0,0,:])
        
        # each row contains strain rate data for sector 1, 2, 3, 4
        self.r_matrix = np.zeros((4, T_ed)); self.r_matrix[:, :] = np.nan
        self.c_matrix = np.zeros((4, T_ed)); self.c_matrix[:, :] = np.nan
        self.l_matrix = np.zeros((4, T_ed)); self.l_matrix[:, :] = np.nan

        # for each segment, we store angles distributions corresponding to positive/negative eigenvalues 
        self.theta1 = np.zeros((4, T_ed), dtype = 'object')  # 'positive' angles (stretch direction)
        self.theta2 = np.zeros((4, T_ed), dtype = 'object')  # 'negative' angles (compression direction)
        self.phi1 = np.zeros((4, T_ed), dtype = 'object') 
        self.phi2 = np.zeros((4, T_ed), dtype = 'object')
        
        # center of mass at t=0
        self.cx_0, self.cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[0, 0, :, :]))
        
        self.d = np.zeros(T_ed)
        
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
        
        sns.set_style("darkgrid", {'font.family': ['sans-serif'], 'font.sans-serif': ['DejaVu Sans']})
        
        print(f'Calculating Strain rate for {ID}...')
        
        run = 1  #
        for t in self.range_:

            # combodata mask 
            mask_t = mask[t, 0, :, :].T  #mask at this timepoint
            mask_ta = mask_a[t, 0, :, :].T  #mask above
            mask_tb = mask_b[t, 0, :, :].T  #mask below
            mask_segment_t = mask_segment[t, 0, :, :].T  #mask at this timepoint
            
            #find center of mass of filled mask (middle of the heart)
            cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
            
            # erode mask 
            mask_e = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
            
            # calculate certainty matrix from normalized magnitude plot
            self.C = M[t, 0, :, :].T/np.max(M[t, 0, :, :].T); C = self.C
            self.Ca = Ma[t, 0, :, :].T/np.max(Ma[t, 0, :, :].T); Ca = self.Ca
            self.Cb = Mb[t, 0, :, :].T/np.max(Mb[t, 0, :, :].T); Cb = self.Cb
            
            sector = 0  # reset sector number
            
            # remove nan's
            self.r_matrix[:, t] = 0; self.c_matrix[:, t] = 0; self.l_matrix[:, t] = 0
            
            # angles from sectors appended here, reset every t
            theta1_ = [[], [], [], []]; theta2_ = [[], [], [], []]
            phi1_ = [[], [], [], []]; phi2_ = [[], [], [], []]
            
            # amount of ellipses in this timepoint in each sector 1-4 is stored here
            e_count = np.zeros(4)
            
            self.vx = ndi.gaussian_filter(V[0, t, 0, :, :].T*C, self.sigma) / ndi.gaussian_filter(C, self.sigma)
            self.vy = ndi.gaussian_filter(V[1, t, 0, :, :].T*C, self.sigma) / ndi.gaussian_filter(C, self.sigma)
            self.vz = ndi.gaussian_filter(V[2, t, 0, :, :].T*C, self.sigma) / ndi.gaussian_filter(C, self.sigma)
            
            # the same fields one slice above and below
            # should we do this for C as well?
            self.vxa = ndi.gaussian_filter(Va[0, t, 0, :, :].T*Ca, self.sigma) / ndi.gaussian_filter(Ca, self.sigma) 
            self.vxb = ndi.gaussian_filter(Vb[0, t, 0, :, :].T*Cb, self.sigma) / ndi.gaussian_filter(Cb, self.sigma)
            
            self.vya = ndi.gaussian_filter(Va[1, t, 0, :, :].T*Ca, self.sigma) / ndi.gaussian_filter(Ca, self.sigma) 
            self.vyb = ndi.gaussian_filter(Vb[1, t, 0, :, :].T*Cb, self.sigma) / ndi.gaussian_filter(Cb, self.sigma)
            
            self.vza = ndi.gaussian_filter(Va[2, t, 0, :, :].T*Ca, self.sigma) / ndi.gaussian_filter(Ca, self.sigma)
            self.vzb = ndi.gaussian_filter(Vb[2, t, 0, :, :].T*Cb, self.sigma) / ndi.gaussian_filter(Cb, self.sigma)
            
            # calculate eigenvalues and vectors
            for x in range(0, self.ax, self.n):
                for y in range(0, self.ay, self.n): 
                    # search in eroded mask to avoid border artifacts
                    # get mask above and below, these are never exactly 0
                    if mask_t[x, y] == 1:
                        ## check if Va[x, y] or Vb[x, y] = 0 here, find nearest non-zero if so
                        if mask_ta[x,y] == 0:
                            # find closest non-zero index
                            idx, idy = self._nearest_nonzero_idx(mask_ta, x, y)
                            self.vxa[x,y] = self.vxa[idx, idy]
                            self.vya[x,y] = self.vya[idx, idy]
                            self.vza[x,y] = self.vza[idx, idy]
                            if all([self.vxa[x,y], self.vya[x,y], self.vza[x,y]]) == False:
                                print('Still Zeroes')
                                continue  # if all surrounding voxels are 0, continue
                        
                        elif mask_tb[x,y] == 0:
                            # find closest non-zero index
                            idx, idy = self._nearest_nonzero_idx(mask_tb, x, y)
                            self.vxb[x,y] = self.vxb[idx, idy]
                            self.vyb[x,y] = self.vyb[idx, idy]
                            self.vzb[x,y] = self.vzb[idx, idy]
                            if all([self.vxb[x,y], self.vyb[x,y], self.vzb[x,y]]) == False:
                                print('Still Zeroes')
                                continue
                            
                        else:
                            pass
                        
                        # SR tensor for point xy 
                        D_ = self._D_ij_3D(x, y, t)
                        
                        # from this point on its besically the same;
                        # 2d slice with but with 3d sr tensors
                        val, vec = np.linalg.eig(D_)
                        
                        # skip this voxel if all eigenvalue signs are equal
                        #if np.sign(val[0]) == np.sign(val[1]) == np.sign(val[2]):
                        #    continue
                        
                        self.d[t] += sum(val)
                        
                        # vector between center of mass and point (x, y, 0) 
                        r = np.array([x - cx, y - cy])
                        z = np.array([0, 0, 1])
                        #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                        
                        # index of eigenvalues
                        val_max_i = np.argmax(val)  # most positive value
                        val_min_i = np.argmin(val)  # most negative
                        val_last_i = 3 - val_min_i - val_max_i
                        
                        # find angle between xy-projections and r
                        theta = theta_rad(r, vec[val_max_i][:-1])  # highest eigenvector 
                        theta_ = theta_rad(r, vec[val_min_i][:-1])  # lowest eigenvector
                        theta__ = theta_rad(r, vec[val_last_i][:-1])  # third eigenvector 
                        
                        # projection angle in (r,z)-plane
                        a,b,c = vec[val_max_i]
                        #phi = theta_rad(z, [a*np.cos(theta), b*np.sin(theta), c])  # angle between highest eigenvector and z-axis
                        phi = theta_rad(z, np.array([a, b, c]))  # angle between highest eigenvector and z-axis
                        
                        a,b,c = vec[val_min_i]
                        #phi_ = theta_rad(z, [a*np.cos(theta_), b*np.sin(theta_), c]) # angle between lowest eigenvector and z-axis
                        phi_ = theta_rad(z, np.array([a, b, c]))
                        
                        a,b,c = vec[val_last_i]
                        #phi__ = theta_rad(z, [a*np.cos(theta__), b*np.sin(theta__), c]) # angle between third eigenvector and z-axis
                        phi__ = theta_rad(z, np.array([a, b, c]))
                        
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
                        
                        # adjacent (green)
                        elif sect_xy in range1 or (sect_xy not in range1_)*any(range1_) \
                            or sect_xy in range11 or (sect_xy not in range11_)*any(range11_):   
                            sector = 1
                            e_count[sector] += 1
                            
                        # medial (blue)                
                        elif sect_xy in range2 or (sect_xy not in range2_)*any(range2_) \
                            or sect_xy in range22 or (sect_xy not in range22_)*any(range22_):   
                            sector = 2
                            e_count[sector] += 1
                        
                        # remote (purple)
                        elif sect_xy in range(p_end, n_end) \
                            or (sect_xy not in range(n_end, p_end))*any(range(n_end, p_end)):
                            sector = 3
                            e_count[sector] += 1
                        
                        else:  # avoid plotting ellipses in invalid ranges
                            continue
                        
                        # longitudinal eigenvalue projection
                        val_l = (val[val_max_i])*abs(np.cos(phi))
                        val_l_ = (val[val_min_i])*abs(np.cos(phi_))
                        val_l__ = (val[val_last_i])*abs(np.cos(phi__))
                        
                        # in-plane projection of eigenvalue vector length
                        val_transverse = np.sqrt(val[val_max_i]**2 - val_l**2) * np.sign(val[val_max_i])
                        val_transverse_ = np.sqrt(val[val_min_i]**2 - val_l_**2) * np.sign(val[val_min_i])
                        val_transverse__ = np.sqrt(val[val_last_i]**2 - val_l__**2) * np.sign(val[val_last_i])
                        
                        # vector decomposition along radial (r) and circumferential (c) axes
                        self.r_matrix[sector, t] += val_transverse*abs(np.cos(theta)) \
                            + val_transverse_*abs(np.cos(theta_)) + val_transverse__*abs(np.cos(theta__))
                        self.c_matrix[sector, t] += val_transverse*abs(np.sin(theta)) \
                            + val_transverse_*abs(np.sin(theta_)) + val_transverse__*abs(np.sin(theta__))
                        self.l_matrix[sector, t] += val_l + val_l_ + val_l__
                        
                        # angle sum collected, scaled to get average angle each t
                        # does not assume that each 2d tensor has a positive and negative eigenvector
                        if val[val_max_i] > 0:
                            theta1_[sector].append(theta)
                            phi1_[sector].append(phi)
                        if val[val_min_i] > 0:
                            theta1_[sector].append(theta_)
                            phi1_[sector].append(phi_)
                        if val[val_last_i] > 0:
                            theta1_[sector].append(theta__)
                            phi1_[sector].append(phi__)
                            
                        if val[val_max_i] < 0:
                            theta2_[sector].append(theta) 
                            phi2_[sector].append(phi)
                        if val[val_min_i] < 0:
                            theta2_[sector].append(theta_)
                            phi2_[sector].append(phi_)
                        if val[val_last_i] < 0:
                            theta2_[sector].append(theta__)
                            phi2_[sector].append(phi__)
                            
            # if any of these are zero, we have likely reached the end of the sector mask. 
            # end at this timepoint, if so, to avoid division by zero.
            if np.all(e_count) == False:
                if run == 1:
                    T_ = t  # solves plotting issues from sector mask bug
                    run = 0
                    print(f'Ellipse count at {t*self.TR*1000} ms: {e_count}. Skipping the last timepoints to avoid division by zero.')
                continue
        
            for sector in range(4):  # scaling for correct units 
                self.r_matrix[sector, t] = self.r_matrix[sector, t] / (e_count[sector]*self.res)
                self.c_matrix[sector, t] = self.c_matrix[sector, t] / (e_count[sector]*self.res)
                self.l_matrix[sector, t] = self.l_matrix[sector, t] / (e_count[sector]*self.res)
            
                # collect angle distributions in degrees
                self.theta1[sector, t] = np.array(theta1_[sector])*180/np.pi
                self.theta2[sector, t] = np.array(theta2_[sector])*180/np.pi
                self.phi1[sector, t] = np.array(phi1_[sector])*180/np.pi
                self.phi2[sector, t] = np.array(phi2_[sector])*180/np.pi
        
        # if no issues, T_ed will be used as endpoint in plots
        if run == 1:
            T_ = T_ed
        
        # add strain rate parameters to dictionary
        r_sr_global = np.sum(self.r_matrix, axis = 0)[:T_] / 4
        c_sr_global = np.sum(self.c_matrix, axis = 0)[:T_] / 4
        l_sr_global = np.sum(self.l_matrix, axis = 0)[:T_] / 4
        
        self.c_sr_max = np.max(c_sr_global)
        self.r_sr_max = np.max(r_sr_global)
        self.c_sr_min = np.min(c_sr_global)
        self.r_sr_min = np.min(r_sr_global)
        
        # global strain arrays
        rs = 100*self._strain(r_sr_global, T_)
        cs = 100*self._strain(c_sr_global, T_)
        ls = 100*self._strain(l_sr_global, T_)
        
        # mean stretch/compression (theta1/theta2) angles
        theta1_mean = np.zeros((4, T_)); theta2_mean = np.zeros((4, T_))
        phi1_mean = np.zeros((4, T_)); phi2_mean = np.zeros((4, T_))
        for t in self.range_[:T_]:
            for sector in range(4):
                theta1_mean[sector, t] = np.mean(self.theta1[sector, t])
                theta2_mean[sector, t] = np.mean(self.theta2[sector, t])
                phi1_mean[sector, t] = np.mean(self.phi1[sector, t])
                phi2_mean[sector, t] = np.mean(self.phi2[sector, t])
             
        # mean angles
        theta1_mean_global = np.sum(theta1_mean, axis = 0) / 4
        theta2_mean_global = np.sum(theta2_mean, axis = 0) / 4
        phi1_mean_global = np.sum(phi1_mean, axis = 0) / 4
        phi2_mean_global = np.sum(phi2_mean, axis = 0) / 4
        
        # max/min of mean curve
        self.theta1_mean_max = np.max(theta1_mean_global)
        self.theta1_mean_min = np.min(theta1_mean_global)
        self.theta2_mean_max = np.max(theta2_mean_global)
        self.theta2_mean_min = np.min(theta2_mean_global)
        
        self.phi1_mean_max = np.max(phi1_mean_global)
        self.phi1_mean_min = np.min(phi1_mean_global)
        self.phi2_mean_max = np.max(phi2_mean_global)
        self.phi2_mean_min = np.min(phi2_mean_global)
        
        # strain curve analysis, synchrony of sectors
        self.c_peakvals = np.zeros(4); self.r_peakvals = np.zeros(4)
        self.c_peaktime = np.zeros(4); self.r_peaktime = np.zeros(4)
        
        for sector in range(4):
            rs = 100*self._strain(self.r_matrix[sector, :], T_)
            cs = 100*self._strain(self.c_matrix[sector, :], T_)
            
            # this regional data can be aquired for segment == 0 as well
            self.r_peakvals[sector] = np.max(rs); self.r_peaktime[sector] = np.argmax(rs)*self.TR*1000
            self.c_peakvals[sector] = np.min(cs); self.c_peaktime[sector] = np.argmin(cs)*self.TR*1000
            
        if plot == 1:
            # plot global strain rate

            plt.figure(figsize=(10, 8))

            plt.title(f'Global Strain Rate ({ID})', fontsize = 15)
            plt.axvline(self.T_es*self.TR*1000, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axhline(0, c = 'k', lw = 1)

            plt.xlim(0, T_*self.TR*1000)#; plt.ylim(0, 50)
            plt.xlabel('Time [ms]', fontsize = 15)
            plt.ylabel('$s^{-1}$', fontsize = 20)
            
            if segment == 1:
                plt.title(f'Regional Strain Rate ({ID})', fontsize = 15)
                
                for sector in range(4):
                    rs = self.r_matrix[sector, :]
                    cs = self.c_matrix[sector, :]
                    
                    plt.plot(self.range_TR, running_average(rs, 4), c = c_cmap(sector), lw=2)
                    plt.plot(self.range_TR, running_average(cs, 4), c = c_cmap(sector), lw=2)
                    
                    #include curve parameters in these plots too?
                    #plt.scatter(self.r_peaktime[sector], self.r_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                    #plt.scatter(self.c_peaktime[sector], self.c_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                
                plt.legend(handles = legend_handles1)
                    
            if segment == 0:
                plt.title(f'Global Strain Rate ({ID})', fontsize = 15)
                
                rsr = running_average(r_sr_global, 4)
                csr = running_average(c_sr_global, 4)
                lsr = running_average(l_sr_global, 4)
                
                plt.plot(self.range_TR[:len(r_sr_global)], rsr, c = 'darkblue', lw=2, label = 'Radial strain')
                plt.plot(self.range_TR[:len(c_sr_global)], csr, c = 'chocolate', lw=2, label = 'Circumferential strain')
                plt.plot(self.range_TR[:len(l_sr_global)], lsr, c = 'darkgreen', ls='--', lw=2, label = 'Longitudinal strain')
                
                plt.legend()

            plt.subplots_adjust(wspace=0.25)
            
            if save == 1:
                if os.path.exists(f'R:\Lasse\plots\MP4\{self.filename}') == False:
                    os.makedirs(f'R:\Lasse\plots\MP4\{self.filename}')
                    
                plt.savefig(f'R:\Lasse\plots\MP4\{self.filename}\{self.filename}_GSR.PNG')
            plt.show()
               
        if plot == 1:
            # plot strain 

            plt.figure(figsize=(8, 6))

            plt.axvline(self.T_es*self.TR*1000, c = 'k', ls = ':', lw = 2, label = 'End Systole')
            plt.axhline(0, c = 'k', lw = 1)

            plt.xlim(0, T_*self.TR*1000)#; plt.ylim(0, 50)
            plt.xlabel('Time [ms]', fontsize = 15)
            plt.ylabel('%', fontsize = 15)
                
            if segment == 1:
                plt.title(f'Regional Strain ({ID})', fontsize = 15)
                
                for sector in range(4):
                    rs = 100*self._strain(self.r_matrix[sector, :], T_)
                    cs = 100*self._strain(self.c_matrix[sector, :], T_)
                    
                    plt.plot(self.range_TR, rs, c = c_cmap(sector), lw=2)
                    plt.plot(self.range_TR, cs, c = c_cmap(sector), lw=2)
                    
                    plt.scatter(self.r_peaktime[sector], self.r_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                    plt.scatter(self.c_peaktime[sector], self.c_peakvals[sector], color = c_cmap(sector), marker = 'x', s = 100)
                
                plt.legend(handles = legend_handles1)
                    
            if segment == 0:
                plt.title(f'Global Strain ({ID})', fontsize = 15)
                
                rs = 100*self._strain(r_sr_global, T_)
                cs = 100*self._strain(c_sr_global, T_)
                ls = 100*self._strain(l_sr_global, T_)
                
                plt.plot(self.range_TR[:len(rsr)], rs, c = 'darkblue', lw=2, label = 'Radial strain')
                plt.plot(self.range_TR[:len(csr)], cs, c = 'chocolate', lw=2, label = 'Circumferential strain')
                plt.plot(self.range_TR[:len(lsr)], ls, c = 'darkgreen', lw=2, ls='--', label = 'Longitudinal strain')
                
                #plt.plot(self.range_TR[:len(cs)], np.gradient(cs), label = 'derived from cs', color = 'r')
                #plt.plot(self.range_TR[:len(rs)], np.gradient(rs), label = 'derived from rs', color = 'b')
                #plt.plot(self.range_TR[:len(ls)], np.gradient(ls), label = 'derived from ls', color = 'g')
                
                #plt.plot(self.range_TR[:len(cs)], r_sr_global[:len(cs)])
                
                plt.legend()
            
            plt.subplots_adjust(wspace=0.25)
            if save == 1:
                plt.savefig(f'R:\Lasse\plots\MP4\{self.filename}\{self.filename}_GS.PNG')
                
            plt.show()
            

            if segment == 1:  # mean angles segments
                c = 'viridis'
                c_cmap = plt.get_cmap(c)
                cmax = np.max(theta1_mean)
                cmin = np.min(theta1_mean)
                norm_ = mpl.colors.Normalize(vmin = cmin, vmax = cmax)
            
                f, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 4))
                f.suptitle(f'Strain rate mean $\\theta$ [degrees] ({ID})')
                
                # smoothed mean stretch / compression
                ax0.imshow(theta1_mean, cmap = c_cmap); ax0.grid(0)
                ax0.text(T_-0.5, 0.7, '∎', color = 'r', fontsize = 20)
                
                im = ax1.imshow(theta2_mean, cmap = c_cmap); ax1.grid(0)
                ax1.text(T_-0.5, 0.7, '∎', color = 'g', fontsize = 20)
                
                ax2.plot(self.range_TR[:T_], theta2_mean_global[:T_], c = 'g', lw = 1.5)
                ax2.plot(self.range_TR[:T_], theta1_mean_global[:T_], c = 'r', lw = 1.5)
                
                #ax2.plot(self.range_TR[:T_], phi1_mean[sector, :][:T_], color = c_cmap(sector))
                #ax2.plot(self.range_TR[:T_], phi2_mean[sector, :][:T_], color = 'lightgray')
                
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                f.colorbar(im, cax = cbar_ax, norm = norm_)
                      
            else:  # global angle distribution 
                fig = plt.figure(figsize=(12, 6))
                mpl.rc_file_defaults()  # remove sns style
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                
                plt.suptitle(f'Strain rate direction ({ID})', fontsize = 15)
                ax1.axvline(self.T_es*self.TR*1000, c = 'k', ls = ':', lw = 2, label = 'End Systole')
                ax1.set_xlim(0, T_*self.TR*1000)
                ax1.set_xlabel('Time [ms]', fontsize = 15)
                ax1.set_ylabel('$\\theta$', fontsize = 17)

                ax2.set_ylabel('$\\phi$', fontsize = 17)
                ax2.axvline(self.T_es*self.TR*1000, c = 'k', ls = ':', lw = 2, label = 'End Systole')
                ax2.set_xlim(0, T_*self.TR*1000)
                ax2.set_xlabel('Time [ms]', fontsize = 15)
                
                for i in self.range_[:T_]:
                    for sector in range(4):
                        #print(i, len(self.theta1[sector, i]), len(self.theta2[sector, i]))
                        ax1.scatter([self.range_TR[i]]*len(self.theta1[sector, i]), self.theta1[sector, i], color = 'r', alpha = 0.006*self.n**2)
                        ax1.scatter([self.range_TR[i]]*len(self.theta2[sector, i]), self.theta2[sector, i], color = 'g', alpha = 0.006*self.n**2)
                        ax2.scatter([self.range_TR[i]]*len(self.phi1[sector, i]), self.phi1[sector, i], color = 'r', alpha = 0.006*self.n**2)
                        ax2.scatter([self.range_TR[i]]*len(self.phi2[sector, i]), self.phi2[sector, i], color = 'g', alpha = 0.006*self.n**2)
            
                ax1.plot(self.range_TR[:T_], theta1_mean_global[:T_], 'r', label = 'Positive eigenvectors (stretch)')
                ax1.plot(self.range_TR[:T_], theta2_mean_global[:T_], 'g', label = 'Negative eigenvectors (compression)')
                ax2.plot(self.range_TR[:T_], phi1_mean_global[:T_], 'r', label = 'Stretch')
                ax2.plot(self.range_TR[:T_], phi2_mean_global[:T_], 'g', label = 'Compression')
                ax2.legend(loc = 'lower right')
                ax1.grid(0); ax2.grid(0)
            
            plt.subplots_adjust(wspace = 0.2)
            plt.show()
        
        if segment == 0:  # turn all return arrays global
            self.r_strain_rate = running_average(r_sr_global, 4)
            self.c_strain_rate = running_average(c_sr_global, 4)
            self.l_strain_rate = running_average(l_sr_global, 4)
            
            self.r_strain = rs
            self.c_strain = cs
            self.l_strain = ls
            
            self.theta1 = theta1_mean_global
            self.theta2 = theta2_mean_global
            self.phi1 = phi1_mean_global
            self.phi2 = phi2_mean_global
            
        else:
            self.theta1 = theta1_mean
            self.theta2 = theta2_mean
            
        if save == 1:
            # save strain/strain rate/angle dist npy files for analysis

            if os.path.exists(f'R:\Lasse\strain data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\strain data\{self.filename}')
                
            np.save(fr'R:\Lasse\strain data\{self.filename}\r_strain', self.r_strain)
            np.save(fr'R:\Lasse\strain data\{self.filename}\c_strain', self.c_strain)
                
            if os.path.exists(f'R:\Lasse\strain rate data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\strain rate data\{self.filename}')
            
            np.save(fr'R:\Lasse\strain rate data\{self.filename}\r_strain_rate', self.r_strain_rate)
            np.save(fr'R:\Lasse\strain rate data\{self.filename}\c_strain_rate', self.c_strain_rate)
            
            if os.path.exists(f'R:\Lasse\\angle distribution data\{self.filename}') == False:
                os.makedirs(f'R:\Lasse\\angle distribution data\{self.filename}')
            
            np.save(fr'R:\Lasse\\angle distribution data\{self.filename}\angle_distribution_pos', self.theta1)
            np.save(fr'R:\Lasse\\angle distribution data\{self.filename}\angle_distribution_neg', self.theta2)
                
        # if save = 0 the parameters can still be collected from return statement without overwriting 
        return self.r_matrix, self.c_matrix, self.theta1, self.theta2
            
        
        
#%%
# example of use
if __name__ == "__main__":
    st = time.time()
    # create instance for input combodata file
    run2 = ComboDataSR_3D('sham_D4-4_41d', n = 1)
    
    # get info/generate data 
    run2.overview()
    #grv2 = run2.velocity(slice_ = 9, dim = '3D', save = 0)  # mostly useful to see how velocity field behaves
    # plot = 1: show strain, strain rate, angle distribution
    # save = 1: save data arrays, videos to folder
    # segment = 1: regional analysis
    # slice: choose a slice between slices
    run2.strain_rate(plot = 1, slice_ = 6, save = 0, segment = 0)
    
    #print(run1.__dict__['r_peaktime'])  # example of dictionary functionality
    
    et = time.time()
    print(f'Time elapsed: {et-st:.3f} s')
    
#%%
    '''    
    d2 = run2.__dict__['d']  # divergence over time
    #d1 = run1.__dict__['d']
    plt.plot(range(len(d2)), d2, label = '3d') 
    plt.plot(range(len(d1)), d1, label = '2d')
    plt.legend(); plt.show()
    
    print(sum(d2))
    print(sum(d1))
    '''
