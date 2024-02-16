"""
Created on Tue Nov 2
Author Lasse Totland

this is an expansion of the initial ellipse plot to try and make it work for 3d 
collection of shax combodata slices (work in progress)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from util import D_ij_2D, theta_rad, running_average, clockwise_angle
from util import gaussian_2d, theta_extreme
#import pandas as pd
import seaborn as sns; sns.set_style("darkgrid", {'font.family': ['sans-serif'], 'font.sans-serif': ['DejaVu Sans']})

#import sklearn

from ComboDataSR_3D import ComboDataSR_3D
from math import ceil, floor

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import convolve2d
import scipy.interpolate as scint
from scipy.integrate import cumtrapz
import imageio
import copy
import mat73, h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file ='mi_D12-8_45d'
#file = 'mi_ten66-m2_'

#data = sio.loadmat(f'R:\Lasse\combodata_3d_shax\{file}.mat')['ComboData']['pss0']
#data = mat73.loadmat(f'R:\Lasse\combodata_3d_shax\{file}.mat')
#data = h5py.File(f'R:\Lasse\combodata_3d_shax\{file}.mat', 'r')['ComboData']

# sorts h5py compatible files and not
try:
    # use this method 
    data = h5py.File(f'R:\Lasse\combodata_3d_shax\{file}.mat', 'r')['ComboData']
    pss0 = [float(data[data['pss0'][i,0]][0,0]) for i in range(len(data['pss0']))]  # dictionary with slice nr?
    # sorted slice order and z positions
    idx, pss0 = zip(*sorted(list(enumerate(pss0)), reverse = True, key = lambda x: x[1]))
    print(idx, pss0)  # slice positions in order
    
    # global parameters
    T_es = float(data[data['TimePointEndSystole'][0,0]][0,0])
    res = float(data[data['Resolution'][0,0]][0,0])  # spatial resolution in cm
    slicethickness = float(data[data['SliceThickness'][0,0]][0,0])  # in mm
    TR = float(data[data['TR'][0,0]][0,0])  # temporal resolution in s
    ShortDesc = data['ShortDesc']
    slices = len(ShortDesc)  # nr of slices in this file

    T_ed = []
    V = {}; M = {}; mask = {}  # dictionary keys for all slices
    slicenr = {}  # dictionary of short descriptions
    for slice_ in range(slices):
        V[f'V{slice_ + 1}'] = np.array(data[data['V'][idx[slice_], 0]])  # velocity field for one slice
        M[f'M{slice_ + 1}'] = np.array(data[data['Magn'][idx[slice_], 0]]) #magnitudes
        mask[f'mask{slice_ + 1}'] = np.array(data[data['Mask'][idx[slice_], 0]]) #mask for non-heart tissue 
        
        T_ed.append(int(data[data['TimePointEndDiastole'][idx[slice_], 0]][0,0]))
        
        # check if mi, collect infarct site mis if so
        '''
        mis = np.nan
        l = file.split('_')
        if (l[0] == 'mi') is True:
            mis = data['InfarctSector'][0,0][0]
            print(f'Infarct Sector at {mis}')
        '''
        
        a = []  # construct ShortDescription
        for i in range(len(data[ShortDesc[0,0]])):
            try: 
                data[ShortDesc[idx[slice_], 0]][i,0]
            except IndexError:
                break
            else:
                a.append(chr(data[ShortDesc[idx[slice_], 0]][i,0]))
        
        desc = ''.join(a)
        snr = desc.split(' ')[-2][-2:].lstrip('0')
        #print(desc)  # slice nr
        slicenr[f'slice {slice_+1}'] = snr
        
        plt.title(f'Slice {desc}')
        plt.imshow(mask[f'mask{slice_ + 1}'][25,0,:,:], origin = 'lower')
        plt.show()
        # dont need to transverse mask? this could lead to indexing confusion later
        # has the structure organization changed from the original combodata?

    T = len(V['V1'][0,:,0,0,0]) #Total amount of time steps
    
except OSError:  
    data = sio.loadmat(f'R:\Lasse\combodata_3d_shax\{file}.mat')['ComboData']
    pss0 = [float(data['PVM_SPackArrSliceOffset'][0,i]) for i in range(len(data['PVM_SPackArrSliceOffset'][0,:]))]  # dictionary with slice nr?
    idx, pss0 = zip(*sorted(list(enumerate(pss0)), reverse = True, key = lambda x: x[1]))
    print(idx, pss0)  # slice positions in order
    
    T_es = float(data['TimePointEndSystole'][0,0])
    T_ed = []
    res = float(data['Resolution'][0,0])  # spatial resolution in cm
    slicethickness = float(data['SliceThickness'][0,0])  # in mm
    TR = float(data['TR'][0,0])  # temporal resolution in s
    ShortDesc = data['ShortDesc']
    slices = len(ShortDesc[0])  # nr of slices in this file
    
    V = {}; M = {}; mask = {}  # dictionary keys for all slices
    slicenr = {}  # dictionary of short descriptions
    for slice_ in range(slices):
        # transpose fields to allign with h5py import
        V[f'V{slice_ + 1}'] = np.array(data['V'][0, idx[slice_]].T)  # velocity field for one slice
        M[f'M{slice_ + 1}'] = np.array(data['Magn'][0, idx[slice_]].T) #magnitudes
        mask[f'mask{slice_ + 1}'] = np.array(data['Mask'][0, idx[slice_]].T) #mask for non-heart tissue 
        
        T_ed.append(int(data['TimePointEndDiastole'][0, idx[slice_]]))
        
        # check if mi, collect infarct site mis if so
        '''
        mis = np.nan
        l = file.split('_')
        if (l[0] == 'mi') is True:
            mis = data['InfarctSector'][0,0][0]
            print(f'Infarct Sector at {mis}')
        '''
        
        plt.title(f'{file}')
        plt.imshow(mask[f'mask{slice_ + 1}'][25,0,:,:], origin = 'lower')
        plt.show()
        # dont need to transverse mask? this could lead to indexing confusion later
        # has the structure organization changed from the original combodata?

    T = len(V['V1'][0,:,0,0,0]) #Total amount of time steps


#%%
#3d plot of mask layers
'''
n = 3
for t in range(T):
    stack = []
    for slice_ in range(1, slices+1):
        stack.append(mask[f'mask{slice_}'][t,0,:,:])
    mask_stack = np.array(stack) #ndi.binary_erosion(np.array(stack))
    
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    
    for slice_ in range(0, slices):
        for i in range(0, len(mask_stack[0, 0]), n):
            for j in range(0, len(mask_stack[0, 1]), n):
                if mask_stack[slice_, i, j] == 1:
                    # in class: color scaled to radial angle and alpha to invariant?
                    ax.scatter(i, j, slice_, color = 'k', alpha = 0.5)
    
    # in class, center on cx0, cy0 for selected slice
    ax.set_xlim([20, 80])
    ax.set_ylim([20, 80])
    ax.set_zlim([0, 8])
    
    plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
    plt.show()
    
# save video in folder named after filename
filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in range(T)]

with imageio.get_writer('R:\Lasse\plots\MP4\\3D heart.gif', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
        '''

#%%
# using 3d class to calculate strain in whole heart

#slice_selection 
total_lsr = []; total_csr = []; total_rsr = []
total_ls = []; total_cs = []; total_rs = []
theta_stretch = []; theta_comp = []
phi_stretch = []; phi_comp = []

slice_selection = np.arange(2, slices) 
#slice_selection = [2,3,4,5,6,7,8,9]
T_ed_min = np.min(np.array(T_ed))

run = ComboDataSR_3D(file, n = 1)
for slice_ in slice_selection:
    run.strain_rate(slice_, ellipse = 0, save = 0, plot = 0)
    print(f'Slice [{slice_} / {slice_selection[-1]}]')
    
    # sector mask issues can be spotted like this:
    if all(np.array(run.__dict__['theta1'])[:T_ed_min]) == True:
        total_ls.append(np.array(run.__dict__['l_strain'])[:T_ed_min])
        total_cs.append(np.array(run.__dict__['c_strain'])[:T_ed_min])
        total_rs.append(np.array(run.__dict__['r_strain'])[:T_ed_min])
        total_lsr.append(np.array(run.__dict__['l_strain_rate'])[:T_ed_min])
        total_csr.append(np.array(run.__dict__['c_strain_rate'])[:T_ed_min])
        total_rsr.append(np.array(run.__dict__['r_strain_rate'])[:T_ed_min])
        
        theta_stretch.append(np.array(run.__dict__['theta1'])[:T_ed_min])
        theta_comp.append(np.array(run.__dict__['theta2'])[:T_ed_min])
        phi_stretch.append(np.array(run.__dict__['phi1'])[:T_ed_min])
        phi_comp.append(np.array(run.__dict__['phi2'])[:T_ed_min])
    
ID = run.__dict__['ID']

#%%

lsr = np.sum(np.array(total_lsr), axis = 0) / len(slice_selection)
csr = np.sum(np.array(total_csr), axis = 0) / len(slice_selection)
rsr = np.sum(np.array(total_rsr), axis = 0) / len(slice_selection)

theta1 = np.sum(np.array(theta_stretch), axis = 0) / len(slice_selection) 
theta2 = np.sum(np.array(theta_comp), axis = 0) / len(slice_selection) 
phi1 = np.sum(np.array(phi_stretch), axis = 0) / len(slice_selection) 
phi2 = np.sum(np.array(phi_comp), axis = 0) / len(slice_selection) 

# regional analysis
# let apical/basal overlap with one level if odd nr of slices
odd = int((len(slice_selection) % 2) == 0)  # odd = 0 if odd, 1 if even
a_length = len(theta1[:ceil((len(slice_selection)/2))])
b_start = len(theta1[:floor((len(slice_selection)/2))])

basal_theta1 = np.sum(np.array(theta_stretch[b_start:]), axis = 0) / a_length
apical_theta1 = np.sum(np.array(theta_stretch[:a_length]), axis = 0) / a_length
basal_theta2 = np.sum(np.array(theta_comp[b_start:]), axis = 0) / a_length
apical_theta2 = np.sum(np.array(theta_comp[:a_length]), axis = 0) / a_length

basal_phi1 = np.sum(np.array(phi_stretch[b_start:]), axis = 0) / a_length
apical_phi1 = np.sum(np.array(phi_stretch[:a_length]), axis = 0) / a_length
basal_phi2 = np.sum(np.array(phi_comp[b_start:]), axis = 0) / a_length
apical_phi2 = np.sum(np.array(phi_comp[:a_length]), axis = 0) / a_length

basal_lsr = np.sum(np.array(total_lsr[b_start:]), axis = 0) / a_length
apical_lsr = np.sum(np.array(total_lsr[:a_length]), axis = 0) / a_length

basal_rsr = np.sum(np.array(total_rsr[b_start:]), axis = 0) / a_length
apical_rsr = np.sum(np.array(total_rsr[:a_length]), axis = 0) / a_length

basal_csr = np.sum(np.array(total_csr[b_start:]), axis = 0) / a_length
apical_csr = np.sum(np.array(total_csr[:a_length]), axis = 0) / a_length

TR = run.__dict__['TR']
range_ = np.arange(0, T_ed_min)
range_TR = range_*TR

# input array of strain rate data
# (used internally by later methods)
def strain(strain_rate, T_ed, weight = 10):  # inherit from 2d class?
    # weighting for integrals in positive/flipped time directions
    # cyclic boundary conditions
    w = np.tanh((T_ed - 1 - range_)/weight) 
    w_f = np.tanh(range_/weight) 

    strain = cumtrapz(strain_rate, range_TR/1000, initial=0)
    strain_flipped = np.flip(cumtrapz(strain_rate[::-1]/1000, range_TR[::-1], initial=0))
    return (w*strain + w_f*strain_flipped)/2

# derive strain from the total sr curve, not sum of strain curves
#ls = strain(lsr, T_ed_min)*100000
bls = strain(basal_lsr, T_ed_min)*100000
als = strain(apical_lsr, T_ed_min)*100000

#cs = strain(csr, T_ed_min)*100000
bcs = strain(basal_csr, T_ed_min)*100000
acs = strain(apical_csr, T_ed_min)*100000

#rs = strain(rsr, T_ed_min)*100000
brs = strain(basal_rsr, T_ed_min)*100000
ars = strain(apical_rsr, T_ed_min)*100000

#%%
# total strain and strain rate

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax2.axhline(0, c = 'k', lw = 1)
#ax2.plot(range(len(ls)), ls, 'darkgreen', label = 'Longitudinal strain')
ax2.plot(range(len(bls)), bls, 'darkgreen', label = 'GLS')
ax2.plot(range(len(als)), als, 'darkgreen', ls = '--')

#ax2.plot(range(len(cs)), cs, 'chocolate', label = 'Circumferential strain')
ax2.plot(range(len(bcs)), bcs, 'chocolate', label = 'GCS')
ax2.plot(range(len(acs)), acs, 'chocolate', ls = '--')

#ax2.plot(range(len(rs)), rs, 'darkblue', label = 'Radial strain')
ax2.plot(range(len(brs)), brs, 'darkblue', label = 'GRS')
ax2.plot(range(len(ars)), ars, 'darkblue', ls = '--')

ax1.axhline(0, c = 'k', lw = 1)
#ax1.plot(range(len(lsr)), lsr, 'darkgreen', label = 'Longitudinal strain rate')
ax1.plot(range(len(basal_lsr)), basal_lsr, 'darkgreen', label = 'GLSR')
ax1.plot(range(len(apical_lsr)), apical_lsr, 'darkgreen', ls = '--')

#ax1.plot(range(len(csr)), csr, 'chocolate', label = 'Circumferential strain rate')
ax1.plot(range(len(basal_csr)), basal_csr, 'chocolate', label = 'GCSR')
ax1.plot(range(len(apical_csr)), apical_csr, 'chocolate', ls = '--')

#ax1.plot(range(len(rsr)), rsr, 'darkblue', label = 'Radial strain rate')
ax1.plot(range(len(basal_rsr)), basal_rsr, 'darkblue', label = 'GRSR')
ax1.plot(range(len(apical_rsr)), apical_rsr, 'darkblue', ls = '--')

plt.suptitle(f'Whole heart strain ({ID}: {len(slice_selection)} slices)', fontsize = 15)
ax1.set_ylabel('$s^{-1}$', fontsize = 15)

ax2.set_ylabel('%', fontsize = 17)
#ax2.set_xlabel('Time [s]', fontsize = 15)


plt.subplots_adjust(wspace=0.3)
ax1.legend(); ax2.legend(); plt.show()

#%%
# strain plots with separated slices
cmax = np.max(total_rs)
cmin = np.min(total_cs)
c = 'inferno'
c_cmap = plt.get_cmap(c)
norm_ = mpl.colors.Normalize(vmin = cmin, vmax = cmax)

yticks = [0, len(slice_selection)-1]
yticks_new = ['A', 'B']

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(f'Whole heart strain [$\%$] ({ID}: {len(slice_selection)} slices)', fontsize = 15)

axs[0].imshow(np.array(total_rs), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[1].imshow(np.array(total_cs), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
im = axs[2].imshow(np.array(total_ls), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[2].set_xlabel('Timepoints', fontsize = 15)

for i in range(3):
    axs[i].set_yticks(yticks); axs[i].set_yticklabels(yticks_new)
    axs[i].grid(0)

axs[0].set_ylabel('Radial'); axs[2].set_ylabel('Longitudinal'); axs[1].set_ylabel('Circumferential')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax = cbar_ax, norm = norm_)

plt.show()

# strain rate plots with separated slices
cmax = np.max(total_rsr)
cmin = np.min(total_csr)
c_cmap = plt.get_cmap(c)
norm_ = mpl.colors.Normalize(vmin = cmin, vmax = cmax)

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(fr'Whole heart strain rate [$1/s$] ({ID}: {len(slice_selection)} slices)', fontsize = 15)

axs[0].imshow(np.array(total_rsr), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[1].imshow(np.array(total_csr), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
im = axs[2].imshow(np.array(total_lsr), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[2].set_xlabel('Timepoints', fontsize = 15)

for i in range(3):
    axs[i].set_yticks(yticks); axs[i].set_yticklabels(yticks_new)
    axs[i].grid(0)

axs[0].set_ylabel('Radial'); axs[2].set_ylabel('Longitudinal'); axs[1].set_ylabel('Circumferential')


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax = cbar_ax, norm = norm_)

plt.show()

# strain rate mean angle plots with separated slices
cmax = np.max(theta1)
cmin = np.min(theta1)
c = 'viridis'
c_cmap = plt.get_cmap(c)
norm_ = mpl.colors.Normalize(vmin = cmin, vmax = cmax)

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(f'Strain rate mean $\\theta$ [degrees] ({ID}: {len(slice_selection)} slices)', fontsize = 15)

axs[0].imshow(np.array(theta_stretch), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[0].text(T_ed_min-0.5, 0.7, '∎', color = 'r', fontsize = 20)
axs[0].axhline(int(len(slice_selection)/2) - odd*0.5, ls = '--', color = 'k')

im = axs[1].imshow(np.array(theta_comp), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[1].text(T_ed_min-0.5, 0.7, '∎', color = 'g', fontsize = 20)
axs[1].axhline(int(len(slice_selection)/2) - odd*0.5, ls = '--', color = 'k')

#axs[2].plot(range(len(theta1)), theta1, 'gray')
axs[2].plot(range(len(theta1)), basal_theta1, 'r-')
axs[2].plot(range(len(theta1)), apical_theta1, 'r--')

#axs[2].plot(range(len(theta2)), theta2, 'g')
axs[2].plot(range(len(theta2)), basal_theta2, 'g-')
axs[2].plot(range(len(theta2)), apical_theta2, 'g--')


axs[2].set_xlabel('Timepoints', fontsize = 15)

for i in range(2):
    axs[i].set_yticks(yticks); axs[i].set_yticklabels(yticks_new)
    axs[i].grid(0)

axs[0].set_ylabel('Stretch'); axs[1].set_ylabel('Compression'); axs[2].set_ylabel('Mean')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax = cbar_ax, norm = norm_)

plt.show()

cmax = np.max(phi2+5)
cmin = np.min(phi2-5)
c_cmap = plt.get_cmap(c)
norm_ = mpl.colors.Normalize(vmin = cmin, vmax = cmax)

fig, axs = plt.subplots(3, sharex=True)
fig.suptitle(f'Strain rate mean $\\phi$ [degrees] ({ID}: {len(slice_selection)} slices)', fontsize = 15)

axs[0].imshow(np.array(phi_stretch), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[0].text(T_ed_min-0.5, 0.7, '∎', color = 'r', fontsize = 20)
axs[0].axhline(int(len(slice_selection)/2) - odd*0.5, ls = '--', color = 'k')
im = axs[1].imshow(np.array(phi_comp), vmin = cmin, vmax = cmax, cmap = c_cmap, aspect = 'auto')
axs[1].text(T_ed_min-0.5, 0.7, '∎', color = 'g', fontsize = 20)
axs[1].axhline(int(len(slice_selection)/2) - odd*0.5, ls = '--', color = 'k')

#axs[2].plot(range(len(theta1)), phi1, 'gray')
axs[2].plot(range(len(phi1)), basal_phi1, 'r-')
axs[2].plot(range(len(phi1)), apical_phi1, 'r--')

#axs[2].plot(range(len(theta2)), phi2, 'g')
axs[2].plot(range(len(phi2)), basal_phi2, 'g-')
axs[2].plot(range(len(phi2)), apical_phi2, 'g--')

axs[2].set_xlabel('Timepoints', fontsize = 15)

for i in range(2):
    axs[i].set_yticks(yticks); axs[i].set_yticklabels(yticks_new)
    axs[i].grid(0)

axs[0].set_ylabel('Stretch'); axs[1].set_ylabel('Compression'); axs[2].set_ylabel('Mean')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax = cbar_ax, norm = norm_)

plt.show()