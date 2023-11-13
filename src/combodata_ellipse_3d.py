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
#import seaborn as sns
#import sklearn

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
#file = 'ComboData_PC(SIMULA_220404_D4-4_s_2017051502)'
file ='ComboData_PC(SIMULA_220407b_D3-2_s_2017050802)'
#data = sio.loadmat(f'R:\Lasse\combodata_3d_shax\{file}.mat')['ComboData']['pss0']
#data = mat73.loadmat(f'R:\Lasse\combodata_3d_shax\{file}.mat')
data = h5py.File(f'R:\Lasse\combodata_3d_shax\{file}.mat', 'r')['ComboData']

pss0 = [float(data[data['pss0'][i,0]][0,0]) for i in range(len(data['pss0']))]  # dictionary with slice nr?

# sorted slice order and z positions
idx, pss0 = zip(*sorted(list(enumerate(pss0)), reverse = True, key = lambda x: x[1]))
print(idx, pss0)  # slice positions in order

#print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
#print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes (the class should collect all requested slices)
# global parameters for this set

T_es = float(data[data['TimePointEndSystole'][0,0]][0,0])
T_ed = float(data[data['TimePointEndDiastole'][0,0]][0,0])
res = float(data[data['Resolution'][0,0]][0,0])  # spatial resolution in cm
slicethickness = float(data[data['SliceThickness'][0,0]][0,0])  # in mm
TR = float(data[data['TR'][0,0]][0,0])  # temporal resolution in s
ShortDesc = data['ShortDesc']
slices = len(ShortDesc)  # nr of slices in this file


V = {}; M = {}; mask = {}  # dictionary keys for all slices
slicenr = {}  # dictionary of short descriptions
for slice_ in range(slices):
    V[f'V{slice_ + 1}'] = np.array(data[data['V'][idx[slice_], 0]])  # velocity field for one slice
    M[f'M{slice_ + 1}'] = np.array(data[data['Magn'][idx[slice_], 0]]) #magnitudes
    mask[f'mask{slice_ + 1}'] = np.array(data[data['Mask'][idx[slice_], 0]]) #mask for non-heart tissue 
    
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

#%%
#3d plot of mask layers

n = 2
for t in range(T):
    stack = []
    for slice_ in range(1, slices+1):
        stack.append(mask[f'mask{slice_}'][t,0,:,:])
    mask_stack = ndi.binary_erosion(np.array(stack))
    
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    
    for slice_ in range(0, slices):
        for i in range(0, len(mask_stack[0, 0]), n):
            for j in range(0, len(mask_stack[0, 1]), n):
                if mask_stack[slice_, i, j] == 1:
                    # in class: color scaled to radial angle and alpha to invariant?
                    ax.scatter(i, j, slice_, color = 'k', alpha = 0.5)
    
    # in class, center on cx0, cy0 for selected slice
    ax.set_xlim([40, 80])
    ax.set_ylim([40, 80])
    ax.set_zlim([0, 8])
    
    plt.savefig(f'R:\Lasse\plots\Vdump\V(t={t}).PNG')
    plt.show()
    
# save video in folder named after filename
filenames = [f'R:\Lasse\plots\Vdump\V(t={t}).PNG' for t in range(T)]

with imageio.get_writer('R:\Lasse\plots\MP4\\3D heart.gif', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)