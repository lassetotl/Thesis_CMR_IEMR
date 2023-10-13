"""
Created on Tue 13.10.23

@author: lassetot

Note (Oct 13): duplicate of combodata_ellipse.py used to experiment with 
segmentation to look at homogeneity in strain and strain rate. A lot of 
troubleshooting features are removed to focus on segments.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from lasse_functions import D_ij_2D, theta_rad, running_average, clockwise_angle
from lasse_functions import gaussian_2d, theta_extreme
#import pandas as pd
import seaborn as sns
#import sklearn

import scipy.io as sio
import scipy.ndimage as ndi 
from scipy.signal import convolve2d
import scipy.interpolate as scint
from scipy.integrate import cumtrapz
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'sham_D7-1_6w'
data = sio.loadmat(f'R:\Lasse\combodata_shax\{file}.mat')["ComboData_thisonly"]

#print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
#print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)
V = data['V'][0,0] #velocity field
M = data['Magn'][0,0] #magnitudes
mask = data['Mask'][0,0] #mask for non-heart tissue

T = len(V[0,0,0,:,0]) #Total amount of time steps
T_es = data['TimePointEndSystole'][0,0][0][0]
T_ed = data['TimePointEndDiastole'][0,0][0][0]
res = data['Resolution'][0,0][0][0]  # temporal resolution, need this for correct SR units?
TR = data['TR'][0,0][0][0]

# check if mi, collect infarct site mis if so
'''
mis = np.nan
l = file.split('_')
if (l[0] == 'mi') is True:
    mis = data['InfarctSector'][0,0][0]
    print(f'Infarct Sector at {mis}')
'''
print(f'{file} overview:')
print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

print(f'End systole at t={T_es}, end diastole at t={T_ed}')

#%%
# visualizing Strain Rate
# 'fov'
f = 100

# plot every n'th ellipse
n = 2

# sigma value of gaussian used in data smoothing
sigma = 2

# cyclic colormap, normalize to radians 
#c_cmap = plt.get_cmap('plasma')

# you can get discrete nr of colors in palette corresponding to defined sectors
c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())

norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)


plt.imshow(np.array([[4, 1], [2, 3]]), cmap = c_cmap, origin = 'lower')
plt.colorbar()

#%%

range_ = np.arange(0, T, 1)
# initialize arrays for each quadrant
r1 = np.zeros(T); r1[:] = np.nan #Graph values
c1 = np.zeros(T); c1[:] = np.nan #Graph values

r2 = np.zeros(T); r2[:] = np.nan #Graph values
c2 = np.zeros(T); c2[:] = np.nan #Graph values

r3 = np.zeros(T); r3[:] = np.nan #Graph values
c3 = np.zeros(T); c3[:] = np.nan #Graph values

r4 = np.zeros(T); r4[:] = np.nan #Graph values
c4 = np.zeros(T); c4[:] = np.nan #Graph values

#divergence
d = np.zeros(T)

# center of mass at t=0
cx_0, cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

sub = 0 # Graph subplot on (1) or off (0)
for t in range_:
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    ax.text(3, 3, f'Gaussian smoothing ($\sigma = {sigma}$)', color = 'w')
    #ax.set_facecolor('b')
    
    # combodata mask 
    mask_t = mask[:, :, 0, t] #mask at this timepoint
    
    #find center of mass of filled mask (middle of the heart)
    cx, cy = ndi.center_of_mass(ndi.binary_fill_holes(mask_t))
    
    # erode mask 
    mask_e = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    # generate strain rate tensor
    #D = D_ij(V=V, M=M[:f, :f, 0, t], t=t, f=f, mask_ = 1)
    
    # plot magnitude M plot, normalize for certainty values
    # transpose to allign with mask
    M_norm = (M[:, :, 0, t]/np.max(M[:, :, 0, t]))
    plt.imshow(M_norm.T, origin = 'lower', cmap = 'gray', alpha = 1)
    #plt.imshow(mask_t.T, origin = 'lower', cmap = 'gray', alpha = 1)

    # ellipse counter in segment, this timepoint
    e_count1 = 0  
    e_count2 = 0
    e_count3 = 0
    e_count4 = 0
    sector = 0  # reset sector number
    
    # remove nan's
    r1[t] = 0; c1[t] = 0
    r2[t] = 0; c2[t] = 0
    r3[t] = 0; c3[t] = 0
    r4[t] = 0; c4[t] = 0
    
    for x in range(0, f, n):
        for y in range(0, f, n): 
            # search in eroded mask to avoid border artifacts
            if mask_e[x, y] == 1:
                # SR tensor for point xy
                D_ = D_ij_2D(x, y, V, M_norm, t, sigma, mask_t)     
                val, vec = np.linalg.eig(D_)
                
                # stop loop if eigenvalue signs are equal
                #if np.sign(val[0]) == np.sign(val[1]):
                    #continue
                
                # sum of eigenvalues represents divergence (?)
                d[t] += val[0] + val[1]
                
                # vector between center of mass and point (x, y) 
                r = np.array([x - cx, y - cy])
                #plt.quiver(cx, cy, r[0], r[1], scale = 50, width = 0.001)
                
                # index of eigenvalues
                val_max_i = np.argmax(val)  # most positive value
                val_min_i = np.argmin(val)  # most negative
                
                # color code in hex notation, from c value
                #I = val[0]**2 + val[1]**2  # invariant
                
                theta = theta_rad(r, vec[val_max_i])  # angle between highest eigenvector and r
                theta_ = theta_rad(r, vec[val_min_i]) # angle between lowest eigenvector and r
                
                # local contribution
                
                if (x > cx) and (y > cy):
                    sector = 0
                    e_count1 += 1
                    r1[t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c1[t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                    
                if (x > cx) and (y < cy):
                    sector = 1
                    e_count2 += 1
                    r2[t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c2[t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                    
                if (x < cx) and (y < cy):
                    sector = 2
                    e_count3 += 1
                    r3[t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c3[t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                    
                if (x < cx) and (y > cy):
                    sector = 3
                    e_count4 += 1
                    r4[t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c4[t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                
                
                # color code after sector 1 to 4
                hx = mpl.colors.rgb2hex(list(c_cmap(sector)))  # code with
                
                #hx = mpl.colors.rgb2hex(c_cmap(I))  # color code with invariant
                
                # angle between eigenvector and x-axis, converted to degrees anti-clockwise
                # clockwise theta needed
                theta_c = clockwise_angle(r, vec[val_max_i])
                e_angle = -(clockwise_angle([1,0], r) + theta_c)*180/np.pi
                
                # draw ellipses that are spanned by eigenvectors
                # eigenvalues are transformed (1 + tanh(val)) to have a circular unit ellipse
                ellipse = patches.Ellipse((x, y), (1 + np.tanh(val[val_max_i])), (1 + np.tanh(val[val_min_i])), 
                                          angle = e_angle, color = hx)
                
                ax.add_artist(ellipse)
    
    #ax.text(3, 6, f'{e_count} Ellipses', color = 'w')
    res_ = round(f*res, 4)
    ax.text(3, 9, f'{res_} x {res_} cm', color = 'w')
    
    # graph subplot values, scale with amount of ellipses
    # count ellipses for each segment?
    
    r1[t] = r1[t]/(e_count1*res) #local radial strain rate this frame
    c1[t] = c1[t]/(e_count1*res)  #local circumferential strain rate
    
    r2[t] = r2[t]/(e_count2*res) 
    c2[t] = c2[t]/(e_count2*res)
    
    r3[t] = r3[t]/(e_count3*res) 
    c3[t] = c3[t]/(e_count3*res)
    
    r4[t] = r4[t]/(e_count4*res) 
    c4[t] = c4[t]/(e_count4*res)
    
    plt.scatter(cx, cy, marker = 'x', c = 'w')
    #plt.scatter(mis[0], mis[1], marker = 'x', c = 'r')
 
    plt.title(f'Strain Rate at t = {t} ({file})', fontsize = 15)
    
    #z = 25
    #plt.xlim(cx_0-z, cx_0+z); plt.ylim(cy_0-z, cy_0+z)
    plt.xlim(0, f); plt.ylim(0, f)
    
    '''
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
    cbar = plt.colorbar(sm, cax = cax)
    cbar.set_label('$\Theta$ (degrees)', fontsize = 15)
    '''
    
    
    if sub == 1:  # subplot graph
        plt.subplot(1, 2, 2) #TSR
        #plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
        plt.title('Global Strain Rate over time')
    
        plt.axhline(0, c = 'k', lw = 1)
        plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        
        plt.plot(range_, r1, c = c_cmap(0), label = 'Sector 1')
        plt.plot(range_, c1, c = c_cmap(0))
        
        plt.plot(range_, r2, c = c_cmap(1), label = 'Sector 2')
        plt.plot(range_, c2, c = c_cmap(1))
        
        plt.plot(range_, r3, c = c_cmap(2), label = 'Sector 3')
        plt.plot(range_, c3, c = c_cmap(2))
        
        plt.plot(range_, r4, c = c_cmap(3), label = 'Sector 4')
        plt.plot(range_, c4, c = c_cmap(3))
        
        plt.xlim(0, T)#; plt.ylim(0, 50)
        plt.xlabel('Timepoints', fontsize = 15)
        plt.ylabel('$s^{-1}$', fontsize = 20)
        
        plt.legend()
    
    plt.tight_layout()
    #plt.autoscale()
    plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
    plt.show()

#%%
#divergence

# temporal resolution is calculated here
range_TR = np.array(range_)*TR

plt.figure(figsize = (10, 8))
plt.title(f'Divergence over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Time [s]', fontsize = 15)

plt.plot(range_TR, d, lw = 2)
plt.axhline(0, c = 'k', lw = 1)

plt.legend(loc = 'upper right')
plt.show()



#%%
#last frame with running average

N = 4 #window

plt.figure(figsize=(8, 6))

plt.title(f'Regional Strain rate ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)

plt.plot(range_TR, running_average(r1, N), c = c_cmap(0), label = 'Sector 1')
plt.plot(range_TR, running_average(c1, N), c = c_cmap(0))

plt.plot(range_TR, running_average(r2, N), c = c_cmap(1), label = 'Sector 2')
plt.plot(range_TR, running_average(c2, N), c = c_cmap(1))

plt.plot(range_TR, running_average(r3, N), c = c_cmap(2), label = 'Sector 3')
plt.plot(range_TR, running_average(c3, N), c = c_cmap(2))

plt.plot(range_TR, running_average(r4, N), c = c_cmap(3), label = 'Sector 4')
plt.plot(range_TR, running_average(c4, N), c = c_cmap(3))

plt.legend()

plt.subplots_adjust(wspace=0.25)

if os.path.exists(f'R:\Lasse\plots\MP4\{file}') == False:
    os.makedirs(f'R:\Lasse\plots\MP4\{file}')
    
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GSR.PNG')
plt.show()

#%%
# integration

# input array of strain rate data
def strain(strain_rate, weight = 10):
    # weighting for integrals in positive/flipped time directions
    w = np.tanh((T_ed-range_)/weight)[:T_ed+1]
    w_f = np.tanh(range_/weight)[:T_ed+1]

    strain = cumtrapz(strain_rate, range_TR, initial=0)[:T_ed+1]
    strain_flipped = np.flip(cumtrapz(strain_rate[:T_ed+1][::-1], range_TR[:T_ed+1][::-1], initial=0))
    
    return (w*strain + w_f*strain_flipped)/2


#%%
#plot strain over time

plt.figure(figsize=(8, 6))

plt.title(f'Regional Strain over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('%', fontsize = 15)

plt.plot(range_TR[:T_ed+1], 100*strain(r1), c = c_cmap(0), lw=2, label = 'Sector 1')
plt.plot(range_TR[:T_ed+1], 100*strain(c1), c = c_cmap(0), lw=2) #walking average

plt.plot(range_TR[:T_ed+1], 100*strain(r2), c = c_cmap(1), lw=2, label = 'Sector 2')
plt.plot(range_TR[:T_ed+1], 100*strain(c2), c = c_cmap(1), lw=2) #walking average

plt.plot(range_TR[:T_ed+1], 100*strain(r3), c = c_cmap(2), lw=2, label = 'Sector 3')
plt.plot(range_TR[:T_ed+1], 100*strain(c3), c = c_cmap(2), lw=2) #walking average

plt.plot(range_TR[:T_ed+1], 100*strain(r4), c = c_cmap(3), lw=2, label = 'Sector 4')
plt.plot(range_TR[:T_ed+1], 100*strain(c4), c = c_cmap(3), lw=2) #walking average


plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()

#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range_]  
  
with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
        
# save strain npy files for analysis

if os.path.exists(f'R:\Lasse\strain data\{file}') == False:
    os.makedirs(f'R:\Lasse\strain data\{file}')

#np.save(fr'R:\Lasse\strain data\{file}\r_strain', r_strain)
#np.save(fr'R:\Lasse\strain data\{file}\c_strain', c_strain)
