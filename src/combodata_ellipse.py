"""
Created on Tue Aug 29 10:46:34 2023

@author: lassetot

Note (Sep 20): a lot of the current code is included for troubleshooting purposes, 
not all are strictly useful at the moment and will be trimmed eventually
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from utility import D_ij_2D, theta_rad, running_average, clockwise_angle
from utility import gaussian_2d, theta_extreme
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'mi_D4-6_1d'
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
print(f'{res}, {TR}')

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
c_cmap = plt.get_cmap('plasma')

# you can get color palettes from seaborn like this
#c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 8).as_hex())

norm_ = mpl.colors.Normalize(vmin = 0, vmax = 90)

#%%

range_ = np.arange(0, T, 1)
g1 = np.zeros(T); g1[:] = np.nan #Graph values
g2 = np.zeros(T); g2[:] = np.nan #Graph values

# temporary, separates positive and negative contributions over time
r1_ = np.zeros(T); r1_[:] = np.nan #Graph values, positive
r2_ = np.zeros(T); r2_[:] = np.nan #Graph values, negative
c1_ = np.zeros(T); c1_[:] = np.nan #Graph values
c2_ = np.zeros(T); c2_[:] = np.nan #Graph values

a1 = np.zeros(T, dtype = 'object') #  most positive angle (stretch)
a1_std = np.zeros(T) # std stored for each t
a2 = np.zeros(T, dtype = 'object') # most negative angle (compression)
a2_std = np.zeros(T)

#divergence
d = np.zeros(T)

# center of mass at t=0
cx_0, cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

sub = 1 # Graph subplot on (1) or off (0)
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
    
    # reset radial and circumferential contributions from last frame / initialize
    rad_e = 0 #radial components of eigenvectors, sum will be saved every t
    circ_e = 0 #circumferential ...
    
    r1_[t] = 0; r2_[t] = 0  # remove nan values for this t
    c1_[t] = 0; c2_[t] = 0
    
    a1_ = []; a2_ = []
    
    #calculate eigenvalues and vectors
    e_count = 0  # ellipse counter in this frame
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
                
                e_count += 1
                
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
                #print(theta, theta_)
                
                # for color coding; smallest angle between long-axis eigenvector (tends to correspond to positive eigenvalue) and r
                # max value is now 90 degrees (pi/2) relative to radial unit vector
                #theta_r = theta_rad(r, vec[np.argmax(val)])
                
                # radial/circumferential contributions from each eigenvector
                # scaled with amount of ellipses, varies because of dynamic mask
                
                #higher eigenvalues weighted higher (abs to not affect direction)
                r1 = (val[val_max_i])*abs(np.cos(theta))
                r2 = (val[val_min_i])*abs(np.cos(theta_))
                
                c1 = (val[val_max_i])*abs(np.sin(theta))
                c2 = (val[val_min_i])*abs(np.sin(theta_))
                
                # filtering positive and negative values
                r1_[t] += r1*int(r1 > 0) + r2*int(r2 > 0) # +
                r2_[t] += r1*int(r1 < 0) + r2*int(r2 < 0) # -
                c1_[t] += c1*int(c1 > 0) + c2*int(c2 > 0)
                c2_[t] += c1*int(c1 < 0) + c2*int(c2 < 0)
                
                # global contribution
                rad_e += r1 + r2
                circ_e += c1 + c2
                
                # angle sum collected, scaled to get average angle each t
                # does not assume that each 2d tensor has a positive and negative eigenvector
                if val[val_max_i] > 0:
                    a1_.append(theta) 
                if val[val_min_i] > 0:
                    a1_.append(theta_)
                    
                if val[val_max_i] < 0:
                    a2_.append(theta) 
                if val[val_min_i] < 0:
                    a2_.append(theta_)
                
                #print(r, vec[val_i], c)
                
                # for class, skip this part if only data is requested 
                
                # hex code, inputs in range (0, 1) so theta is scaled
                rgb = list(c_cmap(theta/(np.pi/2)))
                
                #rgb[3] = 0.03
                # different function to include alpha?
                hx = mpl.colors.rgb2hex(rgb)  # code with
                
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
                
                # eigenvector visualization
                #plt.quiver(x, y, vec[val_max_i][0], vec[val_max_i][1], color = hx, scale = 10/np.sqrt(val[val_max_i]))
                #plt.quiver(x, y, vec[val_min_i][0], vec[val_min_i][1], color = 'k', scale = 10/np.sqrt(val[val_min_i]))
    
    #I[t] = Invariant(I_[0], I_[1], D)
    
    ax.text(3, 6, f'{e_count} Ellipses', color = 'w')
    res_ = round(f*res, 4)
    ax.text(3, 9, f'{res_} x {res_} cm', color = 'w')
    
    # graph subplot values, scale with amount of ellipses
    g1[t] = rad_e/(e_count*res) #global radial strain rate this frame
    g2[t] = circ_e/(e_count*res)  #global circumferential strain rate
    
    # collect average angle in degrees
    a1[t] = np.array(a1_)*180/np.pi #np.mean(a1_)*180/np.pi 
    a2[t] = np.array(a2_)*180/np.pi #np.mean(a2_)*180/np.pi 
    a1_std[t] = (np.mean(a1_))*180/np.pi
    a2_std[t] = (np.mean(a2_))*180/np.pi

    plt.scatter(cx, cy, marker = 'x', c = 'w')
    #plt.scatter(mis[0], mis[1], marker = 'x', c = 'r')
 
    plt.title(f'Strain Rate at t = {t} ({file})', fontsize = 15)
    
    #z = 25
    #plt.xlim(cx_0-z, cx_0+z); plt.ylim(cy_0-z, cy_0+z)
    plt.xlim(0, f); plt.ylim(0, f)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = plt.cm.ScalarMappable(cmap = c_cmap, norm = norm_)
    cbar = plt.colorbar(sm, cax = cax)
    cbar.set_label('$\Theta$ (degrees)', fontsize = 15)
    
    
    
    if sub == 1:  # subplot graph
        plt.subplot(1, 2, 2) #TSR
        #plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
        plt.title('Global Strain Rate over time')
    
        plt.axhline(0, c = 'k', lw = 1)
        plt.axvline(T_es, c = 'k', ls = ':', lw = 2, label = 'End Systole')
        plt.axvline(T_ed, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
        
        plt.plot(range_, r1_, 'darkblue', label = 'Radial+')
        plt.plot(range_, c1_, 'chocolate', label = 'Circumferential+')
        
        
        plt.plot(range_, r2_, 'blue', label = 'Radial-')
        plt.plot(range_, c2_, 'orange', label = 'Circumferential-')
        
        plt.plot(range_, r1_ + r2_, 'blue', ls = '--', alpha = 0.2)
        plt.plot(range_, c1_ + c2_, 'orange', ls = '--', alpha = 0.2)
        
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
#angles over time

plt.figure(figsize = (10, 8))
plt.title(f'Average radial angles over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('Degrees', fontsize = 20)

for i in range_:
    #print((a2[i]))
    plt.scatter([range_TR[i]]*len(a1[i]), a1[i], color = 'r', alpha = 0.03)
    plt.scatter([range_TR[i]]*len(a2[i]), a2[i], color = 'g', alpha = 0.03)

# mean angles
plt.plot(range_TR, a1_std, 'r', label = 'Positive eigenvectors (stretch)')
plt.plot(range_TR, a2_std, 'g', label = 'Negative eigenvectors (compression)')
# difference
plt.plot(range_TR, abs(a1_std - a2_std), 'darkgray', ls = '--', label = 'Difference')

plt.legend(loc = 'upper right')
plt.show()

#%%
#last frame with running average

N = 4 #window
I_g1 = running_average(g1, N)
I_g2 = running_average(g2, N)

plt.figure(figsize=(10, 8))

plt.title(f'Global Strain rate over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)

plt.plot(range_TR, g1, 'lightgrey')
plt.plot(range_TR, g2, 'lightgrey')

plt.plot(range_TR, I_g1, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
plt.plot(range_TR, I_g2, 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average

plt.legend()

plt.subplots_adjust(wspace=0.25)

if os.path.exists(f'R:\Lasse\plots\MP4\{file}') == False:
    os.makedirs(f'R:\Lasse\plots\MP4\{file}')
    
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GSR.PNG')
plt.show()

#%%
#plot strain over time

# weighting for integrals in positive/flipped time directions
w = np.tanh((T_ed-range_)/10)[:T_ed+1]
w_f = np.tanh(range_/10)[:T_ed+1]

r_strain = cumtrapz(g1, range_TR, initial=0)[:T_ed+1]
r_strain_flipped = np.flip(cumtrapz(g1[:T_ed+1][::-1], range_TR[:T_ed+1][::-1], initial=0))

c_strain = cumtrapz(g2, range_TR, initial=0)[:T_ed+1]
c_strain_flipped = np.flip(cumtrapz(g2[:T_ed+1][::-1], range_TR[:T_ed+1][::-1], initial=0))

plt.figure(figsize=(10, 8))

plt.title(f'Global Strain over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
#plt.ylabel('$s^{-1}$', fontsize = 20)

plt.plot(range_TR[:T_ed+1], (w*r_strain + w_f*r_strain_flipped)/2, 'darkblue', lw=2, label = 'Radial (Walking Average)') #walking average
plt.plot(range_TR[:T_ed+1], (w*c_strain + w_f*c_strain_flipped)/2, 'chocolate', lw=2, label = 'Circumferential (Walking Average)') #walking average


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
