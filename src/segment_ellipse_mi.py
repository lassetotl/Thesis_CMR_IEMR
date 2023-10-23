"""
Created on 13.10.23

@author: lassetot

Note (Oct 13): duplicate of combodata_ellipse.py used to experiment with 
segmentation to look at homogeneity in strain and strain rate. A lot of 
troubleshooting features are removed to focus on segments.
"""


import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
from numpy.linalg import norm


from util import D_ij_2D, theta_rad, running_average, clockwise_angle
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
file = 'sham_D3-2_3d'
data = sio.loadmat(f'R:\Lasse\combodata_shax\{file}.mat')["ComboData_thisonly"]

#print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
#print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)
V = data['V'][0,0] #velocity field
M = data['Magn'][0,0] #magnitudes
mask = data['Mask'][0,0] #mask for non-heart tissue
mask_segment = data['MaskS_medium'][0,0]

T = len(V[0,0,0,:,0]) #Total amount of time steps
T_es = data['TimePointEndSystole'][0,0][0][0]
T_ed = data['TimePointEndDiastole'][0,0][0][0]
res = data['Resolution'][0,0][0][0]  # temporal resolution, need this for correct SR units?
TR = data['TR'][0,0][0][0]

# check if mi, collect infarct site mis if so

mis = np.nan
l = file.split('_')
if l[0] == 'mi':
    mis = data['InfarctSector'][0,0][0]
    
# if sham, just set arbitrary mis = [4, 12]
elif l[0] == 'sham':
    mis = [4, 13]
    
# to avoid reverse ranges like 'range(33, 17)'

print(f'Infarct Sector at [{mis[0], mis[1]}]')

print(f'{file} overview:')
print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')
print(f'Segment mask shape: {np.shape(mask_segment)}')

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

#%%

range_ = np.arange(0, T, 1)

# each row contains strain rate data for sector 1, 2, 3, 4
r_matrix = np.zeros((4, T)); r_matrix[:, :] = np.nan
c_matrix = np.zeros((4, T)); c_matrix[:, :] = np.nan

# for each segment, we store angles corresponding to positive/negative eigenvalues 
a1 = np.zeros((4, T), dtype = 'object') # 'positive' angles (stretch direction)
a2 = np.zeros((4, T), dtype = 'object') # 'negative' angles (compression direction)

#divergence
d = np.zeros(T)

# center of mass at t=0
cx_0, cy_0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, 0]))

# segment slices alloted to non-infarct sectors, rounded down to int
if mis[0] < mis[1]:
    infarct_length = mis[1] - mis[0]  # length in nr of sectors
else:
    infarct_length = mis[0] - 36 - mis[1]

sl = int(np.floor((36 - abs(infarct_length))/6))


sub = 0 # Graph subplot on (1) or off (0)
for t in range_[:T_ed+1]:
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    ax.text(3, 3, f'Gaussian smoothing ($\sigma = {sigma}$)', color = 'w')
    #ax.set_facecolor('b')
    
    # amount of ellipses in this timepoint in each sector 1-4 is stored here
    e_count = np.zeros(4)
    
    # combodata mask 
    mask_t = mask[:, :, 0, t] #mask at this timepoint
    mask_segment_t = mask_segment[:, :, 0, t] #mask at this timepoint
    
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
    #plt.imshow((mask_segment_t).T, origin = 'lower', cmap = 'autumn', alpha = 1)

    sector = 0  # reset sector number
    
    # remove nan's
    r_matrix[:, t] = 0; c_matrix[:, t] = 0
    
    # angles from sectors appended here, reset every t
    a1_ = [[], [], [], []]; a2_ = [[], [], [], []]
    
    for x in range(0, f, n):
        for y in range(0, f, n): 
            # search in eroded mask to avoid border artifacts
            if mask_e[x, y] == 1:
                # SR tensor for point xy
                D_ = D_ij_2D(x, y, V, M_norm, t, sigma, mask_t)     
                val, vec = np.linalg.eig(D_)
                
                # skip this voxel if eigenvalue signs are equal
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
                sect_xy = mask_segment_t[x, y]  # sector value in (x,y)
                
                # need two ranges for every segment to counter invalid ranges (f.ex. range(33, 17))
                # all values that add/subtract risk becoming negative or >36, thus %

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
                    r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                    
                
                # adjacent (green)
                elif sect_xy in range1 or (sect_xy not in range1_)*any(range1_) \
                    or sect_xy in range11 or (sect_xy not in range11_)*any(range11_):   
                        
                    sector = 1
                    e_count[sector] += 1
                    r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                    
                    
                # medial (blue)                
                elif sect_xy in range2 or (sect_xy not in range2_)*any(range2_) \
                    or sect_xy in range22 or (sect_xy not in range22_)*any(range22_):   
                        
                    sector = 2
                    e_count[sector] += 1
                    r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                
                
                # remote (purple)
                elif sect_xy in range(p_end, n_end) \
                    or (sect_xy not in range(n_end, p_end))*any(range(n_end, p_end)):
                    sector = 3
                    
                    e_count[sector] += 1
                    r_matrix[sector, t] += (val[val_max_i])*abs(np.cos(theta)) + (val[val_min_i])*abs(np.cos(theta_))
                    c_matrix[sector, t] += (val[val_max_i])*abs(np.sin(theta)) + (val[val_min_i])*abs(np.sin(theta_))
                
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
    ax.text(3, 6, 'Ellipse count:', color = 'w')
    ax.text(17, 6, f'{int(e_count[0])}', color = c_cmap(0))
    ax.text(21, 6, f'{int(e_count[1])}', color = c_cmap(1))
    ax.text(25, 6, f'{int(e_count[2])}', color = c_cmap(2))
    ax.text(29, 6, f'{int(e_count[3])}', color = c_cmap(3))
    
    # graph subplot values, scale with amount of ellipses
    # count ellipses for each segment?
    
    for sector in range(4):
        r_matrix[sector, t] = r_matrix[sector, t]/(e_count[sector]*res) #local radial strain rate this frame
        c_matrix[sector, t] = c_matrix[sector, t]/(e_count[sector]*res)  #local circumferential strain rate
        
        # collect angles in degrees
        a1[sector, t] = np.array(a1_[sector])*180/np.pi
        a2[sector, t] = np.array(a2_[sector])*180/np.pi
    
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
        
        for sector in range(4):
            plt.plot(range_, r_matrix[sector, :], c = c_cmap(sector))
            plt.plot(range_, c_matrix[sector, :], c = c_cmap(sector))
        
        plt.xlim(0, T)#; plt.ylim(0, 50)
        plt.xlabel('Timepoints', fontsize = 15)
        plt.ylabel('$s^{-1}$', fontsize = 20)
        
        plt.legend()
    
    legend_handles1 = [Line2D([0], [0], color = c_cmap(0), lw = 6.3, label = 'Infarct'),
              Line2D([0], [0], color = c_cmap(1), lw = 6.3, label = 'Adjacent'),
              Line2D([0], [0], color = c_cmap(2), lw = 6.3, label = 'Medial'),
              Line2D([0], [0], color = c_cmap(3), lw = 6.3, label = 'Remote')]
    plt.legend(handles = legend_handles1, fontsize = 12, loc = 'lower right')
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

for sector in range(4):
    plt.plot(range_TR, running_average(r_matrix[sector, :], N), c = c_cmap(sector), label = f'Sector {sector}')
    plt.plot(range_TR, running_average(c_matrix[sector, :], N), c = c_cmap(sector))

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
    w = np.tanh((T_ed-range_)/weight)[:T_ed]
    w_f = np.tanh(range_/weight)[:T_ed]

    strain = cumtrapz(strain_rate[:T_ed], range_TR[:T_ed], initial=0)
    strain_flipped = np.flip(cumtrapz(strain_rate[:T_ed][::-1], range_TR[:T_ed][::-1], initial=0))
    
    return (w*strain + w_f*strain_flipped)/2


#%%
#plot strain over time


plt.figure(figsize=(8, 6))

plt.title(f'Regional Strain over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.axhline(0, c = 'k', lw = 1)

plt.xlim(0, T_ed*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('%', fontsize = 15)

plt.plot(range_TR[:T_ed], 100*strain(r_matrix[0, :]), c = c_cmap(0), lw=2, label = 'Infarct')
plt.plot(range_TR[:T_ed], 100*strain(c_matrix[0, :]), c = c_cmap(0), lw=2) #walking average

plt.plot(range_TR[:T_ed], 100*strain(r_matrix[1, :]), c = c_cmap(1), lw=2, label = 'Adjacent')
plt.plot(range_TR[:T_ed], 100*strain(c_matrix[1, :]), c = c_cmap(1), lw=2) #walking average

plt.plot(range_TR[:T_ed], 100*strain(r_matrix[2, :]), c = c_cmap(2), lw=2, label = 'Medial')
plt.plot(range_TR[:T_ed], 100*strain(c_matrix[2, :]), c = c_cmap(2), lw=2) #walking average

plt.plot(range_TR[:T_ed], 100*strain(r_matrix[3, :]), c = c_cmap(3), lw=2, label = 'Remote')
plt.plot(range_TR[:T_ed], 100*strain(c_matrix[3, :]), c = c_cmap(3), lw=2) #walking average


plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()

#%%
#angles over time

plt.figure(figsize = (10, 8))
plt.title(f'Regional radial concentration over time ({file})', fontsize = 15)
plt.axvline(T_es*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(T_ed*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('Degrees', fontsize = 20)

a1_mean = np.zeros((4, T)); a2_mean = np.zeros((4, T))
for t in range_:
    for sector in range(4):
        a1_mean[sector, t] = np.mean(a1[sector, t])
        a2_mean[sector, t] = np.mean(a2[sector, t])

# mean angles
for sector in range(4):
    #plt.plot(range_TR, a1_std, color = c_cmap(sector), label = 'Positive eigenvectors (stretch)')
    #plt.plot(range_TR, a2_std, 'g', label = 'Negative eigenvectors (compression)')
    # difference
    plt.plot(range_TR, abs(a1_mean[sector, :] - a2_mean[sector, :]), color = c_cmap(sector), label = f'Sector {sector}')

plt.legend(loc = 'upper right')
plt.show()
#%%
#Generate mp4

## clear dump folder
filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range_]  
  
with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7, macro_block_size=1) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
        
# save strain npy files for analysis

if os.path.exists(f'R:\Lasse\strain data\{file}') == False:
    os.makedirs(f'R:\Lasse\strain data\{file}')

#np.save(fr'R:\Lasse\strain data\{file}\r_strain', r_strain)
#np.save(fr'R:\Lasse\strain data\{file}\c_strain', c_strain)
