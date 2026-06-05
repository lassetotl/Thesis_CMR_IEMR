# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:00:31 2026

@author: lasse
collect magnitude images at T_ed and T_es at 40 days
"""

import os
import scipy.io as sio
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

from ComboDataSR_2D import ComboDataSR_2D

#%% S1 - hent og lagre bilder
rat = 0
for file in os.listdir(r'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\combodata_shax'):
    file_ = os.path.splitext(file)[0]
    try:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\combodata_shax\\{file_}')["ComboData_thisonly"]
    except KeyError:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\combodata_shax\\{file_}')["ComboData"]
    
    image_day = data['ImageDay'][0,0][0]
    if eval(image_day) < 40:
        continue
    
    rat += 1
    
    condition = 0
    if str(file_[0]) == 'm':
        condition = 1
    
    T_es = data['TimePointEndSystole'][0,0][0][0]
    T_ed = data['TimePointEndDiastole'][0,0][0][0]
    M = data['Magn'][0,0]  # magnitude matrix
    mask = data['Mask'][0,0]  # binary mask of LV myocardium
    
    cy0, cx0 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, T_es]))
    cy1, cx1 = ndi.center_of_mass(ndi.binary_fill_holes(mask[:, :, 0, T_ed]))
    w = 30  # window from LV center 
    
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    #f.suptitle(f'condition {condition} @ {image_day} days')
    
    # magnitude at end systole / diastole
    M_s = M[:, :, 0, T_es]
    M_e = M[:, :, 0, T_ed]
    ax0.imshow(M_s, cmap = 'gray', vmin = np.percentile(M_s, 5)*0.6, vmax = np.percentile(M_s, 95)*1.4)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_xlim(cx0-w, cx0+w); ax0.set_ylim(cy0-w, cy0+w)
    
    ax1.imshow(M[:, :, 0, T_ed], cmap = 'gray', vmin = np.percentile(M_e, 5)*0.6, vmax = np.percentile(M_e, 95)*1.4)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlim(cx1-w, cx1+w); ax1.set_ylim(cy1-w, cy1+w)
    f.tight_layout()
    plt.show()    
    
    f.savefig(fr'C:\Users\lasse\Desktop\IEMR\Lasse\magnitude series\{condition}\{rat}.png')
    
#%% Figure S1

f, ax = plt.subplots(6, 2, figsize = (16, 25), dpi = 400)

for condition in [0, 1]:
    row = 0
    for img in os.listdir(fr'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\\magnitude series\{condition}'):
        img = plt.imread(fr'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\\magnitude series\{condition}\{img}')
        ax[row, condition].imshow(img)
        ax[row, condition].set_xticks([])
        ax[row, condition].set_yticks([])
        ax[row, condition].set_frame_on(0)
        
        row += 1

ax[0,0].set_title(r'$\bf{Sham}$''\nEnd Systole        End Diastole', fontsize = 28)
ax[0,1].set_title(r'$\bf{MI}$''\n End Systole        End Diastole', fontsize = 28)

f.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.05)
plt.axis('off')
plt.show()

#%% FigS2 - effekt av smoothing på målinger ved sigma=0,1,2

a = 'sham_D7-1_40d'

run0 = ComboDataSR_2D(a, n = 1, sigma = 0)
run0.strain_rate(ellipse = 0, plot = 0, save = 0, segment = 0)
R0 = run0.__dict__['rs']; C0 = run0.__dict__['cs']
D0 = run0.__dict__['dispersion_curve']

run1 = ComboDataSR_2D(a, n = 1, sigma = 1)
run1.strain_rate(ellipse = 0, plot = 0, save = 0, segment = 0)
R1 = run1.__dict__['rs']; C1 = run1.__dict__['cs']
D1 = run1.__dict__['dispersion_curve']

run2 = ComboDataSR_2D(a, n = 1, sigma = 2)
run2.strain_rate(ellipse = 0, plot = 0, save = 0, segment = 0)
R2 = run2.__dict__['rs']; C2 = run2.__dict__['cs']
D2 = run2.__dict__['dispersion_curve']

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (13,5))

f.suptitle('Effect of Gaussian smoothing on velocity fields', fontsize = 17)
ax1.plot(R0, color='k', label = '$\sigma$ = 0')
ax1.plot(C0, color='k')

ax1.plot(R1, color='k', linestyle = '--', label = '$\sigma$ = 1')
ax1.plot(C1, color='k', linestyle = '--')

ax1.plot(R2, color='k', linestyle = '-.', label = '$\sigma$ = 2')
ax1.plot(C2, color='k', linestyle = '-.')

ax1.set_ylabel('Strain (%)', fontsize = 12)

ax2.plot(D0, color='k')
ax2.plot(D1, color='k', linestyle='--')
ax2.plot(D2, color='k', linestyle='-.')

ax2.set_ylabel(r'Standard deviation of $\theta$ ($^{\circ}$)', fontsize = 12)
ax1.legend(); plt.show()