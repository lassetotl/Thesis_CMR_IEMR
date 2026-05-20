# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:28:31 2026

@author: lasse

REPETERBARHET 
data - 4 dyr, 2 gjentatte opptak per

Skisse plott (3x2?):  

    målinger av CSR kurver, dispersion kurver (regionalt?) 
    noe om standardavvik eller CV for å kvantifisere repeterbarhet?
    
    første kolonne (curve overlays)
    subplot 1,1 - CSR kurve, overlay test retest
    
    subplot 1,2 - Bland-Altman hvert punkt (bias i label)
    
    (repeat med dispersion)
"""

import numpy as np
import os
import scipy.io as sio
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from ComboDataSR_2D import ComboDataSR_2D

CS_curves = []
RS_curves = []
disp_curves = []

folder = '1'  # 2 files (test and retest) should be in folder

for file in os.listdir(fr'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}'):
    file_ = os.path.splitext(file)[0]
    try:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData_thisonly"]
    except KeyError:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData"]
        
    run = ComboDataSR_2D(file_, n = 1, sigma=0)  # n = 1 should be used for proper analysis
    run.strain_rate(save = 1, plot = 0, ellipse = 0)
    
    CS_curves.append(run.__dict__['c_strain'])
    RS_curves.append(run.__dict__['r_strain'])
    disp_curves.append(run.__dict__['dispersion_curve'])
    
#%%

def bland_altman_plot(data1, data2, label = '', unit = '', *args, **kwargs):
    x_len = min([len(data1), len(data2)])
    data1 = data1[:x_len]; data2 = data2[:x_len]
    print(x_len)
    mean      = (data1 + data2)/2
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.axhline(md,           color='gray')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.scatter(mean, diff, color = 'k', s = 20, alpha = 1, label = f'Bias = {md.round(3)}, SD = {sd.round(3)}', zorder = 4)
    plt.xlabel(f'Mean{label} - test/retest{unit}'); plt.ylabel(f'Diff{label} - test/retest ({unit[2:-1]}$\pm$ 1.96*SD)')
    plt.ylim(min(diff) - sd, max(diff) + 1.5*sd)
    plt.legend()

plt.plot(CS_curves[0], 'k', label = 'Day 1')
plt.plot(CS_curves[1], 'k', ls='--', label = 'Day 2')
plt.ylabel('Circ Strain Rate [s$^{-1}$]', fontsize = 11)
plt.legend(); plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\1.png', bbox_inches='tight', pad_inches=0)
plt.show()

plt.plot(RS_curves[0], 'k', label = 'Day 1')
plt.plot(RS_curves[1], 'k', ls='--', label = 'Day 2')
plt.ylabel('Rad Strain Rate [s$^{-1}$]', fontsize = 11)
plt.legend(); plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\2.png', bbox_inches='tight', pad_inches=0)
plt.show()

plt.plot(disp_curves[0], 'k', label = 'Day 1')
plt.plot(disp_curves[1], 'k', ls='--', label = 'Day 2')
plt.ylabel('Strain Rate Angle Dispersion [$^{\circ}$]', fontsize = 11)
plt.legend(); plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\3.png', bbox_inches='tight', pad_inches=0)
plt.show()


bland_altman_plot(CS_curves[0], CS_curves[1], label = ' Circ SR', unit = ' [s$^{-1}$]')
plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\4.png', bbox_inches='tight', pad_inches=0)
plt.show()

bland_altman_plot(RS_curves[0], RS_curves[1], label = ' Rad SR', unit = ' [s$^{-1}$]')
plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\5.png', bbox_inches='tight', pad_inches=0)
plt.show()

bland_altman_plot(disp_curves[0], disp_curves[1], label = ' Dispersion', unit = ' [$^{\circ}$]')
plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\6.png', bbox_inches='tight', pad_inches=0)
plt.show()

# Figure ?

f, ax = plt.subplots(3, 2, figsize = (10, 11), dpi = 300)


image = 1
for column in [0, 1]:
    for row in [0, 1, 2]:
        img = plt.imread(fr'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\{image}.png')
        ax[row, column].imshow(img)
        ax[row, column].set_xticks([])
        ax[row, column].set_yticks([])
        ax[row, column].set_frame_on(0)
            
        image += 1

#ax[0,0].set_title(r'$\bf{Sham}$''\nEnd Systole        End Diastole', fontsize = 28)
#ax[0,1].set_title(r'$\bf{MI}$''\n End Systole        End Diastole', fontsize = 28)

f.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.axis('off')
plt.show()
