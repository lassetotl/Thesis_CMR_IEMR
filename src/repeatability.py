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
    
filene må ligge i combodata shax (og fjernes ved annen analyse)
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

folder = '4'  # 2 files (test and retest) should be in folder
day = []

for file in os.listdir(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}'):
    file_ = os.path.splitext(file)[0]
    try:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData_thisonly"]
    except KeyError:
        data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData"]
        
    run = ComboDataSR_2D(file_, n = 1, sigma=0)  # n = 1 should be used for proper analysis
    run.strain_rate(save = 1, plot = 0, ellipse = 0)
    print(data['ImageDay'][0,0][0])
    day.append(eval(data['ImageDay'][0,0][0]))
    
    CS_curves.append(run.__dict__['c_strain'])
    RS_curves.append(run.__dict__['r_strain'])
    disp_curves.append(run.__dict__['dispersion_curve'])
    
#%%

def bland_altman_plot(data1, data2, label = '', unit = '', plot = 1, *args, **kwargs):
    x_len = min([len(data1), len(data2)])
    data1 = data1[:x_len]; data2 = data2[:x_len]
    #print(x_len)
    mean      = (data1 + data2)/2
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    if plot == 1:
        plt.axhline(md,           color='gray')
        plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
        plt.scatter(mean, diff, color = 'k', s = 20, alpha = 1, label = f'Bias = {md.round(3)}, SD = {sd.round(3)}', zorder = 4)
        plt.xlabel(f'Mean{label} - test/retest{unit}'); plt.ylabel(f'Diff{label} - test/retest ({unit[2:-1]}$\pm$ 1.96*SD)')
        plt.ylim(min(diff) - sd, max(diff) + 1.5*sd)
        plt.legend()
    
    return md, sd

plt.plot(CS_curves[0], 'k', label = f'Day {day[0]}')
plt.plot(CS_curves[1], 'k', ls='--', label = f'Day {day[1]}')
plt.ylabel('Circ Strain Rate [s$^{-1}$]', fontsize = 11)
plt.legend(); plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\1.png', bbox_inches='tight', pad_inches=0)
plt.show()

plt.plot(RS_curves[0], 'k', label = f'Day {day[0]}')
plt.plot(RS_curves[1], 'k', ls='--', label = f'Day {day[1]}')
plt.ylabel('Rad Strain Rate [s$^{-1}$]', fontsize = 11)
plt.legend(); plt.savefig('C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\2.png', bbox_inches='tight', pad_inches=0)
plt.show()

plt.plot(disp_curves[0], 'k', label = f'Day {day[0]}')
plt.plot(disp_curves[1], 'k', ls='--', label = f'Day {day[1]}')
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

# Figure 7?

f, ax = plt.subplots(3, 2, figsize = (10, 11), dpi = 300)
f.suptitle('43', fontsize=13)

image = 1
for column in [0, 1]:
    for row in [0, 1, 2]:
        img = plt.imread(fr'C:\\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\temp\\{image}.png')
        ax[row, column].imshow(img)
        ax[row, column].set_xticks([])
        ax[row, column].set_yticks([])
        ax[row, column].set_frame_on(0)
            
        image += 1

f.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.axis('off')
plt.show()

#%% TABLE 3
# collect regional and global curves CS, RS, CSR, RSR, dispersion 
# run bland altman, collect bias and SD (formatted like Table 3) [rad, kolonne]

bias = np.zeros((5,5))
SD = np.zeros((5,5))

# inkl 2 senere ...
folders = ['1', '3', '4']
for folder in folders:
    # reset
    print(folder)
    CS_curves = []; RS_curves = []; CSR_curves = []; RSR_curves = []; disp_curves = []
    CSreg_curves = []; RSreg_curves = []; CSRreg_curves = []; RSRreg_curves = []; dispreg_curves = []
    
    for file in os.listdir(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}'):
        file_ = os.path.splitext(file)[0]
        try:
            data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData_thisonly"]
        except KeyError:
            data = sio.loadmat(fr'C:\Users\\lasse\\Desktop\\IEMR\\Lasse\\repeatability testing\\{folder}\\{file_}')["ComboData"]
            
        run = ComboDataSR_2D(file_, n = 1, sigma=0)  # n = 1 should be used for proper analysis
        run.strain_rate(save = 0, plot = 0, ellipse = 0)
        
        # global curves
        CS_curves.append(run.__dict__['cs'])
        RS_curves.append(run.__dict__['rs'])
        CSR_curves.append(run.__dict__['c_strain'])
        RSR_curves.append(run.__dict__['r_strain'])
        disp_curves.append(run.__dict__['dispersion_curve'])
        
        # regional curves
        CSreg_curves.append(run.__dict__['CSreg'])
        RSreg_curves.append(run.__dict__['RSreg'])
        CSRreg_curves.append(run.__dict__['CSRreg'])
        RSRreg_curves.append(run.__dict__['RSRreg'])
        dispreg_curves.append(run.__dict__['dispreg'])
        
        # beregninger
        
        if len(CS_curves) == 2:
            
            # GLOBAL
            CS = bland_altman_plot(CS_curves[0], CS_curves[1], plot=0)
            bias[0, 4] += abs(CS[0]); SD[0, 4] += CS[1]
            
            RS = bland_altman_plot(RS_curves[0], RS_curves[1], plot=0)
            bias[1, 4] += abs(RS[0]); SD[1, 4] += RS[1]
            
            CSR = bland_altman_plot(CSR_curves[0], CSR_curves[1], plot=0)
            bias[2, 4] += abs(CSR[0]); SD[2, 4] += CSR[1]
            
            RSR = bland_altman_plot(RSR_curves[0], RSR_curves[1], plot=0)
            bias[3, 4] += abs(RSR[0]); SD[3, 4] += RSR[1]
            
            disp = bland_altman_plot(disp_curves[0], disp_curves[1], plot=0)
            bias[4, 4] += abs(disp[0]); SD[4, 4] += disp[1]
            
            # REGIONAL
            for sector in range(4):
                CSr = bland_altman_plot(CSreg_curves[0][sector, :], CSreg_curves[1][sector, :])
                bias[0, sector] += abs(CSr[0]); SD[0, sector] += CSr[1]
                
                RSr = bland_altman_plot(RSreg_curves[0][sector, :], RSreg_curves[1][sector, :])
                bias[1, sector] += abs(RSr[0]); SD[1, sector] += RSr[1]
                
                CSRr = bland_altman_plot(CSRreg_curves[0][sector, :], CSRreg_curves[1][sector, :])
                bias[2, sector] += abs(CSRr[0]); SD[2, sector] += CSRr[1]
                
                RSRr = bland_altman_plot(RSRreg_curves[0][sector, :], RSRreg_curves[1][sector, :])
                bias[3, sector] += abs(RSRr[0]); SD[3, sector] += RSRr[1]
                
                dispr = bland_altman_plot(dispreg_curves[0][sector, :], dispreg_curves[1][sector, :])
                bias[4, sector] += abs(dispr[0]); SD[4, sector] += dispr[1]
                
# beregner gjennomsnitt og konverterer SD verdier til 1.96*SD
bias = bias/len(folders); SD = 1.96*SD/len(folders)

print(f'BIAS:\n{bias.round(2)}')
print(f'\nSD:\n{SD.round(2)}')    
