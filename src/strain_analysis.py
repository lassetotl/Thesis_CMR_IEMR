# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:39:12 2023

@author: lassetot

we collect generated global radial strain (GRS) curves and plot them together.
make sure that all used data were generated using the same setup parameters!!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ComboDataSR_2D import ComboDataSR_2D
from scipy.integrate import cumtrapz
from util import running_average

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

# save characteristic time-points to calc average
T_es_list = []
T_ed_list = []

for file in os.listdir('R:\Lasse\combodata_shax'):
    file_ = os.path.splitext(file)
    run = ComboDataSR_2D(file_[0], n = 2)
    run.strain_rate(save = 1, plot = 0)
    
    # collect parameters
    T_es_list.append(run.__dict__['T_es'])
    T_ed_list.append(run.__dict__['T_ed'])


#%%
# strain
T = 63  # timepoints
# one of the clips are longer for some reason, but we force it to stop at timepoint 62
TR = run.__dict__['TR']
range_ = np.arange(0, T)
range_TR = range_*TR

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.set_title('Global Radial Strain over time', fontsize = 15)
ax1.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax1.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax1.axhline(0, c = 'k', lw = 1)
ax1.set_xlim(0, np.max(T_ed_list)*TR)
ax1.set_xlabel('Time [s]', fontsize = 15)
ax1.set_ylabel('%', fontsize = 17)

ax2.set_title('Global Circumferential Strain over time', fontsize = 15)
ax2.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax2.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax2.axhline(0, c = 'k', lw = 1)
ax2.set_xlim(0, np.max(T_ed_list)*TR)
ax2.set_xlabel('Time [s]', fontsize = 15)

for file in os.listdir('R:\Lasse\strain data'):
    r_strain = np.load(fr'R:\Lasse\strain data\{str(file)}\r_strain.npy', allow_pickle = 1)
    c_strain = np.load(fr'R:\Lasse\strain data\{str(file)}\c_strain.npy', allow_pickle = 1)
    
    T_ = len(r_strain)  # stops at respective end diastole
    if str(file[0]) == 'm':  # double check that folder includes only 6w mi
        ax1.plot(range_TR[:T_], r_strain[:T_], lw=1.3, c='lime') #, label = f'({file.split("_")[2]})')
        ax2.plot(range_TR[:T_], c_strain[:T_], lw=1.3, c='gold')
    else:
        
        ax1.plot(range_TR[:T_], r_strain[:T_], lw=1.3, c='darkblue') 
        ax2.plot(range_TR[:T_], c_strain[:T_], lw=1.3, c='chocolate') 
     
legend_handles1 = [Line2D([0], [0], color = 'darkblue', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'lime', lw = 1.3, label = '6w after MI')]

legend_handles2 = [Line2D([0], [0], color = 'chocolate', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'gold', lw = 1.3, label = '6w after MI')]
                       
ax1.legend(handles = legend_handles1, fontsize = 12)
ax2.legend(handles = legend_handles2, fontsize = 12)

plt.subplots_adjust(wspace=0.07)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()


#%%
# strain rate

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.set_title('Global Radial Strain Rate over time', fontsize = 15)
ax1.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax1.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax1.axhline(0, c = 'k', lw = 1)
ax1.set_xlim(0, np.max(T_ed_list)*TR)
ax1.set_xlabel('Time [s]', fontsize = 15)
ax1.set_ylabel('$s^{-1}$', fontsize = 15)

ax2.set_title('Global Circumferential Strain Rate over time', fontsize = 15)
ax2.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax2.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax2.axhline(0, c = 'k', lw = 1)
ax2.set_xlim(0, np.max(T_ed_list)*TR)
ax2.set_xlabel('Time [s]', fontsize = 15)

for file in os.listdir('R:\Lasse\strain rate data'):
    r_strain_rate = np.load(fr'R:\Lasse\strain rate data\{str(file)}\r_strain_rate.npy', allow_pickle = 1)
    c_strain_rate = np.load(fr'R:\Lasse\strain rate data\{str(file)}\c_strain_rate.npy', allow_pickle = 1)
    
    T_ = 63 # stops at respective end diastole
    if str(file[0]) == 'm':
        ax1.plot(range_TR[:T_], r_strain_rate[:T_], 'lime', lw=1.3)
        ax2.plot(range_TR[:T_], c_strain_rate[:T_], 'gold', lw=1.3)
    else:
        
        ax1.plot(range_TR[:T_], r_strain_rate[:T_], 'darkblue', lw=1.3) 
        ax2.plot(range_TR[:T_], c_strain_rate[:T_], 'chocolate', lw=1.3) 
        
legend_handles1 = [Line2D([0], [0], color = 'darkblue', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'lime', lw = 1.3, label = 'MI')]

legend_handles2 = [Line2D([0], [0], color = 'chocolate', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'gold', lw = 1.3, label = 'MI')]
                       
ax1.legend(handles = legend_handles1, fontsize = 12)
ax2.legend(handles = legend_handles2, fontsize = 12)

plt.subplots_adjust(wspace=0.07)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()

#%%
# angle distributions

plt.figure(figsize = (10, 8))
plt.title('Radial angle concentration', fontsize = 15)
plt.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, T*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('Degrees', fontsize = 20)


a1_mean = np.zeros(T); a2_mean = np.zeros(T)
auc_mi = []; auc_sham = []  # to fill with tuples (days, auc)
for file in os.listdir('R:\Lasse\\angle distribution data'):
    a1 = np.load(fr'R:\Lasse\\angle distribution data\{str(file)}\angle_distribution_pos.npy', allow_pickle = 1)
    a2 = np.load(fr'R:\Lasse\\angle distribution data\{str(file)}\angle_distribution_neg.npy', allow_pickle = 1)
    
    diff = running_average(abs(a1 - a2), 4)
    
    # auc at systole/diastole only
    u = int(np.mean(T_es_list))
    u_ = int(np.mean(T_ed_list))
      
    #days = 0
    if str(file[-1]) == 'w':
           days = int(file.split('_')[2].replace('w', ''))*7
    if str(file[-1]) == 'd':
           days = int(file.split('_')[2].replace('d', ''))
           
    if str(file[0]) == 'm':  # compare angle cohesion
        plt.plot(range_TR[:len(diff)], diff, 'r', lw=1.3)
        auc_mi.append([days, sum(cumtrapz(diff[:]))])
    else:
        plt.plot(range_TR[:len(diff)], diff, 'k', lw=1.3)
        auc_sham.append([days, sum(cumtrapz(diff[:]))])

legend_handles1 = [Line2D([0], [0], color = 'k', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'r', lw = 1.3, label = '6w after MI')]

# difference
plt.legend(handles = legend_handles1, loc = 'upper right')
plt.show()

auc_mi = np.array(auc_mi)
auc_sham = np.array(auc_sham)

#%% angle concentration AUC over time 

T_ = 47
plt.figure(figsize = (8, 6))
plt.title('Radial angle concentration', fontsize = 15)
plt.xlim(0, T_)#; plt.ylim(0, 50)
plt.xlabel('Time [days]', fontsize = 15)
plt.ylabel('AUC', fontsize = 15)

plt.scatter(auc_mi[:,0], auc_mi[:,1], c='r')
plt.scatter(auc_sham[:,0], auc_sham[:,1], c='k')

a, b = np.polyfit(auc_sham[:,0], auc_sham[:,1], 1)
c, d = np.polyfit(auc_mi[:,0], auc_mi[:,1], 1)
t = np.arange(0, T_)

plt.plot(t, a*t + b, 'k', label = f'sham linear fit, slope = {np.round(a, 3)}')
plt.plot(t, c*t + d, 'r', label = f'mi linear fit, slope = {np.round(c, 3)}')

plt.legend()
plt.show()