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

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

# save characteristic time-points to calc average
T_es_list = []
T_ed_list = []

for file in os.listdir('R:\Lasse\combodata_shax'):
    file_ = os.path.splitext(file)
    run = ComboDataSR_2D(file_[0], n = 5)
    run.strain_rate(save = 1, plot = 0)
    
    # collect parameters
    T_es_list.append(run.__dict__['T_es'])
    T_ed_list.append(run.__dict__['T_ed'])


#%%
# strain
T = 63  # timepoints
# one of the clips are longer for some reason, but we force it to stop at timepoint 62
TR = run.__dict__['TR']
range_TR = np.arange(0, T)*TR

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
    
    T_ = len(r_strain) # stops at respective end diastole
    if str(file[0]) == 'm':
        ax1.plot(range_TR[:T_], 100*r_strain[:T_], lw=1.3, c='lime') #, label = f'({file.split("_")[2]})')
        ax2.plot(range_TR[:T_], 100*c_strain[:T_], lw=1.3, c='gold')
    else:
        
        ax1.plot(range_TR[:T_], 100*r_strain[:T_], lw=1.3, c='darkblue') 
        ax2.plot(range_TR[:T_], 100*c_strain[:T_], lw=1.3, c='chocolate') 
     
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
ax1.set_xlim(0, T*TR)#; plt.ylim(0, 50)
ax1.set_xlabel('Time [s]', fontsize = 15)
ax1.set_ylabel('$s^{-1}$', fontsize = 15)

ax2.set_title('Global Circumferential Strain Rate over time', fontsize = 15)
ax2.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax2.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax2.axhline(0, c = 'k', lw = 1)
ax2.set_xlim(0, T*TR)#; plt.ylim(0, 50)
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
          Line2D([0], [0], color = 'lime', lw = 1.3, label = '6w after MI')]

legend_handles2 = [Line2D([0], [0], color = 'chocolate', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'gold', lw = 1.3, label = '6w after MI')]
                       
ax1.legend(handles = legend_handles1, fontsize = 12)
ax2.legend(handles = legend_handles2, fontsize = 12)

plt.subplots_adjust(wspace=0.07)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()