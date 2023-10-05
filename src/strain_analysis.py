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

#%%
#with SR analysis class, run all instances here 
#to make sure they use the same setup parameters



#%%
T = 63  # timepoints
# one of the clips are longer for some reason, but we force it to stop at timepoint 62

#plt.figure(figsize=(10, 8))
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(16, 8))

ax1.set_title('Global Radial Strain over time', fontsize = 15)
ax1.axvline(25, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax1.axvline(50, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax1.axhline(0, c = 'k', lw = 1)
ax1.set_xlim(0, T)#; plt.ylim(0, 50)
ax1.set_xlabel('Timepoints', fontsize = 15)

ax2.set_title('Global Circumferential Strain over time', fontsize = 15)
ax2.axvline(25, c = 'k', ls = ':', lw = 2, label = 'End Systole')
ax2.axvline(50, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax2.axhline(0, c = 'k', lw = 1)
ax2.set_xlim(0, T)#; plt.ylim(0, 50)
ax2.set_xlabel('Timepoints', fontsize = 15)

for file in os.listdir('R:\Lasse\strain data'):
    r_strain = np.load(fr'R:\Lasse\strain data\{str(file)}\r_strain.npy', allow_pickle = 1)
    c_strain = np.load(fr'R:\Lasse\strain data\{str(file)}\c_strain.npy', allow_pickle = 1)
    if str(file[0]) == 'm':
        ax1.plot(range(T), r_strain[:T], 'lime', lw=2)
        ax2.plot(range(T), c_strain[:T], 'gold', lw=2)
    else:
        ax1.plot(range(T), r_strain[:T], 'darkblue', lw=2) 
        ax2.plot(range(T), c_strain[:T], 'chocolate', lw=2) 
        
legend_handles1 = [Line2D([0], [0], color = 'darkblue', lw = 2, label = 'Sham'),
          Line2D([0], [0], color = 'lime', lw = 2, label = '6w after MI')]

legend_handles2 = [Line2D([0], [0], color = 'chocolate', lw = 2, label = 'Sham'),
          Line2D([0], [0], color = 'gold', lw = 2, label = '6w after MI')]
                       
ax1.legend(handles = legend_handles1, fontsize = 15)
ax2.legend(handles = legend_handles2, fontsize = 15)

plt.subplots_adjust(wspace=0.03)
plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()
                       