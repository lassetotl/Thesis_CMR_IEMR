#%%
#author Lasse Totland @ Ullevål OUS
#analysis of MR combodatasets
#(streamlining: all of this can be implemented as a more user friendly 
#function/class eventually (methods for V and SR plot?))


#imports

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from numpy.linalg import norm
from Lasse_functions import D_ij, theta
#import pandas as pd
#import seaborn as sns; sns.set()
#import sklearn

import scipy.io as sio
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
dict = sio.loadmat(r'R:\Lasse\combodata\ComboData.mat')
data = dict["ComboData_thisonly"]

print(f'Keys in dictionary: {dict.keys()}') #dict_keys(['StudyData', 'StudyParam'])
print(f'Combodata shape: {np.shape(data)}')

#%%

#velocity V field, and magnitudes

# shape of V is (1, 1), indexing necessary to 'unpack' the correct format (?)
V = data['V'][0,0] #velocity field
M = data['Magn'][0,0] #magnitudes
mask = data['Mask'][0,0] #mask for non-heart tissue

T = len(V[0,0,0,:,0]) #Total amount of time steps
T_es = data['TimePointEndSystole']
T_ed = data['TimePointEndDiastole']

print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

#%%
#mask modification
#dilate original mask and locate additions from dilation
#then weighted average? interpolate?

f = 80
mask_t = mask[:f, :f, 0, 0]
mask_d = binary_dilation(mask_t).astype(mask_t.dtype) - mask_t
#plt.imshow(mask_d, cmap = 'gray')

test = np.zeros((f,f))#; test[:, :] = np.nan
for p in range(len(np.where(mask_d == 1)[0])):
    dy = np.where(mask_d == 1)[1][p]
    dx = np.where(mask_d == 1)[0][p]
    #print(np.where(mask_d == 1)[1][p], np.where(mask_d == 1)[0][p])
    #print(dx, dy)
    test[dx, dy] = 1
    
    #vx = sig.wiener(V[:f, :f, 0, t, 1]*mask[:f, :f, 0, t]) #x components of velocity w mask
    #vy = sig.wiener(-V[:f, :f, 0, t, 0]*mask[:f, :f, 0, t]) #y components (negative?)
    #test[dx, dy] = scint.interpn(mask_t, [vx, vy], [dx, dy])

#print(np.nansum(test))
#plt.imshow(test, cmap = 'inferno')

#%%
#https://stackoverflow.com/questions/18159874/making-image-white-space-transparent-overlay-onto-imshow
my_cmap = copy.copy(plt.cm.get_cmap('plasma')) # get a copy of the gray color map
my_cmap.set_under(alpha=0) # set how the colormap handles 'bad' values

plt.imshow(M[:f, :f, 0, 30]/np.max(M[:f, :f, 0, 30]), cmap = 'gray')

dm = plt.imshow(test, cmap = my_cmap, vmin = 0.01)
plt.colorbar(dm)
plt.show()
#%%
#visualizing Strain Rate

f = 80
frame_ = np.zeros((f,f))
I = np.zeros(T); I[:] = np.nan #Invariant


for t in range(T):
    plt.figure(figsize=(18, 8))
    
    #diluted mask to calculate D_ij
    mask_t = mask[:f, :f, 0, t] #mask at this timepoint
    mask_d = binary_dilation(mask_t).astype(mask_t.dtype) - mask_t
    mask_e = binary_erosion(mask_t).astype(mask_t.dtype)
    
    D = D_ij(V=V, t=t, f=f, mask_ = 1)
    for x in range(f):
        for y in range(f):
            #SR tensor for specific point xy
            D_ = np.array([[D[0,0][x,y], D[1,0][x,y]], [D[0,1][x,y], D[1,1][x,y]]])
            val, vec = np.linalg.eig(D_)
                
            #the highest eigenvalue is saved, no directional info!
            frame_[x,y] = val[np.argmax(abs(val))] #stretch/shortening
                
            #frame_[x,y] = val[0]**2 + val[1]**2 #invariant
            
    #I[t] = Invariant(I_[0], I_[1], D)
    
    frame_e = frame_*mask_e
    I[t] = np.sum(frame_e)/np.sum(mask_e) #total SR scaled by nr of mask pixels
    
    plt.subplot(1, 2, 1) #SR colormap
    ax = plt.gca()
    
    plt.imshow(M[:f, :f, 0, t], cmap = 'gray', alpha = 1)
    
    frame = frame_*mask_t
    frame[frame == 0] = -20
    cm = plt.imshow(frame/abs(np.max(frame)), vmin = -1, vmax = 1, alpha=1, cmap = my_cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cm, cax=cax)
    
    ax.set_title(f'Strain Rate at t = {t}', fontsize = 15)
    #plt.scatter(I_[0], I_[1], color = 'w', marker = 'x', lw = 2, s = 100)
    
    plt.subplot(1, 2, 2) #TSR
    #plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
    plt.title('Total Strain Rate over time')
    plt.grid(1)
    plt.plot(np.arange(0, T, 1), I, 'k')
    plt.axvline(T_es, c = 'm', ls = ':', lw = 2, label = 'End Systole')
    plt.axvline(T_ed, c = 'm', ls = '--', lw = 1.5, label = 'End Diastole')
    plt.xlim(0, T)#; plt.ylim(0, 50)
    plt.xlabel('Timepoints', fontsize = 15)
    plt.ylabel('$s^{-1}$', fontsize = 20)
    plt.legend()
    
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
    plt.show()


#%%
#last frame with running average

N = 4 #window
I_a = np.convolve(I, np.ones(N)/N, mode='same')

plt.figure(figsize=(18, 8))


plt.subplot(1, 2, 1) #SR colormap
ax = plt.gca()

plt.imshow(M[:f, :f, 0, t], cmap = 'gray', alpha = 1)

cm = plt.imshow(frame/abs(np.max(frame)), vmin = -1, vmax = 1, alpha=1, cmap = my_cmap)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(cm, cax=cax)

ax.set_title(f'Strain Rate at t = {t}', fontsize = 15)
#plt.scatter(I_[0], I_[1], color = 'w', marker = 'x', lw = 2, s = 100)

plt.subplot(1, 2, 2) #TSR
#plt.title('Invariant $\lambda_1^2 + \lambda_2^2$ in marked position', fontsize = 15)
plt.title('Total Strain Rate over time')
plt.grid(1)
plt.plot(np.arange(0, T, 1), I, 'lightgrey', ls = '--')
plt.axvline(T_es, c = 'm', ls = ':', lw = 2, label = 'End Systole', alpha = 0.6)
plt.axvline(T_ed, c = 'm', ls = '--', lw = 1.5, label = 'End Diastole', alpha = 0.6)
plt.xlim(0, T)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('$s^{-1}$', fontsize = 20)

plt.plot(np.arange(0, T, 1), I_a, 'indigo', lw=2, label = 'Walking Average') #walking average
plt.legend()

plt.subplots_adjust(wspace=0.25)
plt.savefig(f'R:\Lasse\plots\SR_GSR.PNG')
plt.show()
#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range(T)]

with imageio.get_writer('R:\Lasse\plots\MP4\Strain Rate.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)