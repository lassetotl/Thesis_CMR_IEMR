"""
Created on Tue Aug 29 10:46:34 2023

@author: lassetot
"""
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
import scipy.ndimage as ndi 
from scipy.signal import wiener
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
file = 'ComboData1'
dict = sio.loadmat(f'R:\Lasse\combodata\{file}.mat')
data = dict["ComboData_thisonly"]

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

print(f'{file} overview:')
print(f'Velocity field shape: {np.shape(V)}')
print(f'Magnitudes field shape: {np.shape(M)}')
print(f'Mask shape: {np.shape(mask)}')

print(f'End systole at t={T_es}, end diastole at t={T_ed}')

#%%
#visualizing Strain Rate

f = 100
frame_ = np.zeros((f,f))
I = np.zeros(T); I[:] = np.nan #Graph values

#plot every n'th ellipse
n = 2

#cyclic colormap

c_cmap = plt.get_cmap('plasma')
norm = mpl.colors.Normalize(0, 1)

#%%
for t in range(T):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    #diluted mask to calculate D_ij
    mask_t = mask[:f, :f, 0, t] #mask at this timepoint
    mask_d = ndi.binary_dilation(mask_t).astype(mask_t.dtype) - mask_t
    mask_e = ndi.binary_erosion(mask_t).astype(mask_t.dtype)
    
    #find center of mass of filled mask (middle of the heart)
    cy, cx = ndi.center_of_mass(ndi.binary_fill_holes(mask[:f, :f, 0, t]))
    
    D = D_ij(V=V, t=t, f=f, mask_ = 1)
    
    #plot ellipse every third index
    #ax1 = fig.add_subplot(211, aspect='auto') #SR colormap
    plt.imshow(M[:f, :f, 0, t], cmap = 'gray', alpha = 1)
    
    eigvals = np.zeros((f,f), dtype = object)
    eigvals_m = np.zeros((f,f), dtype = object) #highest values
    eigvecs = np.zeros((f,f), dtype = object)
    
    #calculate eigenvalues and vectors
    for x in range(0, f, n):
        for y in range(0, f, n):
            #SR tensor for specific point xy
            D_ = np.array([[D[0,0][x,y], D[1,0][x,y]], [D[0,1][x,y], D[1,1][x,y]]])
            val, vec = np.linalg.eig(D_)
            eigvals[x, y] = val 
            eigvecs[x, y] = vec
            eigvals_m[x, y] = val[np.argmax(abs(val))]
            
            #the highest eigenvalue is saved, no directional info!
            #frame_[x,y] = val[np.argmax(abs(val))] #stretch/shortening
            
    #create ellipses, normalized eigenvals at mean
    eigvals = eigvals/abs(np.max((eigvals_m))) #scale with max eigval this frame
    for x in range(0, f, n):
        for y in range(0, f, n):       
            #mask choice here
            if mask_e[y, x] == 1: #why does this work? (Switching x and y)
                    
                vec = eigvecs[x, y]
                val = eigvals[x, y]
                print(val)
                #color code in hex notation, from invariant
                I = val[0]**2 + val[1]**2 
                
                c = mpl.colors.rgb2hex(c_cmap(val[0]**2 + val[1]**2))
                
                #directional information lost from eigenvalues
                val = abs(val)
                
                #https://stackoverflow.com/questions/67718828/how-can-i-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-python-matplotl
                theta_ = np.linspace(0, 2*np.pi, 1000);
                #the first factor is arbitrary scaling
                ellipsis = 1*(np.sqrt(val[None,:]) * vec) @ [np.sin(theta_), np.cos(theta_)]
                
                plt.fill(x + ellipsis[0,:], y + ellipsis[1,:], color = c)
                #plt.quiver(x, y, vec[0][0], vec[0][1], color = 'red', scale = 10/np.sqrt(val[0]))
                #plt.quiver(x, y, vec[1][0], vec[1][1], color = 'red', scale = 10/np.sqrt(val[1]))
    #I[t] = Invariant(I_[0], I_[1], D)
    
    #frame_e = frame_*mask_e
    #I[t] = np.sum(frame_e)/np.sum(mask_e) #total SR scaled by nr of mask pixels

    plt.scatter(cx, cy, marker = 'x', c='w')
 
    plt.title(f'Strain Rate at t = {t}', fontsize = 15)
    plt.xlim(0, f-1); plt.ylim(f-1, 0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    sm = plt.cm.ScalarMappable(cmap=c_cmap, norm=norm)
    cbar = plt.colorbar(sm, cax = cax)
    cbar.set_label('$\lambda_1^2 + \lambda_2^2$', fontsize = 15)
    '''
    ax2 = fig.add_subplot(212, aspect='auto') #TSR
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
    '''
    plt.savefig(f'R:\Lasse\plots\SRdump\SR(t={t}).PNG')
    plt.show()

#%%
#Generate mp4

filenames = [f'R:\Lasse\plots\SRdump\SR(t={t}).PNG' for t in range(T)]

with imageio.get_writer(f'R:\Lasse\plots\MP4\{file}\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
