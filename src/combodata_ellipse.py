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
#import pandas as pd
#import seaborn as sns; sns.set()
#import sklearn

import scipy.io as sio
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from scipy.signal import wiener
import scipy.interpolate as scint
import imageio
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
# Converting .mat files to numpy array, dictionary

#converts to dictionary (dict) format
dict = sio.loadmat(r'R:\Lasse\combodata\ComboData1.mat')
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
print(f'End systole at t={T_es}, end diastole at t={T_ed}')


#%%
#Strain rate tensor (xy plane, dim=2) incl mask
#(Selskog et al 2002, hentet 17.08.23)

#V[i, j, 0, t, i]
# xval, yval, ?, timepoint, axis

def D_ij(V, t, f, mask_, dim = 2): #Construct SR tensor
    L = np.zeros((dim, dim), dtype = object) #Jacobian 2D velocity matrices
    
    v_i = 1; x_j = 0 #index 0 is y and 1 is x (?)
    for i in range(dim):
        s = 1
        for j in range(dim):
            #Gathering velocity data and applying gaussian smoothing
            V_ = gaussian_filter(V[:f, :f, 0, t, v_i]*mask_, sigma = 2)
            #V_[V_ == 0] = np.nan
            
            if (j==1) is True: #negative sign on V_y (?)
                s = -1
            L[i, j] = s * np.gradient(V_, axis=x_j, edge_order = 1)
            x_j += 1
        v_i -= 1
        x_j = 0
    
    D_ij = 0.5*(L + L.T) #Strain rate tensor from Jacobian       
    return D_ij

# Note: returns angle in radians between vectos 
def theta(v, w): return np.arccos(v.dot(w)/(norm(v)*norm(w)))

#https://stats.stackexchange.com/questions/9898/how-to-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-r

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
    mask_d = binary_dilation(mask_t).astype(mask_t.dtype) - mask_t
    mask_e = binary_erosion(mask_t).astype(mask_t.dtype)
    
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
                
                #D_Ellipse(x, y, vectors=0)
                    
                vec = eigvecs[x, y]
                val = eigvals[x, y]
                
                #color code in hex notation, from invariant
                I = val[0]**2 + val[1]**2 
                c = mpl.colors.rgb2hex(c_cmap(val[0]**2 + val[1]**2))
                
                #directional information lost from eigenvalues
                val = abs(val/np.max((eigvals_m)))
                
                #https://stackoverflow.com/questions/67718828/how-can-i-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-python-matplotl
                theta_ = np.linspace(0, 2*np.pi, 1000);
                #the first factor is arbitrary scaling
                ellipsis = 1*(np.sqrt(val[None,:]) * vec) @ [np.sin(theta_), np.cos(theta_)]
                #ellipsis = 1*(1/np.sqrt(val[None,:]) * vec) @ [np.sin(theta_), np.cos(theta_)]
                
                plt.fill(x + ellipsis[0,:], y + ellipsis[1,:], color = c)
                #plt.quiver(x, y, vec[0][0], vec[0][1], color = 'red', scale = 10/np.sqrt(val[0]))
                #plt.quiver(x, y, vec[1][0], vec[1][1], color = 'red', scale = 10/np.sqrt(val[1]))
    #I[t] = Invariant(I_[0], I_[1], D)
    
    #frame_e = frame_*mask_e
    #I[t] = np.sum(frame_e)/np.sum(mask_e) #total SR scaled by nr of mask pixels

    
 
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

with imageio.get_writer('R:\Lasse\plots\MP4\Ellipses.mp4', fps=7) as writer:    # inputs: filename, frame per second
    for filename in filenames:
        image = imageio.imread(filename)                         # load the image file
        writer.append_data(image)
