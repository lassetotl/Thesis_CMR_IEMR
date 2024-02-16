# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:39:12 2023

@author: lassetot

we collect generated global radial strain (GRS) curves and plot them together.
make sure that all used data were generated using the same setup parameters!!

curve analysis parameters collected and used to construct a pandas dataframe
for correlation analysis
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ComboDataSR_2D import ComboDataSR_2D
from scipy.integrate import cumtrapz
from scipy import stats
from util import running_average, drop_outliers_IQR
import pandas 
import seaborn as sns

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

# save characteristic time-points to calc average
T_es_list = []
T_ed_list = []

df_list = []

# sham, mi 1d and >40d in separate lists
rs_sham = []; cs_sham = []
rs_mi_1d = []; cs_mi_1d = []
rs_mi_40d = []; cs_mi_40d = []
tp = 60

st = time.time()
filenr = 0
save = 1
for file in os.listdir('R:\Lasse\combodata_shax'):
    file_ = os.path.splitext(file)
    run = ComboDataSR_2D(file_[0], n = 1)  # n = 1 should be used for proper analysis
    run.strain_rate(save = save, plot = 0, ellipse = 0)
    
    # collect parameters
    T_es_list.append(run.__dict__['T_es'])
    T_ed_list.append(run.__dict__['T_ed'])
    
    # collect dataframe parameters
    filename = run.__dict__['filename']
    if str(filename[-1]) == 'w':
           days = int(filename.split('_')[2].replace('w', ''))*7
    if str(filename[-1]) == 'd':
           days = int(filename.split('_')[2].replace('d', ''))
           
    rs = run.__dict__['r_strain']
    rs = np.pad(rs, (0, tp - len(rs)), 'constant', constant_values = (0))
    
    cs = run.__dict__['c_strain']
    cs = np.pad(cs, (0, tp - len(cs)), 'constant', constant_values = (0))
    
    if str(filename[0]) == 'm':
           condition = 1  # mi
           if days == 1:
               rs_mi_1d.append(rs)
               cs_mi_1d.append(cs)
           if days >= 40:
               rs_mi_40d.append(rs)
               cs_mi_40d.append(cs)
               
    else:
           condition = 0 # sham
           rs_sham.append(rs)
           cs_sham.append(cs)
    
    # collect strain curve parameters
    r_strain_peak_mean = np.mean(run.__dict__['r_peakvals'])
    c_strain_peak_mean = np.mean(run.__dict__['c_peakvals'])
    
    # expressed as percentage of cardiac cycle duration
    TR = run.__dict__['TR']
    r_strain_peaktime_std = 100*np.std(run.__dict__['r_peaktime'])/(TR*T_ed_list[-1])
    c_strain_peaktime_std = 100*np.std(run.__dict__['c_peaktime'])/(TR*T_ed_list[-1])
    
    # strain rate parameters
    r_sr_max = run.__dict__['r_sr_max']
    r_sr_min = run.__dict__['r_sr_min']
    c_sr_max = run.__dict__['c_sr_max']
    c_sr_min = run.__dict__['c_sr_min']
    
    # angle dist
    a1_mean_max = run.__dict__['theta1_mean_max']
    a1_mean_min = run.__dict__['theta1_mean_min']
    a2_mean_max = run.__dict__['theta2_mean_max']
    a2_mean_min = run.__dict__['theta2_mean_min']
    
    # dataframe row
    df_list.append([filename, days, r_strain_peak_mean, c_strain_peak_mean, \
                    r_strain_peaktime_std, c_strain_peaktime_std, r_sr_max, \
                        r_sr_min, c_sr_max, c_sr_min, a1_mean_max, a1_mean_min, \
                            a2_mean_max, a2_mean_min, condition])
    filenr += 1
    if os.path.exists(f'R:\Lasse\plots\MP4\{file}') == False:
        os.makedirs(f'R:\Lasse\plots\MP4\{file}')
    
et = time.time()
print(f'Time elapsed for strain rate calculations on {filenr} files: {int((et-st)/60)} minutes')  

#%%
# mean strain with std
T = 77  # timepoints
# one of the clips are longer for some reason, but we force it to stop at timepoint 62
TR = run.__dict__['TR']
range_ = np.arange(0, T)
range_TR = range_*TR

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.set_title('Global Radial Strain over time', fontsize = 15)
ax1.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
#ax1.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax1.axhline(0, c = 'k', lw = 1)
ax1.set_xlim(0, np.mean(T_ed_list)*TR)
ax1.set_xlabel('Time [s]', fontsize = 15)
ax1.set_ylabel('%', fontsize = 17)

ax2.set_title('Global Circumferential Strain over time', fontsize = 15)
ax2.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
#ax2.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
ax2.axhline(0, c = 'k', lw = 1)
ax2.set_xlim(0, np.mean(T_ed_list)*TR)
ax2.set_xlabel('Time [s]', fontsize = 15)

rs_sham_ = np.sum(rs_sham, axis = 0)/len(rs_sham); T_ = len(rs_sham_)
cs_sham_ = np.sum(cs_sham, axis = 0)/len(cs_sham)
rs_mi_1d_ = np.sum(rs_mi_1d, axis = 0)/len(rs_mi_1d)
cs_mi_1d_ = np.sum(cs_mi_1d, axis = 0)/len(rs_mi_1d)
rs_mi_40d_ = np.sum(rs_mi_40d, axis = 0)/len(rs_mi_40d)
cs_mi_40d_ = np.sum(cs_mi_40d, axis = 0)/len(rs_mi_40d)


ax1.plot(range_TR[:T_], rs_sham_[:T_], lw=2, c='darkblue', label = 'Sham')
ax1.plot(range_TR[:T_], rs_mi_1d_[:T_], lw=2, c='purple', label = 'MI 1 day')
ax1.plot(range_TR[:T_], rs_mi_40d_[:T_], lw=2, c='red', label = 'MI 40+ days')

ax2.plot(range_TR[:T_], cs_sham_[:T_], lw=2, c='chocolate', label = 'Sham') 
ax2.plot(range_TR[:T_], cs_mi_1d_[:T_], lw=2, c='orangered', label = 'MI 1 day')
ax2.plot(range_TR[:T_], cs_mi_40d_[:T_], lw=2, c='red', label = 'MI 40+ days')
                       
ax1.legend(fontsize = 12)
ax2.legend(fontsize = 12)

plt.subplots_adjust(wspace=0.07)
if save == 1:
    plt.savefig(f'R:\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()


#%%
# strain
T = 77  # timepoints
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
    # drop this method and save matrices instead?
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
    
    T_ = len(r_strain_rate) # stops at respective end diastole
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
'''
plt.figure(figsize = (10, 8))
plt.title('Mean direction of compression', fontsize = 15)
plt.axvline(np.mean(T_es_list)*TR, c = 'k', ls = ':', lw = 2, label = 'End Systole')
plt.axvline(np.mean(T_ed_list)*TR, c = 'k', ls = '--', lw = 1.5, label = 'End Diastole')
plt.xlim(0, np.max(T_ed_list)*TR)#; plt.ylim(0, 50)
plt.xlabel('Timepoints', fontsize = 15)
plt.ylabel('Degrees', fontsize = 20)


a1_mean = np.zeros(T); a2_mean = np.zeros(T)
auc_mi = []; auc_sham = []  # to fill with tuples (days, auc)
for file in os.listdir('R:\Lasse\\angle distribution data'):
    a1 = np.load(fr'R:\Lasse\\angle distribution data\{str(file)}\angle_distribution_pos.npy', allow_pickle = 1)
    a2 = np.load(fr'R:\Lasse\\angle distribution data\{str(file)}\angle_distribution_neg.npy', allow_pickle = 1)
    
    diff = running_average(a2, 4)
    
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
        auc_mi.append([days, sum(cumtrapz(diff[:u]))])
    else:
        plt.plot(range_TR[:len(diff)], diff, 'k', lw=1.3)
        auc_sham.append([days, sum(cumtrapz(diff[:u]))])

legend_handles1 = [Line2D([0], [0], color = 'k', lw = 1.3, label = 'Sham'),
          Line2D([0], [0], color = 'r', lw = 1.3, label = 'MI')]

# difference
plt.legend(handles = legend_handles1, loc = 'upper right')
plt.show()

auc_mi = np.array(auc_mi)
auc_sham = np.array(auc_sham)
'''
#%%
# dataframe analysis

# Create the pandas DataFrame 
'''
df = pandas.DataFrame(df_list, columns=['Name', 'Day', 'GRS', 'GCS', \
                                        'Rad SDI', 'Circ SDI', 'GRSRs', \
                                            'GRSRd', 'GCSRd', 'GCSRs', \
                                                'a1_mean_max', 'a1_mean_min', \
                                                    'a2_mean_max', 'a2_mean_min', 'Condition']) 
'''
# to analyze a generated csv file instead
df = pandas.read_csv('combodata_analysis')
    
# uncomment to save new csv file
#df.to_csv('combodata_analysis', sep=',', index=False, encoding='utf-8')
    
# display 8 random data samples
print(f'Shape of dataset (instances, features): {df.shape}')
#%%
# correlation analysis
# https://www.kaggle.com/code/datafan07/heart-disease-and-some-scikit-learn-magic/notebook

#Compute pairwise correlation of columns, excluding NA/null values.
correlation = df.corr(method='pearson')

mask = np.triu(correlation) #diagonal + upper triangle redundant
fig=plt.figure(figsize=(14,12))
sns.heatmap(correlation, mask=mask, cmap='coolwarm', center = 0, annot=True, annot_kws={'size':14}, fmt='.2f')
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
fig.get_axes()[1].remove()#; plt.savefig('Corr_Heatmap')
plt.show()

#%%
# internal function that does linear fit on non-outlier data and plot
# only works with global values within this script
def ax_corr(ax, column_name):
    # create temporary dataframes 
    temp_sham = drop_outliers_IQR(df_sham, column_name, 100) 
    temp_mi = drop_outliers_IQR(df_mi, column_name, 100)
    valid_data = pandas.concat([temp_sham[1], temp_mi[1]])
    outliers = pandas.concat([temp_sham[0], temp_mi[0]])
    
    # find correlation and p value with days
    corr_sham, r_sham = stats.pearsonr(temp_sham[1][column_name], temp_sham[1]['Day'])
    corr_mi, r_mi = stats.pearsonr(temp_mi[1][column_name], temp_mi[1]['Day'])
    
    # t-test
    r = stats.ttest_ind(temp_sham[1][column_name], temp_mi[1][column_name])
    if r[1] < 0.001:
        r_str = 'r < 0.001'
    else:
        r_str = f'r = {np.round(r[1], 3)}'


    sns.lmplot(x='Day', y=column_name, hue='Condition', hue_order=[1,0], data = df, palette='Set1')
    '''
    valid_data.plot.scatter(x='Day', y=column_name, c='Condition', cmap=cmap, s=50, ax=ax, alpha=0.8, colorbar = 0)
    outliers.plot.scatter(x='Day', y=column_name, c='Condition', cmap=cmap, s=50, ax=ax, alpha=0.8, marker = 'x', colorbar = 0)
    
    
    
    ax.plot(t, temp_sham[2]*t + temp_sham[3], c = plt.get_cmap(cmap)(0), label = f'slope = {np.round(temp_sham[2], 3)}, p = {np.round(r_sham, 3)}')
    ax.plot(t, temp_mi[2]*t + temp_mi[3], c = plt.get_cmap(cmap)(1000), label = f'slope = {np.round(temp_mi[2], 3)}, p = {np.round(r_mi, 3)}, {r_str}')
    '''
    
# plot linear regression with 95% confidence interval
def sns_plot(column_name, ylabel_):
    s = sns.lmplot(x='Day', y=column_name, hue='Condition', hue_order=[1,0], data = df, \
                    palette='Set1', height=5, aspect=1.1, legend = 0) 
    s.ax.set_ylabel(ylabel_, fontsize = 15)
    s.ax.set_xlabel('Days', fontsize = 15)
    
    temp_sham = drop_outliers_IQR(df_sham, column_name, 100)[1]
    temp_mi = drop_outliers_IQR(df_mi, column_name, 100)[1]
    # t-test
    #r = stats.ttest_ind(temp_sham[1][column_name], temp_mi[1][column_name])
    
    #t test at start and end
    r1 = stats.ttest_ind(temp_sham[temp_sham['Day'] == 1][column_name], temp_mi[temp_mi['Day'] == 1][column_name])
    r40 = stats.ttest_ind(temp_sham[temp_sham['Day'] >= 40][column_name], temp_mi[temp_mi['Day'] >= 40][column_name])
   
    if r1[1] < 0.001:
        r_str1 = r'$p_{1} < 0.001$'
    else:
        r_str1 = fr'$p_{1} = ${np.round(r1[1], 3)}'
        
    if r40[1] < 0.001:
        r_str40 = r'$p_{40} < 0.001$'
    else:
        r_str40 = r'$p_{40} = $' + f'{np.round(r40[1], 3)}'
    # return p value that represents linreg comparison
    s.ax.text(22, np.min(df[column_name]), f'{r_str1}, {r_str40}', size=15, color='k')
    
#%%

df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

sns_plot('GCS', ylabel_ = 'GCS [%]')
sns_plot('GRS', ylabel_ = 'GRS [%]')

sns_plot('GRSRs', ylabel_ = 'GRSRs [$s^{-1}$]')
sns_plot('GRSRd', ylabel_ = 'GRSRd [$s^{-1}$]')
sns_plot('GCSRs', ylabel_ = 'GCSRs [$s^{-1}$]')
sns_plot('GCSRd', ylabel_ = 'GCSRd [$s^{-1}$]')

sns_plot('a1_mean_max', ylabel_ = 'a1_mean_max [Degrees]')
sns_plot('a1_mean_min', ylabel_ = 'a1_mean_min [Degrees]')
sns_plot('a2_mean_max', ylabel_ = 'a2_mean_max [Degrees]')
sns_plot('a2_mean_min', ylabel_ = 'a2_mean_min [Degrees]')

#%%
# peak strain values and dyssynchrony over time

#convert from numeric to categorical for correct label
df['Condition'] = pandas.Categorical(df['Condition'])
T_ = df['Day'].max(); t = np.arange(0, T_)  # x lim


df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

plt.rcParams.update({'font.size': 12})
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13,11))
#plt.title('Regional strain correlation analysis', fontsize = 15)

cmap = 'coolwarm'

ax_corr(ax1, 'GCS')
ax1.set_ylabel('GCS [%]', fontsize=15); ax1.set_xlabel(''); ax1.legend(loc = 4)

ax_corr(ax2, 'Circ SDI')
ax2.set_ylabel('Circumferential SDI [%]', fontsize=15); ax2.set_xlabel(''); ax2.legend(loc = 1)

ax_corr(ax3, 'GRS')
ax3.set_ylabel('GRS [%]', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend(loc = 1)

ax_corr(ax4, 'Rad SDI')
ax4.set_ylabel('Radial SDI [%]', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend(loc = 1)


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()

#%%
# strain rate peaks and direction over time

plt.rcParams.update({'font.size': 12})
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13,11))
#plt.title('Strain rate correlation analysis', fontsize = 15)

ax_corr(ax1, 'GRSRs')
ax1.set_ylabel('GRSRs [$s^{-1}$]', fontsize=15); ax1.set_xlabel(''); ax1.legend(loc = 4)

ax_corr(ax2, 'GRSRd')
ax2.set_ylabel('GRSRd', fontsize=15); ax2.set_xlabel(''); ax2.legend(loc = 1)

ax_corr(ax3, 'GCSRd')
ax3.set_ylabel('GCSRd', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend(loc = 1)

ax_corr(ax4, 'GCSRs')
ax4.set_ylabel('GCSRs', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend(loc = 1)


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()

#%%
# strain rate peaks and direction over time

plt.rcParams.update({'font.size': 12})
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13,11))
#plt.title('Strain rate direction correlation analysis', fontsize = 15)

ax_corr(ax1, 'a1_mean_max')
ax1.set_ylabel('a1_mean_max [Degrees]', fontsize=15); ax1.set_xlabel(''); ax1.legend()

ax_corr(ax2, 'a1_mean_min')
ax2.set_ylabel('a1_mean_min', fontsize=15); ax2.set_xlabel(''); ax2.legend()

ax_corr(ax3, 'a2_mean_max')
ax3.set_ylabel('a2_mean_max', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend()

ax_corr(ax4, 'a2_mean_min')
ax4.set_ylabel('a2_mean_min', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend()


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()