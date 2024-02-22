# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:12:08 2024

@author: lassetot

Curve analysis parameters collected and used to construct a pandas dataframe
for correlation analysis.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ComboDataSR_3D import ComboDataSR_3D

from math import ceil, floor
from scipy.integrate import cumtrapz
from scipy import stats
from util import running_average, drop_outliers_IQR
import pandas 
import seaborn as sns; sns.set()
import statsmodels.api as sm

def strain(strain_rate, T_ed, weight = 10):  # inherit from 2d class?
    # weighting for integrals in positive/flipped time directions
    # cyclic boundary conditions
    w = np.tanh((T_ed - 1 - range_)/weight) 
    w_f = np.tanh(range_/weight) 

    strain = cumtrapz(strain_rate, range_TR/1000, initial=0)
    strain_flipped = np.flip(cumtrapz(strain_rate[::-1]/1000, range_TR[::-1], initial=0))
    return (w*strain + w_f*strain_flipped)/2

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

## 07.02.2024: Time elapsed for 41 files: 33 minutes
## (41 minutes with non-eroded masks)

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
for file in os.listdir('R:\Lasse\combodata_3d_shax'):
    if filenr > 0:
        et = time.time()
        print(f'(( File {filenr + 1}, time elapsed: {int((et-st)/60)} minutes ))')
    else:
        print(f'(( File {filenr + 1} ))')
        
    file_ = os.path.splitext(file)
    
    # collect parameters
    #T_es_list.append(run.__dict__['T_es'])
    #T_ed_list.append(run.__dict__['T_ed'])
    
    run = ComboDataSR_3D(file_[0], n = 1)
    slices = run.__dict__['slices']  # amount of slices
    slice_selection = np.arange(2, slices) 
    
    # all slices
    T_ed = []
    for slice_ in range(1, slices):
        T_ed.append(run.__dict__['T_ed'][f'T_ed{slice_}'])
        
    # curves for parameter analysis
    T_ed_min = np.min(np.array(T_ed))
    total_lsr = []; total_csr = []; total_rsr = []  # strain rate
    total_ls = []; total_cs = []; total_rs = []  # strain
    theta_stretch = []; theta_comp = []  # in-plane sr direction
    phi_stretch = []; phi_comp = []  # out-of-plane sr direction
        
    # all slices with surrounding slices
    for slice_ in slice_selection:
        run.strain_rate(slice_, save = 0, plot = 0)
        print(f'Slice [{slice_} / {slice_selection[-1]}]')
        
        # if there is a sector mask issue, data will not be appended
        if all(np.array(run.__dict__['theta1'])[:T_ed_min]) == True:
            total_ls.append(np.array(run.__dict__['l_strain'])[:T_ed_min])
            total_cs.append(np.array(run.__dict__['c_strain'])[:T_ed_min])
            total_rs.append(np.array(run.__dict__['r_strain'])[:T_ed_min])
            total_lsr.append(np.array(run.__dict__['l_strain_rate'])[:T_ed_min])
            total_csr.append(np.array(run.__dict__['c_strain_rate'])[:T_ed_min])
            total_rsr.append(np.array(run.__dict__['r_strain_rate'])[:T_ed_min])
            
            theta_stretch.append(np.array(run.__dict__['theta1'])[:T_ed_min])
            theta_comp.append(np.array(run.__dict__['theta2'])[:T_ed_min])
            phi_stretch.append(np.array(run.__dict__['phi1'])[:T_ed_min])
            phi_comp.append(np.array(run.__dict__['phi2'])[:T_ed_min])
            
    # total LV strain rate for all slices
    lsr = np.sum(np.array(total_lsr), axis = 0) / len(slice_selection)
    csr = np.sum(np.array(total_csr), axis = 0) / len(slice_selection)
    rsr = np.sum(np.array(total_rsr), axis = 0) / len(slice_selection)
    
    # total/mean sr directions across all slices
    theta_s = np.sum(np.array(theta_stretch), axis = 0) / len(slice_selection)
    theta_c = np.sum(np.array(theta_comp), axis = 0) / len(slice_selection)
    phi_s = np.sum(np.array(phi_stretch), axis = 0) / len(slice_selection)
    phi_c = np.sum(np.array(phi_comp), axis = 0) / len(slice_selection)
    
    TR = run.__dict__['TR']
    range_ = np.arange(0, T_ed_min)
    range_TR = range_*TR
    
    # derive strain from the total sr curve, not sum of collected strain curves
    # to avoid noise accumilation
    ls = strain(lsr, T_ed_min)*100000
    cs = strain(csr, T_ed_min)*100000
    rs = strain(rsr, T_ed_min)*100000
    
    # collect dataframe parameters
    filename = run.__dict__['filename']
    if str(filename[-1]) == 'w':
           days = int(filename.split('_')[2].replace('w', ''))*7
    if str(filename[-1]) == 'd':
           days = int(filename.split('_')[2].replace('d', ''))
    
    if str(filename[0]) == 'm':
           condition = 1  # mi
    else:
           condition = 0 # sham
    
    # collect strain curve parameters
    r_strain_peak = np.max(rs)
    c_strain_peak = np.min(cs)
    l_strain_peak = np.min(ls)
    
    # collect directional parameters
    
    # expressed as percentage of cardiac cycle duration
    # (mean SDI across slices?)
    '''
    r_strain_peaktime_std = 100*np.std(run.__dict__['r_peaktime'])/(TR*T_ed_list[-1])
    c_strain_peaktime_std = 100*np.std(run.__dict__['c_peaktime'])/(TR*T_ed_list[-1])
    '''
    # strain rate parameters
    r_sr_max = np.max(rsr); r_sr_min = np.min(rsr)
    c_sr_max = np.max(csr); c_sr_min = np.min(csr)
    l_sr_max = np.max(lsr); l_sr_min = np.min(lsr)
    
    # theta dist
    ts_max = np.max(theta_s)
    ts_min = np.min(theta_s)
    tc_max = np.max(theta_c)
    tc_min = np.min(theta_c)
    
    # phi dist
    ps_max = np.max(phi_s)
    ps_min = np.min(phi_s)
    pc_max = np.max(phi_c)
    pc_min = np.min(phi_c)
    
    # let apical/basal overlap with one level if odd nr of slices
    #odd = int((len(slice_selection) % 2) == 0)  # odd = 0 if odd, 1 if even
    a_length = len(theta_s[:ceil((len(slice_selection)/2))])
    b_start = len(theta_s[:floor((len(slice_selection)/2))])
    
    basal_theta1 = np.sum(np.array(theta_stretch[b_start:]), axis = 0) / a_length
    apical_theta1 = np.sum(np.array(theta_stretch[:a_length]), axis = 0) / a_length
    basal_theta2 = np.sum(np.array(theta_comp[b_start:]), axis = 0) / a_length
    apical_theta2 = np.sum(np.array(theta_comp[:a_length]), axis = 0) / a_length

    basal_phi1 = np.sum(np.array(phi_stretch[b_start:]), axis = 0) / a_length
    apical_phi1 = np.sum(np.array(phi_stretch[:a_length]), axis = 0) / a_length
    basal_phi2 = np.sum(np.array(phi_comp[b_start:]), axis = 0) / a_length
    apical_phi2 = np.sum(np.array(phi_comp[:a_length]), axis = 0) / a_length
    
    tcs_ = abs(np.max(basal_theta2) - np.max(apical_theta2))
    tss_ = abs(np.min(basal_theta1) - np.min(apical_theta1))
    pcs_ = abs(np.max(basal_phi2) - np.max(apical_phi2))  # switch min/max
    pss_ = abs(np.min(basal_phi1) - np.min(apical_phi1))
    
    # dataframe row
    df_list.append([filename, days, r_strain_peak, c_strain_peak, l_strain_peak, r_sr_max, \
                        r_sr_min, c_sr_max, c_sr_min, l_sr_max, l_sr_min, ts_max, ts_min, \
                            tc_max, tc_min, ps_max, ps_min, pc_max, pc_min, tcs_, tss_, pcs_, pss_, condition])
    filenr += 1
    if os.path.exists(f'R:\Lasse\plots\MP4\{file}') == False:
        os.makedirs(f'R:\Lasse\plots\MP4\{file}')
    
et = time.time()
print(f'Time elapsed for strain rate calculations on {filenr} files: {int((et-st)/60)} minutes') 

#%%
# dataframe analysis

# Create the pandas DataFrame 
'''
df = pandas.DataFrame(df_list, columns=['Name', 'Day', 'GRS', 'GCS', 'GLS', \
                                         'GRSRs', 'GRSRd', 'GCSRd', 'GCSRs', 'GLSRd', 'GLSRs', \
                                                'ts_max', 'ts_min', 'tc_max', 'tc_min', 'ps_max', \
                                                     'ps_min', 'pc_max', 'pc_min', 'tcs_diff', \
                                                         'tss_diff', 'pcs_diff', 'pss_diff', 'Condition']) 
'''
# to analyze a generated csv file instead
df = pandas.read_csv('combodata_analysis_3d')
    
# uncomment to save new csv file
#df.to_csv('combodata_analysis_3d', sep=',', index=False, encoding='utf-8')

print(f'Shape of dataset (instances, features): {df.shape}')

#%%
# correlation matrix
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

    valid_data.plot.scatter(x='Day', y=column_name, c='Condition', cmap=cmap, s=50, ax=ax, alpha=0.8, colorbar = 0)
    outliers.plot.scatter(x='Day', y=column_name, c='Condition', cmap=cmap, s=50, ax=ax, alpha=0.8, marker = 'x', colorbar = 0)
    
    
    ax.plot(t, temp_sham[2]*t + temp_sham[3], c = plt.get_cmap(cmap)(0), label = f'slope = {np.round(temp_sham[2], 3)}, p = {np.round(r_sham, 3)}')
    ax.plot(t, temp_mi[2]*t + temp_mi[3], c = plt.get_cmap(cmap)(1000), label = f'slope = {np.round(temp_mi[2], 3)}, p = {np.round(r_mi, 3)}, {r_str}')
    

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
# peak strain values and dyssynchrony over time

#convert from numeric to categorical for correct label
df['Condition'] = pandas.Categorical(df['Condition'])
T_ = df['Day'].max(); t = np.arange(0, T_)  # x lim


df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

plt.rcParams.update({'font.size': 12})
#fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13,11))
#plt.title('Regional strain correlation analysis', fontsize = 15)

cmap = 'coolwarm'

sns_plot('GCS', ylabel_ = 'GCS [%]')
sns_plot('GRS', ylabel_ = 'GRS [%]')
sns_plot('GLS', ylabel_ = 'GLS [%]')
sns_plot('GLSRs', ylabel_ = 'GLSRs [$s^{-1}$]')
sns_plot('GLSRd', ylabel_ = 'GLSRd [$s^{-1}$]')

sns_plot('GRSRs', ylabel_ = 'GRSRs [$s^{-1}$]')
sns_plot('GRSRd', ylabel_ = 'GRSRd [$s^{-1}$]')
sns_plot('GCSRs', ylabel_ = 'GCSRs [$s^{-1}$]')
sns_plot('GCSRd', ylabel_ = 'GCSRd [$s^{-1}$]')

sns_plot('ts_max', ylabel_ = 'ts_max [Degrees]')
sns_plot('ts_min', ylabel_ = 'ts_min [Degrees]')
sns_plot('tc_max', ylabel_ = 'tc_max [Degrees]')
sns_plot('tc_min', ylabel_ = 'tc_min [Degrees]')

sns_plot('ps_max', ylabel_ = 'ps_max [Degrees]')
sns_plot('ps_min', ylabel_ = 'ps_min [Degrees]')
sns_plot('pc_max', ylabel_ = 'pc_max [Degrees]')
sns_plot('pc_min', ylabel_ = 'pc_min [Degrees]')

sns_plot('tcs_diff', ylabel_ = 'tcs_diff [Degrees]')
sns_plot('tss_diff', ylabel_ = 'tss_diff [Degrees]')
sns_plot('pcs_diff', ylabel_ = 'pcs_diff [Degrees]')
sns_plot('pss_diff', ylabel_ = 'pss_diff [Degrees]')

'''
ax_corr(ax2, 'GLS')
ax2.set_ylabel('GLS [%]', fontsize=15); ax2.set_xlabel(''); ax2.legend(loc = 1)

ax_corr(ax3, 'GRS')
ax3.set_ylabel('GRS [%]', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend(loc = 1)

ax_corr(ax4, 'GLSRs')
ax4.set_ylabel('GLSRs [$s^{-1}$]', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend(loc = 1)


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()
'''
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

ax_corr(ax1, 'ts_max')
ax1.set_ylabel('ts_max [Degrees]', fontsize=15); ax1.set_xlabel(''); ax1.legend()

ax_corr(ax2, 'ts_min')
ax2.set_ylabel('ts_min', fontsize=15); ax2.set_xlabel(''); ax2.legend()

ax_corr(ax3, 'tc_max')
ax3.set_ylabel('tc_max', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend()

ax_corr(ax4, 'tc_min')
ax4.set_ylabel('tc_min', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend()


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()

#%%
# strain rate peaks and direction over time

plt.rcParams.update({'font.size': 12})
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13,11))
#plt.title('Strain rate direction correlation analysis', fontsize = 15)

ax_corr(ax1, 'ps_max')
ax1.set_ylabel('ps_max [Degrees]', fontsize=15); ax1.set_xlabel(''); ax1.legend()

ax_corr(ax2, 'ps_min')
ax2.set_ylabel('ps_min', fontsize=15); ax2.set_xlabel(''); ax2.legend()

ax_corr(ax3, 'pc_max')
ax3.set_ylabel('pc_max', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend()

ax_corr(ax4, 'pc_min')
ax4.set_ylabel('pc_min', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend()


plt.subplots_adjust(wspace=0.25, hspace=0.15)#; plt.savefig('Heart_Scatter')
plt.show()