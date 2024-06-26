# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:12:08 2024

@author: lassetot

Curve analysis parameters collected and used to construct a pandas dataframe
for correlation analysis from the analysis using 3D strain rate tensors.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from ComboDataSR_3D import ComboDataSR_3D

from math import ceil, floor
from scipy.integrate import cumulative_trapezoid
from scipy import stats
from util import drop_outliers_IQR
import pandas 
import seaborn as sns; sns.set()
import statsmodels.api as sm

def strain(strain_rate, T_ed, weight = 10):  # inherit from 2d class?
    # weighting for integrals in positive/flipped time directions
    # cyclic boundary conditions
    # (old weights)
    #w = np.tanh((self.T_ed - 1 - self.range_)/weight) 
    #w_f = np.tanh(self.range_/weight) 
    
    # linear weights
    w1 = range_[:T_ed]*T_ed; w1 = w1/np.max(w1)
    w2 = np.flip(w1); w2 = w2/np.max(w2)

    strain = cumulative_trapezoid(strain_rate, range_TR/1000, initial=0)[:T_ed]
    strain_flipped = np.flip(cumulative_trapezoid(strain_rate[::-1], range_TR[::-1]/1000, initial=0))[:T_ed]
    return w2*strain + w1*strain_flipped

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

## 07.02.2024: Time elapsed for 41 files: 33 minutes
## (41 minutes with non-eroded masks)

# save characteristic time-points to calc average
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
        
    T_es = run.__dict__['T_es']
        
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
    TSd = np.max(theta_s)
    TSs = np.min(theta_s)
    TCs = np.max(theta_c)
    TCd = np.min(theta_c)
    
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
    
    #tcs_ = abs(np.max(basal_theta2) - np.max(apical_theta2))
    #tss_ = abs(np.min(basal_theta1) - np.min(apical_theta1))
    pcs_ = abs(np.max(basal_phi2) - np.max(apical_phi2))  # switch min/max
    pss_ = abs(np.min(basal_phi1) - np.min(apical_phi1))
    
    # systolic theta unity, mean value of inner 80% of time interval
    border = int(T_es*0.2)
    tcs_ = abs(np.mean(basal_theta2[border : T_es-border]) - np.mean(apical_theta2[border : T_es-border]))
    tss_ = abs(np.mean(basal_theta1[border : T_es-border]) - np.mean(apical_theta1[border : T_es-border]))
    
    # dataframe row
    df_list.append([filename, days, r_strain_peak, c_strain_peak, l_strain_peak, r_sr_max, \
                        r_sr_min, c_sr_max, c_sr_min, l_sr_max, l_sr_min, TSd, TSs, \
                            TCs, TCd, ps_max, ps_min, pc_max, pc_min, tcs_, tss_, pcs_, pss_, condition])
    filenr += 1
    if os.path.exists(f'R:\Lasse\plots\MP4\{file}') == False:
        os.makedirs(f'R:\Lasse\plots\MP4\{file}')
    
et = time.time()
print(f'Time elapsed for strain rate calculations on {filenr} files: {int((et-st)/60)} minutes') 

#%%
# dataframe analysis

# TSs = TSmin, TCs = Tcmax, TSd = Tsmax, TCd = Tcmin

# Create the pandas DataFrame 
'''
df = pandas.DataFrame(df_list, columns=['Name', 'Day', 'GRS', 'GCS', 'GLS', \
                                         'GRSRs', 'GRSRd', 'GCSRd', 'GCSRs', 'GLSRd', 'GLSRs', \
                                                'TSd', 'TSs', 'TCs', 'TCd', 'ps_max', \
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
    # linreg scatterplot
    s = sns.lmplot(x='Day', y=column_name, hue='Condition', hue_order=[1,0], data = df, \
                    palette='Set1', height=5, aspect=1.1, legend = 0) 
    s.ax.set_ylabel(ylabel_, fontsize = 15)
    s.ax.set_xlabel('Days', fontsize = 15)
    
    
    
    temp_sham = drop_outliers_IQR(df_sham, column_name, 100)[1]
    temp_mi = drop_outliers_IQR(df_mi, column_name, 100)[1]
    # t-test
    #r = stats.ttest_ind(temp_sham[1][column_name], temp_mi[1][column_name])
    
    # barplot p1 p40
    temp_c1 =  drop_outliers_IQR(df[df['Day'] == 1], column_name, 100)[1]
    temp_c40 =  drop_outliers_IQR(df[df['Day'] >= 40], column_name, 100)[1]
    temp_c40['Day'].replace([41,42,43,44,45], 40, inplace = True)
    
    # grouped days 40+ together
    temp_c = pandas.concat([temp_c1, temp_c40])
    
    # slope
    b1_mi = drop_outliers_IQR(df_mi, column_name, 100)[6]
    b1_sham = drop_outliers_IQR(df_sham, column_name, 100)[6]
    
    # slope p-values
    b_mi = drop_outliers_IQR(df_mi, column_name, 100)[4]
    b_sham = drop_outliers_IQR(df_sham, column_name, 100)[4]
    
    # slope ci
    ci_mi = drop_outliers_IQR(df_mi, column_name, 100)[5]*1.96
    ci_sham = drop_outliers_IQR(df_sham, column_name, 100)[5]*1.96
    
    print(f'beta1 mi pval: {np.round(b_mi, 3)}')
    print(f'beta1 sham pval: {np.round(b_sham, 3)}')
    
    # https://www.econometrics-with-r.org/2.1-random-variables-and-probability-distributions.html
    # https://www.econometrics-with-r.org/5.2-cifrc.html
    print(f'(b1 +- 95ci) mi: {np.round(b1_mi, 3)} {np.round(ci_mi, 3)}')
    print(f'(b1 +- 95ci) sham: {np.round(b1_sham, 3)} {np.round(ci_sham, 3)}')
    
    #t test at start and end
    r1 = stats.ttest_ind(temp_sham[temp_sham['Day'] == 1][column_name], temp_mi[temp_mi['Day'] == 1][column_name])
    r40 = stats.ttest_ind(temp_sham[temp_sham['Day'] >= 40][column_name], temp_mi[temp_mi['Day'] >= 40][column_name])
    
    
    # linreg slope pvalues (for scatter plot)
    if b_mi < 0.001:
        b_str1 = r'$\beta_1 = $' + f'{np.round(b1_mi, 3)},  $p < 0.001$'
    else:
        b_str1 = r'$\beta_1 = $' + f'{np.round(b1_mi, 3)},  p = {np.round(b_mi, 3)}'
        
    if b_sham < 0.001:
        b_str2 = r'$\beta_1 = $' + f'{np.round(b1_sham, 3)},  $p < 0.001$'
    else:
        b_str2 = r'$\beta_1 = $' + f'{np.round(b1_sham, 3)},  p = {np.round(b_sham, 3)}'
    
    # ttest pvalues (for catplot)
    if r1[1] < 0.001:
        r_str1 = 'Day 1 \n ($p < 0.001$)'
    else:
        r_str1 = f'Day 1 \n ($p = ${np.round(r1[1], 3)})'
        
    if r40[1] < 0.001:
        r_str40 = 'Day 40+ \n ($p < 0.001$)'
    else:
        r_str40 = f'Day 40+ \n ($p = ${np.round(r40[1], 3)})'
    # return p value that represents linreg comparison
    #s.ax.text(22, np.min(df[column_name]), f'{b_str1}, {b_str2}', size=15, color='k')
    s.ax.tick_params(axis='both', which='major', labelsize=13)
    
    c_cmap = mpl.colors.ListedColormap(sns.color_palette('Set1').as_hex())
    legend_handles1 = [Line2D([0], [0], color = c_cmap(0), lw = 2, label = b_str1),
              Line2D([0], [0], color = c_cmap(1), lw = 2, label = b_str2)]
    
    plt.legend(s, handles=legend_handles1, prop={'size': 12}); plt.show(s)
    
    
    # catplot
    c = sns.catplot(data = temp_c, x = 'Day', y = column_name, hue='Condition', hue_order=[1,0], \
                    palette='Set1', kind='bar', ci='sd', capsize=.1, alpha = 0.8, legend = 0)
    c.ax.set_ylabel(ylabel_, fontsize = 15)
    c.ax.set_xlabel('', fontsize = 15)
    
    c.ax.set_xticks([0,1], [r_str1, r_str40])
    c.ax.tick_params(axis='both', which='major', labelsize=15)

#%%
# peak strain values and dyssynchrony over time

#convert from numeric to categorical for correct label
df['Condition'] = pandas.Categorical(df['Condition'])
T_ = df['Day'].max(); t = np.arange(0, T_)  # x lim


df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

#plt.rcParams.update({'font.size': 12})
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

sns_plot('TSd', ylabel_ = r'$\theta_{sd}$ [Degrees]')
sns_plot('TSs', ylabel_ = r'$\theta_{ss}$ [Degrees]')
sns_plot('TCs', ylabel_ = r'$\theta_{cs}$ [Degrees]')
sns_plot('TCd', ylabel_ = r'$\theta_{cd}$ [Degrees]')

sns_plot('ps_max', ylabel_ = 'ps_max [Degrees]')
sns_plot('ps_min', ylabel_ = 'ps_min [Degrees]')
sns_plot('pc_max', ylabel_ = 'pc_max [Degrees]')
sns_plot('pc_min', ylabel_ = 'pc_min [Degrees]')

sns_plot('tcs_diff', ylabel_ = r'$\Delta \theta_{cs}$ [Degrees]')
sns_plot('tss_diff', ylabel_ = r'$\Delta \theta_{ss}$ [Degrees]')
sns_plot('pcs_diff', ylabel_ = 'pcs_diff [Degrees]')
sns_plot('pss_diff', ylabel_ = 'pss_diff [Degrees]')

#%%
# table of (mean +- std) for each parameter in df, grouped by condition

column = 'pc_max'
df_ = df[df['Day'] >= 40].groupby(['Condition'], as_index = False).agg({column:[np.mean, np.std]})
df__ = df[df['Day'] == 1].groupby(['Condition'], as_index = False).agg({column:[np.mean, np.std]})

print(f'Day 1: {df__.round(2)}')
print(f'Day 40+: {df_.round(2)}')

#%%

df_mi_40 = df_mi[df_mi['Day'] >= 40]  # chronic stage MI
df_sham_40 = df_sham[df_sham['Day'] >= 40]  # chronic stage MI

#%%
'''
ax_corr(ax2, 'GLS')
ax2.set_ylabel('GLS [%]', fontsize=15); ax2.set_xlabel(''); ax2.legend(loc = 1)

ax_corr(ax3, 'GRS')
ax3.set_ylabel('GRS [%]', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend(loc = 1)

ax_corr(ax4, 'GLSRs')
ax4.set_ylabel('GLSRs [$s^{-1}$]', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend(loc = 1)


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

ax_corr(ax1, 'TSd')
ax1.set_ylabel('TSd [Degrees]', fontsize=15); ax1.set_xlabel(''); ax1.legend()

ax_corr(ax2, 'TSs')
ax2.set_ylabel('TSs', fontsize=15); ax2.set_xlabel(''); ax2.legend()

ax_corr(ax3, 'TCs')
ax3.set_ylabel('TCs', fontsize=15); ax3.set_xlabel('Days', fontsize=15); ax3.legend()

ax_corr(ax4, 'TCd')
ax4.set_ylabel('TCd', fontsize=15); ax4.set_xlabel('Days', fontsize=15); ax4.legend()


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
'''