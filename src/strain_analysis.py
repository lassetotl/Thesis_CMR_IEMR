# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:39:12 2023

@author: lassetot

Curve analysis parameters collected and used to construct a pandas dataframe
for statistical analysis between Sham and MI, and between LV sectors.
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from ComboDataSR_2D import ComboDataSR_2D
from scipy.integrate import cumulative_trapezoid
from scipy import stats
from util import drop_outliers_IQR
import pandas 
import seaborn as sns; sns.set()

#import warnings
#warnings.simplefilter("ignore", DeprecationWarning)

#%%
## This segment will take some time to run, and will overwrite saved data if save = 1 !! ##

# save characteristic time-points to calc average
T_es_list = []
T_ed_list = []

df_list = []
mis_list = []

# sham, mi 1d and >40d in separate lists
rs_sham = []; cs_sham = []
rs_mi_1d = []; cs_mi_1d = []
rs_mi_40d = []; cs_mi_40d = []
tp = 60

st = time.time()
filenr = 0
save = 1
for file in os.listdir(r'C:\Users\lasse\Desktop\IEMR\Lasse\combodata_shax'):
    file_ = os.path.splitext(file)
    run = ComboDataSR_2D(file_[0], n = 1, sigma=0)  # n = 1 should be used for proper analysis
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
    
    condition = 0
    if str(filename[0]) == 'm':
           condition = 1  # mi
           mis_list.append(run.__dict__['mis'])
           if days == 1:
               rs_mi_1d.append(rs)
               cs_mi_1d.append(cs)
           if days >= 40:
               rs_mi_40d.append(rs)
               cs_mi_40d.append(cs)
               
    else:
           rs_sham.append(rs)
           cs_sham.append(cs)
    
    # collect strain curve parameters
    r_strain_peak_mean = np.mean(run.__dict__['r_peakvals'])
    c_strain_peak_mean = np.mean(run.__dict__['c_peakvals'])
    
    r_strain_peak_std = np.std(run.__dict__['r_peakvals'])
    c_strain_peak_std = np.std(run.__dict__['c_peakvals'])
    
    # collect regional strain peaks
    
    # index order - infarct, adjacent, medial, remote
    r_strain_reg = run.__dict__['r_peakvals']
    c_strain_reg = run.__dict__['c_peakvals']
    GCSRs_reg = run.__dict__['GCSRs_peakvals']
    GCSRd_reg = run.__dict__['GCSRd_peakvals']
    GRSRs_reg = run.__dict__['GRSRs_peakvals']
    GRSRd_reg = run.__dict__['GRSRd_peakvals']
    
    TSd_reg = run.__dict__['TSd_peakvals']
    TSs_reg = run.__dict__['TSs_peakvals']
    TCd_reg = run.__dict__['TCd_peakvals']
    TCs_reg = run.__dict__['TCs_peakvals']
    std_s_min = run.__dict__['std_s_min']
    std_e_min = run.__dict__['std_e_min']
    
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
    
    a_std_s = run.__dict__['theta_std_s']
    a_std_e = run.__dict__['theta_std_e']
    t_peak_diff_s = run.__dict__['peaktime_diff_s'] #
    t_peak_diff_e = run.__dict__['peaktime_diff_e'] #
    
    # dataframe row
    df_list.append([filename, days, r_strain_peak_mean, c_strain_peak_mean, \
                    r_strain_peaktime_std, c_strain_peaktime_std, r_sr_max, \
                        r_sr_min, c_sr_max, c_sr_min, a1_mean_max, a1_mean_min, \
                            a2_mean_max, a2_mean_min, r_strain_peak_std, c_strain_peak_std, \
                                r_strain_reg, c_strain_reg, a_std_s, a_std_e, std_s_min, \
                                    std_e_min, t_peak_diff_s, t_peak_diff_e, GCSRs_reg, \
                                        GCSRd_reg, GRSRs_reg, GRSRd_reg, TSs_reg, TSd_reg,\
                                            TCs_reg, TCd_reg, condition])
    filenr += 1
    if os.path.exists(fr'C:\Users\lasse\Desktop\IEMR\Lasse\plots\MP4\{file}') == False:
        os.makedirs(fr'C:\Users\lasse\Desktop\IEMR\Lasse\plots\MP4\{file}')
    
et = time.time()
print(f'Time elapsed for strain rate calculations on {filenr} files: {int((et-st)/60)} minutes')  

#%% mean infarct sectors

#print(mis_list)
'''
m1 = []; m2 = []
for t in mis_list:
    m2.append(max(t))
    m1.append(min(t))
print(f'Mean MI sector: [{int(np.mean(m1))}, {int(np.mean(m2))}]')
'''
# output: Mean MI sector: [5, 18]
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
    plt.savefig(fr'C:\Users\lasse\Desktop\IEMR\Lasse\plots\MP4\{file}\{file}_GS.PNG')
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

for file in os.listdir(r'C:\Users\lasse\Desktop\IEMR\Lasse\strain data'):
    # drop this method and save matrices instead?
    r_strain = np.load(fr'C:\Users\lasse\Desktop\IEMR\Lasse\strain data\{str(file)}\r_strain.npy', allow_pickle = 1)
    c_strain = np.load(fr'C:\Users\lasse\Desktop\IEMR\Lasse\strain data\{str(file)}\c_strain.npy', allow_pickle = 1)
    
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
plt.savefig(fr'C:\Users\lasse\Desktop\IEMR\Lasse\plots\MP4\{file}\{file}_GS.PNG')
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

for file in os.listdir(r'C:\Users\lasse\Desktop\IEMR\Lasse\strain rate data'):
    r_strain_rate = np.load(fr'C:\Users\lasse\Desktop\IEMR\Lasse\strain rate data\{str(file)}\r_strain_rate.npy', allow_pickle = 1)
    c_strain_rate = np.load(fr'C:\Users\lasse\Desktop\IEMR\Lasse\strain rate data\{str(file)}\c_strain_rate.npy', allow_pickle = 1)
    
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
plt.savefig(fr'C:\Users\lasse\Desktop\IEMR\Lasse\plots\MP4\{file}\{file}_GS.PNG')
plt.show()


#%%
# dataframe analysis

# Create the pandas DataFrame 
#'''
df = pandas.DataFrame(df_list, columns=['Name', 'Day', 'GRS', 'GCS', \
                                        'Rad SDI', 'Circ SDI', 'GRSRs', \
                                            'GRSRd', 'GCSRd', 'GCSRs', \
                                                'TSd', 'TSs', 'TCs', 'TCd', \
                                                    'r_std', 'c_std', 'r_reg', 'c_reg', \
                                                        'angle_std_s', 'angle_std_e', \
                                                            'std_s_reg', 'std_e_reg', 't_peak_diff_s', 't_peak_diff_e', 'GCSRs_reg', \
                                                                'GCSRd_reg', 'GRSRs_reg', 'GRSRd_reg', 'TSs_reg', 'TSd_reg',\
                                                                    'TCs_reg', 'TCd_reg','Condition']) 
#'''
# to analyze a generated csv file instead
#df = pandas.read_csv('combodata_analysis')
#df = pandas.read_csv('combodata_analysis_may_2026')
    
# uncomment to save new csv file
#df.to_csv('combodata_analysis_may_2026', sep=',', index=False, encoding='utf-8')
    
# display 8 random data samples
print(f'Shape of dataset (instances, features): {df.shape}')

#%%
# modifiser TSd til å vise avstand som avstand fra 90

TSd_mod = []; TSs_mod = []
TCd_mod = []; TCs_mod = []
for row in range(len(df)):
    TSd_mod.append(abs(90 - df['TSd'][row]))
    TSs_mod.append(abs(df['TSs'][row]))
    
    TCd_mod.append(abs(df['TCd'][row]))
    TCs_mod.append(abs(90 - df['TCs'][row]))
    
df['TSd_mod'] = TSd_mod; df['TSs_mod'] = TSs_mod
df['TCd_mod'] = TCd_mod; df['TCs_mod'] = TCs_mod

TSd_reg_mod = []; TSs_reg_mod = []
TCd_reg_mod = []; TCs_reg_mod = []

for row in range(len(df)):
    TSd_reg_mod.append(abs(90 - 180/np.pi*np.array(df['TSd_reg'][row])))
    TSs_reg_mod.append(abs(180/np.pi*np.array(df['TSs_reg'][row])))
    
    TCd_reg_mod.append(abs(180/np.pi*np.array(df['TCd_reg'][row])))
    TCs_reg_mod.append(abs(90 - 180/np.pi*np.array(df['TCs_reg'][row])))
    
df['TSd_reg_mod'] = TSd_reg_mod; df['TSs_reg_mod'] = TSs_reg_mod
df['TCd_reg_mod'] = TCd_reg_mod; df['TCs_reg_mod'] = TCs_reg_mod
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

df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

sns_plot('GCS', ylabel_ = 'GCS [%]')
sns_plot('GRS', ylabel_ = 'GRS [%]')
sns_plot('Circ SDI', ylabel_ = 'CSDI [%]')
sns_plot('Rad SDI', ylabel_ = 'RSDI [%]')

#sns_plot('r_std', ylabel_ = 'rstd [%]')
#sns_plot('c_std', ylabel_ = 'cstd [%]')

sns_plot('GRSRs', ylabel_ = 'GRSRs [$s^{-1}$]')
sns_plot('GRSRd', ylabel_ = 'GRSRd [$s^{-1}$]')
sns_plot('GCSRs', ylabel_ = 'GCSRs [$s^{-1}$]')
sns_plot('GCSRd', ylabel_ = 'GCSRd [$s^{-1}$]')

sns_plot('TSd', ylabel_ = r'$\theta_{sd}$ [Degrees]')
sns_plot('TSs', ylabel_ = r'$\theta_{ss}$ [Degrees]')
sns_plot('TCs', ylabel_ = r'$\theta_{cs}$ [Degrees]')
sns_plot('TCd', ylabel_ = r'$\theta_{cd}$ [Degrees]')

sns_plot('angle_std_s', ylabel_ = r'$\theta_{cs}$ [Degrees]')
sns_plot('angle_std_e', ylabel_ = r'$\theta_{cd}$ [Degrees]')

sns_plot('t_peak_diff_s', ylabel_ = r'$\theta_{cs}$ [Degrees]')
sns_plot('t_peak_diff_e', ylabel_ = r'$\theta_{cd}$ [Degrees]')

#%%
# table of (mean +- std) for each parameter in df, grouped by condition

column = 'angle_std_s'
df_ = df[df['Day'] >= 40].groupby(['Condition'], as_index = False).agg({column:['mean', 'std']})
df__ = df[df['Day'] == 1].groupby(['Condition'], as_index = False).agg({column:['mean', 'std']})

print(f'Day 1: {df__.round(2)}')
print(f'Day 40+: {df_.round(2)}')

#%%
# chronic sham vs mi, mean, std, pval

column = 'GRS'
#df_mi_1 = df_mi[df_mi['Day'] == 1]
df_mi_40 = df_mi[df_mi['Day'] >= 40]  # chronic stage MI
df_sham_40 = df_sham[df_sham['Day'] >= 40]  # chronic stage MI

meanval_sham = np.mean(df_sham_40[column]); stdval_sham = np.std(df_sham_40[column])
meanval_mi = np.mean(df_mi_40[column]); stdval_mi = np.std(df_mi_40[column])
pval = stats.ttest_ind(df_sham_40[column].dropna(), df_mi_40[column].dropna(), equal_var=False, nan_policy='raise')[1]

print(f'stats - {column}:')
print(fr'Sham: {round(meanval_sham, 3)} $\pm$ {round(stdval_sham, 3)}')
print(fr'MI: {round(meanval_mi, 3)} $\pm$ {round(stdval_mi, 3)}')
print(fr'p-value: {round(pval, 3)}')

#%%

#%%
# box plot MI hearts regional variation (only works with fresh dataframe ??)
# bug: c_reg and r_reg keys turn from list into strings when loading df?
# c_reg or r_reg or TSd_reg or TSs_reg or TCd_reg or TCs_reg or std_s_reg or std_e_reg or 
# GCSRs_reg or TSs_reg_mod (etc)
column = 'c_reg'

# Sham

g1 = []
g2 = []
g3 = []
g4 = []

# c_reg or r_reg
for key, value in df_sham_40[column].items():
    g1.append(value[0])  
    g2.append(value[1])  
    g3.append(value[2])  
    g4.append(value[3])  

mask = ~np.isnan(g1) & ~np.isnan(g2) & ~np.isnan(g3) & ~np.isnan(g4)
g1 = np.array(g1)[mask]
g2 = np.array(g2)[mask]
g3 = np.array(g3)[mask]
g4 = np.array(g4)[mask]

# regional colormap
c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())
norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)

# p values compared with infarct

pa_sham = round(stats.ttest_ind(g1, g2, equal_var=False)[1], 3)
pm_sham = round(stats.ttest_ind(g1, g3, equal_var=False)[1], 3)
pr_sham = round(stats.ttest_ind(g1, g4, equal_var=False)[1], 3)
#print(column, 'sham:', pa_sham,pm_sham,pr_sham)

# Holm-Bonferroni correction
pvals_sham = sorted([pa_sham, pm_sham, pr_sham])
HB_sham = ''
for i in range(3):
    p_corr = 0.05/(3-i)
    if pvals_sham[i] <= p_corr:
        HB_sham += 'V'
    else: 
        HB_sham += 'X'

# line-plot individual animals (each column is one animal)
all_points_sham = np.array([g1, g2, g3, g4])

#plt.xticks([0, 1, 2, 3], ['Sector 1', f'Sector 2 \n ($p =${np.round(pa, 3)})', \
#                          f'Sector 3 \n ($p =${np.round(pm, 3)})', f'Sector 4 \n ($p =${np.round(pr, 3)})'])
    
                          
#plt.scatter([0]*len(df_sham_40[column]), g1, color = 'darkred', s = 40)
#plt.scatter([1]*len(df_sham_40[column]), g2, color = 'darkgreen', s = 40)
#plt.scatter([2]*len(df_sham_40[column]), g3, color = 'darkblue', s = 40)
#plt.scatter([3]*len(df_sham_40[column]), g4, color = 'indigo', s = 40)

#plt.title(f'{column}, sham: a {pa}, m {pm}, r {pr}')

#ymin = plt.axis()[2]
#ymax = plt.axis()[3]
#plt.ylim(ymin, ymax)
plt.show()

#%

# MI

g1_ = []
g2_ = []
g3_ = []
g4_ = []

# c_reg or r_reg
for key, value in df_mi_40[column].items():
    g1_.append(value[0])  
    g2_.append(value[1])  
    g3_.append(value[2])  
    g4_.append(value[3])  

mask = ~np.isnan(g1_) & ~np.isnan(g2_) & ~np.isnan(g3_) & ~np.isnan(g4_)
infarct = np.array(g1_)[mask]
adjacent = np.array(g2_)[mask]
medial = np.array(g3_)[mask]
remote = np.array(g4_)[mask]

# regional colormap
c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())
norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)

# p values compared with infarct

pa = round(stats.ttest_ind(infarct, adjacent, equal_var=False)[1], 3)
pm = round(stats.ttest_ind(infarct, medial, equal_var=False)[1], 3)  # first value in medial was NaN, remove first infarct index
pr = round(stats.ttest_ind(infarct, remote, equal_var=False)[1], 4)
print(column, 'mi:', pa,pm,pr)

# Holm-Bonferroni correction
pvals_mi = sorted([pa, pm, pr])
HB_mi = ''
for i in range(3):
    p_corr = 0.05/(3-i)
    print(p_corr)
    if pvals_mi[i] <= p_corr:
        HB_mi += 'V'
    else: 
        HB_mi += 'X'

# line-plot individual animals (each column is one animal)
all_points_mi = np.array([infarct, adjacent, medial, remote])


plt.figure(figsize=(6, 2.5), dpi=200)
#plt.title('GRS Regional variation MI')

ax1=sns.stripplot(data = [g1, g2, g3, g4, infarct, adjacent, medial, remote], size=5, \
            palette = [c_cmap(0), c_cmap(1), c_cmap(2), c_cmap(3)]*2, alpha = 1, jitter=False)

for i in range(len(g1)):
    ax1.plot(list(range(4)), all_points_sham[:,i], color='gray', zorder=1, alpha=0.6, lw=0.9)
    
for i in range(len(infarct)):
    ax1.plot(list(range(4,8)), all_points_mi[:,i], color='gray', zorder=1, alpha=0.6, lw=0.9)


ax2=sns.pointplot(data = [g1, g2, g3, g4, infarct, adjacent, medial, remote], errorbar=None, marker='_', \
              markersize=20, markeredgewidth=3, color='k', join=False)
    
plt.setp(ax1.collections, zorder=2)
plt.setp(ax2.collections, zorder=2)
plt.yticks(fontsize=8)
    
# uncomment to include p values relative to infarct
#plt.xticks([0, 1, 2, 3], ['Infarct', f'Adjacent \n ($p =${np.round(pa, 3)})', \
#                          f'Medial \n ($p =${np.round(pm, 3)})', f'Remote \n ($p =${np.round(pr, 3)})'])
    
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Infarct', 'Adjacent', 'Medial', 'Remote'], size=7)
#plt.scatter([0]*len(infarct), infarct, color = 'darkred', s = 40)
#plt.scatter([1]*len(adjacent), adjacent, color = 'darkgreen', s = 40)
#plt.scatter([2]*len(medial), medial, color = 'darkblue', s = 40)
#plt.scatter([3]*len(remote), remote, color = 'indigo', s = 40)

ymax = max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
ymin = min([ax1.get_ylim()[0], ax2.get_ylim()[0]])

#plt.ylim(ymin, ymax + abs(max([abs(ymax), abs(ymin)]))*0.4)
plt.gca().set_ylim(top = ymax + abs(max([abs(ymax), abs(ymin)]))*0.4)
plt.axvline(3.5, color='white', linewidth=7)
plt.title(f'{column}, sham:, 2) {pa_sham}, 3) {pm_sham}, 4) {pr_sham} {HB_sham[::-1]}- mi: a {pa}, m {pm}, r {pr} {HB_mi[::-1]}', fontsize = 8)
plt.show()

#% Sham vs MI @ 6w

''' save manually:
sham_SR = np.concatenate([g2, g3, g4], axis = None)
mi_SR = np.concatenate([adjacent, medial, remote], axis = None)
sham_mean_SR = np.mean([g2, g3, g4], axis=0)
mi_mean_SR = np.mean([adjacent, medial, remote], axis=0)
'''
sham = np.concatenate([g2, g3, g4], axis = None)
mi = np.concatenate([adjacent, medial, remote], axis = None)
sham_mean = np.mean([g2, g3, g4], axis=0)
mi_mean = np.mean([adjacent, medial, remote], axis=0)

meanval_sham = np.mean(sham); stdval_sham = np.std(sham)
meanval_mi = np.mean(mi); stdval_mi = np.std(mi)
pval = stats.ttest_ind(sham, mi)[1]

print(f'stats, viable myocardium - {column}:')
print(fr'Sham: {round(meanval_sham, 3)} $\pm$ {round(stdval_sham, 3)}')
print(fr'MI: {round(meanval_mi, 3)} $\pm$ {round(stdval_mi, 3)}')
print(fr'p-value: {round(pval, 3)}')

#%% viable correlation - spearman

color_ = ['#373C9B']*len(sham_mean_SR) + ['#B03D3E']*len(mi_mean_SR)
SR = np.concatenate([sham_mean_SR, mi_mean_SR])
std = np.concatenate([sham_mean*180/np.pi, mi_mean*180/np.pi])

spear, p_s = stats.spearmanr(std, SR).statistic, stats.spearmanr(std, SR).pvalue
print(spear, p_s)

plt.scatter(std, SR, color = color_)
plt.title(f'Spearman correlation, six weeks post-MI (r = {spear.round(3)}, p = {p_s.round(3)})')
plt.xlabel('angle_std_e'); plt.ylabel('GRSRd')
plt.show()

#%% identify outlier mi

