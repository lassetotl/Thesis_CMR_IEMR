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
    
    r_strain_peak_std = np.std(run.__dict__['r_peakvals'])
    c_strain_peak_std = np.std(run.__dict__['c_peakvals'])
    
    # collect regional strain peaks
    
    # index order - infarct, adjacent, medial, remote
    r_strain_reg = run.__dict__['r_peakvals']
    c_strain_reg = run.__dict__['c_peakvals']
    
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
                            a2_mean_max, a2_mean_min, r_strain_peak_std, c_strain_peak_std, \
                                r_strain_reg, c_strain_reg, condition])
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
        auc_mi.append([days, sum(cumulative_trapezoid(diff[:u]))])
    else:
        plt.plot(range_TR[:len(diff)], diff, 'k', lw=1.3)
        auc_sham.append([days, sum(cumulative_trapezoid(diff[:u]))])

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
#'''
df = pandas.DataFrame(df_list, columns=['Name', 'Day', 'GRS', 'GCS', \
                                        'Rad SDI', 'Circ SDI', 'GRSRs', \
                                            'GRSRd', 'GCSRd', 'GCSRs', \
                                                'TSd', 'TSs', 'TCs', 'TCd', \
                                                    'r_std', 'c_std', 'r_reg', 'c_reg', \
                                                        'Condition']) 
#'''
# to analyze a generated csv file instead
#df = pandas.read_csv('combodata_analysis')
    
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

#%%
# table of (mean +- std) for each parameter in df, grouped by condition

column = 'TSs'
df_ = df[df['Day'] >= 40].groupby(['Condition'], as_index = False).agg({column:[np.mean, np.std]})
df__ = df[df['Day'] == 1].groupby(['Condition'], as_index = False).agg({column:[np.mean, np.std]})

print(f'Day 1: {df__.round(2)}')
print(f'Day 40+: {df_.round(2)}')

#%%
# box plot MI hearts regional variation
# bug: c_reg and r_reg keys turn from list into strings when loading df?


df_mi_1 = df_mi[df_mi['Day'] == 1]
df_mi_40 = df_mi[df_mi['Day'] >= 40]  # chronic stage MI
df_sham_40 = df_sham[df_sham['Day'] >= 40]  # chronic stage MI

# c_reg or r_reg
column = 'r_reg'

# Sham

g1 = []
g2 = []
g3 = []
g4 = []

# c_reg or r_reg
for key, value in df_sham_40[column].iteritems():
    g1.append(value[0])  
    g2.append(value[1])  
    g3.append(value[2])  
    g4.append(value[3])  

# regional colormap
c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())
norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)

# p values compared with infarct

pa = stats.ttest_ind(g1, g2)[1]
pm = stats.ttest_ind(g1, g3)[1]
pr = stats.ttest_ind(g1, g4)[1]
print(pa,pm,pr)


# scatter/violin plot MI regional variation
plt.figure(figsize=(7, 6), dpi=300)
#plt.title('GRS Regional variation Sham')
sns.barplot(data = [g1, g2, g3, g4], ci='sd', capsize=.4, \
            palette = [c_cmap(0), c_cmap(1), c_cmap(2), c_cmap(3)], errwidth = 1.4)

#plt.xticks([0, 1, 2, 3], ['Sector 1', f'Sector 2 \n ($p =${np.round(pa, 3)})', \
#                          f'Sector 3 \n ($p =${np.round(pm, 3)})', f'Sector 4 \n ($p =${np.round(pr, 3)})'])
    
plt.xticks([0, 1, 2, 3], ['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4'])
                          
plt.scatter([0]*len(df_sham_40[column]), g1, color = 'darkred', s = 40)
plt.scatter([1]*len(df_sham_40[column]), g2, color = 'darkgreen', s = 40)
plt.scatter([2]*len(df_sham_40[column]), g3, color = 'darkblue', s = 40)
plt.scatter([3]*len(df_sham_40[column]), g4, color = 'indigo', s = 40)

if column == 'c_reg':
    plt.ylabel('GCS [%]', fontsize = 17)
else:
    plt.ylabel('GRS [%]', fontsize = 17)

plt.ylim(ymin, ymax)

#ymin = plt.axis()[2]
#ymax = plt.axis()[3]

plt.show()

#%%

# MI

infarct = []
adjacent = []
medial = []
remote = []

for key, value in df_mi_40[column].iteritems():
    infarct.append(value[0])  
    adjacent.append(value[1])  
    medial.append(value[2])  
    remote.append(value[3])  

# regional colormap
c_cmap = mpl.colors.ListedColormap(sns.color_palette('hls', 4).as_hex())
norm_ = mpl.colors.Normalize(vmin = 1, vmax = 4)

# p values compared with infarct

pa = stats.ttest_ind(infarct, adjacent)[1]
pm = stats.ttest_ind(infarct, medial)[1]
pr = stats.ttest_ind(infarct, remote)[1]
print(pa,pm,pr)

# scatter/violin plot MI regional variation
'''
plt.figure(figsize=(6, 5))
plt.title('GCS Regional variation MI')

plt.scatter([0]*len(df_mi_40[column]), infarct, color = c_cmap(0))
plt.scatter([1]*len(df_mi_40[column]), adjacent, color = c_cmap(1))
plt.scatter([2]*len(df_mi_40[column]), medial, color = c_cmap(2))
plt.scatter([3]*len(df_mi_40[column]), remote, color = c_cmap(3))

plt.xticks([0, 1, 2, 3], ['Infarct', 'Adjacent', 'Medial', 'Remote'])
plt.ylabel('%', fontsize = 17)

plt.show()

'''

plt.figure(figsize=(7, 6), dpi=300)
#plt.title('GRS Regional variation MI')
sns.barplot(data = [infarct, adjacent, medial, remote], ci='sd', capsize=.4, \
            palette = [c_cmap(0), c_cmap(1), c_cmap(2), c_cmap(3)], errwidth = 1.4)

# uncomment to include p values relative to infarct
#plt.xticks([0, 1, 2, 3], ['Infarct', f'Adjacent \n ($p =${np.round(pa, 3)})', \
#                          f'Medial \n ($p =${np.round(pm, 3)})', f'Remote \n ($p =${np.round(pr, 3)})'])
    
plt.xticks([0, 1, 2, 3], ['Infarct', 'Adjacent', 'Medial', 'Remote'])
plt.scatter([0]*len(df_mi_40[column]), infarct, color = 'darkred', s = 40)
plt.scatter([1]*len(df_mi_40[column]), adjacent, color = 'darkgreen', s = 40)
plt.scatter([2]*len(df_mi_40[column]), medial, color = 'darkblue', s = 40)
plt.scatter([3]*len(df_mi_40[column]), remote, color = 'indigo', s = 40)

if column == 'c_reg':
    plt.ylabel('GCS [%]', fontsize = 17)
else:
    plt.ylabel('GRS [%]', fontsize = 17)

#plt.ylim(ymin, ymax)

ymin = plt.axis()[2]
ymax = plt.axis()[3]

plt.show()