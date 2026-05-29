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
import matplotlib.patches as mpatches
from ComboDataSR_2D import ComboDataSR_2D
from scipy.integrate import cumulative_trapezoid
from scipy import stats
from util import drop_outliers_IQR
import pandas 
import seaborn as sns; sns.set()


import pandas as pd

from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from scipy.stats import pearsonr, linregress, spearmanr
import pingouin as pg

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
df_sham = df[df['Condition'] == 0]
df_mi = df[df['Condition'] == 1]

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
# table of (mean +- std) for each parameter in df, grouped by condition

column = 'angle_std_s'
df_ = df[df['Day'] >= 40].groupby(['Condition'], as_index = False).agg({column:['mean', 'std']})
df__ = df[df['Day'] == 1].groupby(['Condition'], as_index = False).agg({column:['mean', 'std']})

print(f'Day 1: {df__.round(2)}')
print(f'Day 40+: {df_.round(2)}')

#%%
# chronic sham vs mi, mean, std, pval

column = 'angle_std_s'
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
# fix std_e _s units

df_mi_40['std_s_reg'] = df_mi_40['std_s_reg']*180/np.pi
df_sham_40['std_s_reg'] = df_sham_40['std_s_reg']*180/np.pi

df_mi_40['std_e_reg'] = df_mi_40['std_e_reg']*180/np.pi
df_sham_40['std_e_reg'] = df_sham_40['std_e_reg']*180/np.pi

#%%
# box plot MI hearts regional variation (only works with fresh dataframe ??)
# bug: c_reg and r_reg keys turn from list into strings when loading df?
# c_reg or r_reg or TSd_reg or TSs_reg or TCd_reg or TCs_reg or std_s_reg or std_e_reg or 
# GCSRs_reg or TSs_reg_mod (etc)
column = 'std_e_reg'

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
    if (HB_sham.count('X') == 0) and (pvals_sham[i] <= p_corr):
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
    print(p_corr, pvals_mi[i])
    if (HB_mi.count('X') == 0) and (pvals_mi[i] <= p_corr):
        HB_mi += 'V'
    else: 
        HB_mi += 'X'

# line-plot individual animals (each column is one animal)
all_points_mi = np.array([infarct, adjacent, medial, remote])


plt.figure(figsize=(6, 2.5), dpi=300)
#plt.title('GRS Regional variation MI')

ax1=sns.stripplot(data = [g1, g2, g3, g4, infarct, adjacent, medial, remote], size=5, \
            palette = [c_cmap(0), c_cmap(1), c_cmap(2), c_cmap(3)]*2, alpha = 1, jitter=False)

for i in range(len(g1)):
    ax1.plot(list(range(4)), all_points_sham[:,i], color='gray', zorder=1, alpha=0.5, lw=0.9)
    
for i in range(len(infarct)):
    ax1.plot(list(range(4,8)), all_points_mi[:,i], color='gray', zorder=1, alpha=0.5, lw=0.9)


ax2=sns.pointplot(data = [g1, g2, g3, g4, infarct, adjacent, medial, remote], errorbar=None, marker='_', \
              markersize=17, markeredgewidth=2, color='k', join=False)
    
plt.setp(ax1.collections, zorder=2)
plt.setp(ax2.collections, zorder=2)
plt.yticks(fontsize=7.5)
    
# uncomment to include p values relative to infarct
#plt.xticks([0, 1, 2, 3], ['Infarct', f'Adjacent \n ($p =${np.round(pa, 3)})', \
#                          f'Medial \n ($p =${np.round(pm, 3)})', f'Remote \n ($p =${np.round(pr, 3)})'])
    
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Infarct', 'Adjacent', 'Medial', 'Remote'], size=7.5)
#plt.scatter([0]*len(infarct), infarct, color = 'darkred', s = 40)
#plt.scatter([1]*len(adjacent), adjacent, color = 'darkgreen', s = 40)
#plt.scatter([2]*len(medial), medial, color = 'darkblue', s = 40)
#plt.scatter([3]*len(remote), remote, color = 'indigo', s = 40)

ymax = max([ax1.get_ylim()[1], ax2.get_ylim()[1]])
ymin = min([ax1.get_ylim()[0], ax2.get_ylim()[0]])

#plt.ylim(ymin, ymax + abs(max([abs(ymax), abs(ymin)]))*0.4)
plt.gca().set_ylim(top = ymax + abs(max([abs(ymax), abs(ymin)]))*0.3)
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
std = np.concatenate([sham_mean, mi_mean])

spear, p_s = stats.spearmanr(std, SR).statistic, stats.spearmanr(std, SR).pvalue
print(spear, p_s)

plt.figure(dpi=300)
plt.scatter(std, SR, color = color_)
plt.title(f'Spearman correlation, six weeks post-MI')
plt.xlabel('angle_std_e'); plt.ylabel('GRSRd')
empty_patch = mpatches.Patch(color='none', label = f'r = {spear.round(3)}, p = {p_s.round(3)}') 
plt.legend(handles=[empty_patch])
plt.show()

#%% dataframe where with viable means instead of global
# if reg parameters dont work, create a fresh df instead of import ("strain_analysis.py")

std_s_v = []; std_e_v = []
GCSRs_v = []; GCSRd_v = []
GRSRs_v = []; GRSRd_v = []
for i in range(len(df['std_e_reg'])):
    std_s_v.append(np.mean(df['std_s_reg'][i][1:4])*180/np.pi)
    std_e_v.append(np.mean(df['std_e_reg'][i][1:4])*180/np.pi)
    GCSRs_v.append(np.mean(df['GCSRs_reg'][i][1:4]))
    GCSRd_v.append(np.mean(df['GCSRd_reg'][i][1:4]))
    GRSRs_v.append(np.mean(df['GRSRs_reg'][i][1:4]))
    GRSRd_v.append(np.mean(df['GRSRd_reg'][i][1:4]))
    
df['std_s_v'] = std_s_v
df['std_e_v'] = std_e_v
df['GCSRs_v'] = GCSRs_v
df['GCSRd_v'] = GCSRd_v
df['GRSRs_v'] = GRSRs_v
df['GRSRd_v'] = GRSRd_v 

#%%

# legger til en egen kolonne med IDer
ID = []
for row in range(len(df)):
    ID.append(df['Name'][row].split('_')[1])
df['ID'] = ID
#df = df.set_index('ID')

# create another column of integers mapped to IDs
mapping = {item:i for i, item in enumerate(df['ID'].unique())}
df['ID_int'] = df['ID'].apply(lambda x: mapping[x])

#%% correlation heatmap

df_num = df.copy()
df_num = df_num.drop(columns = ['Name', 'std_s_reg', 'std_e_reg', 'GCSRs_reg', 'GCSRd_reg', 'GRSRs_reg', 't_peak_diff_s',
't_peak_diff_e', 'GRSRd_reg', 'r_std', 'c_std', 'r_reg', 'c_reg', 'Rad SDI', 'Circ SDI', 'TSd', 'TSs', 'TCs', 'TCd'])

corr = df_num.corr(method='pearson')
mask = np.triu(corr)  # diagonal + upper triangle redundant
fig = plt.figure(figsize=(14,12))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, norm='linear', annot_kws={'size':14}, fmt='.2f')
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
fig.get_axes()[1].remove()
plt.show()

#%%
# paletter, html-koder
mi_palette = ['#852F30', '#9B3637', '#B03D3E', '#C1494A', '#C95D5E', '#D07273', '#D88788', '#DF9C9C', '#E6B1B1']
sham_palette = ['#373C9B', '#3E44B1', '#4B51C1', '#5F64C9', '#7478D0', '#898CD8', '#9DA1DF', '#B3B5E6', '#C8CAEE']

# mi x7, sham x6
palette_ = mi_palette[:6] + sham_palette[:6]
markers_ = ['v']*6 + ['o']*6

#%% correlation plot (use pg.rm_corr)

param = ['angle_std_s', 'GRSRs']

df_num_s = df[df['Condition']==0]
df_num_mi = df[df['Condition']==1]

x_s = df_num_s[param[0]]; y_s = df_num_s[param[1]]
mask = ~np.isnan(x_s) & ~np.isnan(y_s)
x_s = x_s[mask]; y_s = y_s[mask]

a_s, b_s = linregress(x_s, y_s)[:2]
pear_s, p_s = pearsonr(x_s, y_s).statistic, pearsonr(x_s, y_s).pvalue

x_mi = df_num_mi[param[0]]; y_mi = df_num_mi[param[1]]
mask = ~np.isnan(x_mi) & ~np.isnan(y_mi)
x_mi = x_mi[mask]; y_mi = y_mi[mask]

a_mi, b_mi = linregress(x_mi, y_mi)[:2]
pear_mi, p_mi = pearsonr(x_mi, y_mi).statistic, pearsonr(x_mi, y_mi).pvalue

plt.scatter(x_s, y_s, c='b', label = f'Sham, pearson {round(pear_s, 3)} (p = {round(p_s, 3)})')
plt.scatter(x_mi, y_mi, c='r', label = f'MI, pearson {round(pear_mi, 3)} (p = {round(p_mi, 3)})')

x = np.linspace(min([min(x_mi), min(x_s)]), max([max(x_mi), min(x_s)]), 1000)
plt.plot(x, a_mi*x + b_mi, 'r')
plt.plot(x, a_s*x + b_s, 'b')

plt.xlabel(param[0]); plt.ylabel(param[1])
plt.legend(); plt.show()

#repeated measures correlation

rm_corr = pg.rm_corr(data=df, x=param[0], y=param[1], subject='ID')
r = rm_corr['r'].iloc[0]
p = rm_corr['pval'].iloc[0]
dof = rm_corr['dof'].iloc[0]
ci_lower, ci_upper = rm_corr['CI95%'].iloc[0]
power = rm_corr['power'].iloc[0]
if p < 0.001:
    p = '< 0.001'
else:
    p = f'= {p.round(3)}'

fig = pg.plot_rm_corr(data=df, x=param[0], y=param[1], subject='ID', \
                      kwargs_facetgrid={'aspect': 1.2, 'height': 4.5, 'palette':palette_},\
                          kwargs_scatter={'edgecolors':'None'})
    
plt.title(f'RM correlation (r = {r.round(3)}, p {p})')
plt.show()

#%% spearman 6w
param = ['angle_std_s', 'GCSRs']
df_num_6w = df[df['Day']>=40]


x = df_num_6w[param[0]]
y = df_num_6w[param[1]]
spear, p_s = spearmanr(x, y).statistic, spearmanr(x, y).pvalue
print(spear, p_s)

plt.scatter(x, y, color = palette_)
plt.title(f'Spearman correlation @ 40+ days (r = {spear.round(3)}, p = {p_s.round(3)})')
plt.xlabel(param[0]); plt.ylabel(param[1])
plt.show()

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
#%% mixed linear models longitudinal data

param = 'angle_std_e'
formula = f'{param} ~ Day + ID'

#filter outliers TSs and TCs (only apply for those measurements!!)
#df = df[(df['TSs_mod'] < 50) & (df['TCs_mod'] < 50)]

df_sham = df[df['Condition']==0]
df_sham = df_sham.dropna()

df_mi = df[df['Condition']==1]
df_mi = df_mi.dropna()


#mixed linear model
md = mixedlm(f'{param} ~ Day', data=df_sham, groups=df_sham["ID"])
mdf = md.fit()
print(mdf.summary())

md_mi = mixedlm(f'{param} ~ Day', data=df_mi, groups=df_mi["ID"])
mdf_mi = md_mi.fit()
print(mdf_mi.summary())

# plotte linjer over tid for hvert individ
individer = set(df['ID'])
mi_i = 0; sham_i = 0
for id_ in individer:
    #print(id_)
    days = list(df[df['ID']==id_]['Day'])
    param_ = list(df[df['ID']==id_][param])
    days_param_zip = sorted(zip(days, param_))
    days_sorted, param_sorted = zip(*days_param_zip)
    #print(days_sorted, param_sorted)
    if df[df['ID']==id_]['Condition'].any()==1:
        color = mi_palette[mi_i]
        mi_i += 1
        marker = 'v'
    else:
        color = sham_palette[sham_i]
        sham_i += 1
        marker = 'o'
    plt.plot(days_sorted, param_sorted, c=color, marker=marker)

plt.xlabel('Days'); plt.ylabel(param)
#plt.ylabel(r'GC-SRs [$s^{-1}$]')
#plt.ylabel(r'$\theta_{compression, diastole} \ [^{\circ}]$', fontsize = 15)
#plt.legend(handles=legend_handles1, prop={'size': 12})
#plt.legend(handles=legend_handles1, prop={'size': 12}, loc='upper right', bbox_to_anchor=(0.99, 0.8))
plt.show()
