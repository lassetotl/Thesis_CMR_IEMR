# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:23:43 2025

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from matplotlib.lines import Line2D
import pingouin as pg
from scipy.stats import pearsonr, linregress, spearmanr

''' 
data = {'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time_point': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'measurement': [10, 12, 15, 8, 9, 11, 13, 16, 18]}
df = pd.DataFrame(data)
print(df)
'''
#subject og dag må sorteres ut fra strings, testen gjentas for hver parameter
df = pd.read_csv('combodata_analysis_des_2025')

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
# legger til en egen kolonne med ID'er

ID = []
for row in range(len(df)):
    ID.append(df['Name'][row].split('_')[1])
df['ID'] = ID
#df = df.set_index('ID')

# create another column of integers mapped to IDs
mapping = {item:i for i, item in enumerate(df['ID'].unique())}
df['ID_int'] = df['ID'].apply(lambda x: mapping[x])

#df.dropna() # angle std values are not nan



#%%
# paletter, html-koder
mi_palette = ['#852F30', '#9B3637', '#B03D3E', '#C1494A', '#C95D5E', '#D07273', '#D88788', '#DF9C9C', '#E6B1B1']
sham_palette = ['#373C9B', '#3E44B1', '#4B51C1', '#5F64C9', '#7478D0', '#898CD8', '#9DA1DF', '#B3B5E6', '#C8CAEE']

# mi x7, sham x6
palette_ = mi_palette[:7] + sham_palette[:6]
markers_ = ['v']*7 + ['o']*6

#%% correlation plot (use pg.rm_corr)

param = ['std_s_v', 'GCSRs_v']

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
param = ['angle_std_s', 'GCSRd']
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
#%% lm ANOVA

param = 'TCd_mod'
formula = f'{param} ~ Day + ID'

df_sham = df[df['Condition']==0]
df_sham = df_sham.dropna()

'''
model_sham = ols(formula, data=df_sham).fit()
#print(model_sham.summary())
slope_sham = model_sham.params.iloc[-1] # indexing to exclude intercept and Day
std_sham = model_sham.bse.iloc[-1]
print(f'OLS regression, sham: ({slope_sham.round(3)} \pm {std_sham.round(3)})')

anova_table_sham = anova_lm(model_sham)
print(f'ANOVA results {param} (sham): \n', anova_table_sham, '\n')
P_sham = anova_table_sham['PR(>F)']['Day']  # P-verdi for endring over dager

'''
df_mi = df[df['Condition']==1]
df_mi = df_mi.dropna()
'''
model_mi = ols(formula, data=df_mi).fit()
anova_table_mi = anova_lm(model_mi)
#print(model_mi.summary())
slope_mi = model_mi.params.iloc[-1] # indexing to exclude intercept and Day
std_mi = model_mi.bse.iloc[-1]
print(f'OLS regression, mi: ({slope_mi.round(3)} \pm {std_mi.round(3)})')

print(f'ANOVA results {param} (mi): \n', anova_table_mi, '\n')
P_mi = anova_table_mi['PR(>F)']['Day']  # P-verdi for endring over dager

print(f'Endring over tid for {param}: \n Sham: {np.round(P_sham, 3)} \n MI: {np.round(P_mi, 3)}')


f = plt.figure(figsize=(6, 5), dpi=200)
#plt.title('Repeated measures ANOVA, OLS linear model')


# labels
legend_handles1 = [Line2D([0], [0], color = sham_palette[1], lw = 2, label = fr'$\beta_1$ = {np.round(slope_sham, 3)}, p = {np.round(P_sham, 3)}', marker = 'o'),
          Line2D([0], [0], color = mi_palette[1], lw = 2, label = fr'$\beta_1$ = {np.round(slope_mi,3)}, p = {np.round(P_mi, 3)}', marker = 'v')]
'''
'''
# RM ANOVA
#df_sham input
rm_anova_sham = AnovaRM(data=df_sham, depvar=param, subject='ID', within=['Day'])
rm_results_sham = rm_anova_sham.fit()
print(rm_results_sham.anova_table)
'''

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

#%%
# pingouin - sphericity corrected RM ANOVA
'''
spher, W, chisq, dof, pval = pg.sphericity(df_sham, dv=param, within='Day', subject = 'ID_int')
print('Sphericity, df_sham:', spher, round(W, 3), round(chisq, 3), dof, round(pval, 3))

spher, W, chisq, dof, pval = pg.sphericity(df_mi)
print('Sphericity, df_mi:', spher, round(W, 3), round(chisq, 3), dof, round(pval, 3))
'''
pd.options.display.max_columns = 90
pg.rm_anova(df_mi)

#print(pg.rm_anova(data = df_mi, dv = param, within = 'Day', subject = 'ID_int', correction = 'GG'))