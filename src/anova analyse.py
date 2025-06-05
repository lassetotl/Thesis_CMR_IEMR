# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:23:43 2025

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()

''' 
data = {'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time_point': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'measurement': [10, 12, 15, 8, 9, 11, 13, 16, 18]}
df = pd.DataFrame(data)
print(df)
'''
#subject og dag må sorteres ut fra strings, testen gjentas for hver parameter
df = pd.read_csv('combodata_analysis')

#%%
# legger til en egen kolonne med ID'er

ID = []
for row in range(len(df)):
    ID.append(df['Name'][row].split('_')[1])
df['ID'] = ID
#df = df.set_index('ID')
#%%

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

param = 'c_std'
formula = f'{param} ~ Day + ID'

df_sham = df[df['Condition']==0]
model_sham = ols(formula, data=df_sham).fit()
anova_table_sham = anova_lm(model_sham)
print(f'ANOVA results {param} (sham): \n', anova_table_sham, '\n')
P_sham = anova_table_sham['PR(>F)']['Day']  # P-verdi for endring over dager


df_mi = df[df['Condition']==1]
model_mi = ols(formula, data=df_mi).fit()
anova_table_mi = anova_lm(model_mi)
print(f'ANOVA results {param} (mi): \n', anova_table_mi, '\n')
P_mi = anova_table_mi['PR(>F)']['Day']  # P-verdi for endring over dager

print(f'Endring over tid for {param}: \n Sham: {np.round(P_sham, 3)} \n MI: {np.round(P_mi, 3)}')


plt.figure(figsize=(7, 6), dpi=300)
plt.title(f'repeated measures ANOVA, linear model \n Sham: p = {np.round(P_sham, 3)} \n MI: p = {np.round(P_mi, 3)}')

# paletter, html-koder
mi_palette = ['#852F30', '#9B3637', '#B03D3E', '#C1494A', '#C95D5E', '#D07273', '#D88788', '#DF9C9C', '#E6B1B1']
sham_palette = ['#373C9B', '#3E44B1', '#4B51C1', '#5F64C9', '#7478D0', '#898CD8', '#9DA1DF', '#B3B5E6', '#C8CAEE']

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
    #plt.plot(df_sham[])
    
plt.show()

#%%
#POST HOC analyse
# identifisere for hvilke rotter målingene endres signifikant
#tukey_oneway_sham = pairwise_tukeyhsd(endog = df_sham[param], groups = df_sham['Day'])
#tukey_oneway_mi = pairwise_tukeyhsd(endog = df_mi[param], groups = df_mi['Day'])
