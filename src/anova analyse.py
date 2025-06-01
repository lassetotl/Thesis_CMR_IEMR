# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:23:43 2025

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns; sns.set()

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

param = 'TSd'
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

#%%
plt.figure(figsize=(7, 6), dpi=300)
#plt.title('GRS Regional variation Sham')

# plotte linjer over tid for hvert individ

individer = set(df.index.tolist())
for id_ in individer:
    print(id_)
    days = df[df['ID']==id_]
    plt.plot()
    #plt.plot(df_sham[])