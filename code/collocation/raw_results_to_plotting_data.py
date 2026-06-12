from copy import deepcopy as cp
import pandas as pd
from clspde.utils import plot, eval_dict
from clspde.prepare import from_file, prepare_settings
from clspde.solution import Solution
import itertools
import numpy as np
import copy
import yaml
import pickle as pkl
from random import gauss as random


columns = ['noise', 'n_measurements', 'rel_S', 'rel_I', 'rel_uS', 'rel_uI', 'rel_beta', 'residual', 'residual_S', 'residual_I', 'residual_uS', 'residual_uI',]

df = pd.read_csv('result.csv')
df = df.dropna()
df.columns = columns

for noise in np.unique(df['noise']):
    for n in np.unique(df['n_measurements']):
        df_part = df[df['noise'] == noise]
        df_part = df_part[df_part['n_measurements'] == n]
        df_mean = np.mean(df_part, axis=0)
        std_bet = np.std(df_part['rel_beta'])
        print(noise, n, df_mean['rel_beta'], std_bet)
#new_df
print(df.head())

ni = len(np.unique(df['noise']))
nj = len(np.unique(df['n_measurements']))
res_mean = np.zeros((ni,nj))
res_std = np.zeros((ni,nj))


for i, noise in enumerate(np.unique(df['noise'])):
    for j, n in enumerate(np.unique(df['n_measurements'])):
        df_part = df[df['noise'] == noise]
        df_part = df_part[df_part['n_measurements'] == n]
        df_mean = np.mean(df_part, axis=0)
        std_bet = np.std(df_part['rel_beta'])
        print(noise, n, df_mean['rel_beta'], std_bet)
        
        res_mean[i,j] = df_mean['rel_beta']
        res_std[i,j] = std_bet
     
import matplotlib.pyplot as plt     
plt.plot(np.log(res_mean.T))
plt.show()

plt.plot(np.log(res_std.T))
plt.show()
#new_df

df_old = df
df = pd.DataFrame(res_mean.T)
df.columns = np.unique(df_old['noise'])
df['x'] = 50*np.array([2**i for i in range(len(df.index))])
df.to_csv('rel_means.csv')

df = pd.DataFrame(res_std.T)
df.columns = np.unique(df_old['noise'])
df['x'] = 50*np.array([2**i for i in range(len(df.index))])
df.to_csv('rel_stds.csv')
