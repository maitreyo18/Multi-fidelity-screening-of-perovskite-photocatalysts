import numpy as np
import csv
import copy
import random
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import sklearn
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.metrics import mean_squared_error as mse
from rgf.sklearn import RGFRegressor

import warnings
warnings.filterwarnings("ignore")


# Band-gap predictor

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./Datasets/PBE_and_HSE_mf1.csv')

All_desc = pd.DataFrame(df, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'Cubic', 'Tetra', 'Ortho', 'Hex', 'PBE', 'HSE', 'A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
decomp = df['Decomposition Energy']

X = All_desc.copy()
y = decomp.copy()


# Enumerated data

outside = pd.read_csv('./Datasets/Tol_screened.csv')



cols_original = ['Formula','Pred_decomp_sum', 'Pred_decomp_avg']
decomp_original = pd.DataFrame(0, index=np.arange(len(df.index)), columns=cols_original)
decomp_original['Formula'] = df.Name


outside_cols = ["Formula", "Decomp(HSE)_sum", "Decomp(HSE)_avg", "Decomp(PBE)_sum", "Decomp(PBE)_avg"]
outside_df = pd.DataFrame(0, index=np.arange(len(outside.index)), columns=outside_cols)
outside_df['Formula'] = outside.Formula



# Creating an ensemble of 4000 models
# Data split
n_iter = 4000

for i in range(n_iter):
    print("Iteration: ", i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    #Define RGF Regressor

    rgf_reg = RGFRegressor(loss ='LS',
                           algorithm ='RGF',
                           l2 = 0.1,
                           learning_rate = 0.1,
                           n_iter = 20,
                           sl2 = 0.2,
                           memory_policy = 'generous',
                           min_samples_leaf = 25,
                           max_leaf = 1000,
                        )
    param_grid = {
    #    "learning_rate": [0.05, 0.1],
    #    "max_leaf": [500, 800, 1000],
    #    "min_samples_leaf": [5, 15, 25],
    #    "n_iter": [5, 10, 20],
     }
    param_grid = {
    #    "learning_rate": [0.05, 0.1],
    #    "max_leaf": [500, 800, 1000],
    #    "min_samples_leaf": [5, 15, 25],
    #    "n_iter": [5, 10, 20],
        }

    rgf_decomp = GridSearchCV(rgf_reg, param_grid=param_grid, cv=5)
    rgf_decomp.fit(X_train, y_train)

    # Dataset values
    Pred_decomp_org = rgf_decomp.predict(X)

    decomp_original['Pred_decomp_sum'] += Pred_decomp_org


    # Enumerated data
    enum_predict_hse =  pd.DataFrame(outside, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'Cubic', 'Tetra', 'Ortho', 'Hex', 'PBE', 'HSE', 'A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
    enum_predict_hse['PBE'] = 0
    enum_predict_hse['HSE'] = 1


    enum_predict_pbe = pd.DataFrame(outside, columns=['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'Cl', 'Br', 'I', 'Cubic', 'Tetra', 'Ortho', 'Hex', 'PBE', 'HSE', 'A_ion_rad', 'A_BP', 'A_MP', 'A_dens', 'A_at_wt', 'A_EA', 'A_IE', 'A_hof', 'A_hov', 'A_En', 'A_at_num', 'A_period', 'B_ion_rad', 'B_BP', 'B_MP', 'B_dens', 'B_at_wt', 'B_EA', 'B_IE', 'B_hof', 'B_hov', 'B_En', 'B_at_num', 'B_period', 'X_ion_rad', 'X_BP', 'X_MP', 'X_dens', 'X_at_wt', 'X_EA', 'X_IE', 'X_hof', 'X_hov', 'X_En', 'X_at_num', 'X_period'])
    enum_predict_pbe['PBE'] = 1
    enum_predict_pbe['HSE'] = 0


    Pred_decomp_hse = rgf_decomp.predict(enum_predict_hse)
    Pred_decomp_pbe = rgf_decomp.predict(enum_predict_pbe)

    outside_df['Decomp(HSE)_sum'] += Pred_decomp_hse
    outside_df['Decomp(PBE)_sum'] += Pred_decomp_pbe


decomp_original['Pred_decomp_avg'] = decomp_original['Pred_decomp_sum']/n_iter
outside_df['Decomp(HSE)_avg'] = outside_df['Decomp(HSE)_sum']/n_iter
outside_df['Decomp(PBE)_avg'] = outside_df['Decomp(PBE)_sum']/n_iter

decomp_original.to_csv("./Decomp_original_4000.csv")
outside_df.to_csv("./Screened_decomp_4000.csv")

print(decomp_original)
print(outside_df)
