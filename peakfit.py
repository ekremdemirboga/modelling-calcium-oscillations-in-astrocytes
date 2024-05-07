import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from data import data_matrix,data_to_embedding
from grid_search import param_init2,initial_conditions
from solver import evolve
def func(time,k_CaI):
    parameters = param_init2()
    parameters[5] = k_CaI
    # parameters[6] = k_CaA
    R = initial_conditions
    #simulation parameters
    t_initial = 1
    t_final = 600
    dt = 1/128
    #main part
    t,X,_,_ = evolve(t_initial, t_final, R, dt, parameters)
    np.interp(time,t, X)
    return np.interp(time,t, X)
ith_cell=71

X_data = data_matrix[ith_cell]
t_data = range(len(X_data))

popt, pcov = curve_fit(func, t_data, X_data)

plt.plot(t_data, func(t_data, *popt), 'r-')