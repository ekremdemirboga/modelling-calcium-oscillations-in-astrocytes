from data import data_matrix,data_to_embedding,smooth
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from solver import evolve
from grid_search import param_init2,initial_conditions,param_init
from scipy.optimize import curve_fit


ith_cell=73


def param_init_peak():
    # vm2 = 0.553
    # vm3 = 200
    # v_in = 0.05
    # v_p = 0.05
    # k_2 = 0.1
    # k_CaA = 2.42
    # k_CaI = 0.129
    # k_ip3 = 0.331
    # k_p = 0.069
    # k_deg = 0.260
    # k_out = 0.5
    # k_f = 0.016
    # n = 2.02
    # m = 2.2
    vm2 = 15
    vm3 = 40
    v_in = 0.05
    v_p = 0.05
    k_2 = 0.1
    k_CaA = 0.15
    k_CaI = 0.15
    k_ip3 = 0.1
    k_p = 0.3
    k_deg = 0.08
    k_out = 0.5
    k_f = 0.5
    n = 2.02
    m = 2.2
    params = [vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, 
              k_ip3, k_p, k_deg, k_out, k_f, n, m]
    return np.asarray(params)


def objective_func(time,vm2, vm3, v_in, v_p, k_2, k_CaA, k_CaI, k_ip3, k_p, k_deg, k_out, k_f, n, m):
    parameters = param_init_peak()
    parameters[5] = k_CaA
    parameters[6] = k_CaI
    parameters[0] = vm2
    parameters[1] = vm3
    parameters[8] = k_p
    parameters[11] = k_f
    parameters[9] = k_deg
    parameters[7] = k_ip3
    #
    parameters[2] = v_in 
    parameters[3] = v_p
    parameters[4] = k_2
    parameters[10] = k_out
    parameters[12] = n
    parameters[13] = m

    initial_conditions = [0.1,4.9,0.01]
    t_initial = 0
    t_final = 45
    dt = 1/128
    t,X,_,_ = evolve(t_initial, t_final, initial_conditions, dt, parameters)
    interpolated = CubicSpline(t,X)
    return interpolated(time)

def compare_plots():
    parameters = param_init()
    initial_conditions = [0.1,4.9,0.01]
    t_initial = 0
    t_final = 45
    dt = 1/128
    t,X,_,_ = evolve(t_initial, t_final, initial_conditions, dt, parameters)

    X_data = data_matrix[ith_cell][60:106]
    X_data_smoothed = smooth(X_data,box_pts=2,interpolate=False)
    X_data_interpolated = CubicSpline(np.arange(len(X_data_smoothed)),X_data_smoothed)
    t_data = np.arange(len(X_data))
    plt.plot( t,X,'k--',linewidth=2,label = 'simulation')
    plt.plot( t,X_data_interpolated(t),'k',linewidth=3,label = 'data')
    plt.legend(frameon=True)
    plt.grid(True)
    plt.show()
    

X_data = data_matrix[ith_cell][200:246]
X_data_smoothed = smooth(X_data,box_pts=2,interpolate=False)
X_data_interpolated = CubicSpline(np.arange(len(X_data_smoothed)),X_data_smoothed)
t_data = np.arange(len(X_data))
X_data_interpolated(t_data)[X_data_interpolated(t_data) < 0] = 0

#k_CaI,k_CaA,vm2,vm3,k_p,k_f,k_deg,k_ip3,                               v_in,v_p,k_2,k_out,n,m
initial_guess = param_init_peak()#[2.42,0.129,15,40,0.3,0.5,0.08,0.1,0.05,0.05,0.1,0.5,2.02,2.2]

#k_CaI,k_CaA,vm2,vm3,k_p,k_f,k_deg,k_ip3,                               v_in,v_p,k_2,k_out,n,m
#                vm2, vm3, v_in, v_p, k_2,k_CaA,k_CaI,k_ip3,k_p,k_deg, k_out, k_f,   n, m
upper = np.array([16, 220, 0.15, 0.15, 0.5 , 3.0, 0.5, 0.5,   0.4 ,0.4  ,0.8   ,0.6   ,2.2 ,2.4])
lower = np.array([0.1, 35,0.01, 0.01, 0.01, 0.05,0.05,0.05,   0.01,0.01,0.05,  0.01  ,1.9,2.0])


# upper = np.array([0.35, 3.0, 16, 200, 0.5, 1, 0.3, 0.5,  0.15,0.15,0.5,1,3,3])
# lower = np.array([0.05, 0.05, 0.1, 35, 0.01,0.01,0.05,0.05,   0.01,0.01,0.05,0.1,1.5,1.5])

bnds = (lower,upper)
popt, pcov = curve_fit(objective_func, t_data, X_data_interpolated(t_data),p0 =initial_guess,check_finite=False,bounds=bnds)
#                vm2, vm3, v_in, v_p, k_2,k_CaA,k_CaI,k_ip3,k_p,k_deg, k_out, k_f,   n, m
plt.plot(t_data, objective_func(t_data, *popt), 'r--',
         label='fit: vm2=%5.3f, vm3=%5.3f, v_in=%5.3f,v_p=%5.3f, \n k_2=%5.3f, kCaA=%5.3f, kCaI=%5.3f,  kip3=%5.3f,\n k_p=%5.3f, kdeg=%5.3f, kout=%5.3f,  kf=%5.3f,\n n=%5.3f,  m=%5.3f,' % tuple(popt))
plt.plot( t_data,X_data_interpolated(t_data),'k',linewidth=3,label = 'data')
plt.grid(True)
plt.legend(fancybox=True)
plt.show()